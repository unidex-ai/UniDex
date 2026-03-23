import os
import cv2
import torch
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import glob
from typing import Optional, Dict

# --- Import WiLoR, SAM2, and Utils ---
from torchvision.ops import nms
from ultralytics import YOLO
from wilor.models import load_wilor
from wilor.utils import recursive_to
from wilor.datasets.vitdet_dataset import ViTDetDataset
from wilor.utils.renderer import cam_crop_to_full
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# --- Configuration ---
MODEL_PATHS = {
    'sam2_checkpoint': 'pretrained/sam2/sam2.1_hiera_large.pt',
    'sam2_config': 'sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml',
    'wilor_checkpoint': 'pretrained/WiLoR/wilor_final.ckpt',
    'wilor_config': 'pretrained/WiLoR/model_config.yaml',
    'yolo_checkpoint': 'pretrained/WiLoR/detector.pt'
}

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Helper Functions from Source ---
def project_full_img(points, cam_trans, focal_length, img_res): 
    camera_center = [img_res[0] / 2., img_res[1] / 2.]
    K = np.eye(3)
    K[0,0] = focal_length
    K[1,1] = focal_length
    K[0,2] = camera_center[0]
    K[1,2] = camera_center[1]
    points = points + cam_trans
    points = points / points[..., -1:] 
    V_2d = (K @ points.T).T 
    return V_2d[..., :-1]

class HandMaskGenerator:
    def __init__(self, model_paths):
        print(f"Loading models on {DEVICE}...")
        self.device = torch.device(DEVICE)
        
        # 1. Setup SAM2
        self.sam2_model = build_sam2(model_paths['sam2_config'], model_paths['sam2_checkpoint'], device=self.device)
        self.predictor = SAM2ImagePredictor(self.sam2_model)
        
        # 2. Setup WiLoR
        self.wilor_model, self.wilor_cfg = load_wilor(model_paths['wilor_checkpoint'], model_paths['wilor_config'])
        self.wilor_model = self.wilor_model.to(self.device)
        self.wilor_model.eval()
        
        # 3. Setup YOLO
        self.detector = YOLO(model_paths['yolo_checkpoint'])
        self.detector = self.detector.to(self.device)
        
        # Set precision
        if self.device.type == "cuda":
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

    def generate_mask(self, color_image):
        """
        Returns a mask: 255 (non-zero) for background, 0 for hand.
        """
        height, width = color_image.shape[:2]
        
        # 1. Detect Hands (YOLO)
        detections = self.detector(color_image, conf=0.3, verbose=False)[0]
        bboxes, scores, is_right = [], [], []
        
        for det in detections:
            bboxes.append(det.boxes.data.cpu().detach().squeeze().numpy()[:4].tolist())
            scores.append(det.boxes.conf.cpu().detach().squeeze().item())
            is_right.append(det.boxes.cls.cpu().detach().squeeze().item())

        # Initialize "No Hand" mask (All 255)
        full_mask = np.ones((height, width), dtype=np.uint8) * 255
        
        if len(bboxes) == 0:
            return full_mask

        # 2. NMS
        boxes_tensor = torch.tensor(bboxes, dtype=torch.float32)
        scores_tensor = torch.tensor(scores, dtype=torch.float32)
        keep = nms(boxes_tensor, scores_tensor, iou_threshold=0.3)
        
        boxes = boxes_tensor[keep].numpy()
        right = np.array(is_right)[keep.numpy()]
        
        # 3. WiLoR Inference
        dataset = ViTDetDataset(self.wilor_cfg, color_image, boxes, right, rescale_factor=2.0) # Check rescale factor if needed
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=len(boxes), shuffle=False, num_workers=0)
        
        input_points_batch = []
        
        for batch in dataloader:
            batch = recursive_to(batch, self.device)
            with torch.no_grad():
                out = self.wilor_model(batch)
            
            # Project 3D joints to 2D
            multiplier = (2 * batch['right'] - 1)
            pred_cam = out['pred_cam']
            pred_cam[:, 1] = multiplier * pred_cam[:, 1]
            
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            
            scaled_focal_length = self.wilor_cfg.EXTRA.FOCAL_LENGTH / self.wilor_cfg.MODEL.IMAGE_SIZE * img_size.max()
            pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()
            
            batch_size = batch['img'].shape[0]
            for n in range(batch_size):
                verts = out['pred_vertices'][n].detach().cpu().numpy()
                is_r_val = batch['right'][n].cpu().numpy()
                verts[:, 0] = (2 * is_r_val - 1) * verts[:, 0]
                
                cam_t = pred_cam_t_full[n]
                kpts_2d = project_full_img(verts, cam_t, scaled_focal_length, img_size[n])
                
                # Logic from source: Pick points 0 (wrist) and 778 (index tip?)
                # Check shapes to be safe
                if kpts_2d.shape[0] > 778:
                    pts = np.vstack([kpts_2d[0], kpts_2d[778]])
                else:
                    pts = kpts_2d[0].reshape(1, 2)
                input_points_batch.append(pts)

        # 4. SAM2 Segmentation
        if not input_points_batch:
            return full_mask
            
        self.predictor.set_image(color_image)
        
        # Combine masks from all detected hands
        combined_hand_mask = np.zeros((height, width), dtype=bool)

        for points in input_points_batch:
            input_point = np.array(points, dtype=np.float32)
            input_label = np.ones(input_point.shape[0], dtype=int)
            
            masks, scores, logits = self.predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True,
            )
            # Take best mask
            sorted_ind = np.argsort(scores)[::-1]
            best_mask = masks[sorted_ind][0].astype(bool)
            
            # Combine
            combined_hand_mask = np.logical_or(combined_hand_mask, best_mask)
            
        # 5. Final Formatting
        # Request: 1 (Non zero) for no hand, 0 for hand region
        # combined_hand_mask is True for Hand.
        # So where combined_hand_mask is True -> 0
        # Where combined_hand_mask is False -> 255
        full_mask = np.where(combined_hand_mask, 0, 255).astype(np.uint8)
        
        return full_mask

# --- Original Taco Processing Logic ---

def extract_video_frames(video_path, ext='jpg', is_depth=False):
    if not os.path.exists(video_path):
        return None # Return None if skipped
    
    # Logic adjustment: Ensure we handle path objects correctly
    video_path = str(video_path)
    output_dir = os.path.join(os.path.dirname(video_path), 'depthframes' if is_depth else 'colorframes')

    # Basic check to skip existing
    if os.path.exists(output_dir) and len(os.listdir(output_dir)) > 0:
        # print(f"Skipping {video_path}, output directory not empty: {output_dir}")
        return output_dir

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if width != 1920 or height != 1080:
        print(f"Skipping {video_path}, unexpected resolution: {width}x{height}")
        cap.release()
        return None

    os.makedirs(output_dir, exist_ok=True)
    
    if is_depth:
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    for i in tqdm(range(frame_count), desc=f"Extracting {os.path.basename(video_path)}", leave=False):
        ret, frame = cap.read()
        if not ret:
            break
        
        output_path = os.path.join(output_dir, f"{i:05d}.{ext}")
        
        if is_depth:
            if len(frame.shape) == 3:
                frame = frame[:, :, 0]
            frame = frame.astype(np.uint16)
            cv2.imwrite(output_path, frame)
        else:
            cv2.imwrite(output_path, frame)
            
    cap.release()
    return output_dir

def process_dataset(root_dir):
    # Initialize Mask Generator once
    mask_generator = HandMaskGenerator(MODEL_PATHS)
    
    root_path = Path(root_dir)
    
    rgb_root_path = root_path / 'Egocentric_RGB_Videos'
    depth_root_path = root_path / 'Egocentric_Depth_Videos'
    mask_root_path = root_path / 'Hand_Masks'
    
    rgb_files = sorted(list(rgb_root_path.rglob('color.mp4')))
    relative_files = [vf.relative_to(rgb_root_path) for vf in rgb_files]
    
    print(f"Found {len(rgb_files)} sessions.")
    
    for rel_file in tqdm(relative_files, desc="Processing sessions"):
        # Setup paths
        rgb_file = rgb_root_path / rel_file
        depth_video = depth_root_path / rel_file.parent / 'egocentric_depth.avi'
        
        # 1. Process Color Video
        color_frames_dir = extract_video_frames(rgb_file, ext='jpg', is_depth=False)
        
        # 2. Process Depth Video
        extract_video_frames(depth_video, ext='png', is_depth=True)
        
        # 3. Process Masks
        if color_frames_dir:
            # Construct Mask Output Directory: Hand_Masks/<scene>/<session>/
            session_mask_dir = mask_root_path / rel_file.parent
            os.makedirs(session_mask_dir, exist_ok=True)
            
            # Get list of generated generated color frames
            frame_paths = sorted(glob.glob(os.path.join(color_frames_dir, "*.jpg")))
            
            if len(os.listdir(session_mask_dir)) == len(frame_paths):
                print(f"Masks exist for {rel_file.parent}, skipping...")
                continue

            for frame_path in tqdm(frame_paths, desc="Generating Masks", leave=False):
                # Filename management (keep same index 00000.png)
                fname = os.path.basename(frame_path)
                fname_no_ext = os.path.splitext(fname)[0]
                mask_out_path = session_mask_dir / f"{fname_no_ext}.png"
                
                # Load image
                color_image = cv2.imread(frame_path)
                if color_image is None:
                    continue
                
                # Generate mask
                # Returns 0 for hand, 255 for bg
                mask = mask_generator.generate_mask(color_image)
                
                # Save
                cv2.imwrite(str(mask_out_path), mask)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data/Taco", help="Path to Taco dataset root")
    args = parser.parse_args()
    
    process_dataset(args.data_root)