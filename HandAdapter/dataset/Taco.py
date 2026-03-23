import open3d as o3d
import numpy as np
import os
import glob
import zarr
import pickle
import cv2
from pathlib import Path
import torch
from manopth.manolayer import ManoLayer

from .base import AbsSequence, AbsDataset

transform = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

right_manolayer = ManoLayer(mano_root='manopth/mano/models', use_pca=False, ncomps=45, side='right', center_idx=0)
left_manolayer = ManoLayer(mano_root='manopth/mano/models', use_pca=False, ncomps=45, side='left', center_idx=0)

def to_k3d(theta, beta, trans, side='right'):
    """
    Convert MANO parameters to 3D keypoints and vertices in meters.
    
    Args:
        theta: MANO pose parameters
        beta: MANO shape parameters
        trans: MANO translation parameters
        side: 'right' or 'left' hand.
    Returns:
        kps3d: 3D keypoints in meters, shape (21, 3)
    """
    manolayer = right_manolayer if side == 'right' else left_manolayer
    theta = torch.FloatTensor(theta).unsqueeze(0)
    beta = torch.FloatTensor(beta).unsqueeze(0)
    trans = torch.FloatTensor(trans).unsqueeze(0)
    hand_verts, hand_joints = manolayer(theta, beta)
    kps3d = hand_joints / 1000.0 + trans.unsqueeze(1) # in meters
    return kps3d.squeeze(0).numpy()

class TacoSequence(AbsSequence):
    def __init__(self, data_dir, task, session, intrinsics_dir, load_pcd, index):
        """
        Initialize the TacoSequence.
        """
        self.data_dir = Path(data_dir)
        self.task = task
        self.session = session
        self.load_pcd = load_pcd
        self.index = index
        self.intrinsics_dir = intrinsics_dir
        self.intrinsics_dict = None

        # Paths
        self.rgb_dir = self.data_dir / 'Egocentric_RGB_Videos' / task / session / 'colorframes'
        self.depth_dir = self.data_dir / 'Egocentric_Depth_Videos' / task / session / 'depthframes'
        self.mask_dir = self.data_dir / 'Hand_Masks' / task / session
        self.left_hand_pose = self.data_dir / 'Hand_Poses' / task / session / 'left_hand.pkl'
        self.left_hand_pose_beta = self.data_dir / 'Hand_Poses' / task / session / 'left_hand_shape.pkl'
        self.right_hand_pose = self.data_dir / 'Hand_Poses' / task / session / 'right_hand.pkl'
        self.right_hand_pose_beta = self.data_dir / 'Hand_Poses' / task / session / 'right_hand_shape.pkl'

        self.extrinsics_dir = self.data_dir / 'Egocentric_Camera_Parameters' / task / session / 'egocentric_frame_extrinsic.npy'

        # Find all zarr directories to determine valid frames
        all_color_frames = sorted(glob.glob(os.path.join(self.rgb_dir, '*.jpg')))
        all_depth_frames = sorted(glob.glob(os.path.join(self.depth_dir, '*.png')))
        self.frames = min(len(all_color_frames), len(all_depth_frames))
    
    @property
    def intrinsics(self):
        if self.intrinsics_dict is None:
            intrinsics = np.loadtxt(self.intrinsics_dir)
            self.intrinsics_dict = {
                "fx": intrinsics[0,0],
                "fy": intrinsics[1,1],
                "cx": intrinsics[0,2],
                "cy": intrinsics[1,2],
                "width": 1920,
                "height": 1080
            }
        return self.intrinsics_dict

    def __len__(self):
        """
        Returns the length of the sequence.
        """
        return self.frames

    def __getitem__(self, index):
        """
        Returns the item at the specified index in the sequence.
        {
            "pcd": pcd, o3d.geometry.PointCloud if load_pcd
            "left_pose": pose, np.ndarray or None, (21,3)
            "right_pose": pose, np.ndarray or None, (21,3)
            "left_R_hand_world": R_hand_world, np.ndarray or None, (3,3)
            "right_R_hand_world": R_hand_world, np.ndarray or None, (3,3)
        }
        """
        result = {
            "left_pose": None,
            "right_pose": None,
            "left_R_hand_world": None,
            "right_R_hand_world": None,
        }

        if self.load_pcd:
            rgb_path = self.rgb_dir / f"{index:05d}.jpg"
            depth_path = self.depth_dir / f"{index:05d}.png"
            mask_path = self.mask_dir / f"{index:05d}.png"

            rgb = cv2.imread(str(rgb_path))
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            depth = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
            depth = depth.astype(np.float32) / 4000.0

            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            # Apply mask
            depth[mask == 0] = 0
            
            # Generate PCD
            pcd_array = self._rgbd_to_pointcloud(rgb, depth, self.intrinsics)
            
            # Convert to Open3D point cloud
            pcd = o3d.geometry.PointCloud()
            if len(pcd_array) > 0:
                pcd.points = o3d.utility.Vector3dVector(pcd_array[:, :3])
                pcd.colors = o3d.utility.Vector3dVector(pcd_array[:, 3:])
            
            result["pcd"] = pcd

        extrinsics = np.load(self.extrinsics_dir)[index]
        R = extrinsics[:3, :3]
        t = extrinsics[:3, 3:]

        with open(self.left_hand_pose, 'rb') as f:
            left_hand_data = pickle.load(f)[f'{index + 1:05d}']
        with open(self.left_hand_pose_beta, 'rb') as f:
            left_hand_shape_data = pickle.load(f)
        left_pose = to_k3d(left_hand_data['hand_pose'], left_hand_shape_data['hand_shape'], left_hand_data['hand_trans'], side='left')
        left_pose = (transform @ (R @ left_pose.T + t)).T
        result["left_pose"] = left_pose
        result["left_R_hand_world"] = self.palm_R(left_pose)
            
        with open(self.right_hand_pose, 'rb') as f:
            right_hand_data = pickle.load(f)[f'{index + 1:05d}']
        with open(self.right_hand_pose_beta, 'rb') as f:
            right_hand_shape_data = pickle.load(f)
        right_pose = to_k3d(right_hand_data['hand_pose'], right_hand_shape_data['hand_shape'], right_hand_data['hand_trans'], side='right')
        right_pose = (transform @ (R @ right_pose.T + t)).T
        result["right_pose"] = right_pose
        result["right_R_hand_world"] = self.palm_R(right_pose)

        return result
    
    def get_intrinsics(self):
        return self.intrinsics

    def palm_R(self, j: np.ndarray):
        w, i, m, r = j[0], j[8], j[11], j[14]
        x = (i + m + r) / 3 - w; x /= np.linalg.norm(x)
        z = np.cross(x, m - i);  z /= np.linalg.norm(z)
        y = np.cross(z, x);      y /= np.linalg.norm(y)
        return np.stack([x, y, z], 1)

class TacoDataset(AbsDataset):
    SEQUENCE_CLASS = TacoSequence
    
    def __init__(self, data_dir, load_pcd=True, cache_path=None):
        """
        Initialize the TacoDataset with the path to the dataset.
        
        Args:
            data_dir: Base directory of Taco dataset
            load_pcd: Whether to load point cloud data
            cache_path: Path to cached sequences file
        """
        self.data_dir = Path(data_dir)
        self.load_pcd = load_pcd
        
        if cache_path is not None and os.path.exists(cache_path):
            # Load from cache
            self.sequences = self.load_from_cache(cache_path)
        else:
            # Load from original data
            self.sequences = self._find_sequences()

    def _find_sequences(self):
        """Find all valid sequences based on directory structure"""
        all_sequences = []
        root_dir = self.data_dir / 'Egocentric_Depth_Videos'
        
        for scene in sorted(root_dir.iterdir()):
            if not scene.is_dir(): continue
            for session in sorted(scene.iterdir()):
                if not session.is_dir(): continue
                # Check if we have RGB video and Depth folder
                rgb_dir = self.data_dir / 'Egocentric_RGB_Videos' / scene.name / session.name / 'colorframes'
                depth_dir = self.data_dir / 'Egocentric_Depth_Videos' / scene.name / session.name / 'depthframes'
                
                if rgb_dir.exists() and depth_dir.exists():
                    index = f"{scene.name}/{session.name}"
                    intrinsics_dir = self.data_dir / 'Egocentric_Camera_Parameters' / index / 'egocentric_intrinsic.txt'
                    all_sequences.append(TacoSequence(self.data_dir, scene.name, session.name, intrinsics_dir, self.load_pcd, index))
        
        return all_sequences

    def __len__(self):
        """
        Returns the number of sequences in the dataset.
        """
        return len(self.sequences)

    def __getitem__(self, index):
        """
        Returns the sequence at the specified index in the dataset.
        """
        return self.sequences[index]
