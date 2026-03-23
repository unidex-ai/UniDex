import open3d as o3d
import numpy as np
import os
import glob
import zarr
import pickle
import cv2
import torch
from scipy.spatial.transform import Rotation as R
from pathlib import Path
from manopth.manolayer import ManoLayer

from .base import AbsSequence, AbsDataset

transform = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

right_manolayer = ManoLayer(mano_root='manopth/mano/models', use_pca=False, ncomps=45, flat_hand_mean=True, side='right')
left_manolayer = ManoLayer(mano_root='manopth/mano/models', use_pca=False, ncomps=45, flat_hand_mean=True, side='left')

def pkl_to_k3d(pkl_path, side='right'):
    """
    Convert a MANO pickle file to 3D keypoints and vertices in meters.
    
    Args:
        pkl_path: Path to the MANO pickle file.
        side: 'right' or 'left' hand.
    Returns:
        kps3d: 3D keypoints in meters, shape (21, 3)
    """
    manolayer = right_manolayer if side == 'right' else left_manolayer
    with open(pkl_path, 'rb') as f:
        hand_info = pickle.load(f, encoding='latin1')
    theta = torch.FloatTensor(hand_info['poseCoeff']).unsqueeze(0)
    beta = torch.FloatTensor(hand_info['beta']).unsqueeze(0)
    trans = torch.FloatTensor(hand_info['trans']).unsqueeze(0)
    hand_verts, hand_joints = manolayer(theta, beta)
    kps3d = hand_joints / 1000.0 + trans.unsqueeze(1) # in meters
    return kps3d.squeeze(0).numpy()

class HOI4DSequence(AbsSequence):
    def __init__(self, rgb_dir, depth_dir, mask_dir, left_hand_pose_dir, right_hand_pose_dir, intrinsics_dir, load_pcd, index):
        """
        Initialize the HOI4DSequence with the path to the sequence data.
        """
        self.rgb_dir = rgb_dir
        self.depth_dir = depth_dir
        self.mask_dir = mask_dir
        self.left_hand_pose_dir = left_hand_pose_dir
        self.right_hand_pose_dir = right_hand_pose_dir
        self.load_pcd = load_pcd
        self.intrinsics_dir = intrinsics_dir
        self.index = index
        self.intrinsics_dict = None
        self.tag = []

        self.path = []
        
        # Find RGB files (support jpg and png)
        rgb_files = sorted(glob.glob(os.path.join(rgb_dir, '*.jpg')))
        if not rgb_files:
            rgb_files = sorted(glob.glob(os.path.join(rgb_dir, '*.png')))
            
        for i, rgb_file in enumerate(rgb_files):
            # Assume depth and mask have same name but png extension
            name = os.path.splitext(os.path.basename(rgb_file))[0]
            squeezed_name = name.lstrip('0')  # Remove leading zeros for pose files
            if squeezed_name == '':
                squeezed_name = '0'
            
            self.path.append((
                rgb_file,
                depth_dir / f'{name}.png',
                mask_dir / f'{name}.png',
                left_hand_pose_dir / f'{squeezed_name}.pickle',
                right_hand_pose_dir / f'{squeezed_name}.pickle'
            ))
            self.tag.append(name)

    @property
    def intrinsics(self):
        if self.intrinsics_dict is None:
            intrinsics = np.load(self.intrinsics_dir)
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
        return len(self.path)

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
        rgb_file, depth_file, mask_file, left_hand_file, right_hand_file = self.path[index]

        result = {
            "left_pose": None,
            "right_pose": None,
            "left_R_hand_world": None,
            "right_R_hand_world": None,
        }

        if self.load_pcd:
            rgb = cv2.imread(str(rgb_file))
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            depth = cv2.imread(str(depth_file), cv2.IMREAD_UNCHANGED) / 1000.0
            mask_color = cv2.imread(str(mask_file), cv2.IMREAD_COLOR)
            mask = ((mask_color[:, :, 0] == 0) & (mask_color[:, :, 2] == 0)) | ((mask_color[:, :, 0] == 0) & (mask_color[:, :, 1] == 128) & (mask_color[:, :, 2] == 128))
            mask = ~mask
            mask = mask.astype(np.uint8)
            
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

        if os.path.exists(left_hand_file):
            left_joints = pkl_to_k3d(left_hand_file, side='left')
            left_joints = (transform @ left_joints.T).T
            result["left_pose"] = left_joints
            with open(left_hand_file, 'rb') as f:
                data = pickle.load(f)
            left_R_hand_world = R.from_rotvec(np.array(data['poseCoeff'][:3])).as_matrix()
            left_R_hand_world = (transform @ left_R_hand_world)
            result["left_R_hand_world"] = left_R_hand_world
            
        if os.path.exists(right_hand_file):
            right_joints = pkl_to_k3d(right_hand_file, side='right')
            right_joints = (transform @ right_joints.T).T
            result["right_pose"] = right_joints
            with open(right_hand_file, 'rb') as f:
                data = pickle.load(f)
            right_R_hand_world = R.from_rotvec(np.array(data['poseCoeff'][:3])).as_matrix()
            right_R_hand_world = (transform @ right_R_hand_world)
            result["right_R_hand_world"] = right_R_hand_world
        return result
    
    def get_intrinsics(self):
        return self.intrinsics

class HOI4DDataset(AbsDataset):
    SEQUENCE_CLASS = HOI4DSequence
    
    def __init__(self, data_dir, load_pcd=True, cache_path=None):
        """
        Initialize the HOI4DDataset with the path to the dataset.
        
        Args:
            data_dir: Base directory of HOI4D dataset
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
        rgb_root = self.data_dir / 'HOI4D_release'
        
        for h1 in sorted(rgb_root.iterdir()):
            if not h1.is_dir():
                continue
            for h2 in sorted(h1.iterdir()):
                if not h2.is_dir():
                    continue
                for h3 in sorted(h2.iterdir()):
                    if not h3.is_dir():
                        continue
                    for h4 in sorted(h3.iterdir()):
                        if not h4.is_dir():
                            continue
                        for h5 in sorted(h4.iterdir()):
                            if not h5.is_dir():
                                continue
                            for h6 in sorted(h5.iterdir()):
                                if not h6.is_dir():
                                    continue
                                for h7 in sorted(h6.iterdir()):
                                    if not h7.is_dir():
                                        continue
        
                                    rgb_dir = h7 / 'align_rgb'
                                    depth_dir = h7 / 'align_depth'
                                    mask_dir = h7 / '2Dseg' / 'mask' if h1.name == 'ZY20210800001' else h7 / '2Dseg' / 'shift_mask'
                                    
                                    left_hand_pose_dir = self.data_dir / 'Hand_pose' / 'handpose_left_hand' / h1.name / h2.name / h3.name / h4.name / h5.name / h6.name / h7.name
                                    right_hand_pose_dir = self.data_dir / 'Hand_pose' / 'handpose_right_hand' / h1.name / h2.name / h3.name / h4.name / h5.name / h6.name / h7.name
                                    intrinsics_dir = self.data_dir / 'camera' / h1.name / 'intrin.npy'
                                    
                                    if rgb_dir.exists() and depth_dir.exists() and mask_dir.exists() and (left_hand_pose_dir.exists() or right_hand_pose_dir.exists()):
                                        index = f"{h1.name}/{h2.name}/{h3.name}/{h4.name}/{h5.name}/{h6.name}/{h7.name}"
                                        all_sequences.append(HOI4DSequence(rgb_dir, depth_dir, mask_dir, left_hand_pose_dir, right_hand_pose_dir, intrinsics_dir, self.load_pcd, index))

                            
        
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
