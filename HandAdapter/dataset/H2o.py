import open3d as o3d
import numpy as np
import os
import glob
import cv2
from pathlib import Path

from .base import AbsSequence, AbsDataset

transform = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

class H2oSequence(AbsSequence):
    def __init__(self, rgb_dir, depth_dir, mask_dir, hand_pos_dir, load_pcd, index=None):
        """
        Initialize the H2oSequence with the path to the sequence data.
        """
        self.rgb_dir = rgb_dir
        self.depth_dir = depth_dir
        self.mask_dir = mask_dir
        self.hand_pos_dir = hand_pos_dir
        self.load_pcd = load_pcd
        self.index = index

        self.path = []
        self.tag = []
        
        rgb_files = sorted(glob.glob(os.path.join(rgb_dir, '*.png')))
        depth_files = sorted(glob.glob(os.path.join(depth_dir, '*.png')))
        mask_files = sorted(glob.glob(os.path.join(mask_dir, '*.png')))
        hand_files = sorted(glob.glob(os.path.join(hand_pos_dir, '*.txt')))
        
        # Ensure we have matching files
        min_len = min(len(rgb_files), len(depth_files), len(mask_files), len(hand_files))
        rgb_files = rgb_files[:min_len]
        depth_files = depth_files[:min_len]
        mask_files = mask_files[:min_len]
        hand_files = hand_files[:min_len]

        for rgb, depth, mask, hand in zip(rgb_files, depth_files, mask_files, hand_files):
            self.path.append((rgb, depth, mask, hand))
            self.tag.append(os.path.basename(rgb).split('.')[0])

    @property
    def intrinsics(self):
        return {
            "fx": 635.283881879317,
            "fy": 636.251953125,
            "cx": 636.6593017578125,
            "cy": 366.8740353496978,
            "width": 1280,
            "height": 720
        }

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

        pose = np.fromfile(self.path[index][3], sep=' ')

        result = {
            "left_pose": None,
            "right_pose": None,
            "left_R_hand_world": None,
            "right_R_hand_world": None,
        }
        if self.load_pcd:
            rgb = cv2.imread(self.path[index][0])
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            depth = cv2.imread(self.path[index][1], cv2.IMREAD_UNCHANGED) / 1000.0
            mask = cv2.imread(self.path[index][2], cv2.IMREAD_GRAYSCALE)
            
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

        if pose[0] == 1:
            left_joints = pose[1:64].reshape(21, 3)
            left_joints = (transform @ left_joints.T).T
            result["left_pose"] = left_joints
            result["left_R_hand_world"] = self.palm_R(left_joints)
        if pose[64] == 1:
            right_joints = pose[65:128].reshape(21, 3)
            right_joints = (transform @ right_joints.T).T
            result["right_pose"] = right_joints
            result["right_R_hand_world"] = self.palm_R(right_joints)

        return result
    
    def palm_R(self, joints):
        wrist = joints[0]
        index_mcp = joints[5]
        pinky_mcp = joints[17]
        x_axis = index_mcp - wrist
        x_axis /= np.linalg.norm(x_axis)
        y_axis = pinky_mcp - wrist
        y_axis /= np.linalg.norm(y_axis)
        z_axis = np.cross(x_axis, y_axis)
        z_axis /= np.linalg.norm(z_axis)
        y_axis = np.cross(z_axis, x_axis)
        return np.stack([x_axis, y_axis, z_axis], 1)
        
    def get_intrinsics(self):
        return self.intrinsics

class H2oDataset(AbsDataset):
    SEQUENCE_CLASS = H2oSequence
    
    def __init__(self, data_dir, load_pcd=True, cache_path=None):
        """
        Initialize the H2oDataset with the path to the dataset.
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
        """Find all valid sequences based on filters"""
        all_sequences = []
        
        # Look through all subject directories
        for subject_dir in sorted((self.data_dir / 'pose').glob('subject*_ego')):
            subject = subject_dir.name
                
            # Look through all session types
            for session_dir in sorted(subject_dir.glob('*')):
                session = session_dir.name
                
                # Look through all sequence numbers
                for seq_dir in sorted(session_dir.glob('*')):
                    sequence = seq_dir.name
                    
                    # Look through all cameras
                    for cam_dir in sorted(seq_dir.glob('cam*')):
                        camera = cam_dir.name
                        
                        # Verify required data exists
                        rgb_dir = self.data_dir / 'all_img' / subject / session / sequence / camera / 'rgb'
                        depth_dir = self.data_dir / 'all_img' / subject / session / sequence / camera / 'depth'
                        mask_dir = self.data_dir / 'all_img' / subject / session / sequence / camera / 'mask'

                        if camera == 'cam4':
                            hand_pos_dir = self.data_dir / 'pose' / subject / session / sequence / camera / 'hand_pose'
                            if rgb_dir.exists() and depth_dir.exists() and mask_dir.exists() and hand_pos_dir.exists():
                                index = f"{subject}/{session}/{sequence}/{camera}"
                                all_sequences.append(H2oSequence(rgb_dir, depth_dir, mask_dir, hand_pos_dir, self.load_pcd, index))
        
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