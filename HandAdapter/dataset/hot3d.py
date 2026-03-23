import open3d as o3d
import numpy as np
import os
import glob
import zarr
import pickle
import cv2
from pathlib import Path

from .base import AbsSequence, AbsDataset

transform = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

class hot3dSequence(AbsSequence):
    def __init__(self, rgb_dir, depth_dir, mask_dir, left_hand_pose_dir, right_hand_pose_dir, left_wrist_dir, right_wrist_dir, cam_dir, intrinsics_dir, load_pcd, index=None):
        """
        Initialize the hot3dSequence with the path to the sequence data.
        """
        self.rgb_dir = rgb_dir
        self.depth_dir = depth_dir
        self.mask_dir = mask_dir
        self.left_hand_pose_dir = left_hand_pose_dir
        self.right_hand_pose_dir = right_hand_pose_dir
        self.left_wrist_dir = left_wrist_dir
        self.right_wrist_dir = right_wrist_dir
        self.cam_dir = cam_dir
        self.load_pcd = load_pcd
        self.index = index
        self.intrinsics_dir = intrinsics_dir
        self.intrinsics_dict = None
        self.tag = []

        self.path = []
        
        # Find RGB files (support jpg and png)
        depth_files = sorted(glob.glob(os.path.join(depth_dir, '*.tiff')))
            
        for depth_file in depth_files:
            tag = os.path.splitext(os.path.basename(depth_file))[0]
            self.path.append((
                rgb_dir / f'{tag}.jpg',
                depth_dir / f'{tag}.tiff',
                mask_dir / f'{tag}.jpg',
                left_hand_pose_dir / f'{tag}.txt',
                right_hand_pose_dir / f'{tag}.txt',
                left_wrist_dir / f'{tag}.txt',
                right_wrist_dir / f'{tag}.txt',
                cam_dir / f'{tag}.txt'
            ))
            self.tag.append(tag)

    @property
    def intrinsics(self):
        if self.intrinsics_dict is None:
            with open(self.intrinsics_dir / 'intrinsics.txt', 'r') as f:
                lines = f.readlines()
                self.intrinsics_dict = {
                    'fx': float(lines[0].split(':')[1].strip().split()[0][1:]),
                    'fy': float(lines[0].split(':')[1].strip().split()[1][:-1]),
                    'cx': float(lines[1].split(':')[1].strip().split()[0][1:]),
                    'cy': float(lines[1].split(':')[1].strip().split()[1][:-1]),
                    'width': int(lines[2].split(':')[1].strip().split()[0][1:]),
                    'height': int(lines[2].split(':')[1].strip().split()[1][:-1])
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
        rgb_file, depth_file, mask_file, left_hand_file, right_hand_file, left_wrist_file, right_wrist_file, cam_file = self.path[index]

        result = {
            "left_pose": None,
            "right_pose": None,
            "left_R_hand_world": None,
            "right_R_hand_world": None,
        }

        if self.load_pcd:
            rgb = cv2.imread(str(rgb_file))
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            depth = cv2.imread(str(depth_file), cv2.IMREAD_UNCHANGED)
            mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
            
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

        cam = np.loadtxt(cam_file)
        if os.path.exists(left_hand_file):
            left_pose = np.loadtxt(left_hand_file)
            left_pose = (np.linalg.inv(cam) @ np.hstack([left_pose, np.ones((21, 1))]).T).T[:, :3]
            left_pose = (transform @ left_pose.T).T
            result["left_pose"] = left_pose
            left_R_hand_world = np.loadtxt(left_wrist_file)
            left_R_hand_world = (np.linalg.inv(cam) @ left_R_hand_world)
            left_R_hand_world = (transform @ left_R_hand_world[:3, :3])
            result["left_R_hand_world"] = left_R_hand_world
            
        if os.path.exists(right_hand_file):
            right_pose = np.loadtxt(right_hand_file)
            right_pose = (np.linalg.inv(cam) @ np.hstack([right_pose, np.ones((21, 1))]).T).T[:, :3]
            right_pose = (transform @ right_pose.T).T
            result["right_pose"] = right_pose
            right_R_hand_world = np.loadtxt(right_wrist_file)
            right_R_hand_world = (np.linalg.inv(cam) @ right_R_hand_world)
            right_R_hand_world = (transform @ right_R_hand_world[:3, :3])
            result["right_R_hand_world"] = right_R_hand_world

        return result
    
    def get_intrinsics(self):
        return self.intrinsics

    def palm_R(self, j: np.ndarray):
        w, i, m, r = j[0], j[8], j[11], j[14]
        x = (i + m + r) / 3 - w; x /= np.linalg.norm(x)
        z = np.cross(x, m - i);  z /= np.linalg.norm(z)
        y = np.cross(z, x);      y /= np.linalg.norm(y)
        return np.stack([x, y, z], 1)

class hot3dDataset(AbsDataset):
    SEQUENCE_CLASS = hot3dSequence
    
    def __init__(self, data_dir, load_pcd=True, cache_path=None):
        """
        Initialize the hot3dDataset with the path to the dataset.
        
        Args:
            data_dir: Base directory of hot3d dataset
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
        
        for seq in sorted(self.data_dir.iterdir()):
            if not seq.is_dir():
                continue

            rgb_dir = seq / 'RGB'
            depth_dir = seq / 'depth'
            mask_dir = seq / 'masks'

            if not (rgb_dir.exists() and depth_dir.exists() and mask_dir.exists() and
                    (seq / 'left_hand_pose').exists() and
                    (seq / 'right_hand_pose').exists() and
                    (seq / 'left_wrist').exists() and
                    (seq / 'right_wrist').exists() and
                    (seq / 'cam_extrinsic').exists()):
                continue

            all_sequences.append(
                hot3dSequence(
                    rgb_dir=rgb_dir,
                    depth_dir=depth_dir,
                    mask_dir=mask_dir,
                    left_hand_pose_dir=seq / 'left_hand_pose',
                    right_hand_pose_dir=seq / 'right_hand_pose',
                    left_wrist_dir=seq / 'left_wrist',
                    right_wrist_dir=seq / 'right_wrist',
                    cam_dir=seq / 'cam_extrinsic',
                    intrinsics_dir=seq / 'intrinsics',
                    load_pcd=self.load_pcd,
                    index=seq.name
                )
            )
            print(f"Found sequence: {seq.name}, total sequences: {len(all_sequences)}")

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
