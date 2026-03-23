import abc
import pickle
import os
import numpy as np
import open3d as o3d
from typing import Dict, Tuple, Optional

class AbsSequence(abc.ABC):
    """
    Abstract base class for a sequence of data.
    """
    
    # Transform matrix for camera coordinate conversion
    CV_TO_CAM = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)

    TRANSFORM = np.eye(4, dtype=np.float32)
    
    # Default pointcloud mask (can be overridden by subclasses)
    PCD_MASK = None

    def _rgbd_to_pointcloud(self, rgb: np.ndarray, depth: np.ndarray, camera_intrinsics: Dict[str, float]) -> np.ndarray:
        """Convert RGBD to pointcloud using Open3D and camera parameters"""
        # Ensure data types are correct
        rgb = rgb.astype(np.uint8)
        depth = depth.astype(np.float32)

        # Apply depth truncation
        depth[depth > 1.15] = 0

        # Extract camera intrinsics
        fx = camera_intrinsics['fx']
        fy = camera_intrinsics['fy']
        cx = camera_intrinsics['cx']
        cy = camera_intrinsics['cy']
        width = camera_intrinsics['width']
        height = camera_intrinsics['height']

        # Create Open3D image objects
        rgb_o3d = o3d.geometry.Image(rgb)
        depth_o3d = o3d.geometry.Image(depth)

        # Set camera intrinsics
        camera_intrinsics_o3d = o3d.camera.PinholeCameraIntrinsic(
            width=width,
            height=height,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy
        )

        # Convert RGB-D data to pointcloud
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color=rgb_o3d,
            depth=depth_o3d,
            depth_scale=1.0,
            depth_trunc=3.0,
            convert_rgb_to_intensity=False
        )

        point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image,
            camera_intrinsics_o3d
        )

        # Apply transformation
        point_cloud.transform(self.CV_TO_CAM)
        
        # Get points and colors
        points = np.asarray(point_cloud.points) @ self.TRANSFORM[:3, :3].T
        colors = np.asarray(point_cloud.colors)
        
        # Apply PCD_MASK if defined
        if self.PCD_MASK is not None and len(points) > 0:
            # This creates a bounding box filter
            if self.PCD_MASK.shape == (2, 3):
                min_bounds = self.PCD_MASK[0]
                max_bounds = self.PCD_MASK[1]
                
                # Create mask for points within bounds
                mask = np.any(points < min_bounds, axis=1) | np.any(points > max_bounds, axis=1)
                mask = ~mask
                
                # Filter points and colors
                points = points[mask]
                colors = colors[mask]

        # Combine points and colors
        if len(points) > 0:
            pc = np.concatenate([points, colors], axis=1)  # (N, 6)
        else:
            pc = np.empty((0, 6))

        return pc

    @abc.abstractmethod
    def __len__(self):
        """
        Returns the length of the sequence.
        """
        pass

    @abc.abstractmethod
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
        pass

class AbsDataset(abc.ABC):
    """
    Abstract base class for a dataset containing multiple sequences.
    """
    SEQUENCE_CLASS = None

    @abc.abstractmethod
    def __len__(self):
        """
        Returns the number of sequences in the dataset.
        """
        pass

    @abc.abstractmethod
    def __getitem__(self, index):
        """
        Returns the sequence at the specified index in the dataset.
        """
        pass

    def save(self, path):
        """
        Save the dataset to the specified path.
        """
        sequences = [self[i].__dict__ for i in range(len(self))]
        with open(path, 'wb') as f:
            pickle.dump(sequences, f)
    
    def load_from_cache(self, cache_path):
        """
        Load sequences from cached file.
        """
        with open(cache_path, 'rb') as f:
            sequences_data = pickle.load(f)
        
        loaded_sequences = []
        for seq_data in sequences_data:
            # Create a new sequence instance without calling __init__
            seq = self.SEQUENCE_CLASS.__new__(self.SEQUENCE_CLASS)
            # Restore all the sequence attributes directly
            seq.__dict__.update(seq_data)
            loaded_sequences.append(seq)
        
        return loaded_sequences