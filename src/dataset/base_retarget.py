"""
Base retarget dataset class for unified handling of retarget datasets.
Provides common functionality while allowing subclasses to implement specific
sequence finding and data loading logic.

HDF5 File Structure:
├── complete                                    # Dataset[bool]: Processing completion flag
├── metadata/                                   # Group: Sequence metadata
│   ├── dataset                                 # Attr[str]: Dataset name (e.g., "H2o", "HOI4D")
│   ├── hand_type                              # Attr[str]: Hand type (e.g., "Leap", "Inspire")
│   ├── sequence_index                         # Attr[str]: Sequence identifier
│   ├── total_frames                           # Attr[int]: Total number of frames
│   ├── camera_intrinsics/                     # Group: Camera parameters
│   │   ├── fx, fy, cx, cy                     # Attr[float]: Camera intrinsic parameters
│   │   ├── width, height                      # Attr[int]: Image dimensions
│   ├── tip_indices                            # Dataset[int]: MANO fingertip joint indices [4,8,12,16,20]
│   ├── joint_names/                           # Group: Joint name mappings
│   │   ├── left                               # Dataset[bytes]: Left hand joint names
│   │   └── right                              # Dataset[bytes]: Right hand joint names
│   └── pose_names/                            # Group: Pose link names
│       ├── left                               # Dataset[bytes]: Left hand pose link names
│       └── right                              # Dataset[bytes]: Right hand pose link names
└── frames/                                    # Group: Frame data
    ├── frame_ids                              # Dataset[int32]: Frame indices [0,1,2,...]
    ├── timestamps                             # Dataset[str]: Frame timestamps
    ├── rgb_images                             # Dataset[uint8]: RGB images (N,H,W,3)
    ├── depth_images                           # Dataset[float16]: Depth images (N,H,W)
    ├── joint_values/                          # Group: Joint angle values
    │   ├── left                               # Dataset[float32]: Left hand joint angles (N,num_joints)
    │   └── right                              # Dataset[float32]: Right hand joint angles (N,num_joints)
    └── poses/                                 # Group: 4x4 transformation matrices
        ├── left                               # Dataset[float32]: Left hand poses (N,num_poses,4,4)
        └── right                              # Dataset[float32]: Right hand poses (N,num_poses,4,4)
"""

import os
import json
import hashlib
import h5py
import cv2
import numpy as np
import open3d as o3d
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from .base import BaseDataset
from ..utils import hand_utils, pose as pose_utils


class BaseRetargetDataset(BaseDataset, ABC):
    """
    Base class for retarget datasets that load data from HDF5 files.
    Provides common functionality for loading states, actions, and pointclouds
    from HDF5 format while allowing subclasses to implement specific data loading.
    """
    
    # Transform matrix for camera coordinate conversion
    CV_TO_CAM = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)

    TRANSFORM = np.eye(4, dtype=np.float32)
    
    # Pose matrix for unseen/missing hands
    UNSEEN_POSE = np.eye(4, dtype=np.float32)
    
    # Default pointcloud mask (can be overridden by subclasses)
    PCD_MASK = None

    @abstractmethod
    def _find_sequences(self) -> List[Dict[str, Any]]:
        """
        Find all valid retarget sequences.
        
        Returns:
            List of sequence dictionaries with keys:
            - h5_path: Path to HDF5 file
            - relative_path: Relative path for loading masked RGBD
            - hand_type: Type of hand model
            - sequence_name: Name of sequence
        """
        pass

    @abstractmethod
    def _build_window(self, seq_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Build all windows metadata for a sequence.
        
        Args:
            seq_info: Sequence information dictionary
            
        Returns:
            List of window dictionaries
        """
        pass

    @abstractmethod
    def _load_prompt(self, window) -> str:
        """
        Generate text prompt for a sequence.
        
        Args:
            window: window dictionary containing seq_info, start_idx, end_idx
            
        Returns:
            Text description of the sequence
        """
        pass

    @abstractmethod
    def _load_scene_rgbd(self, relative_path: str, frame_idx: int, mask_hand: bool = False) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Load masked RGB, depth for a specific frame.
        
        Args:
            relative_path: Relative path for the sequence
            frame_idx: Frame index
            
        Returns:
            Tuple of (rgb, depth) as numpy arrays
        """
        pass

    def get_sequence_intrinsics(self, seq_idx: int) -> Optional[Dict[str, Any]]:
        """
        Get camera intrinsics for a specific sequence index.
        
        Args:
            seq_idx: The index of the sequence in self.sequences.
            
        Returns:
            A dictionary with camera intrinsic parameters or None if not found.
        """
        if seq_idx < 0 or seq_idx >= len(self.sequences):
            return None
        
        seq_info = self.sequences[seq_idx]
        h5_path = seq_info.get('h5_path')
        
        if not h5_path:
            return None
            
        try:
            with h5py.File(h5_path, 'r') as f:
                cam_group = f['metadata/camera_intrinsics']
                intrinsics = {
                    'fx': cam_group.attrs['fx'],
                    'fy': cam_group.attrs['fy'],
                    'cx': cam_group.attrs['cx'],
                    'cy': cam_group.attrs['cy'],
                    'width': cam_group.attrs['width'],
                    'height': cam_group.attrs['height'],
                }
                return intrinsics
        except Exception as e:
            self.log(f"Warning: Could not load intrinsics from {h5_path}: {e}")
            return None

    def _reorder_joint_values(self, joint_values: np.ndarray, joint_names: List[str], hand_type: str, hand_side: str) -> np.ndarray:
        """
        Reorder joint values according to JOINT_MAP order and apply retarget scaling.
        
        Args:
            joint_values: Original joint values array
            joint_names: List of joint names from HDF5
            hand_type: Hand type (e.g., 'Inspire')
            hand_side: Hand side ('left' or 'right')
            
        Returns:
            Reordered and scaled joint values
        """
        if hand_type not in hand_utils.JOINT_MAP:
            raise ValueError(f"Unknown hand type: {hand_type}")
        
        joint_map = hand_utils.JOINT_MAP[hand_type]
        
        # Create mapping from original order to desired order
        reordered_values = np.zeros((joint_values.shape[0], len(joint_map)))
        
        for desired_idx, desired_name in enumerate(joint_map.keys()):
            if desired_name in joint_names:
                original_idx = joint_names.index(desired_name)
                reordered_values[:, desired_idx] = joint_values[:, original_idx]
        
        return reordered_values

    def _merge_rgbd_images(self, hand_rgb: np.ndarray, hand_depth: np.ndarray, 
                          scene_rgb: np.ndarray, scene_depth: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Merge hand RGBD with scene RGBD images.
        
        Args:
            hand_rgb: Hand RGB image from HDF5 (H, W, 3)
            hand_depth: Hand depth image from HDF5 (H, W)
            scene_rgb: Scene RGB image (H, W, 3)
            scene_depth: Scene depth image (H, W)
            
        Returns:
            Tuple of (merged_rgb, merged_depth)
        """
        # Start with scene images as base
        merged_rgb = scene_rgb.copy()
        merged_depth = scene_depth.copy()
        
        # Determine where to use hand data
        hand_valid = (hand_depth > 0) & ((scene_depth == 0) | (hand_depth < scene_depth))
        # Apply hand data to valid regions
        merged_rgb[hand_valid] = hand_rgb[hand_valid]
        merged_depth[hand_valid] = hand_depth[hand_valid]
        return merged_rgb, merged_depth

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

    def _load_raw_pointcloud(self, pcd_paths) -> np.ndarray:
        """
        Load raw pointcloud data by merging hand RGBD from HDF5 with scene RGBD.
        
        Args:
            pcd_paths: Tuple containing (h5_path, relative_path, frame_idx)
            
        Returns:
            Raw pointcloud array of shape (N, 6) containing xyz+rgb data
        """
        h5_path, relative_path, frame_idx = pcd_paths
        
        # Load hand RGBD from HDF5
        with h5py.File(h5_path, 'r') as f:
            hand_rgb = f['frames']['rgb_images'][frame_idx]  # (H, W, 3)
            hand_depth = f['frames']['depth_images'][frame_idx]  # (H, W)
            camera_intrinsics = {
                'fx': f['metadata']['camera_intrinsics'].attrs['fx'],
                'fy': f['metadata']['camera_intrinsics'].attrs['fy'],
                'cx': f['metadata']['camera_intrinsics'].attrs['cx'],
                'cy': f['metadata']['camera_intrinsics'].attrs['cy'],
                'width': f['metadata']['camera_intrinsics'].attrs['width'],
                'height': f['metadata']['camera_intrinsics'].attrs['height']
            }
        
        # Load scene RGBD from dataset-specific paths
        scene_rgb, scene_depth = self._load_scene_rgbd(relative_path, frame_idx, mask_hand=True)
        
        # Merge hand and scene RGBD
        merged_rgb, merged_depth = self._merge_rgbd_images(hand_rgb, hand_depth, scene_rgb, scene_depth)
        
        # Generate pointcloud from merged RGBD
        pointcloud = self._rgbd_to_pointcloud(merged_rgb, merged_depth, camera_intrinsics)
        
        return pointcloud

    def _load_state(self, window: Dict[str, Any]) -> np.ndarray:
        """Load state for the window"""
        h5_path = window['h5_path']
        hand_type = window['hand_type']
        frame_indices = window['state_frame_indices']
        
        states = []
        with h5py.File(h5_path, 'r') as f:
            # Get metadata for joint names and pose names
            joint_names = {}
            pose_names = {}
            for side in ['right', 'left']:
                joint_names[side] = [name.decode('utf-8') for name in f['metadata']['joint_names'][side][:]]
                pose_names[side] = [name.decode('utf-8') for name in f['metadata']['pose_names'][side][:]]
            
            # Get mapping from HDF5 pose names to universal names
            hand_tip_mapping = hand_utils.HAND_TIP_TO_UNIVERSAL[hand_type]
            
            for frame_idx in frame_indices:
                state = []
                
                # Load poses for both hands and create pose dictionaries
                poses = {}
                pose_dicts = {}
                wrists = {}
                
                for side in ['right', 'left']:
                    poses[side] = f['frames']['poses'][side][frame_idx]  # (num_poses, 4, 4)
                    pose_dicts[side] = {}
                    
                    # Create pose dictionary for current frame
                    for i, pose_name in enumerate(pose_names[side]):
                        if pose_name in hand_tip_mapping:
                            universal_name = hand_tip_mapping[pose_name]
                            pose_dicts[side][universal_name] = self.TRANSFORM @ poses[side][i]
                    
                    # Get wrist pose and apply transforms
                    wrist_pose = pose_dicts[side]['wrist']
                    if wrist_pose is None or np.isnan(wrist_pose).any():
                        wrists[side] = self.UNSEEN_POSE.copy()
                    else:
                        wrists[side] = wrist_pose @ hand_utils.HAND_TRANSFORMS[hand_type][side]
                
                # Convert wrist poses to 9D representation
                state.extend(pose_utils.mat_to_pose9d(wrists['right']).tolist())  # Right wrist
                state.extend(pose_utils.mat_to_pose9d(wrists['left']).tolist())   # Left wrist

                hand_info = {'right': True, 'left': True}
                
                # Use joint values
                joints = {}
                reordered_joints = {}
                
                for side in ['right', 'left']:
                    joints[side] = f['frames']['joint_values'][side][frame_idx]
                    
                    # Check for NaN values in joint data
                    if np.isnan(joints[side]).any():
                        # Use zero joint values if hand is missing
                        reordered_joints[side] = np.zeros(len(hand_utils.JOINT_MAP[hand_type]))
                        hand_info[side] = False
                    else:
                        # Reorder according to JOINT_MAP
                        reordered_joints[side] = self._reorder_joint_values(
                            joints[side].reshape(1, -1), joint_names[side], hand_type, side
                        ).flatten()
                
                state.extend(reordered_joints['right'].tolist())
                state.extend(reordered_joints['left'].tolist())
                
                states.append(np.array(state, dtype=np.float32))
        
        return np.array(states, dtype=np.float32), hand_info

    def _load_action_sequence(self, window: Dict[str, Any]) -> np.ndarray:
        """Load action sequence for the window"""
        h5_path = window['h5_path']
        hand_type = window['hand_type']
        current_frame_idx = window['state_frame_indices'][-1]
        action_frame_indices = window['frame_indices']
        
        actions = []
        with h5py.File(h5_path, 'r') as f:
            # Get metadata for joint names and pose names
            joint_names = {}
            pose_names = {}
            for side in ['right', 'left']:
                joint_names[side] = [name.decode('utf-8') for name in f['metadata']['joint_names'][side][:]]
                pose_names[side] = [name.decode('utf-8') for name in f['metadata']['pose_names'][side][:]]
            
            # Get mapping from HDF5 pose names to universal names
            hand_tip_mapping = hand_utils.HAND_TIP_TO_UNIVERSAL[hand_type]
            
            # Get current state poses and create pose dictionaries
            current_poses = {}
            current_pose_dicts = {}
            current_wrists = {}
            
            for side in ['right', 'left']:
                current_poses[side] = f['frames']['poses'][side][current_frame_idx]
                current_pose_dicts[side] = {}
                
                # Create current pose dictionaries
                for i, pose_name in enumerate(pose_names[side]):
                    if pose_name in hand_tip_mapping:
                        universal_name = hand_tip_mapping[pose_name]
                        current_pose_dicts[side][universal_name] = self.TRANSFORM @ current_poses[side][i]
                
                # Get wrist poses and apply transforms
                current_wrist_pose = current_pose_dicts[side]['wrist']
                if current_wrist_pose is None or np.isnan(current_wrist_pose).any():
                    current_wrists[side] = self.UNSEEN_POSE.copy()
                else:
                    current_wrists[side] = current_wrist_pose @ hand_utils.HAND_TRANSFORMS[hand_type][side]
            
            for frame_idx in action_frame_indices:
                action = []
                
                # Load target poses
                target_poses = {}
                for side in ['right', 'left']:
                    target_poses[side] = f['frames']['poses'][side][frame_idx]
                
                # Update pose dictionaries for target frame
                target_pose_dicts = {}
                for side in ['right', 'left']:
                    target_pose_dicts[side] = {}
                    for i, pose_name in enumerate(pose_names[side]):
                        if pose_name in hand_tip_mapping:
                            universal_name = hand_tip_mapping[pose_name]
                            target_pose_dicts[side][universal_name] = self.TRANSFORM @ target_poses[side][i]
                
                # Get target wrist poses and apply transforms
                target_wrists = {}
                for side in ['right', 'left']:
                    target_wrist_pose = target_pose_dicts[side].get('wrist')
                    if target_wrist_pose is None or np.isnan(target_wrist_pose).any():
                        target_wrists[side] = self.UNSEEN_POSE.copy()
                    else:
                        target_wrists[side] = target_wrist_pose @ hand_utils.HAND_TRANSFORMS[hand_type][side]
                
                # Compute relative transformations
                right_rel = np.linalg.inv(current_wrists['right']) @ target_wrists['right']
                left_rel = np.linalg.inv(current_wrists['left']) @ target_wrists['left']
                
                action.extend(pose_utils.mat_to_pose9d(right_rel).tolist())
                action.extend(pose_utils.mat_to_pose9d(left_rel).tolist())
                
                # Use joint values
                target_joints = {}
                reordered_joints = {}
                
                for side in ['right', 'left']:
                    target_joints[side] = f['frames']['joint_values'][side][frame_idx]
                    
                    # Check for NaN values in joint data
                    if np.isnan(target_joints[side]).any():
                        # Use zero joint values if hand is missing
                        reordered_joints[side] = np.zeros(len(hand_utils.JOINT_MAP[hand_type]))
                    else:
                        # Reorder according to JOINT_MAP
                        reordered_joints[side] = self._reorder_joint_values(
                            target_joints[side].reshape(1, -1), joint_names[side], hand_type, side
                        ).flatten()
                
                action.extend(reordered_joints['right'].tolist())
                action.extend(reordered_joints['left'].tolist())
                
                actions.append(np.array(action, dtype=np.float32))
        
        return np.array(actions, dtype=np.float32)


    def _get_initial_action(self, window: Dict[str, Any]) -> np.ndarray:
        """Get initial action corresponding to current state frame"""
        h5_path = window['h5_path']
        hand_type = window['hand_type']
        current_frame_idx = window['state_frame_indices'][-1]
        
        with h5py.File(h5_path, 'r') as f:
            # Get metadata for joint names and pose names
            joint_names = {}
            pose_names = {}
            for side in ['right', 'left']:
                joint_names[side] = [name.decode('utf-8') for name in f['metadata']['joint_names'][side][:]]
                pose_names[side] = [name.decode('utf-8') for name in f['metadata']['pose_names'][side][:]]
            
            # Get mapping from HDF5 pose names to universal names
            hand_tip_mapping = hand_utils.HAND_TIP_TO_UNIVERSAL[hand_type]
            
            # Create pose mappings for efficient lookup (universal name -> pose matrix)
            current_poses = {}
            pose_dicts = {}
            current_wrists = {}
            
            for side in ['right', 'left']:
                current_poses[side] = f['frames']['poses'][side][current_frame_idx]
                pose_dicts[side] = {}
                
                # Create current pose dictionaries
                for i, pose_name in enumerate(pose_names[side]):
                    if pose_name in hand_tip_mapping:
                        universal_name = hand_tip_mapping[pose_name]
                        pose_dicts[side][universal_name] = self.TRANSFORM @ current_poses[side][i]
                
                # Get wrist poses and apply transforms
                current_wrist_pose = pose_dicts[side].get('wrist')
                if current_wrist_pose is None or np.isnan(current_wrist_pose).any():
                    current_wrists[side] = self.UNSEEN_POSE.copy()
                else:
                    current_wrists[side] = current_wrist_pose @ hand_utils.HAND_TRANSFORMS[hand_type][side]
            
            action = []
            
            # Identity relative transformations for initial action
            rh_rel = np.eye(4)
            lh_rel = np.eye(4)
            action.extend(pose_utils.mat_to_pose9d(rh_rel).tolist())
            action.extend(pose_utils.mat_to_pose9d(lh_rel).tolist())
            
            # Use current joint values
            current_joints = {}
            reordered_joints = {}
            
            for side in ['right', 'left']:
                current_joints[side] = f['frames']['joint_values'][side][current_frame_idx]
                
                # Check for NaN values in joint data
                if np.isnan(current_joints[side]).any():
                    # Use zero joint values if hand is missing
                    reordered_joints[side] = np.zeros(len(hand_utils.JOINT_MAP[hand_type]))
                else:
                    # Reorder according to JOINT_MAP
                    reordered_joints[side] = self._reorder_joint_values(
                        current_joints[side].reshape(1, -1), joint_names[side], hand_type, side
                    ).flatten()
            
            action.extend(reordered_joints['right'].tolist())
            action.extend(reordered_joints['left'].tolist())
        
        return np.array(action, dtype=np.float32)

    def _generate_window_hash(self, window: dict) -> str:
        """Generate a unique hash for the window based on its key information"""
        key_info = {
            'seq_info': str(window['seq_info']),
            'h5_path': str(window['h5_path']),
            'start_frame': window['start_frame'],
            'end_frame': window['end_frame'],
            'chunk_size': self.chunk_size,
            'sample_stride': self.sample_stride,
            'pointcloud_size': self.pointcloud_size,
            'state_horizon': self.state_horizon,
            'pcd_horizon': self.pcd_horizon,
        }
        key_str = str(sorted(key_info.items()))
        return hashlib.md5(key_str.encode()).hexdigest()[:16]

    def _generate_pcd_hash(self, pcd_paths) -> str:
        """Generate a unique hash for the pointcloud file(s)"""
        h5_path, relative_path, frame_idx = pcd_paths
        key_info = {
            'h5_path': str(h5_path),
            'relative_path': str(relative_path),
            'frame_idx': frame_idx,
            'pointcloud_size': self.pointcloud_size,
        }
        key_str = str(sorted(key_info.items()))
        return hashlib.md5(key_str.encode()).hexdigest()[:16]

