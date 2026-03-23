"""
HOI4D Retarget Dataset implementation
"""

import os
import glob
import torch
import numpy as np
import open3d as o3d
from typing import List, Optional, Tuple
import hashlib
import zarr
import cv2
import h5py
from pathlib import Path

import src.utils.pose as pose_utils
import src.utils.hand_utils as hand_utils
from src.utils.normalizers import Normalizer
from src.dataset.base_retarget import BaseRetargetDataset

torch.multiprocessing.set_start_method('spawn', force=True)

class HOI4DDataset(BaseRetargetDataset):
    # Pointcloud mask for HOI4D dataset
    PCD_MASK = np.array([[-0.4, -0.3, -1.2], [0.4, 0.2, -0.4]])
    
    TASKS = {
        "C1": {
            "T1": "pick and place toy car",
            "T2": "push toy car",
            "T3": "put the toy car in the drawer",
            "T4": "take the toy car out of the drawer",
        },
        "C2": {
            "T1": "pick and place mug",
            "T2": "put the mug in the drawer",
            "T3": "take the mug out of the drawer",
            "T4": "fill the mug with water by a kettle",
            "T5": "pour water into another mug",
            "T6": "pick and place the mug (with water)"
        },
        "C3": {
            "T1": "pick and place laptop",
            "T2": "open and close the laptop display",
        },
        "C4": {
            "T1": "open and close the drawer of the storage furniture",
            "T2": "open and close the door of the storage furniture",
            "T3": "put the drink in the door of the storage furniture",
            "T4": "put the drink in the drawer of the storage furniture",
        },
        "C5": {
            "T1": "pick and place bottle",
            "T2": "pour all the water from the bottle into a mug",
            "T3": "put the bottle in the drawer",
            "T4": "take the bottle out of the drawer",
            "T5": "reposition the bottle",
            "T6": "pick and place the bottle (with water)"
        },
        "C6": {
            "T1": "open and close the safe door",
            "T2": "put something in the safe",
            "T3": "take something out of the safe",
        },
        "C7": {
            "T1": "pick and place bowl",
            "T2": "put the bowl in the drawer",
            "T3": "take the bowl out of the drawer",
            "T4": "take the ball out of the bowl",
            "T5": "put the ball in the bowl",
            "T6": "pick and place the bowl (with ball)"
        },
        "C8": {
            "T1": "pick and place bucket",
            "T2": "pour water from the bucket into another bucket",
        },
        "C9": {
            "T1": "pick and place scissors",
            "T2": "cut something with the scissors",
        },
        "C11": {
            "T1": "pick and place pliers",
            "T2": "take the pliers out of the drawer",
            "T3": "put the pliers in the drawer",
            "T4": "clamp something with the pliers",
        },
        "C12": {
            "T1": "pick and place kettle",
            "T2": "pour water from the kettle into a mug",
        },
        "C13": {
            "T1": "pick and place knife",
            "T2": "take the knife out of the drawer",
            "T3": "put the knife in the drawer",
            "T4": "cut apple with the knife",
        },
        "C14": {
            "T1": "open and close the trash can",
            "T2": "throw something in the trash can",
        },
        "C17": {
            "T1": "pick and place lamp",
            "T2": "turn and fold the lamp",
            "T3": "turn on and turn off the lamp",
        },
        "C18": {
            "T1": "pick and place stapler",
            "T2": "bind the paper with the stapler",
        },
        "C20": {
            "T1": "pick and place the chair to the original position",
            "T2": "pick and place the chair to the original position",
            "T3": "pick and place the chair to a new position",
            "T4": "pick and place the chair to a new position",
        }
    }

    def _find_sequences(self):
        """Find all valid retarget sequences for HOI4D dataset"""
        sequences = []
        
        # Look for HDF5 files in retarget_RGBD directory
        retarget_dir = Path(self.data_dir) / 'retarget_RGBD'
        
        for hand_type in self.hands:
            # Find all H5 files for this hand type
            h5_pattern = f"**/{hand_type}.h5"
            h5_files = list(sorted(retarget_dir.glob(h5_pattern)))
            
            for h5_file in h5_files:
                # Extract relative path from retarget_RGBD
                relative_path = h5_file.relative_to(retarget_dir).parent
                
                # Parse path components for filtering
                path_parts = str(relative_path).split('/')
                session_id = path_parts[2]
                camera_id = path_parts[6]
                prompt = self.TASKS[session_id][camera_id]
                
                seq_info = {
                    'h5_path': str(h5_file),
                    'relative_path': str(relative_path),
                    'hand_type': hand_type,
                    'prompt': f"Use {hand_type} hands to {prompt}."
                }
                sequences.append(seq_info)
                
        return sequences

    def _build_window(self, seq_info):
        """Build window data for HOI4D sequence"""
        try:
            # Load only total frames from metadata (memory efficient)
            with h5py.File(seq_info['h5_path'], 'r') as f:
                total_frames = f['metadata'].attrs['total_frames']
            
            # Apply stride to reduce frame rate
            available_frames = list(range(0, total_frames, self.sample_stride))
            min_frames = len(available_frames)
            
            # Calculate required minimum frames based on state_horizon and pcd_horizon only
            min_required_frames = self.chunk_size + max(self.state_horizon, self.pcd_horizon)
            if min_frames < min_required_frames:
                return []
                
            windows = []
            
            # Create windows with sliding window
            # Start from max(state_horizon, pcd_horizon) for window selection
            max_horizon = max(self.state_horizon, self.pcd_horizon)
            for start_idx in range(max_horizon, 
                                 min_frames - self.chunk_size + 1, 
                                 self.chunk_size // 6):
                end_idx = start_idx + self.chunk_size
                
                window = {
                    'seq_info': seq_info,
                    'h5_path': seq_info['h5_path'],
                    'relative_path': seq_info['relative_path'],
                    'hand_type': seq_info['hand_type'],
                    'start_frame': start_idx,
                    'end_frame': end_idx,
                    'frame_indices': available_frames[start_idx:end_idx],
                    'state_frame_indices': [available_frames[start_idx - i] for i in range(self.state_horizon, 0, -1)],
                    'pcd_paths': [(seq_info['h5_path'], seq_info['relative_path'], available_frames[start_idx - i]) for i in range(self.pcd_horizon, 0, -1)],
                }
                
                windows.append(window)
                
            return windows
            
        except Exception as e:
            self.log(f"Error building window for {seq_info['h5_path']}: {e}")
            return []

    def _load_prompt(self, window):
        """Load prompt for HOI4D sequence"""
        return window['seq_info']['prompt']

    def _load_scene_rgbd(self, relative_path: str, frame_idx: int, mask_hand: bool = False) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Load RGB, depth, and mask images for HOI4D dataset"""
        # HOI4D paths:
        # rgb: data/HOI4D/HOI4D_release/ZY20210800001/H1/C1/N19/S100/s02/T1/align_rgb/00000.jpg
        # depth: data/HOI4D/HOI4D_release/ZY20210800001/H1/C1/N19/S100/s02/T1/align_depth/00000.png
        # mask: data/HOI4D/HOI4D_release/ZY20210800001/H1/C1/N19/S100/s02/T1/2Dseg/mask/00000.png
        
        rgb_dir = Path(self.data_dir) / 'HOI4D_release' / relative_path / 'align_rgb'
        depth_dir = Path(self.data_dir) / 'HOI4D_release' / relative_path / 'align_depth'
        mask_dir = Path(self.data_dir) / 'HOI4D_release' / relative_path / '2Dseg' / 'mask' if relative_path.split('/')[0] == 'ZY20210800001' else Path(self.data_dir) / 'HOI4D_release' / relative_path / '2Dseg' / 'shift_mask'
        
        # Format frame index as 5-digit number
        frame_str = f"{frame_idx:05d}.jpg"  # HOI4D uses jpg for RGB
        depth_str = f"{frame_idx:05d}.png"  # PNG for depth
        mask_str = f"{frame_idx:05d}.png"   # PNG for mask
        
        rgb_path = rgb_dir / frame_str
        depth_path = depth_dir / depth_str
        mask_path = mask_dir / mask_str
        
        # Load RGB
        if rgb_path.exists():
            rgb = cv2.imread(str(rgb_path))
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        else:
            raise FileNotFoundError(f"RGB file not found: {rgb_path}")
        
        # Load depth
        if depth_path.exists():
            depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            depth = depth.astype(np.float32) / 1000.0  # Convert to meters
        else:
            raise FileNotFoundError(f"Depth file not found: {depth_path}")
        
        # Load mask if available
        mask = None
        if mask_path.exists():
            mask_color = cv2.imread(str(mask_path), cv2.IMREAD_COLOR)
            mask = ((mask_color[:, :, 0] == 0) & (mask_color[:, :, 2] == 0)) | ((mask_color[:, :, 0] == 0) & (mask_color[:, :, 1] == 128) & (mask_color[:, :, 2] == 128))
            mask = ~mask
            mask = mask.astype(np.uint8)
        
        # Apply mask if requested
        if mask_hand and mask is not None:
            rgb = rgb * mask[:, :, None]
            depth = depth * mask
        
        return rgb, depth