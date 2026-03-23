"""
Taco Retarget Dataset implementation
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

class TacoDataset(BaseRetargetDataset):
    # Pointcloud mask for Taco dataset  
    PCD_MASK = np.array([[-0.4, -0.3, -0.9], [0.3, 0.4, -0.45]])

    def _find_sequences(self):
        """Find all valid retarget sequences for Taco dataset"""
        sequences = []
        
        # Look for HDF5 files in retarget_RGBD directory
        retarget_dir = Path(self.data_dir) / 'retarget_RGBD'
        
        for hand_type in self.hands:
            # Find all H5 files for this hand type
            h5_pattern = f"**/{hand_type}.h5"
            h5_files = list(retarget_dir.glob(h5_pattern))
            
            for h5_file in h5_files:
                # Extract relative path from retarget_RGBD
                relative_path = h5_file.relative_to(retarget_dir).parent
                prime_relative_path = str(relative_path).split('/')[0]
                words = prime_relative_path[1:-1].split(',')
                prompt_content = f'{words[0].strip()} the {words[2].strip()}'
                
                prompt = f"Use {hand_type} hands to {prompt_content}."
                
                seq_info = {
                    'h5_path': str(h5_file),
                    'relative_path': str(relative_path),
                    'hand_type': hand_type,
                    'prompt': prompt
                }
                sequences.append(seq_info)
                
        return sequences

    def _build_window(self, seq_info):
        """Build window data for Taco sequence"""
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
        """Load prompt for Taco sequence"""
        return window['seq_info']['prompt']

    def _load_scene_rgbd(self, relative_path: str, frame_idx: int, mask_hand: bool = False) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Load RGB, depth, and mask images for Taco dataset"""
        # Taco paths:
        # rgb: data/Taco/Egocentric_RGB_Videos/(brush, brush, bowl)/20230919_036/colorframes/00000.jpg
        # depth: data/Taco/Egocentric_Depth_Videos/(brush, brush, bowl)/20230919_036/depthframes/00000.png
        # mask: data/Taco/Hand_Masks/(brush, brush, bowl)/20230927_027/00000.png
        
        # Extract task and session from relative_path
        # relative_path format: (brush, brush, bowl)/20230919_036
        parts = str(relative_path).split('/')
        task_name = parts[0]
        session_name = parts[1]
        
        rgb_dir = Path(self.data_dir) / 'Egocentric_RGB_Videos' / task_name / session_name / 'colorframes'
        depth_dir = Path(self.data_dir) / 'Egocentric_Depth_Videos' / task_name / session_name / 'depthframes'
        mask_dir = Path(self.data_dir) / 'Hand_Masks' / task_name / session_name
        
        rgb_path = rgb_dir / f"{frame_idx:05d}.jpg"
        depth_path = depth_dir / f"{frame_idx:05d}.png"
        mask_path = mask_dir / f"{frame_idx:05d}.png"
        
        # Load RGB
        if rgb_path.exists():
            rgb = cv2.imread(str(rgb_path))
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        else:
            raise FileNotFoundError(f"RGB file not found: {rgb_path}")
        
        # Load depth
        if depth_path.exists():
            depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            depth = depth.astype(np.float32) / 4000.0
        else:
            raise FileNotFoundError(f"Depth file not found: {depth_path}")
        
        # Load mask if available
        mask = None
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            mask = (mask > 0).astype(np.uint8)
        
        # Apply mask if requested
        if mask_hand and mask is not None:
            rgb = rgb * mask[:, :, None]
            depth = depth * mask
        
        return rgb, depth
