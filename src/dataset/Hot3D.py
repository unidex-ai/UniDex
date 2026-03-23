"""
Hot3D Retarget Dataset implementation
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

class Hot3DDataset(BaseRetargetDataset):
    # Pointcloud mask for Hot3D dataset
    PCD_MASK = None  # Hot3D doesn't use PCD_MASK in original
    
    # Transform matrix for coordinate conversion
    TRANSFORM = np.array([
        [0, 1, 0, 0],
        [-1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    def _find_sequences(self):
        """Find all valid retarget sequences for Hot3D dataset"""
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
                
                # Hot3D has prompt.txt files for each sequence
                prompt_file = Path(self.data_dir) / relative_path / 'prompt.txt'
                if prompt_file.exists():
                    # Parse prompt file to create subsequences
                    subsequences = self._parse_prompt_file(prompt_file, h5_file, relative_path, hand_type)
                    sequences.extend(subsequences)
                else:
                    # If no prompt file, create a single sequence
                    self.log(f"Warning: Prompt file not found for {h5_file}, skipping.")
                    
                
        return sequences
    
    def _parse_prompt_file(self, prompt_file, h5_file, relative_path, hand_type):
        """Parse prompt.txt file to create meaningful subsequences"""
        subsequences = []
        
        try:
            with open(prompt_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Load total frames from H5 file
            with h5py.File(h5_file, 'r') as f:
                total_frames = int(f['metadata'].attrs['total_frames'])
                timestamps = f['frames']['timestamps'][:]
            timestamps = [int(ts.decode('utf-8')) for ts in timestamps]
            
            for i, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                    
                # Parse format: "action_description start_frame end_frame"
                line = ' '.join(line.split())  # Normalize spaces
                parts = line.rsplit(' ', 2)  # Split from right to get last 2 numbers
                if len(parts) != 3:
                    continue
                    
                action_description = parts[0]

                try:
                    start_ts = int(parts[1])
                    end_ts = int(parts[2])
                    if not (timestamps[0] <= start_ts <= timestamps[-1]) and (timestamps[0] <= end_ts <= timestamps[-1]) and (start_ts <= end_ts):
                        self.log(f"Invalid timestamps {parts[1], parts[2]} in line {i+1} of {prompt_file}: {line}")
                        continue
                    start_frame = total_frames - 1
                    end_frame = total_frames - 1
                    for idx in range(total_frames - 1, -1, -1):
                        if timestamps[idx] > start_ts:
                            start_frame = idx
                        if timestamps[idx] > end_ts:
                            end_frame = idx
                except ValueError:
                    self.log(f"Invalid frame indices {parts[1], parts[2]} in line {i+1} of {prompt_file}: {line}")
                    continue

                # Verify the frame range is valid
                if not (0 <= start_frame < total_frames and 0 <= end_frame < total_frames and start_frame <= end_frame):
                    self.log(f"Invalid frame range {start_frame}-{end_frame} in line {i+1} of {prompt_file}")
                    continue
                    
                # Check if subsequence is long enough
                subseq_length = end_frame - start_frame + 1
                if subseq_length < self.chunk_size + max(self.state_horizon, self.pcd_horizon):
                    self.log(f"Subsequence too short for {h5_file.name} action {i+1}: {subseq_length} frames")
                    continue
                
                # Create subsequence info
                seq_key = f"{h5_file.stem}_action_{i:02d}"
                prompt = f"Use {hand_type} hands to {action_description}."
                subseq_info = {
                    'seq_key': seq_key,
                    'h5_path': str(h5_file),
                    'relative_path': str(relative_path),
                    'hand_type': hand_type,
                    'prompt': prompt,
                    'start_frame': start_frame,
                    'end_frame': end_frame,
                }
                subsequences.append(subseq_info)
                
        except Exception as e:
            self.log(f"Error parsing prompt file {prompt_file}: {e}")
            
        return subsequences

    def _build_window(self, seq_info):
        """Build window data for Hot3D sequence"""
        try:
            # Load only total frames from metadata (memory efficient)
            with h5py.File(seq_info['h5_path'], 'r') as f:
                total_frames = f['metadata'].attrs['total_frames']
            
            # Apply stride to reduce frame rate
            available_frames = list(range(0, total_frames, self.sample_stride))
            
            # If this is a subsequence based on action labels, apply frame range
            if 'start_frame' in seq_info and 'end_frame' in seq_info:
                subseq_start = max(0, seq_info['start_frame'] // self.sample_stride)
                subseq_end = min(len(available_frames), (seq_info['end_frame'] + 1) // self.sample_stride)
                available_frames = available_frames[subseq_start:subseq_end]
            
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
        """Load prompt for Hot3D sequence based on parsed prompt file"""
        return window['seq_info']['prompt']

    def _load_scene_rgbd(self, relative_path: str, frame_idx: int, mask_hand: bool = False) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Load RGB, depth, and mask images for Hot3D dataset"""
        # Hot3D paths:
        # rgb: data/hot3d/P0001_4bf4e21a/RGB/59211074860121.jpg
        # depth: data/hot3d/P0001_4bf4e21a/depth/59211074860121.tiff
        # mask: data/hot3d/P0001_4bf4e21a/masks/59212241516144.jpg
        
        rgb_dir = Path(self.data_dir) / relative_path / 'RGB'
        depth_dir = Path(self.data_dir) / relative_path / 'depth'
        mask_dir = Path(self.data_dir) / relative_path / 'masks'
        
        h5_file = (Path(self.data_dir) / 'retarget_RGBD' / relative_path).glob('*.h5')
        h5_file = list(h5_file)[0]
        with h5py.File(h5_file, 'r') as f:
            timestamps = f['frames']['timestamps'][:]
        timestamps = [int(ts.decode('utf-8')) for ts in timestamps]
        frame_name = timestamps[frame_idx]
        frame_str = f"{frame_name}.jpg"  # Hot3D RGB uses jpg
        depth_str = f"{frame_name}.tiff"  # Hot3D depth uses tiff
        mask_str = f"{frame_name}.jpg"   # Hot3D mask uses jpg
        
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
            depth = depth.astype(np.float32)
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