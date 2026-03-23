import os
import glob
import torch
import numpy as np
from typing import List, Optional, Tuple
import cv2
import h5py
from pathlib import Path

from src.dataset.base_retarget import BaseRetargetDataset

torch.multiprocessing.set_start_method('spawn', force=True)

class H2oDataset(BaseRetargetDataset):
    # Action label mapping from H2O dataset
    ACTION_LABELS = {
        0: "background",
        1: "grab book", 2: "grab espresso", 3: "grab lotion", 4: "grab spray",
        5: "grab milk", 6: "grab cocoa", 7: "grab chips", 8: "grab cappuccino",
        9: "place book", 10: "place espresso", 11: "place lotion", 12: "place spray",
        13: "place milk", 14: "place cocoa", 15: "place chips", 16: "place cappuccino",
        17: "open lotion", 18: "open milk", 19: "open chips",
        20: "close lotion", 21: "close milk", 22: "close chips",
        23: "pour milk",
        24: "take out espresso", 25: "take out cocoa", 26: "take out chips", 27: "take out cappuccino",
        28: "put in espresso", 29: "put in cocoa", 30: "put in cappuccino",
        31: "apply lotion", 32: "apply spray",
        33: "read book", 34: "read espresso",
        35: "spray spray", 36: "squeeze lotion"
    }
    PCD_MASK = np.array([[-1.0, -0.3, -1.0], [0.6, 0.5, 0.3]])
    
    def _parse_action_labels(self, action_label_path):
        """Parse action labels using skip sampling and binary search for change detection, filter out all-0 label seqs"""
        if not action_label_path or not action_label_path.exists():
            return []
        action_label_files = sorted(glob.glob(str(action_label_path / '*.txt')))
        if not action_label_files:
            return []
        skip_step = 64
        sampled_labels = []
        sampled_indices = []
        for i in range(0, len(action_label_files), skip_step):
            try:
                with open(action_label_files[i], 'r') as f:
                    label = int(f.read().strip())
                    sampled_labels.append(label)
                    sampled_indices.append(i)
            except (ValueError, FileNotFoundError):
                sampled_labels.append(0)
                sampled_indices.append(i)
        full_labels = [0] * len(action_label_files)
        for idx in range(len(sampled_labels) - 1):
            current_idx = sampled_indices[idx]
            next_idx = sampled_indices[idx + 1]
            current_label = sampled_labels[idx]
            next_label = sampled_labels[idx + 1]
            full_labels[current_idx] = current_label
            if current_label != next_label:
                change_point = self._binary_search_label_change_in_files(
                    action_label_files, current_idx, next_idx, current_label
                )
                for i in range(current_idx + 1, change_point):
                    full_labels[i] = current_label
                for i in range(change_point, next_idx):
                    full_labels[i] = next_label
            else:
                for i in range(current_idx + 1, next_idx):
                    full_labels[i] = current_label
        if sampled_indices:
            last_idx = sampled_indices[-1]
            last_label = sampled_labels[-1]
            full_labels[last_idx] = last_label
            for i in range(last_idx + 1, len(action_label_files)):
                full_labels[i] = last_label
        if all(l == 0 for l in full_labels):
            return []
        return full_labels

    def _binary_search_label_change_in_files(self, action_label_files, start_idx, end_idx, target_label):
        left, right = start_idx, end_idx
        
        while left < right:
            mid = (left + right) // 2
            try:
                with open(action_label_files[mid], 'r') as f:
                    label = int(f.read().strip())
            except (ValueError, FileNotFoundError):
                self.log(f"Warning: Failed to read action label file {action_label_files[mid]}")
                label = 0
                
            if label == target_label:
                left = mid + 1
            else:
                right = mid
                
        return left

    def _split_sequence_by_action_labels(self, base_seq_info, action_labels):
        """Split a sequence into subsequences based on action labels"""
        if not action_labels:
            return []
        
        subsequences = []
        current_label = None
        start_idx = 0
        
        for i, label in enumerate(action_labels):
            # Start a new subsequence when label changes
            if label != current_label:
                # Save previous subsequence if it was non-background and had sufficient length
                if current_label is not None and current_label != 0 and i - start_idx >= self.chunk_size + max(self.state_horizon, self.pcd_horizon):
                    subseq_info = base_seq_info.copy()
                    subseq_info.update({
                        'action_label': current_label,
                        'start_frame': start_idx,
                        'end_frame': i - 1,
                        'prompt': f"Use {base_seq_info['hand_type']} hands to {self.ACTION_LABELS[current_label]}."
                    })
                    subsequences.append(subseq_info)
                
                # Start new subsequence
                current_label = label
                start_idx = i
        
        # Handle the last subsequence
        if current_label is not None and current_label != 0 and len(action_labels) - start_idx >= self.chunk_size + max(self.state_horizon, self.pcd_horizon):
            subseq_info = base_seq_info.copy()
            subseq_info.update({
                'action_label': current_label,
                'start_frame': start_idx,
                'end_frame': len(action_labels) - 1,
                'prompt': f"Use {base_seq_info['hand_type']} hands to {self.ACTION_LABELS[current_label]}."
            })
            subsequences.append(subseq_info)
        
        return subsequences

    def _find_sequences(self):
        """Find all valid retarget sequences for H2o dataset and split by action labels"""
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
                
                # Load total frames to determine sequence length
                try:
                    with h5py.File(h5_file, 'r') as f:
                        total_frames = f['metadata'].attrs['total_frames']
                except Exception as e:
                    self.log(f"Warning: Failed to read metadata from {h5_file}: {e}")
                    continue
                
                # Parse action labels from external files
                # Convert retarget path to action label path
                # relative_path format: subject1_ego/h1/0/cam4
                path_parts = str(relative_path).split('/')
                if len(path_parts) >= 4:
                    subject, session, sequence, camera = path_parts[0], path_parts[1], path_parts[2], path_parts[3]
                    action_label_path = Path(self.data_dir) / 'annotation' / subject / session / sequence / camera / 'action_label'
                    action_labels = self._parse_action_labels(action_label_path)
                else:
                    self.log(f"Warning: Invalid relative path format: {relative_path}")
                    action_labels = []
                
                base_seq_info = {
                    'h5_path': str(h5_file),
                    'relative_path': str(relative_path),
                    'hand_type': hand_type,
                }
                
                # Split sequence by action labels if available
                subsequences = self._split_sequence_by_action_labels(base_seq_info, action_labels)
                if not subsequences:
                    # If no action-based subsequences, add the original sequence
                    sequences.append(base_seq_info)
                else:
                    sequences.extend(subsequences)
        return sequences

    def _build_window(self, seq_info):
        """Build window data for H2o sequence"""
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
        """Load prompt for H2o sequence based on action labels"""
        return window['seq_info']['prompt']

    def _load_scene_rgbd(self, relative_path: str, frame_idx: int, mask_hand: bool = False) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Load RGB, depth, and mask images for H2o dataset"""
        # H2o paths:
        # rgb: data/H2o/all_img/subject1_ego/h1/0/cam4/rgb/000000.png
        # depth: data/H2o/all_img/subject1_ego/h1/0/cam4/depth/000000.png
        # mask: data/H2o/all_img/subject1_ego/h1/0/cam4/mask/000000.png
        
        rgb_dir = Path(self.data_dir) / 'all_img' / relative_path / 'rgb'
        depth_dir = Path(self.data_dir) / 'all_img' / relative_path / 'depth'
        mask_dir = Path(self.data_dir) / 'all_img' / relative_path / 'mask'
        
        # Format frame index as 6-digit number
        frame_str = f"{frame_idx:06d}.png"
        
        rgb_path = rgb_dir / frame_str
        depth_path = depth_dir / frame_str
        mask_path = mask_dir / frame_str
        
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
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            mask = (mask > 0).astype(np.uint8)
        
        # Apply mask if requested
        if mask_hand and mask is not None:
            rgb = rgb * mask[:, :, None]
            depth = depth * mask
        
        return rgb, depth