"""
Real Dataset implementation for loading real-world manipulation data from zarr files.
"""
import os
import cv2
import zarr
import numpy as np
import open3d as o3d
import hashlib
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from src.dataset.base import BaseDataset
from src.utils.normalizers import Normalizer
from src.utils import pose as pose_utils, hand_utils
from scipy.spatial.transform import Rotation as R


class RealDataset(BaseDataset):
    """
    Real Dataset for loading real-world manipulation data from zarr files.
    
    Expected zarr structure (unified format):
    ├── data
    │   ├── action (N, 26) float64
    │   ├── camera0_intrinsic (N, 3, 3) float64
    │   ├── camera0_pointcloud (N, 10000, 6) float32
    │   ├── camera0_real_timestamp (N,) float64
    │   ├── camera0_rgb (N, 224, 224, 3) uint8
    │   ├── embodiment (N, 1) float64
    │   ├── gripper0_gripper_force (N, 20) float64
    │   ├── gripper0_gripper_pose (N, 20) float64
    │   ├── robot0_eef_pos (N, 3) float64
    │   ├── robot0_eef_rot_axis_angle (N, 3) float64
    │   ├── robot0_joint_pos (N, 7) float64
    │   ├── robot0_joint_vel (N, 7) float64
    │   ├── source_idx (N, 1) float64
    │   └── timestamp (N,) float64
    └── meta
        └── window_ends (K,) int64
    """
    
    # Transform matrix for camera coordinate conversion
    CV_TO_CAM = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)
    
    PCD_MASK = np.array([-0.04138259,  0.3303094 ,  0.88850776, 0.78]) # To clip table and background points
    EXTRINSICS = np.array([
        [0.992855, 0.116778, -0.024527, 0.079333],
        [-0.098651, 0.918941, 0.381858, -0.005191],
        [0.067132, -0.376710, 0.923896, -0.924145],
        [0.000000, 0.000000, 0.000000, 1.000000]
    ], dtype=np.float32) # From calibration

    
    def __init__(
        self,
        data_dir: str,
        data_dirs: List[Dict],  # List of dicts with keys: relative_path, hand_type, action, sample_stride
        chunk_size: int = 30,
        pointcloud_size: int = 1024,
        cache_data: bool = True,
        use_cached_metadata: bool = False,
        cache_dir: Optional[str] = None,
        sample_stride: int = 1,  # Default sample_stride, can be overridden per zarr
        state_horizon: int = 1,
        pcd_horizon: int = 1,
        normalizer: Normalizer | None = None,
        interpolation_factor: int = 1,  # Default interpolation factor, can be overridden per zarr
        hands: Optional[List[str]] = None,
        use_generated_data: bool = True,  # New parameter: use generated augmented data when available
        camera_angle_peturb: float = 0.0,
        camera_pos_peturb: float = 0.0,
        **kwargs
    ):
        """
        Initialize Real Dataset
        
        Args:
            data_dir: Base directory containing zarr files
            data_dirs: List of dicts with keys:
                      - relative_path: path to zarr file relative to data_dir
                      - hand_type: hand type name (e.g., 'Wuji', 'Inspire', etc.)
                      - action: action description (e.g., 'spray', 'grasp', etc.)
                      - sample_stride: (optional) custom sample stride for this zarr file
                      - generated_data_dir: (optional) path to generated data dir for this zarr
            sample_stride: Default sample stride used if not specified per zarr file
            use_generated_data: If True, also include generated augmented data when available
            Other args: Same as BaseDataset
        """
        
        # Convert data_dirs format and store with sample_stride, generated_data_dir, and interpolation_factor info
        self.data_dirs_with_stride = []
        for data_dir_info in data_dirs:
            rel_path = data_dir_info['relative_path']
            hand_type = data_dir_info['hand_type']
            hand_side = data_dir_info.get('hand_side', 'Right')  # New: explicit side
            action = data_dir_info['action']
            stride = data_dir_info['sample_stride']
            gen_data_dir = data_dir_info['generated_data_dir']
            interp_factor = data_dir_info['interpolation_factor']
            sequence_num = data_dir_info.get('sequence_num', None) # Get sequence_num, default to None

            self.data_dirs_with_stride.append({
                'relative_path': rel_path,
                'hand_type': hand_type,
                'hand_side': hand_side.lower(),
                'action': action,
                'sample_stride': stride,
                'generated_data_dir': gen_data_dir,
                'interpolation_factor': interp_factor,
                'sequence_num': sequence_num, # Store sequence_num
                'zarr_format_version': 1, # Default to v1, will be updated in _find_sequences
            })

        # Keep original data_dirs format for compatibility
        self.data_dirs = [(info['relative_path'], info['hand_type'], info['action']) 
                          for info in self.data_dirs_with_stride]
        self.use_generated_data = use_generated_data
        self.camera_angle_peturb = camera_angle_peturb
        self.camera_pos_peturb = camera_pos_peturb
        
        # Call parent constructor
        super().__init__(
            data_dir=data_dir,
            chunk_size=chunk_size,
            pointcloud_size=pointcloud_size,
            cache_data=cache_data,
            use_cached_metadata=use_cached_metadata,
            cache_dir=cache_dir,
            sample_stride=sample_stride,
            state_horizon=state_horizon,
            pcd_horizon=pcd_horizon,
            normalizer=normalizer,
            hands=hands,
            interpolation_factor=interpolation_factor,
            **kwargs
        )
        
        # Additional filtering: keep only sequences/windows that are in data_dirs
        self._filter_by_data_dirs()

    def _filter_by_data_dirs(self):
        """Filter sequences & windows by current config AND enforce per-zarr sequence_num limits.

        Rationale:
        - Cached metadata may contain more sequences than currently desired (sequence_num newly added or changed).
        - We first filter by presence in current config (data_dirs), then apply sequence_num truncation per relative_path.
        - sequence_num counts both original and generated sequences in the same ordering used during _find_sequences
          (original sequences first, then generated sequences ordered by generation_idx, window_idx).
        """
        valid_paths = {str(self.data_dir / rel_path) for rel_path, _, _ in self.data_dirs}
        original_seq_count = len(self.sequences)
        self.sequences = [
            seq for seq in self.sequences
            if str(seq['zarr_path']) in valid_paths or (
                seq.get('is_generated') and str(seq.get('original_zarr_path')) in valid_paths
            )
        ]

        # Apply use_generated_data filter
        if not self.use_generated_data:
            self.sequences = [
                seq for seq in self.sequences if not seq.get('is_generated', False)
            ]


        original_ep_count = len(self.windows)
        self.windows = [
            ep for ep in self.windows
            if str(ep['seq_info']['zarr_path']) in valid_paths or (
                ep['seq_info'].get('is_generated') and str(ep['seq_info'].get('original_zarr_path')) in valid_paths
            )
        ]

        # Apply use_generated_data filter to windows
        if not self.use_generated_data:
            self.windows = [
                ep for ep in self.windows if not ep['seq_info'].get('is_generated', False)
            ]

        relpath_to_limit: dict[str, int | None] = {}
        relpath_order: list[str] = []
        for info in self.data_dirs_with_stride:
            relpath_order.append(info['relative_path'])
            relpath_to_limit[info['relative_path']] = info.get('sequence_num', None)

        if any(limit is not None for limit in relpath_to_limit.values()):
            seqs_by_rel = {}
            for seq in self.sequences:
                seqs_by_rel.setdefault(seq['relative_path'], []).append(seq)

            new_sequences = []
            truncation_logs = []
            for rel_path in relpath_order:
                if rel_path not in seqs_by_rel:
                    continue
                seq_list = seqs_by_rel[rel_path]
                before = len(seq_list)
                limit = relpath_to_limit.get(rel_path)
                if limit is not None:
                    seq_list = seq_list[:limit]
                after = len(seq_list)
                if before != after:
                    truncation_logs.append(f"{rel_path}: {before}->{after} (limit={limit})")
                new_sequences.extend(seq_list)
            self.sequences = new_sequences

            # Filter windows to only those whose seq_info key remains
            allowed_keys = set(
                (s['relative_path'], s['window_idx'], s.get('is_generated', False), s.get('generation_idx', -1))
                for s in self.sequences
            )
            ep_before_limit = len(self.windows)
            self.windows = [
                ep for ep in self.windows
                if (ep['seq_info']['relative_path'], ep['seq_info']['window_idx'], ep['seq_info'].get('is_generated', False), ep['seq_info'].get('generation_idx', -1)) in allowed_keys
            ]
            if truncation_logs:
                self.log("Applied sequence_num limits: " + "; ".join(truncation_logs))
                self.log(f"windows after sequence_num limiting: {ep_before_limit}->{len(self.windows)}")

        self.log(f"Filtered sequences (path filter): {original_seq_count} -> {len(self.sequences)}")
        self.log(f"Filtered windows (path filter & seq limits): {original_ep_count} -> {len(self.windows)}")

        # 4. Recompute pcd_needed
        self.pcd_needed = [pcd_paths for ep in self.windows for pcd_paths in ep['pcd_paths']]

    def _validate_zarr_format_v1(self, root: zarr.Group) -> bool:
        """Validate zarr file for the original unified format (teleop data)."""
        if 'data' not in root:
            return False
        
        data_keys = set(root['data'].keys())
        
        # Check for required keys in new format
        required_keys = {
            'action', 'camera0_intrinsic', 'camera0_pointcloud', 'camera0_rgb',
            'gripper0_gripper_pose', 'robot0_eef_pos', 'robot0_eef_rot_axis_angle'
        }
        
        if not required_keys.issubset(data_keys):
            missing = required_keys - data_keys
            self.log(f"V1 format validation failed. Missing keys: {missing}")
            return False
        
        return True

    def _validate_zarr_format_v2(self, root: zarr.Group) -> bool:
        """Validate zarr file for the second format (retarget data)."""
        if 'data' not in root:
            return False
            
        data_keys = set(root['data'].keys())
        
        required_keys = {
            'pointcloud', 'right_joint_poses', 'right_joint_values'
        }
        
        if not required_keys.issubset(data_keys):
            missing = required_keys - data_keys
            self.log(f"V2 format validation failed. Missing keys: {missing}")
            return False
            
        return True

    def _validate_zarr_format(self, zarr_path: Path) -> bool:
        """
        Validate zarr file format - supports both v1 and v2 formats.
        """
        try:
            root = zarr.open(str(zarr_path), mode='r')
            
            if 'data' not in root:
                return False
            
            # Check for either v1 or v2 format
            if self._validate_zarr_format_v1(root) or self._validate_zarr_format_v2(root):
                return True
            else:
                self.log(f"Validation failed for both v1 and v2 formats.")
                return False
            
        except Exception as e:
            self.log(f"Error validating format for {zarr_path}: {e}")
            return False

    def _find_sequences(self) -> List[Dict]:
        """Find all valid sequences from zarr files, with specific ordering for sequence_num."""
        sequences = []
        
        for i in range(len(self.data_dirs_with_stride)):
            data_info = self.data_dirs_with_stride[i]
            rel_path = data_info['relative_path']
            hand_type = data_info['hand_type']
            hand_side = data_info.get('hand_side', 'right').lower()
            action = data_info['action']
            sample_stride = data_info['sample_stride']
            interpolation_factor = data_info['interpolation_factor']
            sequence_num_limit = data_info.get('sequence_num')

            zarr_path = Path(self.data_dir) / rel_path
            
            if not zarr_path.exists():
                self.log(f"Warning: zarr file not found: {zarr_path}")
                continue
            
            version = self._validate_zarr_file(zarr_path)
            if version == 0:
                self.log(f"Warning: Invalid zarr structure for both v1 and v2: {zarr_path}")
                continue
            
            self.data_dirs_with_stride[i]['zarr_format_version'] = version
            data_info['zarr_format_version'] = version

            # Temporary lists for sorting according to the new logic
            original_sequences_for_zarr = []
            generated_sequences_for_zarr = []

            # 1. Collect original data sequences
            root = zarr.open(str(zarr_path), mode='r')
            window_ends = root['meta']['window_ends'][:]
            window_starts = [0] + window_ends[:-1].tolist()
            
            for ep_idx, (start_frame, end_frame) in enumerate(zip(window_starts, window_ends)):
                original_sequences_for_zarr.append({
                    'zarr_path': zarr_path,
                    'relative_path': rel_path,
                    'hand_type': hand_type,
                    'action': action,
                    'hand_side': hand_side,
                    'window_idx': ep_idx,
                    'start_frame': start_frame,
                    'end_frame': end_frame,
                    'total_frames': end_frame - start_frame,
                    'sample_stride': sample_stride,
                    'interpolation_factor': interpolation_factor,
                    'is_generated': False,
                    'generation_idx': -1, # Use -1 for sorting originals first
                    'original_start_frame': None,
                    'zarr_format_version': version,
                })

            # 2. Collect generated augmented data
            generated_data_dir = data_info['generated_data_dir']
            if self.use_generated_data and generated_data_dir:
                generated_dir = zarr_path / generated_data_dir
                if generated_dir.exists():
                    generated_zarr_files = list(generated_dir.glob("*.zarr"))
                    
                    for gen_zarr_path in generated_zarr_files:
                        if self._validate_generated_zarr_file(gen_zarr_path):
                            filename = gen_zarr_path.name
                            parts = filename.replace('.zarr', '').split('_')
                            
                            ep_idx = None
                            for part in parts:
                                if part.startswith('ep') and part[2:].isdigit():
                                    ep_idx = int(part[2:])
                                    break
                            
                            if ep_idx is None:
                                self.log(f"Warning: Could not parse window index from {filename}")
                                continue

                            gen_root = zarr.open(str(gen_zarr_path), mode='r')
                            gen_window_ends = gen_root['meta']['window_ends'][:]
                            gen_window_starts = [0] + gen_window_ends[:-1].tolist()
                            
                            original_window_starts = [0] + root['meta']['window_ends'][:-1].tolist()
                            
                            for gen_idx, (start_frame, end_frame) in enumerate(zip(gen_window_starts, gen_window_ends)):
                                generated_sequences_for_zarr.append({
                                    'zarr_path': gen_zarr_path,
                                    'relative_path': str(gen_zarr_path.relative_to(self.data_dir)),
                                    'hand_type': hand_type,
                                    'action': action,
                                    'hand_side': hand_side,
                                    'window_idx': ep_idx,
                                    'start_frame': start_frame,
                                    'end_frame': end_frame,
                                    'total_frames': end_frame - start_frame,
                                    'sample_stride': sample_stride,
                                    'interpolation_factor': interpolation_factor,
                                    'is_generated': True,
                                    'generation_idx': gen_idx,
                                    'original_zarr_path': zarr_path,
                                    'original_start_frame': original_window_starts[ep_idx],
                                    'zarr_format_version': version,
                                })
                        else:
                            self.log(f"Warning: Invalid generated zarr structure: {gen_zarr_path}")
                else:
                    self.log(f"Generated data directory not found: {generated_dir}")

            # 3. Sort generated sequences by generation_idx, then window_idx
            generated_sequences_for_zarr.sort(key=lambda s: (s['generation_idx'], s['window_idx']))
            
            # 4. Combine and apply limit
            all_sequences_for_zarr = original_sequences_for_zarr + generated_sequences_for_zarr
            
            if sequence_num_limit is not None:
                sequences.extend(all_sequences_for_zarr[:sequence_num_limit])
            else:
                sequences.extend(all_sequences_for_zarr)
        
        # Log statistics
        original_count = sum(1 for seq in sequences if not seq['is_generated'])
        generated_count = sum(1 for seq in sequences if seq['is_generated'])
        self.log(f"Found {len(sequences)} total sequences: {original_count} original + {generated_count} generated")
        
        return sequences

    def _validate_generated_zarr_file(self, zarr_path: Path) -> bool:
        """Validate generated zarr file structure"""
        try:
            root = zarr.open(str(zarr_path), mode='r')
            
            # Check required top-level groups
            if 'data' not in root or 'meta' not in root:
                return False
            
            # Check for required generated data keys
            data_keys = set(root['data'].keys())
            required_keys = {
                'pointcloud_camera0_base',  # Generated pointcloud data
                'state_camera0_base'        # Generated wrist poses
            }
            
            if not required_keys.issubset(data_keys):
                missing = required_keys - data_keys
                self.log(f"Missing required generated data keys: {missing}")
                return False
            
            # Check meta keys
            if 'window_ends' not in root['meta']:
                return False
            
            return True
            
        except Exception as e:
            self.log(f"Error validating generated zarr format for {zarr_path}: {e}")
            return False

    def _validate_zarr_file(self, zarr_path: Path) -> int:
        """Validate zarr file structure, returning version number (1 or 2) or 0 for invalid."""
        try:
            root = zarr.open(str(zarr_path), mode='r')
            
            # Check required top-level groups
            if 'data' not in root or 'meta' not in root:
                return 0
            
            if self._validate_zarr_format_v1(root):
                self.log(f"Zarr file {zarr_path} validated as format v1")
                return 1
            
            if self._validate_zarr_format_v2(root):
                self.log(f"Zarr file {zarr_path} validated as format v2")
                return 2
            
            return 0
                
        except Exception as e:
            self.log(f"Error validating zarr file {zarr_path}: {e}")
            return 0

    def _build_window(self, seq_info) -> List[Dict]:
        """Build window data from a single window sequence, supporting both original and generated data"""
        zarr_path = seq_info['zarr_path']
        start_frame = seq_info['start_frame']
        end_frame = seq_info['end_frame']
        is_generated = seq_info['is_generated']
        seq_sample_stride = seq_info['sample_stride']
        seq_interpolation_factor = seq_info['interpolation_factor']
        
        effective_chunk_size = self.chunk_size // seq_interpolation_factor
        assert effective_chunk_size > 0, f"chunk_size {self.chunk_size} must be divisible by interpolation_factor {seq_interpolation_factor}"
        
        available_frames = list(range(start_frame, end_frame, seq_sample_stride))
        
        if is_generated:
            self.log(f"Generated window {seq_info['window_idx']}-{seq_info['generation_idx']}: using stride {seq_sample_stride}, interpolation {seq_interpolation_factor}, {len(available_frames)} frames")
        else:
            self.log(f"Original window {seq_info['window_idx']}: using stride {seq_sample_stride}, interpolation {seq_interpolation_factor}, {len(available_frames)} frames")
        
        min_required_frames = effective_chunk_size + max(self.state_horizon, self.pcd_horizon)
            
        if len(available_frames) < min_required_frames:
            self.log(f"window {seq_info} too short: {len(available_frames)} frames (need {min_required_frames})")
            return []
        
        windows = []
        
        # Use consistent overlap strategy for both original and generated data
        step_size = max(1, effective_chunk_size // 6)
        min_start_idx = max(self.state_horizon, self.pcd_horizon)
            
        for start_idx in range(min_start_idx, 
                             len(available_frames) - effective_chunk_size + 1,
                             step_size):
            actual_start_frame = available_frames[start_idx - max(self.state_horizon, self.pcd_horizon)]
            actual_end_frame = available_frames[start_idx + effective_chunk_size - 1]
            
            # Build pcd_paths for this window
            pcd_paths = []
            for t in range(start_idx - self.pcd_horizon, start_idx):
                frame_idx = available_frames[t]
                pcd_paths.append((zarr_path, frame_idx))
            
            window = {
                'seq_info': seq_info.copy(),
                'window_idx': seq_info['window_idx'],
                'start_idx': start_idx,
                'end_idx': start_idx + effective_chunk_size,
                'chunk_size': effective_chunk_size,
                'interpolation_factor': seq_interpolation_factor,
                'sample_stride': seq_sample_stride, 
                'available_frames': available_frames,
                'actual_start_frame': actual_start_frame,
                'actual_end_frame': actual_end_frame,
                'pcd_paths': pcd_paths,
                'is_generated': is_generated,
                'generation_idx': seq_info['generation_idx'],
                'zarr_format_version': seq_info['zarr_format_version'],
            }
            
            windows.append(window)
        
        window_type = "generated" if is_generated else "original"
        generation_info = f"-{seq_info['generation_idx']}" if is_generated else ""
        self.log(f"Built {len(windows)} sub-windows from {window_type} window {seq_info['window_idx']}{generation_info} ({zarr_path.name})")
        return windows

    def _load_raw_pointcloud(self, pcd_paths) -> np.ndarray:
        """Load pointcloud data from the unified format or generated data"""
        zarr_path, frame_idx = pcd_paths
        
        root = zarr.open(str(zarr_path), mode='r')
        
        # Check if this is generated data by looking for generated-specific keys
        if 'pointcloud_camera0_base' in root['data']:
            # This is generated data, use the generated pointcloud
            pointcloud = root['data']['pointcloud_camera0_base'][frame_idx]  # (10000, 6)
        elif 'camera0_pointcloud' in root['data']:
            pointcloud = root['data']['camera0_pointcloud'][frame_idx]  # (10000, 6)
        else:
            pointcloud = root['data']['pointcloud'][frame_idx]
            pointcloud[:, 3:6] = pointcloud[:, 3:6][:, ::-1]
        
        return pointcloud


    def _axis_angle_to_matrix(self, position: np.ndarray, axis_angle: np.ndarray) -> np.ndarray:
        """Convert position and axis-angle rotation to 4x4 transformation matrix"""
        # Convert axis-angle to rotation matrix using scipy
        rotation_matrix = R.from_rotvec(axis_angle).as_matrix()
        
        # Create 4x4 transformation matrix
        transform_matrix = np.eye(4, dtype=np.float32)
        transform_matrix[:3, :3] = rotation_matrix
        transform_matrix[:3, 3] = position
        
        return self.CV_TO_CAM @ transform_matrix

    def _load_state(self, window) -> np.ndarray:
        """Load state data (wrist pose + joint values) for window using the new helper functions."""
        zarr_path = window['seq_info']['zarr_path']
        available_frames = window['available_frames']
        start_idx = window['start_idx']
        is_generated = window['seq_info']['is_generated']
        version = window['seq_info']['zarr_format_version']
        
        states = []
        hand_side = window['seq_info']['hand_side']  # 'left' or 'right'
        hand_info = {'right': (hand_side == 'right'), 'left': (hand_side == 'left')}
        for t in range(start_idx - self.state_horizon, start_idx):
            frame_idx = available_frames[t]
            
            original_zarr_path = window['seq_info'].get('original_zarr_path', zarr_path)
            original_start_frame = window['seq_info'].get('original_start_frame', 0)
            original_frame_idx = original_start_frame + frame_idx - window['seq_info']['start_frame'] if is_generated else frame_idx

            # Load wrist pose and joint values using helper functions
            active_wrist_pose = self._load_wrist_pose(zarr_path, original_zarr_path, is_generated, frame_idx, original_frame_idx, version)
            joint_values = self._load_joint_values(zarr_path, original_zarr_path, is_generated, frame_idx, original_frame_idx, version)

            # Identity for the inactive hand
            inactive_wrist_pose = np.eye(4, dtype=np.float32)

            # Convert poses to 9D representation
            active_wrist_9d = pose_utils.mat_to_pose9d(active_wrist_pose)
            inactive_wrist_9d = pose_utils.mat_to_pose9d(inactive_wrist_pose)

            hand_type = window['seq_info']['hand_type']
            zero_joints = np.zeros(len(hand_utils.JOINT_MAP[hand_type]), dtype=np.float32)

            if hand_side == 'right':
                state = np.concatenate([active_wrist_9d, inactive_wrist_9d, joint_values, zero_joints])
            else:
                state = np.concatenate([inactive_wrist_9d, active_wrist_9d, zero_joints, joint_values])
            states.append(state)
        
        return np.stack(states, axis=0), hand_info

    def _load_wrist_pose(self, zarr_path: Path, original_zarr_path: Path, is_generated: bool, frame_idx: int, original_frame_idx: int, version: int) -> np.ndarray:
        """
        Loads a single 4x4 wrist pose matrix, handling generated data and different zarr formats.

        Args:
            zarr_path: Path to the zarr file to load from (can be original or generated).
            original_zarr_path: Path to the original zarr file (used if is_generated).
            is_generated: Flag indicating if the data is from a generated file.
            frame_idx: The frame index in the zarr_path.
            original_frame_idx: The corresponding frame index in the original_zarr_path.
            version: The format version of the zarr file providing the primary pose data.

        Returns:
            A 4x4 numpy array representing the right wrist pose.
        """
        root = zarr.open(str(zarr_path), 'r')

        if is_generated:
            original_root = zarr.open(str(original_zarr_path), 'r')
            # The passed 'version' for generated data is the version of the original file.
            if version == 1:
                wrist_rot = original_root['data']['robot0_eef_rot_axis_angle'][original_frame_idx]
                # Position is a placeholder as it will be overwritten
                wrist_pos_dummy = original_root['data']['robot0_eef_pos'][original_frame_idx]
                right_wrist_pose = self._axis_angle_to_matrix(wrist_pos_dummy, wrist_rot)
            elif version == 2:
                right_wrist_pose = original_root['data']['right_joint_poses'][original_frame_idx].copy()
            else:
                raise ValueError(f"Unknown original zarr format version: {version}")

            # Override the position with the generated position
            generated_pos = root['data']['state_camera0_base'][frame_idx]
            right_wrist_pose[:3, 3] = generated_pos
        else:
            # For original data, use the passed version directly
            if version == 1:
                wrist_pos = root['data']['robot0_eef_pos'][frame_idx]
                wrist_rot = root['data']['robot0_eef_rot_axis_angle'][frame_idx]
                right_wrist_pose = self._axis_angle_to_matrix(wrist_pos, wrist_rot)
            elif version == 2:
                right_wrist_pose = root['data']['right_joint_poses'][frame_idx]
            else:
                raise ValueError(f"Unknown zarr format version: {version} for path {zarr_path}")
        
        return right_wrist_pose

    def _load_joint_values(self, zarr_path: Path, original_zarr_path: Path, is_generated: bool, frame_idx: int, original_frame_idx: int, version: int) -> np.ndarray:
        """
        Loads a single joint values array, handling generated data and different zarr formats.

        Args:
            zarr_path: Path to the zarr file to load from (can be original or generated).
            original_zarr_path: Path to the original zarr file (used if is_generated).
            is_generated: Flag indicating if the data is from a generated file.
            frame_idx: The frame index in the zarr_path.
            original_frame_idx: The corresponding frame index in the original_zarr_path.
            version: The format version of the zarr file providing the joint data.

        Returns:
            A numpy array of joint values.
        """
        if is_generated:
            original_root = zarr.open(str(original_zarr_path), 'r')
            # The passed 'version' for generated data is the version of the original file.
            if version == 1:
                joint_values = original_root['data']['gripper0_gripper_pose'][original_frame_idx]
            elif version == 2:
                joint_values = original_root['data']['right_joint_values'][original_frame_idx]
            else:
                raise ValueError(f"Unknown original zarr format version: {version}")
        else:
            root = zarr.open(str(zarr_path), 'r')
            # For original data, use the passed version directly
            if version == 1:
                joint_values = root['data']['gripper0_gripper_pose'][frame_idx]
            elif version == 2:
                joint_values = root['data']['right_joint_values'][frame_idx]
            else:
                raise ValueError(f"Unknown zarr format version: {version} for path {zarr_path}")
        
        return joint_values

    def _load_action_sequence(self, window) -> np.ndarray:
        """Load action sequence for window with relative wrist transforms using helper functions."""
        zarr_path = window['seq_info']['zarr_path']
        available_frames = window['available_frames']
        start_idx = window['start_idx'] 
        chunk_size = window['chunk_size']
        is_generated = window['seq_info']['is_generated']
        version = window['seq_info']['zarr_format_version']
        
        # Get current frame as reference
        current_frame_idx = available_frames[start_idx - 1]
        original_zarr_path = window['seq_info'].get('original_zarr_path', zarr_path)
        original_start_frame = window['seq_info'].get('original_start_frame', 0)
        original_current_frame_idx = original_start_frame + current_frame_idx - window['seq_info']['start_frame'] if is_generated else current_frame_idx

        # Load current wrist pose
        hand_side = window['seq_info']['hand_side']
        current_active_wrist = self._load_wrist_pose(zarr_path, original_zarr_path, is_generated, current_frame_idx, original_current_frame_idx, version)
        current_inactive_wrist = np.eye(4, dtype=np.float32)  # Inactive hand identity
        
        actions = []
        for t in range(start_idx, start_idx + chunk_size):
            frame_idx = available_frames[t]
            original_frame_idx = original_start_frame + frame_idx - window['seq_info']['start_frame'] if is_generated else frame_idx
            
            # Load target wrist pose and joint values
            target_active_wrist = self._load_wrist_pose(zarr_path, original_zarr_path, is_generated, frame_idx, original_frame_idx, version)
            active_joint_values = self._load_joint_values(zarr_path, original_zarr_path, is_generated, frame_idx, original_frame_idx, version)
            target_inactive_wrist = np.eye(4, dtype=np.float32)
            inactive_joint_values = np.zeros_like(active_joint_values)

            active_rel_wrist = np.linalg.inv(current_active_wrist) @ target_active_wrist
            inactive_rel_wrist = np.linalg.inv(current_inactive_wrist) @ target_inactive_wrist
            active_wrist_9d = pose_utils.mat_to_pose9d(active_rel_wrist)
            inactive_wrist_9d = pose_utils.mat_to_pose9d(inactive_rel_wrist)

            active_joints = active_joint_values.astype(np.float32)
            inactive_joints = inactive_joint_values.astype(np.float32)

            if hand_side == 'right':
                formatted_action = np.concatenate([active_wrist_9d, inactive_wrist_9d, active_joints, inactive_joints])
            else:
                formatted_action = np.concatenate([inactive_wrist_9d, active_wrist_9d, inactive_joints, active_joints])
            actions.append(formatted_action)
        
        return np.stack(actions, axis=0)

    def _get_initial_action(self, window) -> np.ndarray:
        """Get initial action corresponding to current state frame using helper functions."""
        zarr_path = window['seq_info']['zarr_path']
        available_frames = window['available_frames']
        start_idx = window['start_idx']
        frame_idx = available_frames[start_idx]
        is_generated = window['seq_info']['is_generated']
        version = window['seq_info']['zarr_format_version']
        
        # Identity transformations for relative actions
        hand_side = window['seq_info']['hand_side']
        wrist_identity_9d = pose_utils.mat_to_pose9d(np.eye(4, dtype=np.float32))
        
        original_zarr_path = window['seq_info'].get('original_zarr_path', zarr_path)
        original_start_frame = window['seq_info'].get('original_start_frame', 0)
        original_frame_idx = original_start_frame + frame_idx - window['seq_info']['start_frame'] if is_generated else frame_idx

        # Load joint values using the helper function
        joint_values = self._load_joint_values(zarr_path, original_zarr_path, is_generated, frame_idx, original_frame_idx, version)
        
        hand_type = window['seq_info']['hand_type']
        zero_joints = np.zeros(len(hand_utils.JOINT_MAP[hand_type]), dtype=np.float32)
        if hand_side == 'right':
            formatted_action = np.concatenate([wrist_identity_9d, wrist_identity_9d, joint_values, zero_joints])
        else:
            formatted_action = np.concatenate([wrist_identity_9d, wrist_identity_9d, zero_joints, joint_values])
        return formatted_action


    def _load_prompt(self, window) -> str:
        """Generate text prompt for the window"""
        hand_type = window['seq_info']['hand_type']
        action = window['seq_info']['action']
        window_idx = window['window_idx']
        
        # Generate descriptive prompt using hand type and action
        prompt = f"Use {hand_type} hands to {action}"
        if not prompt.endswith('.'):
            prompt += '.'
        return prompt

    def _generate_window_hash(self, window: dict) -> str:
        """Generate unique hash for window, including generated data info"""
        zarr_path = str(window['seq_info']['zarr_path'])
        window_idx = window['window_idx']
        start_idx = window['start_idx']
        end_idx = window['end_idx']
        # Use the sequence-specific sample_stride for hash generation
        seq_sample_stride = window['seq_info']['sample_stride']
        is_generated = window['is_generated']
        generation_idx = window['generation_idx']
        
        key_info = {
            'zarr_path': zarr_path,
            'window_idx': window_idx,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'chunk_size': window['chunk_size'],  # Use window-specific chunk_size
            'sample_stride': seq_sample_stride,  # Use sequence-specific stride
            'interpolation_factor': window['interpolation_factor'],  # Use window-specific interpolation factor
            'pointcloud_size': self.pointcloud_size,
            'state_horizon': self.state_horizon,
            'pcd_horizon': self.pcd_horizon,
            'is_generated': is_generated,
            'generation_idx': generation_idx,
            'hand_side': window['seq_info'].get('hand_side', 'right'),
        }
        key_str = str(sorted(key_info.items()))
        return hashlib.md5(key_str.encode()).hexdigest()[:16]

    def _generate_pcd_hash(self, pcd_paths) -> str:
        """Generate unique hash for pointcloud paths"""
        zarr_path, frame_idx = pcd_paths
        hash_str = f"{zarr_path}_{frame_idx}"
        return hashlib.md5(hash_str.encode()).hexdigest()