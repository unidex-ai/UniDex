import os
import glob
import torch
import numpy as np
import open3d as o3d
from pathlib import Path
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple, Any
import pickle
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from pytorch3d.ops import sample_farthest_points
import time
from tqdm import tqdm
import threading
import queue
import subprocess
import sys

import src.utils.hand_utils as hand_utils
from src.utils.normalizers import Normalizer

torch.multiprocessing.set_start_method('spawn', force=True)

class BaseDataset(Dataset, ABC):
    """
    Base dataset class that provides common functionality for all hand manipulation datasets.
    
    Subclasses only need to implement:
    - _find_sequences(): Find all valid sequences in the dataset
    - _load_pointcloud(): Load pointcloud data for a specific window
    - _load_state(): Load state data for a specific window  
    - _load_action_sequence(): Load action sequence for a specific window
    - _get_initial_action(): Get initial action corresponding to current state frame
    - _load_prompt(): Generate text prompt for an window
    
    Optional class attributes that subclasses can define:
    - ACTION_LABELS: Dictionary mapping action indices to descriptions (if needed)
    - TIPS: List of fingertip names (if needed)
    """
    
    # Default values - subclasses can override these if needed
    UNSEEN_POSE = np.eye(4)

    def __init__(
        self,
        data_dir: str,
        chunk_size: int = 50,
        pointcloud_size: int = 1024,
        cache_data: bool = True,
        use_cached_metadata: bool = False,
        cache_dir: Optional[str] = None,
        sample_stride: int = 1,
        state_horizon: int = 1,
        pcd_horizon: int = 1,
        normalizer: Normalizer | None = None,
        hands: List[str] = ["Inspire"],
        interpolation_factor: int = 1,
        **kwargs  # Allow subclass-specific parameters
    ):
        """
        Base dataset initialization
        
        Args:
            data_dir: Base directory of dataset
            chunk_size: Maximum sequence length for action chunks
            pointcloud_size: Number of points to sample from pointcloud
            cache_data: Whether to cache processed data
            use_cached_metadata: If True, load cached sequences and windows metadata if available
            cache_dir: Directory to store cached data (if None, uses data_dir.parent/.cache)
            sample_stride: Stride for sampling frames (default 1 means no stride)
            state_horizon: Number of state history frames
            pcd_horizon: Number of pointcloud history frames
            normalizer: Normalizer instance for scaling actions and states
            hands: List of hand types to include (['Inspire'], ['Leap'], ['Wuji'], or combinations)
            joint_maps: Dictionary mapping hand types to joint remapping, e.g., {"Inspire": [0,1,2,3,4,5], "Leap": [0,2,1,4,3,5]}
            interpolation_factor: Factor to interpolate actions (3 = 10fps -> 30fps)
            **kwargs: Additional dataset-specific parameters
        """
        self.data_dir = Path(data_dir)
        self.pointcloud_size = pointcloud_size
        self.cache_dir = Path(cache_dir) if cache_dir else self.data_dir.parent / '.cache'
        self.cache_data = cache_data
        self.use_cached_metadata = use_cached_metadata
        self.sample_stride = sample_stride
        self.state_horizon = state_horizon
        self.pcd_horizon = pcd_horizon
        self.pcd_needed = []
        self.normalizer = normalizer
        self.hands = hands
        self.interpolation_factor = interpolation_factor
        assert interpolation_factor >= 1, "interpolation_factor must be >= 1"
        assert chunk_size % interpolation_factor == 0, "chunk_size must be divisible by interpolation_factor"
        self.chunk_size = chunk_size // interpolation_factor

        # Store additional subclass-specific parameters
        self.kwargs = kwargs

        # Validate hands parameter
        valid_hands = list(hand_utils.JOINT_MAP.keys())
        for hand in self.hands:
            if hand not in valid_hands:
                raise ValueError(f"Unsupported hands type: {hand}. Supported: {valid_hands}")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Find all available sequences
        if self.use_cached_metadata and os.path.exists(self.cache_dir / 'sequences.pkl'):
            ok = False
            while not ok:
                try:
                    with open(self.cache_dir / 'sequences.pkl', 'rb') as f:
                        self.sequences = pickle.load(f)
                    self.log(f"Loaded cached sequences from {self.cache_dir / 'sequences.pkl'}")
                    ok = True
                except:
                    self.log(f"Retrying loading sequences.pkl ...")

        else:
            self.sequences = self._find_sequences()
            with open(self.cache_dir / 'sequences.pkl', 'wb') as f:
                pickle.dump(self.sequences, f)
            self.log(f"Saved {len(self.sequences)} sequences to {self.cache_dir / 'sequences.pkl'}")
        self.log(f"Found {len(self.sequences)} sequences")

        # Build window list
        if self.use_cached_metadata and os.path.exists(self.cache_dir / 'windows.pkl'):
            ok = False
            while not ok:
                try:
                    with open(self.cache_dir / 'windows.pkl', 'rb') as f:
                        self.windows = pickle.load(f)
                    self.log(f"Loaded cached windows from {self.cache_dir / 'windows.pkl'}")
                    ok = True
                except:
                    self.log(f"Retrying loading windows.pkl ...")
        else:
            self.windows = []
            for seq in self.sequences:
                window_data = self._build_window(seq)
                if window_data:
                    self.windows.extend(window_data)

            with open(self.cache_dir / 'windows.pkl', 'wb') as f:
                pickle.dump(self.windows, f)
            self.log(f"Saved {len(self.windows)} windows to {self.cache_dir / 'windows.pkl'}")

        self.windows = [ep for ep in self.windows if ep['seq_info']['hand_type'] in self.hands]
        self.log(f"Filtered to {len(self.windows)} windows after hand type filtering: {self.hands}")
        
        self.pcd_needed = [pcd_paths for ep in self.windows for pcd_paths in ep['pcd_paths']]
        
        self.log(f"initialized with {len(self.windows)} windows")

        with open(self.cache_dir / 'sequences.pkl', 'wb') as f:
            pickle.dump(self.sequences, f)
        with open(self.cache_dir / 'windows.pkl', 'wb') as f:
            pickle.dump(self.windows, f)
    # ============================================================================
    # Abstract methods that subclasses must implement
    # ============================================================================
    
    @abstractmethod
    def _find_sequences(self) -> List[Dict]:
        """
        Find all valid sequences in the dataset.
        
        Returns:
            List of sequence info dictionaries. Each dict should contain
            information needed to load data for that sequence.
        """
        pass

    @abstractmethod
    def _build_window(self, seq_info) -> List[Dict]:
        """
        Build window data for a sequence.
        Each dataset can implement its own window building logic.
        
        Args:
            seq_info: Sequence information dictionary
            
        Returns:
            List of window dictionaries. Each window should contain:
            - 'seq_info': The sequence information
            - 'start_idx': Start frame index
            - 'end_idx': End frame index
            - 'chunk_size': Size of the window chunk
        """
        pass

    @abstractmethod
    def _load_raw_pointcloud(self, pcd_paths) -> np.ndarray:
        """
        Load raw pointcloud data for a specific frame/location.
        This method should be implemented by subclasses to handle their specific data format.
        
        Args:
            pcd_paths: Dataset-specific path/key information to locate the pointcloud data
            
        Returns:
            Raw pointcloud array of shape (N, 6) where N varies, containing xyz+rgb data
        """
        pass

    @abstractmethod
    def _load_raw_image(self, pcd_paths) -> np.ndarray:
        """
        Load raw image data for a specific frame/location.
        This method should be implemented by subclasses to handle their specific data format.
        
        Args:
            image_paths: Dataset-specific path/key information to locate the image data
        Returns:
            Raw image array of shape (H, W, 3) containing RGB data
        """
        pass

    @abstractmethod
    def _load_state(self, window) -> np.ndarray:
        """
        Load state data for an window.
        
        Args:
            window: window dictionary containing seq_info, start_idx, end_idx
            
        Returns:
            State array of shape (state_horizon, state_dim)
        """
        pass

    @abstractmethod
    def _load_action_sequence(self, window) -> np.ndarray:
        """
        Load action sequence for an window.
        
        Args:
            window: window dictionary containing seq_info, start_idx, end_idx
            
        Returns:
            Action array of shape (chunk_size, action_dim)
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
    def _get_initial_action(self, window) -> np.ndarray:
        """
        Get the initial action corresponding to the current state frame.
        
        Args:
            window: window dictionary containing seq_info, start_idx, end_idx
            
        Returns:
            Initial action array of shape (action_dim,)
        """
        pass

    @abstractmethod
    def _generate_window_hash(self, window: dict) -> str:
        """
        Generate a unique hash for the window based on its key information.
        
        Args:
            window: window dictionary containing seq_info, start_idx, end_idx
            
        Returns:
            Unique hash string for caching
        """
        pass

    @abstractmethod
    def _generate_pcd_hash(self, pcd_paths) -> str:
        """
        Generate a unique hash for the pointcloud file(s).
        
        Args:
            pcd_paths: Path(s) to pointcloud data (can be string, tuple, or list)
            
        Returns:
            Unique hash string for caching
        """
        pass

    # ============================================================================
    # Common utility methods (used by all datasets)
    # ============================================================================
    def log(self, message: str):
        """
        Log a message with the dataset name prefix.
        
        Args:
            message: Message to log
        """
        print(f"[{self.__class__.__name__[:-7]}] {message}")

    def is_window_cached(self, window):
        robot_cache_exists = self._generate_window_hash(window) in self.cached_filenames['robot']
        pcd_cache_exists = all(
            self._generate_pcd_hash(pcd_paths) in self.cached_filenames['pcd']
            for pcd_paths in window['pcd_paths']
        )
        return robot_cache_exists and pcd_cache_exists

    def _interpolate_actions(self, initial_action, actions):
        """
        Interpolate action sequence based on initial action and subsequent actions
        
        Args:
            initial_action: Initial action corresponding to current state frame
            actions: Subsequent action sequence (list of action lists)
            
        Returns:
            Interpolated action sequence with length equal to action_horizon
        """
        if self.interpolation_factor <= 1:
            return actions
        
        all_actions = [initial_action]
        all_actions.extend(actions)
        interpolated_actions = []
        
        # Interpolate between consecutive actions
        for i in range(len(all_actions) - 1):
            current_action = np.array(all_actions[i])
            next_action = np.array(all_actions[i + 1])
            
            # Add interpolated actions between current and next
            for j in range(1, self.interpolation_factor):
                alpha = j / self.interpolation_factor
                interpolated_action = (1 - alpha) * current_action + alpha * next_action
                interpolated_actions.append(interpolated_action)
            
            interpolated_actions.append(next_action)
        
        return interpolated_actions

    def _get_robot_cache_path(self, window_hash):
        """Get cache path for robot data (state + action)"""
        cache_subdir = self.cache_dir / "robot"
        os.makedirs(cache_subdir, exist_ok=True)
        return cache_subdir / f"{window_hash}.pkl"

    def _get_pcd_cache_path(self, pcd_hash):
        """Get cache path for individual pointcloud data"""
        pcd_cache_dir = self.cache_dir / 'pcd'
        os.makedirs(pcd_cache_dir, exist_ok=True)
        return pcd_cache_dir / f"{pcd_hash}.npy"


    def _load_pointcloud(self, pcd_paths):
        """Load and combine hand and scene pointclouds from zarr files"""
        pcd_hash = self._generate_pcd_hash(pcd_paths)
            
        cache_path = self._get_pcd_cache_path(pcd_hash)
        
        # Check if cached
        if not os.path.exists(cache_path):
            index = self.pcd_needed.index(pcd_paths)
            self._load_pointcloud_batch_and_cache(index)
        
        pointcloud = np.load(cache_path)
        return pointcloud
    
    
    def _load_pointcloud_cached(self, window) -> np.ndarray:
        """
        Load pointcloud data for an window, using caching if enabled.
        This is the common implementation that works with window['pcd_paths'].
        
        Args:
            window: window dictionary containing pcd_paths
            
        Returns:
            Pointcloud array of shape (pcd_horizon, pointcloud_size, 6) containing xyz+rgb data
        """
        pcd_paths_list = window['pcd_paths']
        pointclouds = []
        
        for pcd_paths in pcd_paths_list:
            pointcloud = self._load_pointcloud(pcd_paths)
            pointclouds.append(pointcloud)
        
        return np.stack(pointclouds, axis=0)
    

    def _load_pointcloud_batch_and_cache(self, index: int, batch_size: int = 16):
        """
        Load and cache pointclouds in batches with FPS sampling.
        This is a generic implementation that uses _load_raw_pointcloud.
        """
        original_index = index
        index = (index // batch_size) * batch_size  # Ensure index is aligned with batch size
        batch_pcd_needed = self.pcd_needed[index:index + batch_size]
        uncached = []
        uncached_indices = []
        hashes = []
        
        for i, pcd_paths in enumerate(batch_pcd_needed):
            pcd_hash = self._generate_pcd_hash(pcd_paths)
            hashes.append(pcd_hash)
            cache_path = self._get_pcd_cache_path(pcd_hash)
            if not os.path.exists(cache_path):
                uncached.append(pcd_paths)
                uncached_indices.append(i)
        
        if not uncached:
            return
            
        # Generate pointclouds for uncached items
        pcs = []
        for pcd_paths in uncached:
            # Use subclass-specific method to load raw pointcloud
            pointcloud = self._load_raw_pointcloud(pcd_paths)
            pcs.append(pointcloud)  
        # Find max number of points and pad all pointclouds
        max_n = max([len(pc) for pc in pcs])
        padded = []
        for pc in pcs:
            if len(pc) == 0:
                pc = np.zeros((1, 6))
            if len(pc) < max_n:
                pad_n = max_n - len(pc)
                idx = np.random.choice(len(pc), pad_n, replace=True)
                pad = pc[idx]
                pc = np.vstack([pc, pad])
            padded.append(pc)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pcs_tensor = torch.tensor(np.stack(padded), dtype=torch.float32).to(device)
        fps_xyz = pcs_tensor[..., :3]  # (B, N, 3)
        _, fps_idx = sample_farthest_points(fps_xyz, K=self.pointcloud_size)
        
        # Sample points using FPS indices
        sampled = []
        for i, idxs in enumerate(fps_idx):
            pc = pcs_tensor[i][idxs]  # (pointcloud_size, 6)
            sampled.append(pc.cpu().numpy())

        # Cache all sampled pointclouds
        for i, pcd_paths in enumerate(uncached):
            pcd_hash = self._generate_pcd_hash(pcd_paths)
            cache_path = self._get_pcd_cache_path(pcd_hash)
            np.save(cache_path, sampled[i])
        
        self.log(f"Loaded pointcloud batch: {len(os.listdir(self.cache_dir / 'pcd'))}/{len(self.pcd_needed)}")

    def _load_robot_data(self, window):
        """
        Load robot-related data (state, actions, prompt) and handle caching.
        """
        window_hash = self._generate_window_hash(window)
        robot_cache_path = self._get_robot_cache_path(window_hash)

        # self.log(f"{robot_cache_path}: {os.path.exists(robot_cache_path)}")

        # Check if robot data is cached
        if os.path.exists(robot_cache_path) and self.cache_data:
            try:
                with open(robot_cache_path, 'rb') as f:
                    robot_data = pickle.load(f)
                state = np.array(robot_data['state'])
                actions = np.array(robot_data['actions'])
                prompt = robot_data['prompt']

                if not prompt.endswith('\n'):
                    prompt += '\n'
                
                result = {
                    'state': state,
                    'action': actions,
                    'prompt': prompt
                }
                
                return result
            except Exception as e:
                self.log(f"Error loading cached robot data for window {window_hash}: {e}")
                # If cache is corrupted, fall back to loading from disk
                os.remove(robot_cache_path)

        state, hand_info = self._load_state(window)
        actions = self._load_action_sequence(window)
        prompt = self._load_prompt(window)
        
        if not prompt.endswith('\n'):
            prompt += '\n'
        
        if self.interpolation_factor > 1:
            # Interpolate actions if needed
            initial_action = self._get_initial_action(window)
            actions = self._interpolate_actions(initial_action, actions)

        state, actions = np.array(state), np.array(actions)
        state = self._apply_scale_shift(state, window['seq_info']['hand_type'], hand_info)
        actions = self._apply_scale_shift(actions, window['seq_info']['hand_type'], hand_info)

        # Cache robot data separately
        if self.cache_data:
            robot_data = {
                'state': state,
                'actions': actions,
                'prompt': prompt
            }
            with open(robot_cache_path, 'wb') as f:
                pickle.dump(robot_data, f)

        result = {
            'state': state,
            'action': actions,
            'prompt': prompt
        }
        return result

    def _apply_scale_shift(self, actions: np.ndarray, hand_type: str, hand_info: dict) -> np.ndarray:
        right_scale, right_offset = np.array(list(hand_utils.RETARGET_JOINT_MAP_SCALE[hand_type]['right'].values())), np.array(list(hand_utils.RETARGET_JOINT_MAP_OFFSET[hand_type]['right'].values()))
        left_scale, left_offset = np.array(list(hand_utils.RETARGET_JOINT_MAP_SCALE[hand_type]['left'].values())), np.array(list(hand_utils.RETARGET_JOINT_MAP_OFFSET[hand_type]['left'].values()))

        hand_joint_dim = hand_utils.JOINT_DIMENSIONS[hand_type]
        if hand_info['right']:
            actions[:, 18:18 + hand_joint_dim] = actions[:, 18:18 + hand_joint_dim] * right_scale + right_offset
        if hand_info['left']:
            actions[:, 18 + hand_joint_dim:18 + hand_joint_dim * 2] = actions[:, 18 + hand_joint_dim:18 + hand_joint_dim * 2] * left_scale + left_offset
        return actions
    
    def _apply_action_map(self, actions: np.ndarray, hand_type: str) -> np.ndarray:
        """
        Apply action mapping for a batch of actions for a specific hand type to remap action values.

        Args:
            actions: NumPy array of shape (batch_size, action_dim), where action_dim includes wrist poses and joint values.
            hand_type: Hand type (e.g., "Inspire", "Leap", "Wuji").

        Returns:
            Mapped actions as a NumPy array of shape (batch_size, mapped_action_dim).
        """
        joint_map = hand_utils.JOINT_MAP[hand_type]
        batch_size, action_dim = actions.shape

        # Extract wrist poses (first 18 dimensions: 9D for each hand)
        wrist_poses = actions[:, :18]

        # Extract joint values for both hands
        hand_joint_dim = len(joint_map.keys())
        right_hand_joints = actions[:, 18:18 + hand_joint_dim]
        left_hand_joints = actions[:, 18 + hand_joint_dim : 18 + 2 * hand_joint_dim]
        additional_values = actions[:, 18 + 2 * hand_joint_dim:]
        assert additional_values.shape[1] <= 2 * (hand_utils.MAPPED_JOINT_DIM - hand_utils.JOINT_DIM_IN_USE)
        additional_values = [additional_values[:, :hand_utils.MAPPED_JOINT_DIM - hand_utils.JOINT_DIM_IN_USE], additional_values[:, hand_utils.MAPPED_JOINT_DIM - hand_utils.JOINT_DIM_IN_USE:]]
        additional_values[0] = np.pad(additional_values[0], ((0, 0), (0, hand_utils.MAPPED_JOINT_DIM - hand_utils.JOINT_DIM_IN_USE - additional_values[0].shape[1])), mode='constant')
        additional_values[1] = np.pad(additional_values[1], ((0, 0), (0, hand_utils.MAPPED_JOINT_DIM - hand_utils.JOINT_DIM_IN_USE - additional_values[1].shape[1])), mode='constant')

        # Initialize mapped joint arrays
        mapped_right_joints = np.zeros((batch_size, hand_utils.MAPPED_JOINT_DIM), dtype=np.float32)
        mapped_left_joints = np.zeros((batch_size, hand_utils.MAPPED_JOINT_DIM), dtype=np.float32)

        mapped_right_joints[:, hand_utils.JOINT_DIM_IN_USE:] = additional_values[0]
        mapped_left_joints[:, hand_utils.JOINT_DIM_IN_USE:] = additional_values[1]

        # Map joint values for both hands
        for idx, mapped_idx in enumerate(joint_map.values()):
            mapped_right_joints[:, mapped_idx] = right_hand_joints[:, idx]
            mapped_left_joints[:, mapped_idx] = left_hand_joints[:, idx]

        # Concatenate wrist poses and mapped joint values
        mapped_actions = np.concatenate([wrist_poses, mapped_right_joints, mapped_left_joints], axis=1)

        return torch.tensor(mapped_actions, dtype=torch.float32)
    
    # ============================================================================
    # Standard Dataset interface (implemented by base class)
    # ============================================================================

    @property
    def shape(self):
        """
        Get the output shapes for state and action tensors based on predict_pose setting
        
        Returns:
            dict: Dictionary containing shape information
                - state_shape: tuple of (state_horizon, state_dim)
                - action_shape: tuple of (chunk_size, action_dim)
                - state_dim: int, dimension per state frame
                - action_dim: int, dimension per action frame
        """
        state_dim = 18 + 2 * hand_utils.MAPPED_JOINT_DIM
        action_dim = 18 + 2 * hand_utils.MAPPED_JOINT_DIM
        
        return {
            'state': (self.state_horizon, state_dim),
            'action': (self.chunk_size, action_dim),
            'pointcloud': (self.pcd_horizon, self.pointcloud_size, 6),  # (horizon, points, xyz+rgb)
        }

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        """
        Override base __getitem__ to support return_image functionality and per-window interpolation.
        When return_image=True, replace pointcloud data with RGB images.
        Also handles per-window interpolation_factor.
        """
        window = self.windows[idx]
            
        try:
            data = {}
            
            # Load pointcloud data
            data['pointcloud'] = self._load_pointcloud_cached(window)  # (pcd_horizon, pointcloud_size, 6)
            data['pointcloud'] = data['pointcloud'].astype(np.float32)
            
            # Load robot data (state + action)
            robot_data = self._load_robot_data(window)
            data.update({
                'prompt': robot_data['prompt'],
                'action': robot_data['action'],
                'state': robot_data['state'],
            })
                
            # Apply action mapping
            data['action'] = self._apply_action_map(data['action'], window['seq_info']['hand_type'])
            data['state'] = self._apply_action_map(data['state'], window['seq_info']['hand_type'])
            
            # Apply normalizer if available
            if self.normalizer is not None:
                data = self.normalizer.normalize(data)
            
            return data
            
        except Exception as e:
            self.log(f"Error loading data for index {idx}: {e}. Now loading a random index instead.")
            random_idx = np.random.randint(0, len(self))
            return self.__getitem__(random_idx)
