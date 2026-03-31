"""
Hand Processor for IK solving and RGBD projection
"""

import os
import copy
import json
import numpy as np
import open3d as o3d
import pybullet as p
import pybullet_data
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from scipy.spatial.transform import Rotation as R
import xml.etree.ElementTree as ET
import cv2
from dataclasses import dataclass
import hydra
import argparse
import h5py
import random
from tqdm import tqdm

# Transform matrix from Bullet to Open3D coordinate system
B2O = np.array([[ 1, 0, 0, 0],
                [ 0, 0, 1, 0],
                [ 0,-1, 0, 0],
                [ 0, 0, 0, 1]])

DATASET_CONFIG = {
    "H2o": {
        "_target_": "dataset.H2o.H2oDataset",
        "data_dir": "data/H2o",
        "load_pcd": False,
        "cache_path": "HandAdapter/dataset_cache/H2o.pkl"
    },
    "HOI4D": {
        "_target_": "dataset.HOI4D.HOI4DDataset", 
        "data_dir": "data/HOI4D",
        "load_pcd": False,
        "cache_path": "HandAdapter/dataset_cache/HOI4D.pkl"
    },
    "hot3d": {
        "_target_": "dataset.hot3d.hot3dDataset",
        "data_dir": "data/hot3d",
        "load_pcd": False,
        "cache_path": "HandAdapter/dataset_cache/hot3d.pkl"
    },
    "Taco": {
        "_target_": "dataset.Taco.TacoDataset",
        "data_dir": "data/Taco",
        "load_pcd": False,
        "cache_path": "HandAdapter/dataset_cache/Taco.pkl"
    }
}

MANO_TIP_INDEX_MAP = {
    "H2o": [4, 8, 12, 16, 20],
    "HOI4D": [4, 8, 12, 16, 20],
    "hot3d": [16, 17, 18, 19, 20],
    "Taco": [4, 8, 12, 16, 20]
}
                
@dataclass
class CameraIntrinsics:
    """Camera intrinsics parameters"""
    fx: float  # Focal length in x
    fy: float  # Focal length in y  
    cx: float  # Principal point x
    cy: float  # Principal point y
    width: int  # Image width
    height: int  # Image height
    
    def to_matrix(self) -> np.ndarray:
        """Convert to 3x3 intrinsic matrix"""
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ])
    
    @classmethod
    def from_params(cls, fx: float, fy: float, cx: float, cy: float, width: int, height: int) -> 'CameraIntrinsics':
        """Create from individual parameters"""
        return cls(fx=fx, fy=fy, cx=cx, cy=cy, width=width, height=height)
    
    @classmethod  
    def from_matrix(cls, K: np.ndarray, width: int, height: int) -> 'CameraIntrinsics':
        """Create from intrinsic matrix"""
        return cls(fx=K[0,0], fy=K[1,1], cx=K[0,2], cy=K[1,2], width=width, height=height)

    def _to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for JSON serialization"""
        return {
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "width": self.width,
            "height": self.height
        }

def _load_mesh(filename: str, scale: List[float]) -> o3d.geometry.TriangleMesh:
    """Load and scale a mesh file"""
    mesh = o3d.io.read_triangle_mesh(filename)
    mesh.compute_vertex_normals()
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices) * np.asarray(scale))
    return mesh

def parse_visual_offsets_from_urdf(urdf_path: str) -> Dict[str, np.ndarray]:
    """Parse visual offsets from URDF file"""
    origin_map = {}
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    for link in root.findall('link'):
        name = link.get('name')
        visual = link.find('visual')
        if visual is not None:
            origin = visual.find('origin')
            if origin is not None:
                xyz = [float(v) for v in origin.get('xyz', '0 0 0').split()]
                rpy = [float(v) for v in origin.get('rpy', '0 0 0').split()]
                T = np.eye(4)
                T[:3, 3] = xyz
                T[:3, :3] = R.from_euler('xyz', rpy).as_matrix()
                origin_map[name] = T
    return origin_map

def save_all_poses(poses, output_dir, name, hand_type):
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"{name}.txt")
    with open(filename, 'w') as f:
        f.write(f"Poses for {hand_type} hand (timestamp {name})\n")
        f.write("Format: [R00 R01 R02 t0]\n")
        f.write("        [R10 R11 R12 t1]\n")
        f.write("        [R20 R21 R22 t2]\n")
        f.write("        [0   0   0   1 ]\n\n")
        for pname, pose in poses.items():
            f.write(f"{pname}:\n")
            np.savetxt(f, pose, fmt='%.6f')
            f.write("\n")

def save_joint_values(joint_names, joint_values, output_dir, name):
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"{name}.txt")
    with open(filename, 'w') as f:
        for jname, value in zip(joint_names, joint_values):
            f.write(f"{jname}: {value:.6f}\n")

class HandProcessor:
    """Hand processor for IK solving and RGBD projection supporting both left and right hands"""
    def __init__(self, left_urdf_path: str = None, right_urdf_path: str = None):
        """
        Initialize the HandProcessor with URDF files for left and/or right hand
        
        Args:
            left_urdf_path: Path to the left hand URDF file (optional)
            right_urdf_path: Path to the right hand URDF file (optional)
        """
        if left_urdf_path is None and right_urdf_path is None:
            raise ValueError("At least one hand URDF path must be provided")
        
        self.left_urdf_path = str(Path(left_urdf_path).resolve()) if left_urdf_path else None
        self.right_urdf_path = str(Path(right_urdf_path).resolve()) if right_urdf_path else None
        
        # Initialize PyBullet
        self.physics_client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, 0)
        p.setPhysicsEngineParameter(enableFileCaching=0)
        
        # IK parameters
        self.mano_scale = 1.0
        self.max_iterations = 1000
        self.residual_threshold = 1e-3
        self.ik_damping = 0.1
        self.mimic_iterations = 50
        self.mimic_step = 5
        
        # Initialize hand data structures
        self.hands = {}  # Store left/right hand data separately
        
        # Initialize OffscreenRenderer (will be resized when needed)
        self.renderer = None
        self.current_renderer_size = None
        print("OffscreenRenderer will be created when needed")
        
        # Load left hand if provided
        if left_urdf_path:
            self.hands['left'] = self._load_hand(left_urdf_path, 'left')
            print(f"Left hand loaded from: {left_urdf_path}")
        
        # Load right hand if provided
        if right_urdf_path:
            self.hands['right'] = self._load_hand(right_urdf_path, 'right')
            print(f"Right hand loaded from: {right_urdf_path}")
        
        print(f"HandProcessor initialized with {len(self.hands)} hand(s)")
    
    def __del__(self):
        if hasattr(self, 'physics_client'):
            p.disconnect(self.physics_client)
    
    def _load_hand(self, urdf_path: str, side: str) -> Dict:
        """Load a single hand URDF and return hand data"""
        # Load configuration
        config_path = Path(urdf_path).parent.parent / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Load robot
        robot = p.loadURDF(urdf_path, useFixedBase=True)
        
        # Get actuated joints
        actuated = [j for j in range(p.getNumJoints(robot))
                   if p.getJointInfo(robot, j)[2] in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC)]
        info = [p.getJointInfo(robot, j) for j in actuated]
        lowers = [i[8] for i in info]
        uppers = [i[9] for i in info]
        rests = [0.0] * len(actuated)
        
        # Get link names
        link_names = [p.getJointInfo(robot, j)[12].decode() for j in range(p.getNumJoints(robot))]
        
        # Get tip links
        tips = config["tips"]
        tip_links = [link_names.index(n) for n in tips if n in link_names]

        # Get pose links
        tips_and_wrist = config["poses"]
        pose_links = [link_names.index(n) for n in tips_and_wrist if n in link_names]
        
        # Get joint names
        joint_names = [p.getJointInfo(robot, j)[1].decode() for j in actuated]
        pose_names = [link_names[j] for j in pose_links]
        
        # Load mesh bank
        urdf_visual_offsets = parse_visual_offsets_from_urdf(urdf_path)
        mesh_bank: Dict[int, Tuple[o3d.geometry.TriangleMesh, np.ndarray]] = {}
        for vs in p.getVisualShapeData(robot):
            _, link_idx, geom_type, scale, fname, *_ = vs
            if geom_type != p.GEOM_MESH or fname == b'':
                continue
            fname = fname.decode('utf-8')
            assert os.path.exists(fname), f"File not found: {fname}"
            mesh = _load_mesh(fname, scale)
            link_name = link_names[link_idx] if 0 <= link_idx < len(link_names) else 'base'
            T_local = urdf_visual_offsets.get(link_name, np.eye(4))
            mesh_bank[link_idx] = (mesh, T_local)
        
        return {
            'config': config,
            'robot': robot,
            'actuated': actuated,
            'lowers': lowers,
            'uppers': uppers,
            'rests': rests,
            'link_names': link_names,
            'tip_links': tip_links,
            'pose_links': pose_links,
            'joint_names': joint_names,
            'pose_names': pose_names,
            'mesh_bank': mesh_bank,
            'urdf_path': urdf_path
        }
    
    def _ensure_renderer_size(self, width: int, height: int):
        """Ensure renderer has the correct size, recreate if needed"""
        if self.current_renderer_size != (width, height):
            # Create new renderer with correct size
            self.renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
            self.current_renderer_size = (width, height)
            print(f"OffscreenRenderer created with size {width}x{height}")
    
    def _setup_renderer_camera(self, camera_intrinsics: CameraIntrinsics):
        """Setup camera intrinsics for the renderer"""
        if self.renderer is not None:
            # Create Open3D camera intrinsics
            intrinsic = o3d.camera.PinholeCameraIntrinsic(
                width=camera_intrinsics.width,
                height=camera_intrinsics.height,
                fx=camera_intrinsics.fx,
                fy=camera_intrinsics.fy,
                cx=camera_intrinsics.cx,
                cy=camera_intrinsics.cy
            )
            
            extrinsic = np.array([[1, 0, 0, 0],
                                  [0, -1, 0, 0],
                                  [0, 0, -1, 0],
                                  [0, 0, 0, 1]])
            self.renderer.setup_camera(
                intrinsics=intrinsic, 
                extrinsic_matrix=extrinsic
            )
            
            # print(f"Renderer camera intrinsics set: fx={camera_intrinsics.fx}, fy={camera_intrinsics.fy}, cx={camera_intrinsics.cx}, cy={camera_intrinsics.cy}")
    
    def _clear_renderer(self):
        """Clear all geometries from the renderer"""
        if self.renderer is not None:
            self.renderer.scene.clear_geometry()
    
    def solve_ik(self, targets: List[np.ndarray], side: str) -> np.ndarray:
        """
        Solve inverse kinematics for given target positions
        
        Args:
            targets: List of target positions for fingertips
            side: 'left' or 'right' hand
            
        Returns:
            Joint angles array
        """
        if side not in self.hands:
            raise ValueError(f"Hand side '{side}' not available. Available: {list(self.hands.keys())}")
        
        hand_data = self.hands[side]
        
        # Get mimic relations from config
        MIMIC_RELATIONS = hand_data['config'].get("mimic_relations", {})
        
        # Map joint names to indices
        joint_name_to_idx = {name: i for i, name in enumerate(hand_data['joint_names'])}
        slave_pairs = []
        for slave_name, (master_name, mult, offset) in MIMIC_RELATIONS.items():
            if slave_name in joint_name_to_idx and master_name in joint_name_to_idx:
                slave_idx = joint_name_to_idx[slave_name]
                master_idx = joint_name_to_idx[master_name]
                slave_pairs.append((slave_idx, master_idx, float(mult), float(offset)))
        
        # Prepare targets for PyBullet
        targets = targets[:len(hand_data['tip_links'])]
        target_positions = [np.asarray(t, dtype=float).tolist() for t in targets]
        
        # Setup IK parameters
        num_joints = p.getNumJoints(hand_data['robot'])
        joint_ranges = [(u - l) if np.isfinite(u) and np.isfinite(l) else 2.0
                        for l, u in zip(hand_data['lowers'], hand_data['uppers'])]
        joint_damping = [float(self.ik_damping)] * num_joints
        
        if MIMIC_RELATIONS:
            best_error = float('inf')
            best_q = None
            for _ in range(self.mimic_iterations):
                rest_poses = [p.getJointState(hand_data['robot'], j)[0] for j in range(num_joints)]
                
                # Solve IK
                q = p.calculateInverseKinematics2(
                    bodyUniqueId=hand_data['robot'],
                    endEffectorLinkIndices=hand_data['tip_links'],
                    targetPositions=target_positions,
                    lowerLimits=hand_data['lowers'],
                    upperLimits=hand_data['uppers'],
                    jointRanges=joint_ranges,
                    restPoses=rest_poses,
                    jointDamping=joint_damping,
                    maxNumIterations=self.mimic_step,
                    residualThreshold=self.residual_threshold,
                )
                
                q = np.asarray(q, dtype=float)
                q = np.clip(q[:len(hand_data['actuated'])], hand_data['lowers'], hand_data['uppers'])
                
                # Apply mimic constraints
                for slave_idx, master_idx, mult, offset in slave_pairs:
                    if 0 <= master_idx < len(q) and 0 <= slave_idx < len(q):
                        q[slave_idx] = q[master_idx] * mult + offset
                
                # Clip again after mimic
                q = np.clip(q[:len(hand_data['actuated'])], hand_data['lowers'], hand_data['uppers'])
                for j_idx, angle in zip(hand_data['actuated'], q):
                    p.resetJointState(hand_data['robot'], j_idx, angle)
                
                # Calculate error
                error = 0.0
                for link_idx, target in zip(hand_data['tip_links'], target_positions):
                    pos, _ = p.getLinkState(hand_data['robot'], link_idx, computeForwardKinematics=True)[4:6]
                    error += np.linalg.norm(np.array(pos) - np.array(target))
                
                if error < best_error:
                    best_error = error
                    best_q = q
                if error < self.residual_threshold * len(hand_data['tip_links']):
                    break
            q = best_q
        else:
            rest_poses = [p.getJointState(hand_data['robot'], j)[0] for j in range(num_joints)]
            q = p.calculateInverseKinematics2(
                bodyUniqueId=hand_data['robot'],
                endEffectorLinkIndices=hand_data['tip_links'],
                targetPositions=target_positions,
                lowerLimits=hand_data['lowers'],
                upperLimits=hand_data['uppers'],
                jointRanges=joint_ranges,
                restPoses=rest_poses,
                jointDamping=joint_damping,
                maxNumIterations=self.max_iterations,
                residualThreshold=self.residual_threshold,
            )
            q = np.asarray(q, dtype=float)
            q = np.clip(q[:len(hand_data['actuated'])], hand_data['lowers'], hand_data['uppers'])

        # print(f"IK solved for {side} hand: {q}")
            
        return q
    
    def get_tip_poses(self, side):
        poses = {}
        # for tip_link in self.hands[side]['tip_links']:
        for tip_link in self.hands[side]['pose_links']:
            link_state = p.getLinkState(self.hands[side]['robot'], tip_link, computeForwardKinematics=True)
            pos_pb = np.array(link_state[4])
            orn_pb = np.array(link_state[5])
            
            R_link_pb = R.from_quat(orn_pb).as_matrix()
            T_link_pb = np.eye(4)
            T_link_pb[:3, :3] = R_link_pb
            T_link_pb[:3, 3] = pos_pb
            
            fingertip_name = self.hands[side]['link_names'][tip_link]
            poses[fingertip_name] = (B2O @ T_link_pb)
        return poses
    
    def get_joint_values(self, side):
        # print(f"Getting joint values for {side} hand: {[p.getJointState(self.hands[side]['robot'], j)[0] for j in self.hands[side]['actuated']]}")
        return [p.getJointState(self.hands[side]['robot'], j)[0] for j in self.hands[side]['actuated']]
    
    def get_hand_mesh(self, pose: np.ndarray, R_hand_world: np.ndarray, side: str,
                      tip_indices: List[int] = [4, 8, 12, 16, 20]) -> o3d.geometry.TriangleMesh:
        """
        Get the complete hand mesh for given pose
        
        Args:
            pose: Hand pose (21x3 joint positions)
            R_hand_world: Hand rotation matrix (3x3)
            side: 'left' or 'right' hand
            tip_indices: Indices of fingertip joints
            
        Returns:
            Combined hand mesh
        """
        if side not in self.hands:
            raise ValueError(f"Hand side '{side}' not available. Available: {list(self.hands.keys())}")
        
        hand_data = self.hands[side]
        
        # Apply scaling if needed
        j_cam = pose.copy()
        if self.mano_scale != 1.0:
            j_cam = (j_cam - j_cam[0]) * self.mano_scale + j_cam[0]
        
        wrist_w = j_cam[0]
        R_hand_w = R_hand_world
        
        # Transform to PyBullet coordinate system
        R_o2b = B2O[:3, :3].T
        t_o2b = -R_o2b @ B2O[:3, 3]
        R_hand_w_pb = R_o2b @ R_hand_w
        wrist_ik_pb = R_o2b @ wrist_w + t_o2b
        
        # Set robot base position and orientation
        p.resetBasePositionAndOrientation(
            hand_data['robot'], wrist_ik_pb.tolist(), R.from_matrix(R_hand_w_pb).as_quat().tolist())
        
        # Get target positions for IK
        tgt_w = [j_cam[i] for i in tip_indices]
        goals_bullet = [R_o2b @ g + t_o2b for g in tgt_w]
        
        # Step 1: Solve IK first
        q = self.solve_ik([list(g) for g in goals_bullet], side)
        
        # Step 2: Reset joint states with IK solution
        for j_idx, angle in zip(hand_data['actuated'], q):
            p.resetJointState(hand_data['robot'], j_idx, angle)
        
        # Step 3: Get all link meshes and combine them (after setting joint states)
        combined_mesh = o3d.geometry.TriangleMesh()
        
        for link_idx, (original_mesh, T_local) in hand_data['mesh_bank'].items():
            # Get link state after joint states have been set
            pos, orn = (p.getLinkState(hand_data['robot'], link_idx, computeForwardKinematics=True)[4:6]
                       if link_idx >= 0 else p.getBasePositionAndOrientation(hand_data['robot']))
            R_link = R.from_quat(orn).as_matrix()
            T_link = np.eye(4)
            T_link[:3, :3] = R_link
            T_link[:3, 3] = pos
            T_mesh = (B2O @ T_link) @ T_local
            
            # Transform mesh
            mesh = copy.deepcopy(original_mesh)
            mesh.transform(T_mesh)
            
            # Add to combined mesh
            combined_mesh += mesh
        
        return combined_mesh
    
    def project_both_hands_to_rgbd(self, 
                               left_pose: Optional[np.ndarray] = None, 
                               left_R_hand_world: Optional[np.ndarray] = None,
                               right_pose: Optional[np.ndarray] = None,
                               right_R_hand_world: Optional[np.ndarray] = None,
                               tip_indices: List[int] = [4, 8, 12, 16, 20]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project both hands to camera plane and generate combined RGBD image
        """
        # Clear previous geometries
        self._clear_renderer()

        # Process left hand if available
        if left_pose is not None and left_R_hand_world is not None and 'left' in self.hands:
            left_mesh = self.get_hand_mesh(left_pose, left_R_hand_world, 'left', tip_indices)
            left_mesh.paint_uniform_color([0.0, 0.0, 0.0])
            
            # Add mesh to renderer scene
            material = o3d.visualization.rendering.MaterialRecord()
            material.shader = "defaultUnlit"
            self.renderer.scene.add_geometry("left_hand", left_mesh, material)

        # Process right hand if available
        if right_pose is not None and right_R_hand_world is not None and 'right' in self.hands:
            right_mesh = self.get_hand_mesh(right_pose, right_R_hand_world, 'right', tip_indices)
            right_mesh.paint_uniform_color([0.0, 0.0, 0.0])
            
            # Add mesh to renderer scene
            material = o3d.visualization.rendering.MaterialRecord()
            material.shader = "defaultUnlit"
            self.renderer.scene.add_geometry("right_hand", right_mesh, material)

        # Set lighting
        self.renderer.scene.set_lighting(o3d.visualization.rendering.Open3DScene.LightingProfile.NO_SHADOWS, [0, 0, -1])

        # Render RGB and depth images
        rgb_image = np.asarray(self.renderer.render_to_image())
        depth_image = np.asarray(self.renderer.render_to_depth_image(z_in_view_space=True)).astype(np.float16)

        return rgb_image, depth_image

    def process_hand_pose(self, 
                         left_pose: Optional[np.ndarray],
                         right_pose: Optional[np.ndarray],
                         left_R_hand_world: Optional[np.ndarray],
                         right_R_hand_world: Optional[np.ndarray],
                         tip_indices: List[int]) -> Dict:
        """
        Main processing function: solve IK for both hands and project to RGBD
        
        Args:
            left_pose: Left hand pose (21x3 joint positions)
            right_pose: Right hand pose (21x3 joint positions)
            left_R_hand_world: Left hand rotation matrix (3x3)
            right_R_hand_world: Right hand rotation matrix (3x3)
            height: Image height (optional if using CameraIntrinsics)
            tip_indices: Indices of fingertip joints
            
        Returns:
            Dictionary containing rgb_image, depth_image, and joint_angles for both hands
        """
        # Handle different input formats
        result = {}
        
        # Project both hands to RGBD
        rgb_image, depth_image = self.project_both_hands_to_rgbd(
            left_pose, left_R_hand_world, right_pose, right_R_hand_world,
            tip_indices=tip_indices
        )
        
        result['rgb_image'] = rgb_image
        result['depth_image'] = depth_image
        result['poses'] = {
            'left': self.get_tip_poses('left') if left_pose is not None and 'left' in self.hands else None,
            'right': self.get_tip_poses('right') if right_pose is not None and 'right' in self.hands else None
        }
        result['joint_values'] = {
            'left': self.get_joint_values('left') if left_pose is not None and 'left' in self.hands else None,
            'right': self.get_joint_values('right') if right_pose is not None and 'right' in self.hands else None
        }
        
        return result
    
    def _solve_hand_ik(self, pose: np.ndarray, R_hand_world: np.ndarray, side: str, tip_indices: List[int]) -> np.ndarray:
        """Solve IK for a single hand"""
        hand_data = self.hands[side]
        
        # Apply scaling if needed
        j_cam = pose.copy()
        if len(j_cam.shape) == 1:
            j_cam = j_cam.reshape(-1, 3)
        if self.mano_scale != 1.0:
            j_cam = (j_cam - j_cam[0]) * self.mano_scale + j_cam[0]
        
        wrist_w = j_cam[0]
        R_hand_w = R_hand_world
        
        # Transform to PyBullet coordinate system
        R_o2b = B2O[:3, :3].T
        t_o2b = -R_o2b @ B2O[:3, 3]
        R_hand_w_pb = R_o2b @ R_hand_w
        wrist_ik_pb = R_o2b @ wrist_w + t_o2b
        
        # Set robot base position and orientation
        p.resetBasePositionAndOrientation(
            hand_data['robot'], wrist_ik_pb.tolist(), R.from_matrix(R_hand_w_pb).as_quat().tolist())
        
        # Get target positions for IK
        tgt_w = [j_cam[i] for i in tip_indices]
        goals_bullet = [R_o2b @ g + t_o2b for g in tgt_w]
        
        # Solve IK
        joint_angles = self.solve_ik([list(g) for g in goals_bullet], side)
        
        # Set joint states
        for j_idx, angle in zip(hand_data['actuated'], joint_angles):
            p.resetJointState(hand_data['robot'], j_idx, angle)
        
        return joint_angles

def process_sequence(dataset, hand_type, processor, base_path, sequence, continue_process=False):
    """
    Process a sequence and save to HDF5 file with the following structure:
    
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
    
    Args:
        dataset: Dataset name
        hand_type: Type of robotic hand 
        processor: HandProcessor instance
        base_path: Base output directory
        sequence: Sequence object with frames
        continue_process: Whether to continue processing existing files
    """
    sequence_path = base_path / sequence.index
    camera_intrinsics = CameraIntrinsics.from_params(**sequence.intrinsics)
    print(f"Processing sequence: {sequence_path}")
    processor._ensure_renderer_size(camera_intrinsics.width, camera_intrinsics.height)
    processor._setup_renderer_camera(camera_intrinsics)
    os.makedirs(sequence_path, exist_ok=True)
    
    hdf5_file_path = sequence_path / f"{hand_type}.h5"
    if continue_process and hdf5_file_path.exists():
        try:
            with h5py.File(hdf5_file_path, 'a') as f:
                if 'complete' not in f:
                    f.create_dataset('complete', data=True)
                    print(f"Marked existing file for {sequence.index} as complete.")
                    f.flush()
                if 'complete' in f and f['complete'][()]:
                    print(f"Sequence {sequence.index} already processed completely, skipping.")
                    return
                else:
                    print(f"Sequence {sequence.index} incomplete or corrupted, reprocessing...")
                    hdf5_file_path.unlink()
        except (OSError, Exception) as e:
            print(f"Error reading existing file for {sequence.index}: {e}, reprocessing...")
            if hdf5_file_path.exists():
                hdf5_file_path.unlink()
    elif hdf5_file_path.exists():
        hdf5_file_path.unlink()
    
    # Create metadata dictionary with all necessary information
    metadata = {
        'dataset': dataset,
        'hand_type': hand_type,
        'sequence_index': sequence.index,
        'total_frames': len(sequence),
        'camera_intrinsics': camera_intrinsics._to_dict(),
        'tip_indices': MANO_TIP_INDEX_MAP[dataset],
        'joint_names': {
            'left': processor.hands['left']['joint_names'] if 'left' in processor.hands else None,
            'right': processor.hands['right']['joint_names'] if 'right' in processor.hands else None
        },
        'pose_names': {
            'left': processor.hands['left']['pose_names'] if 'left' in processor.hands else None,
            'right': processor.hands['right']['pose_names'] if 'right' in processor.hands else None
        }
    }

    # Create HDF5 file and write data in streaming fashion to avoid memory issues
    with h5py.File(hdf5_file_path, 'w') as f:
        # Write metadata to HDF5 structure
        write_metadata_to_hdf5(f, metadata)
        
        # Create frames group to store all frame data
        frames_group = f.create_group('frames')
        
        # Process first frame to determine data shapes for pre-allocation
        print("Processing first frame to determine data shapes...")
        data = sequence[0]
        result = processor.process_hand_pose(
            left_pose=data['left_pose'],
            right_pose=data['right_pose'],
            left_R_hand_world=data['left_R_hand_world'],
            right_R_hand_world=data['right_R_hand_world'],
            tip_indices=MANO_TIP_INDEX_MAP[dataset]
        )
        
        # Get shapes for efficient pre-allocation
        rgb_shape = result['rgb_image'].shape      # e.g., (480, 640, 3)
        depth_shape = result['depth_image'].shape  # e.g., (480, 640)
        total_frames = len(sequence)
        
        # Create datasets for frame metadata
        frame_ids_ds = frames_group.create_dataset('frame_ids', (total_frames,), dtype='int32')
        timestamps_ds = frames_group.create_dataset('timestamps', (total_frames,), dtype=h5py.string_dtype())
        
        # Create datasets for images with chunking for efficient I/O
        rgb_ds = frames_group.create_dataset('rgb_images', (total_frames,) + rgb_shape, dtype=np.uint8, 
                                           chunks=True)
        depth_ds = frames_group.create_dataset('depth_images', (total_frames,) + depth_shape, dtype=np.float16,
                                              chunks=True)
        
        # Create groups for joint and pose data
        joint_values_group = frames_group.create_group('joint_values')
        poses_group = frames_group.create_group('poses')
        
        # Create datasets for each hand side
        for side in ['left', 'right']:
            # Joint values: (num_frames, num_joints) - stores joint angles in radians
            if metadata['joint_names'][side] is not None:
                num_joints = len(metadata['joint_names'][side])
                joint_values_group.create_dataset(side, (total_frames, num_joints), dtype=np.float32,
                                                 fillvalue=np.nan, chunks=True)
                
            # Poses: (num_frames, num_poses, 4, 4) - stores 4x4 transformation matrices
            if metadata['pose_names'][side] is not None:
                num_poses = len(metadata['pose_names'][side])
                poses_group.create_dataset(side, (total_frames, num_poses, 4, 4), dtype=np.float32,
                                         fillvalue=np.nan, chunks=True)
        
        # Write first frame data
        frame_ids_ds[0] = 0
        timestamps_ds[0] = sequence.tag[0].encode('utf-8')
        rgb_ds[0] = result['rgb_image']
        depth_ds[0] = result['depth_image']
        
        # Write joint values and poses for first frame
        for side in ['left', 'right']:
            if result['joint_values'][side] is not None:
                joint_values_group[side][0] = result['joint_values'][side]
            
            if result['poses'][side] is not None:
                for j, pname in enumerate(metadata['pose_names'][side]):
                    if pname in result['poses'][side]:
                        poses_group[side][0, j] = result['poses'][side][pname]
        
        print(f"Frame 1/{total_frames} processed and saved")
        
        # Process remaining frames with progress bar
        for i in tqdm(range(1, total_frames), desc=f"Processing {sequence.index}", initial=1, total=total_frames):
            data = sequence[i]

            # Process current frame
            result = processor.process_hand_pose(
                left_pose=data['left_pose'],
                right_pose=data['right_pose'],
                left_R_hand_world=data['left_R_hand_world'],
                right_R_hand_world=data['right_R_hand_world'],
                tip_indices=MANO_TIP_INDEX_MAP[dataset]
            )

            # Write frame data to HDF5
            frame_ids_ds[i] = i
            timestamps_ds[i] = sequence.tag[i].encode('utf-8')
            rgb_ds[i] = result['rgb_image']
            depth_ds[i] = result['depth_image']
            
            # Write joint values and poses for current frame
            for side in ['left', 'right']:
                if result['joint_values'][side] is not None:
                    joint_values_group[side][i] = result['joint_values'][side]
                
                if result['poses'][side] is not None:
                    for j, pname in enumerate(metadata['pose_names'][side]):
                        if pname in result['poses'][side]:
                            poses_group[side][i, j] = result['poses'][side][pname]
            
            # Flush data to disk every 10 frames to prevent memory buildup
            if (i + 1) % 10 == 0:
                f.flush()

        # Mark processing as complete
        f['complete'][()] = True
        f.flush()

    print(f"Sequence saved to: {hdf5_file_path}")

def write_metadata_to_hdf5(f, metadata):
    f.create_dataset('complete', data=False)
    
    metadata_group = f.create_group('metadata')
    metadata_group.attrs['dataset'] = metadata['dataset']
    metadata_group.attrs['hand_type'] = metadata['hand_type']
    metadata_group.attrs['sequence_index'] = metadata['sequence_index']
    metadata_group.attrs['total_frames'] = metadata['total_frames']
    
    cam_group = metadata_group.create_group('camera_intrinsics')
    for key, value in metadata['camera_intrinsics'].items():
        cam_group.attrs[key] = value
    
    metadata_group.create_dataset('tip_indices', data=metadata['tip_indices'])
    
    joint_names_group = metadata_group.create_group('joint_names')
    pose_names_group = metadata_group.create_group('pose_names')
    for side in ['left', 'right']:
        if metadata['joint_names'][side] is not None:
            joint_names_group.create_dataset(side, data=[name.encode('utf-8') for name in metadata['joint_names'][side]])
        if metadata['pose_names'][side] is not None:
            pose_names_group.create_dataset(side, data=[name.encode('utf-8') for name in metadata['pose_names'][side]])

def save_sequence_to_hdf5(sequence_data, file_path):
    print("Warning: Using legacy save_sequence_to_hdf5 function. Consider using streaming approach.")
    
    with h5py.File(file_path, 'w') as f:
        write_metadata_to_hdf5(f, sequence_data['metadata'])
        
        frames_group = f.create_group('frames')
        frames = sequence_data['frames']
        
        frame_ids = [frame['frame_id'] for frame in frames]
        timestamps = [frame['timestamp'].encode('utf-8') for frame in frames]
        frames_group.create_dataset('frame_ids', data=frame_ids, dtype='int32')
        frames_group.create_dataset('timestamps', data=timestamps)
        
        rgb_images = np.stack([frame['rgb_image'] for frame in frames])
        depth_images = np.stack([frame['depth_image'] for frame in frames])
        frames_group.create_dataset('rgb_images', data=rgb_images.astype(np.uint8))
        frames_group.create_dataset('depth_images', data=depth_images.astype(np.float16))
        
        joint_values_group = frames_group.create_group('joint_values')

        for side in ['left', 'right']:
            if sequence_data['metadata']['joint_names'][side] is not None:
                num_joints = len(sequence_data['metadata']['joint_names'][side])
                joint_vals_array = np.full((len(frames), num_joints), np.nan, dtype=np.float32)
                for i, frame in enumerate(frames):
                    if frame['joint_values'][side] is not None:
                        joint_vals_array[i] = frame['joint_values'][side]
                joint_values_group.create_dataset(side, data=joint_vals_array)
        
        poses_group = frames_group.create_group('poses')
        
        for side in ['left', 'right']:
            if sequence_data['metadata']['pose_names'][side] is not None:
                num_poses = len(sequence_data['metadata']['pose_names'][side])
                poses_array = np.full((len(frames), num_poses, 4, 4), np.nan, dtype=np.float32)
                for i, frame in enumerate(frames):
                    if frame['poses'][side] is not None:
                        for j, pname in enumerate(sequence_data['metadata']['pose_names'][side]):
                            if pname in frame['poses'][side]:
                                poses_array[i, j] = frame['poses'][side][pname]
                poses_group.create_dataset(side, data=poses_array)
        
        f['complete'][()] = True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--hand_type', type=str, required=True)
    parser.add_argument('--cont', action='store_true', help='Continue processing existing files')
    parser.add_argument('--randperm', action='store_true', help='Randomly permute dataset sequences. Useful for simple parallelization.')

    args = parser.parse_args()
    dataset_name = args.dataset
    hand_type = args.hand_type
    continue_process = args.cont
    random_permute = args.randperm

    dataset = hydra.utils.instantiate(DATASET_CONFIG[dataset_name])
    base_path = Path(f'data/{dataset_name}/retarget_RGBD')
    os.makedirs(base_path, exist_ok=True)
    left_urdf = f"HandAdapter/urdf/{dataset_name}/{hand_type}/left/main.urdf"
    right_urdf = f"HandAdapter/urdf/{dataset_name}/{hand_type}/right/main.urdf"
    
    processor = HandProcessor(left_urdf_path=left_urdf, right_urdf_path=right_urdf)

    if random_permute:
        dataset = list(dataset)
        random.shuffle(dataset)
    
    # Convert to list if not already for progress bar
    if not isinstance(dataset, list):
        dataset = list(dataset)
    
    for sequence in tqdm(dataset, desc=f"Processing {dataset_name} sequences"):
        process_sequence(dataset_name, hand_type, processor, base_path, sequence, continue_process)

if __name__ == "__main__":
    main()