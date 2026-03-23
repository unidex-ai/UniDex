"""
Web-based Interactive Hand Visualizer
"""

import os
import glob
import copy
import shutil
import json
import numpy as np
import open3d as o3d
import pybullet as p
import pybullet_data
from flask import Flask, render_template, request, jsonify, send_from_directory
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from scipy.spatial.transform import Rotation as R
import xml.etree.ElementTree as ET
import hydra

B2O = np.array([[ 1, 0, 0, 0],
                [ 0, 0, 1, 0],
                [ 0,-1, 0, 0],
                [ 0, 0, 0, 1]])

DATASET_CONFIG = {
    "H2o": {
        "_target_": "dataset.H2o.H2oDataset",
        "data_dir": "data/H2o",
        "load_pcd": True,
        "cache_path": "HandAdapter/dataset_cache/H2o.pkl"
    },
    "HOI4D": {
        "_target_": "dataset.HOI4D.HOI4DDataset", 
        "data_dir": "data/HOI4D",
        "load_pcd": True,
        "cache_path": "HandAdapter/dataset_cache/HOI4D.pkl"
    },
    "hot3d": {
        "_target_": "dataset.hot3d.hot3dDataset",
        "data_dir": "data/hot3d",
        "load_pcd": True,
        "cache_path": "HandAdapter/dataset_cache/hot3d.pkl"
    },
    "Taco": {
        "_target_": "dataset.Taco.TacoDataset",
        "data_dir": "data/Taco",
        "load_pcd": True,
        "cache_path": "HandAdapter/dataset_cache/Taco.pkl"
    }
}

MANO_TIP_INDEX_MAP = {
    "H2o": [4, 8, 12, 16, 20],
    "HOI4D": [4, 8, 12, 16, 20],
    "hot3d": [16, 17, 18, 19, 20],
    "Taco": [4, 8, 12, 16, 20]
}
MANO_TIP_INDEX = None

# Available hand types
HAND_TYPES = ["Inspire", "Leap", "Wuji", "Shadow", "Xhand", "Allegro", "Oymotion", "Ability"]

# Base URDF directory
URDF_BASE_DIR = Path("HandAdapter/urdf")

def get_urdf_path(dataset_name: str, hand_type: str, side: str) -> str:
    """
    Get URDF path for a specific dataset, hand type, and side.
    If the specific path doesn't exist, copy from base and return the path.
    
    Args:
        dataset_name: Name of the dataset (e.g., "H2o", "HOI4D")
        hand_type: Type of hand (e.g., "Inspire", "Leap", "Wuji")
        side: Hand side ("left" or "right")
    
    Returns:
        Path to the URDF file
    """
    # Target path: urdf/{dataset}/{hand_type}/{side}/main.urdf
    target_dir = URDF_BASE_DIR / dataset_name / hand_type / side
    target_urdf = target_dir / "main.urdf"
    
    # Base path: urdf/base/{hand_type}/{side}/main.urdf
    base_dir = URDF_BASE_DIR / "base" / hand_type / side
    base_urdf = base_dir / "main.urdf"
    
    # If target doesn't exist, copy from base
    if not target_urdf.exists():
        if base_urdf.exists():
            target_hand_dir = target_dir.parent
            base_hand_dir = base_urdf.parent.parent

            print(f"Copying directory {base_hand_dir} to {target_hand_dir}")
            shutil.copytree(base_hand_dir, target_hand_dir)
        else:
            raise FileNotFoundError(f"Base URDF not found: {base_urdf}")
    
    return str(target_urdf)

def load_dataset(dataset_name: str):
    """Load dataset using hydra configuration"""
    if dataset_name not in DATASET_CONFIG:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    config = DATASET_CONFIG[dataset_name]
    dataset = hydra.utils.instantiate(config)
    dataset.load_pcd = True  # Ensure point clouds are loaded
    for seq in dataset.sequences:
        seq.load_pcd = True
    return dataset

def _load_mesh(filename: str, scale: List[float]) -> o3d.geometry.TriangleMesh:
    mesh = o3d.io.read_triangle_mesh(filename)
    mesh.compute_vertex_normals()
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices) * np.asarray(scale))
    return mesh

def parse_visual_offsets_from_urdf(urdf_path: str) -> Dict[str, np.ndarray]:
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

class WebHandVisualizer:
    """Web version of hand visualizer"""
    physics_client = None

    def __init__(self, urdf: str, is_left: bool = False):
        self.urdf_path = str(Path(urdf).resolve())
        config_path = Path(urdf).parent.parent / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.is_left = is_left

        self.base_xyz, self.base_rpy = self._parse_joint_origin(urdf, self.config["base_joint"])

        self.mano_scale = 1.0
        self.max_iterations = 1000
        self.residual_threshold = 1e-3
        self.ik_damping = 0.1
        self.mimic_iterations = 50  # Number of mimic iterations
        self.mimic_step = 5        # Max iterations per mimic step
        self.mesh_bank = None
        self.visible = True  # Initialize as visible by default
        
        if WebHandVisualizer.physics_client is None:
            WebHandVisualizer.physics_client = p.connect(p.DIRECT)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.setGravity(0, 0, 0)
            p.setPhysicsEngineParameter(enableFileCaching=0)
        
        self._load_robot()
        self.pose: Optional[np.ndarray] = None
        self.R_hand_world: Optional[np.ndarray] = None

    def _parse_joint_origin(self, urdf_path: str, joint_name: str):
        """Parse the xyz and rpy of a specific joint from the URDF file."""
        import xml.etree.ElementTree as ET
        print(f"Parsing joint '{joint_name}' from URDF: {urdf_path}")
        tree = ET.parse(urdf_path)
        root = tree.getroot()

        for joint in root.findall('joint'):
            # print(f"Found joint: {joint.get('name')}")
            if joint.get('name') == joint_name:
                origin = joint.find('origin')
                if origin is not None:
                    xyz = [float(v) for v in origin.get('xyz', '0 0 0').split()]
                    rpy = [float(v) for v in origin.get('rpy', '0 0 0').split()]
                    print(f"Joint '{joint_name}' found at index {joint.get('name')}")
                    print(f"Position: {xyz}, Orientation: {rpy}")
                    return xyz, rpy
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]

    def get_initial_parameters(self):
        """Get initial parameters for the browser."""
        return {
            'base_xyz': self.base_xyz,
            'base_rpy': self.base_rpy,
            'mano_scale': self.mano_scale,
            'max_iterations': self.max_iterations,
            'residual_threshold': self.residual_threshold,
            'ik_damping': self.ik_damping,
            'mimic_iterations': self.mimic_iterations,
            'mimic_step': self.mimic_step,
        }

    def _load_robot(self):
        """Load robot URDF"""
        if hasattr(self, 'robot'):
            p.removeBody(self.robot)

        self.robot = p.loadURDF(self.urdf_path, useFixedBase=True)

        if self.mesh_bank is not None:
            return
        
        self.actuated = [j for j in range(p.getNumJoints(self.robot))
                         if p.getJointInfo(self.robot, j)[2] in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC)]
        info = [p.getJointInfo(self.robot, j) for j in self.actuated]
        self.lowers = [i[8] for i in info]
        self.uppers = [i[9] for i in info]
        self.rests = [0.0] * len(self.actuated)
        
        self.link_names = [p.getJointInfo(self.robot, j)[12].decode() for j in range(p.getNumJoints(self.robot))]
        
        tips = self.config["tips"]
        self.tip_links = [self.link_names.index(n) for n in tips if n in self.link_names]
        print(self.actuated, self.tip_links)
        
        self.joint_names = [p.getJointInfo(self.robot, j)[1].decode() for j in self.actuated]
        
        urdf_visual_offsets = parse_visual_offsets_from_urdf(self.urdf_path)
        self.mesh_bank: Dict[int, Tuple[o3d.geometry.TriangleMesh, np.ndarray]] = {}
        for vs in p.getVisualShapeData(self.robot):
            _, link_idx, geom_type, scale, fname, *_ = vs
            if geom_type != p.GEOM_MESH or fname == b'':
                continue
            fname = fname.decode('utf-8')
            if not os.path.isabs(fname):
                fname = str(Path(self.urdf_path).parent / fname)
            if os.path.exists(fname):
                mesh = _load_mesh(fname, scale)
                link_name = self.link_names[link_idx] if 0 <= link_idx < len(self.link_names) else 'base'
                T_local = urdf_visual_offsets.get(link_name, np.eye(4))
                self.mesh_bank[link_idx] = (mesh, T_local)

    def _update_urdf_joint_origin(self, joint_name: str, xyz: List[float], rpy: List[float]):
        """Update the xyz and rpy of a specific joint in the URDF file."""
        import xml.etree.ElementTree as ET
        # print(f"Updating joint '{joint_name}' - XYZ: {xyz}, RPY: {rpy}")
        tree = ET.parse(self.urdf_path)
        root = tree.getroot()

        for joint in root.findall('joint'):
            if joint.get('name') == joint_name:
                origin = joint.find('origin')
                if origin is not None:
                    origin.set('xyz', ' '.join(map(str, xyz)))
                    origin.set('rpy', ' '.join(map(str, rpy)))
                    break

        tree.write(self.urdf_path)

    def update_parameters(self, **kwargs):
        if 'mano_scale' in kwargs:
            self.mano_scale = kwargs['mano_scale']
        if 'base_xyz' in kwargs:
            self.base_xyz = kwargs['base_xyz']
        if 'base_rpy' in kwargs:
            self.base_rpy = kwargs['base_rpy']
        if 'max_iterations' in kwargs:
            self.max_iterations = kwargs['max_iterations']
        if 'residual_threshold' in kwargs:
            self.residual_threshold = kwargs['residual_threshold']
        if 'ik_damping' in kwargs:
            self.ik_damping = kwargs['ik_damping']
        if 'mimic_iterations' in kwargs:
            self.mimic_iterations = kwargs['mimic_iterations']
        if 'mimic_step' in kwargs:
            self.mimic_step = kwargs['mimic_step']

        # Update URDF file if base_xyz or base_rpy is updated
        if 'base_xyz' in kwargs or 'base_rpy' in kwargs:
            self._update_urdf_joint_origin(self.config["base_joint"], self.base_xyz, self.base_rpy)

        self._load_robot()

    def load_hand_pose_and_rot(self, pose_data, R_hand_world_data, is_left: bool):
        """
        Load pose data from the new format
        
        Args:
            pose_data: 3D joint positions array (21, 3) or None
            R_hand_world_data: Hand rotation matrix (3, 3) or None  
            is_left: Whether this is left hand
        """
        hand_side = "left" if is_left else "right"
        print(f"Loading {hand_side} hand pose: pose_data shape={pose_data.shape if pose_data is not None else None}")
        
        if pose_data is None:
            print(f"No {hand_side} hand pose data")
            # Set a default pose to avoid mesh generation errors, but mark as invisible
            self.pose = None
            self.R_hand_world = None
            self.visible = False
            return
        
        self.pose = pose_data
        self.visible = True

        assert R_hand_world_data is not None
        self.R_hand_world = R_hand_world_data


        print(f"{hand_side.capitalize()} hand pose loaded successfully: {self.pose.shape}")
    
    def solve_ik(
        self,
        targets,
    ):
        # --- Mimic map: slave = master * mult + offset ---
        MIMIC_RELATIONS = self.config["mimic_relations"]
        print(f"Solving IK with targets: {targets}, mimic relations: {MIMIC_RELATIONS}")
        # Map joint names → indices *within the actuated vector order*
        joint_name_to_idx = {name: i for i, name in enumerate(self.joint_names)}
        slave_pairs = []
        for slave_name, (master_name, mult, offset) in MIMIC_RELATIONS.items():
            if slave_name in joint_name_to_idx and master_name in joint_name_to_idx:
                slave_idx  = joint_name_to_idx[slave_name]
                master_idx = joint_name_to_idx[master_name]
                slave_pairs.append((slave_idx, master_idx, float(mult), float(offset)))

        # Ensure Python lists for PyBullet
        targets = targets[:len(self.tip_links)]
        target_positions = [np.asarray(t, dtype=float).tolist() for t in targets]

        # Warm start & ranges for better convergence (actuated-only arrays)
        num_joints = p.getNumJoints(self.robot)
        joint_ranges = [(u - l) if np.isfinite(u) and np.isfinite(l) else 2.0
                        for l, u in zip(self.lowers, self.uppers)]
        joint_damping = [float(self.ik_damping)] * num_joints

        if MIMIC_RELATIONS != {}:
            best_error = float('inf')
            best_q = None
            for _ in range(self.mimic_iterations):
                rest_poses = [p.getJointState(self.robot, j)[0] for j in range(num_joints)]

                # Solve all tip links at once
                q = p.calculateInverseKinematics2(
                    bodyUniqueId=self.robot,
                    endEffectorLinkIndices=self.tip_links,
                    targetPositions=target_positions,
                    # targetOrientations=target_orientations,   # enable if you want orientation constraints
                    lowerLimits=self.lowers,
                    upperLimits=self.uppers,
                    jointRanges=joint_ranges,
                    restPoses=rest_poses,
                    jointDamping=joint_damping,
                    maxNumIterations=self.mimic_step,
                    residualThreshold=self.residual_threshold,
                )

                q = np.asarray(q, dtype=float)
                q = np.clip(q[:len(self.actuated)], self.lowers, self.uppers)

                # Apply mimic: overwrite slaves from their masters
                for slave_idx, master_idx, mult, offset in slave_pairs:
                    # guard in case something changes in URDF later
                    if 0 <= master_idx < len(q) and 0 <= slave_idx < len(q):
                        q[slave_idx] = q[master_idx] * mult + offset

                # Clip again after mimic so slaves obey limits too
                q = np.clip(q, self.lowers, self.uppers)
                for j_idx, angle in zip(self.actuated, q):
                    p.resetJointState(self.robot, j_idx, angle)

                error = 0.0
                for link_idx, target in zip(self.tip_links, target_positions):
                    pos, _ = p.getLinkState(self.robot, link_idx, computeForwardKinematics=True)[4:6]
                    error += np.linalg.norm(np.array(pos) - np.array(target))
                if error < best_error:
                    best_error = error
                    best_q = q
                if error < self.residual_threshold * len(self.tip_links):
                    q = best_q
                    print("Mimic IK converged")
                    break
            q = best_q
        else:
            rest_poses = [p.getJointState(self.robot, j)[0] for j in range(num_joints)]
            # Solve all tip links at once without mimic
            q = p.calculateInverseKinematics2(
                bodyUniqueId=self.robot,
                endEffectorLinkIndices=self.tip_links,
                targetPositions=target_positions,
                lowerLimits=self.lowers,
                upperLimits=self.uppers,
                jointRanges=joint_ranges,
                restPoses=rest_poses,
                jointDamping=joint_damping,
                maxNumIterations=self.max_iterations,
                residualThreshold=self.residual_threshold,
            )

            q = np.asarray(q, dtype=float)
            q = np.clip(q[:len(self.actuated)], self.lowers, self.uppers)

        print(f"IK solution: {q}, actuated joints: {self.actuated}")
        return q
    
    def get_mesh_data_for_web(self, simplify_ratio: float = 1.0) -> Dict:
        print(f"get_mesh_data_for_web called: pose={'available' if self.pose is not None else 'None'}")
        
        if self.pose is None:
            print("ERROR: No pose data available for mesh generation")
            raise ValueError("No pose data available")
        
        print(f"Starting mesh generation with pose shape: {self.pose.shape}")
        print(f"Mesh bank has {len(self.mesh_bank)} items")
        
        j_cam = self.pose.copy()
        if self.mano_scale != 1.0:
            j_cam = (j_cam - j_cam[0]) * self.mano_scale + j_cam[0]

        wrist_w = j_cam[0]
        R_hand_w = self.R_hand_world

        R_o2b = B2O[:3, :3].T
        t_o2b = -R_o2b @ B2O[:3, 3]
        R_hand_w_pb = R_o2b @ R_hand_w
        wrist_ik_pb = R_o2b @ wrist_w + t_o2b

        p.resetBasePositionAndOrientation(
            self.robot, wrist_ik_pb.tolist(), R.from_matrix(R_hand_w_pb).as_quat().tolist())

        idxs = MANO_TIP_INDEX
        tgt_w = [j_cam[i] for i in idxs]
        goals_bullet = [R_o2b @ g + t_o2b for g in tgt_w]
        print(f"Target positions for IK: {goals_bullet}")
        q = self.solve_ik([list(g) for g in goals_bullet])
        for j_idx, angle in zip(self.actuated, q):
            p.resetJointState(self.robot, j_idx, angle)

        all_link_meshes = []

        for link_idx, (original_mesh, T_local) in self.mesh_bank.items():
            pos, orn = (p.getLinkState(self.robot, link_idx, computeForwardKinematics=True)[4:6]
                       if link_idx >= 0 else p.getBasePositionAndOrientation(self.robot))
            R_link = R.from_quat(orn).as_matrix()
            T_link = np.eye(4)
            T_link[:3, :3] = R_link
            T_link[:3, 3] = pos
            T_mesh = (B2O @ T_link) @ T_local
            
            mesh = copy.deepcopy(original_mesh)
            if len(mesh.triangles) > 100 and simplify_ratio < 1.0:
                print(f"Simplifying mesh for link {link_idx} with {len(mesh.triangles)} triangles")
                target_triangles = max(50, int(len(mesh.triangles) * simplify_ratio))
                mesh = mesh.simplify_quadric_decimation(target_triangles)
                print(f"Simplified to {len(mesh.triangles)} triangles")

            # Send original vertices and faces, transform separately
            vertices = np.asarray(mesh.vertices)
            faces = np.asarray(mesh.triangles)
            colors = np.asarray(mesh.vertex_colors) if mesh.has_vertex_colors() else np.tile([0.8, 0.8, 0.8], (len(vertices), 1))
            
            all_link_meshes.append({
                'link_idx': link_idx,
                'vertices': vertices.tolist(),
                'faces': faces.tolist(),
                'colors': colors.tolist(),
                'transform': T_mesh.T.reshape(16).tolist()  # Transpose matrix for Three.js column-major format
            })

        all_vertices = []
        all_faces = []
        all_colors = []
        vertex_offset = 0
        for i, g in enumerate(tgt_w):
            sphere_vertices = np.array([
                [0, 0, 0.005], [0, 0, -0.005],
                [0.005, 0, 0], [-0.005, 0, 0],
                [0, 0.005, 0], [0, -0.005, 0]
            ]) + g
            
            sphere_faces = np.array([
                [0, 2, 4], [0, 4, 3], [0, 3, 5], [0, 5, 2],
                [1, 4, 2], [1, 3, 4], [1, 5, 3], [1, 2, 5]
            ]) + vertex_offset
            
            sphere_colors = np.tile([0.0, 1.0, 0.0], (6, 1))
            
            all_vertices.extend(sphere_vertices.tolist())
            all_faces.extend(sphere_faces.tolist())
            all_colors.extend(sphere_colors.tolist())
            vertex_offset += 6

        result = {
            'link_meshes': all_link_meshes,
            'other_mesh': {
                "vertices": all_vertices,
                "faces": all_faces,
                "colors": all_colors
            }
        }
        
        print(f"Mesh generation completed: {len(all_link_meshes)} link meshes, {len(all_vertices)} target vertices")
        return result
    
    def get_link_transforms_for_web(self) -> Dict:
        """Only get link transforms for web visualization"""
        if self.pose is None:
            raise ValueError("No pose data available")
        
        j_cam = self.pose.copy()
        if self.mano_scale != 1.0:
            j_cam = (j_cam - j_cam[0]) * self.mano_scale + j_cam[0]

        wrist_w = j_cam[0]
        R_hand_w = self.R_hand_world

        R_o2b = B2O[:3, :3].T
        t_o2b = -R_o2b @ B2O[:3, 3]
        R_hand_w_pb = R_o2b @ R_hand_w
        wrist_ik_pb = R_o2b @ wrist_w + t_o2b

        p.resetBasePositionAndOrientation(
            self.robot, wrist_ik_pb.tolist(), R.from_matrix(R_hand_w_pb).as_quat().tolist())

        idxs = MANO_TIP_INDEX
        tgt_w = [j_cam[i] for i in idxs]
        goals_bullet = [R_o2b @ g + t_o2b for g in tgt_w]
        q = self.solve_ik([list(g) for g in goals_bullet])
        for j_idx, angle in zip(self.actuated, q):
            p.resetJointState(self.robot, j_idx, angle)

        all_transforms = []

        for link_idx, (_, T_local) in self.mesh_bank.items():
            pos, orn = (p.getLinkState(self.robot, link_idx, computeForwardKinematics=True)[4:6]
                       if link_idx >= 0 else p.getBasePositionAndOrientation(self.robot))
            R_link = R.from_quat(orn).as_matrix()
            T_link = np.eye(4)
            T_link[:3, :3] = R_link
            T_link[:3, 3] = pos
            T_mesh = (B2O @ T_link) @ T_local
            
            all_transforms.append({
                'link_idx': link_idx,
                'transform': T_mesh.T.reshape(16).tolist()  # Transpose matrix for Three.js column-major format
            })

        return all_transforms

app = Flask(__name__)

data_manager = {
    'current_dataset': 'H2o',
    'current_hand_type': 'Inspire',
    'available_datasets': list(DATASET_CONFIG.keys()),
    'available_hand_types': HAND_TYPES,
    'datasets': {},  # Cache loaded datasets
    'sequences': [],  # Current dataset sequences
    'current_sequence': 0,
    'current_frame': 0,
    'right_hand': None,
    'left_hand': None
}

SIMPLIFY_RATIO = 1.0  # Default simplification ratio for meshes
DOWNSAMPLE_RATIO = 20  # Default downsample ratio for point clouds

def initialize_data():
    """Initialize data for the visualizer"""
    print("Initializing data...")
    
    # Load default dataset
    load_dataset_sequences(data_manager['current_dataset'])
    
    # Initialize hand visualizers with current hand type
    setup_hand_visualizers(data_manager['current_hand_type'])
    
    print("Data initialization completed")

def load_dataset_sequences(dataset_name: str):
    """Load sequences from a specific dataset"""
    if dataset_name not in data_manager['datasets']:
        print(f"Loading dataset: {dataset_name}")
        try:
            dataset = load_dataset(dataset_name)
            data_manager['datasets'][dataset_name] = dataset
            print(f"Dataset {dataset_name} loaded with {len(dataset)} sequences")
        except Exception as e:
            print(f"Error loading dataset {dataset_name}: {e}")
            return False
    
    dataset = data_manager['datasets'][dataset_name]
    data_manager['sequences'] = [dataset[i] for i in range(len(dataset))]
    data_manager['current_sequence'] = 0
    data_manager['current_frame'] = 0
    global MANO_TIP_INDEX
    MANO_TIP_INDEX = MANO_TIP_INDEX_MAP.get(dataset_name)
    
    print(f"Loaded {len(data_manager['sequences'])} sequences from {dataset_name}")
    return True

def setup_hand_visualizers(hand_type: str):
    """Setup hand visualizers for a specific hand type"""
    try:
        dataset_name = data_manager['current_dataset']
        
        right_urdf = get_urdf_path(dataset_name, hand_type, "right")
        left_urdf = get_urdf_path(dataset_name, hand_type, "left")
        
        print(f"Initializing {hand_type} hand visualizers...")
        print(f"Right URDF: {right_urdf}")
        print(f"Left URDF: {left_urdf}")
        
        data_manager['right_hand'] = WebHandVisualizer(right_urdf, is_left=False)
        data_manager['left_hand'] = WebHandVisualizer(left_urdf, is_left=True)
        
        print(f"{hand_type} hand visualizers initialized successfully")
        
    except Exception as e:
        print(f"Error initializing {hand_type} hand visualizers: {e}")

@app.route('/api/frame_count')
def get_frame_count():
    """Get total frame count for current sequence"""
    if data_manager['sequences'] and data_manager['current_sequence'] < len(data_manager['sequences']):
        current_seq = data_manager['sequences'][data_manager['current_sequence']]
        return jsonify({'count': len(current_seq)})
    return jsonify({'count': 0})

@app.route('/api/datasets')
def get_datasets():
    """Get available datasets"""
    return jsonify({
        'datasets': data_manager['available_datasets'],
        'current': data_manager['current_dataset']
    })

@app.route('/api/hand_types')
def get_hand_types():
    """Get available hand types"""
    return jsonify({
        'hand_types': data_manager['available_hand_types'],
        'current': data_manager['current_hand_type']
    })

@app.route('/api/sequences')
def get_sequences():
    """Get available sequences"""
    return jsonify({
        'count': len(data_manager['sequences']),
        'current': data_manager['current_sequence']
    })

@app.route('/api/switch_dataset', methods=['POST'])
def switch_dataset():
    """Switch to a different dataset"""
    try:
        dataset_name = request.json.get('dataset')
        if dataset_name not in data_manager['available_datasets']:
            return jsonify({'error': f'Unknown dataset: {dataset_name}'}), 400
        
        if load_dataset_sequences(dataset_name):
            data_manager['current_dataset'] = dataset_name
            # Reinitialize hand visualizers for new dataset
            setup_hand_visualizers(data_manager['current_hand_type'])
            return jsonify({'status': 'success', 'dataset': dataset_name})
        else:
            return jsonify({'error': 'Failed to load dataset'}), 500
            
    except Exception as e:
        print(f"Error switching dataset: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/switch_hand_type', methods=['POST'])
def switch_hand_type():
    """Switch to a different hand type"""
    try:
        hand_type = request.json.get('hand_type')
        if hand_type not in data_manager['available_hand_types']:
            return jsonify({'error': f'Unknown hand type: {hand_type}'}), 400
        
        data_manager['current_hand_type'] = hand_type
        setup_hand_visualizers(hand_type)
        return jsonify({'status': 'success', 'hand_type': hand_type})
        
    except Exception as e:
        print(f"Error switching hand type: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/switch_sequence', methods=['POST'])
def switch_sequence():
    """Switch to a different sequence"""
    try:
        sequence_idx = request.json.get('sequence')
        if sequence_idx < 0 or sequence_idx >= len(data_manager['sequences']):
            return jsonify({'error': 'Invalid sequence index'}), 400
        
        data_manager['current_sequence'] = sequence_idx
        data_manager['current_frame'] = 0
        return jsonify({'status': 'success', 'sequence': sequence_idx})
        
    except Exception as e:
        print(f"Error switching sequence: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/frame/<int:frame_idx>')
def get_frame_data(frame_idx):
    """Get data for a specific frame index, including point cloud and hand poses."""
    print(f"Requesting frame {frame_idx}")
    
    # Check if we have valid sequence and frame
    if not data_manager['sequences'] or data_manager['current_sequence'] >= len(data_manager['sequences']):
        return jsonify({'error': 'No valid sequence available'}), 400
    
    current_seq = data_manager['sequences'][data_manager['current_sequence']]
    if frame_idx >= len(current_seq):
        print(f"Invalid frame index: {frame_idx} >= {len(current_seq)}")
        return jsonify({'error': 'Invalid frame index'}), 400
    
    try:
        # Get frame data from current sequence
        frame_data = current_seq[frame_idx]
        print(f"Frame data keys: {frame_data.keys()}")
        
        pcd = frame_data.get('pcd')
        left_pose = frame_data.get('left_pose')
        right_pose = frame_data.get('right_pose')
        left_R_hand_world = frame_data.get('left_R_hand_world')
        right_R_hand_world = frame_data.get('right_R_hand_world')
        
        print(f"Left pose shape: {left_pose.shape if left_pose is not None else 'None'}")
        print(f"Right pose shape: {right_pose.shape if right_pose is not None else 'None'}")
        
        # Process point cloud
        pcd_data = {'points': [], 'colors': []}
        if pcd is not None:
            print(f"Processing pointcloud with {len(pcd.points)} points")
            try:
                # Apply downsampling
                pcd_downsampled = pcd.uniform_down_sample(every_k_points=DOWNSAMPLE_RATIO)
                if len(np.asarray(pcd_downsampled.points)) > 0:
                    pcd_data = {
                        'points': np.asarray(pcd_downsampled.points).tolist(),
                        'colors': np.asarray(pcd_downsampled.colors).tolist() if pcd_downsampled.has_colors() else []
                    }
                    print(f"Pointcloud processed: {len(pcd_data['points'])} points")
                else:
                    print("Empty pointcloud after downsampling")
            except Exception as e:
                print(f"Error processing pointcloud: {e}")
        else:
            print("No pointcloud data available")
        
        # Process hand poses
        hand_data = {}
        
        # Process right hand
        if data_manager['right_hand']:
            print("Processing right hand...")
            data_manager['right_hand'].load_hand_pose_and_rot(right_pose, right_R_hand_world, is_left=False)
            # Only generate mesh if pose is available
            if data_manager['right_hand'].pose is not None:
                right_mesh = data_manager['right_hand'].get_mesh_data_for_web(SIMPLIFY_RATIO)
                hand_data['right'] = right_mesh
                print(f"Right hand mesh: {len(right_mesh['link_meshes'])} links")
            else:
                print("Right hand skipped: no pose data")
        
        # Process left hand
        if data_manager['left_hand']:
            print("Processing left hand...")
            data_manager['left_hand'].load_hand_pose_and_rot(left_pose, left_R_hand_world, is_left=True)
            # Only generate mesh if pose is available
            if data_manager['left_hand'].pose is not None:
                left_mesh = data_manager['left_hand'].get_mesh_data_for_web(SIMPLIFY_RATIO)
                hand_data['left'] = left_mesh
                print(f"Left hand mesh: {len(left_mesh['link_meshes'])} links")
            else:
                print("Left hand skipped: no pose data")
        
        result = {
            'frame_idx': frame_idx,
            'pointcloud': pcd_data,
            'hands': hand_data,
            'dataset': data_manager['current_dataset'],
            'hand_type': data_manager['current_hand_type'],
            'sequence': data_manager['current_sequence']
        }
        
        print(f"Frame {frame_idx} data prepared successfully")
        return jsonify(result)
        
    except Exception as e:
        print(f"Error loading frame {frame_idx}: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/update_params', methods=['POST'])
def update_parameters():
    """Update parameters for hand visualizers"""
    try:
        params = request.json
        # print(f"Received parameter update request: {params}")
        
        # Separate left and right hand parameters
        common_params = {}
        right_params = {}
        left_params = {}
        
        # Common parameters
        for key in ['mano_scale', 'max_iterations', 'residual_threshold', 'ik_damping']:
            if key in params:
                common_params[key] = params[key]
        
        # Right hand parameters
        right_params.update(common_params)
        if 'right_base_xyz' in params:
            right_params['base_xyz'] = params['right_base_xyz']
        elif 'base_xyz' in params:  # Backward compatibility
            right_params['base_xyz'] = params['base_xyz']
        if 'right_base_rpy' in params:
            right_params['base_rpy'] = params['right_base_rpy']
        elif 'base_rpy' in params:  # Backward compatibility
            right_params['base_rpy'] = params['base_rpy']
        
        # Left hand parameters
        left_params.update(common_params)
        if 'left_base_xyz' in params:
            left_params['base_xyz'] = params['left_base_xyz']
        elif 'base_xyz' in params:  # Backward compatibility
            left_params['base_xyz'] = params['base_xyz']
        if 'left_base_rpy' in params:
            left_params['base_rpy'] = params['left_base_rpy']
        elif 'base_rpy' in params:  # Backward compatibility
            left_params['base_rpy'] = params['base_rpy']
        
        if data_manager['right_hand']:
            # print(f"Updating right hand parameters: {right_params}")
            data_manager['right_hand'].update_parameters(**right_params)
            print("Right hand parameters updated successfully")

        if data_manager['left_hand']:
            # print(f"Updating left hand parameters: {left_params}")
            data_manager['left_hand'].update_parameters(**left_params)
            print("Left hand parameters updated successfully")
        
        print("Parameter update completed")
        return jsonify({'status': 'success'})
    
    except Exception as e:
        print(f"Error updating parameters: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/transforms_only/<int:frame_idx>')
def get_transforms_only(frame_idx):
    """Get only link transforms for a specific frame index."""
    print(f"Requesting hands only for frame {frame_idx}")
    
    # Check if we have valid sequence and frame
    if not data_manager['sequences'] or data_manager['current_sequence'] >= len(data_manager['sequences']):
        return jsonify({'error': 'No valid sequence available'}), 400
    
    current_seq = data_manager['sequences'][data_manager['current_sequence']]
    if frame_idx >= len(current_seq):
        return jsonify({'error': 'Invalid frame index'}), 400
    
    try:
        # Get frame data from current sequence
        frame_data = current_seq[frame_idx]
        left_pose = frame_data.get('left_pose')
        right_pose = frame_data.get('right_pose')
        left_R_hand_world = frame_data.get('left_R_hand_world')
        right_R_hand_world = frame_data.get('right_R_hand_world')
        
        transform_data = {}
        
        if data_manager['right_hand']:
            data_manager['right_hand'].load_hand_pose_and_rot(right_pose, right_R_hand_world, is_left=False)
            # Only get transforms if pose is available
            if data_manager['right_hand'].pose is not None:
                right_transform = data_manager['right_hand'].get_link_transforms_for_web()
                transform_data['right'] = right_transform
        
        if data_manager['left_hand']:
            data_manager['left_hand'].load_hand_pose_and_rot(left_pose, left_R_hand_world, is_left=True)
            # Only get transforms if pose is available
            if data_manager['left_hand'].pose is not None:
                left_transform = data_manager['left_hand'].get_link_transforms_for_web()
                transform_data['left'] = left_transform
        
        return jsonify({
            'frame_idx': frame_idx,
            'transforms': transform_data
        })
    
    except Exception as e:
        print(f"Error processing hands for frame {frame_idx}: {e}")
        return jsonify({'error': str(e)}), 500

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Interactive Hand Visualizer</title>
    <style>
        body { margin: 0; font-family: Arial, sans-serif; overflow: hidden; }
        #container { display: flex; height: 100vh; }
        #controls { width: 300px; padding: 20px; background: #f0f0f0; overflow-y: auto; }
        #viewer { flex: 1; }
        .control-group { margin-bottom: 15px; }
        .control-group label { display: block; margin-bottom: 5px; font-weight: bold; }
        .control-group input { width: 100%; }
        .status { background: #e8f4fd; padding: 10px; border-radius: 5px; margin: 10px 0; }
        button { padding: 8px 16px; margin: 5px; cursor: pointer; }
    </style>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/TrackballControls.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/FlyControls.js"></script>
    <script>
        // Ensure controllers are loaded correctly
        if (typeof THREE.OrbitControls === 'undefined') {
            console.error('OrbitControls failed to load');
        }
        if (typeof THREE.TrackballControls === 'undefined') {
            console.error('TrackballControls failed to load');
        }
        if (typeof THREE.FlyControls === 'undefined') {
            console.error('FlyControls failed to load');
        }
    </script>
</head>
<body>
    <div id="container">
        <div id="controls">
            <h3>Interactive Hand Visualizer</h3>
            
            <h4>Dataset & Hand Selection</h4>
            <div class="control-group">
                <label>Dataset:</label>
                <select id="dataset-selector">
                    <option value="H2o">H2O Dataset</option>
                    <option value="HOI4D">HOI4D Dataset</option>
                    <option value="hot3d">hot3d Dataset</option>
                    <option value="Taco">Taco Dataset</option>
                </select>
            </div>
            
            <div class="control-group">
                <label>Hand Type:</label>
                <select id="hand-type-selector">
                    <!-- HAND_OPTIONS_PLACEHOLDER -->
                </select>
            </div>
            
            <div class="control-group">
                <label>Sequence: <span id="sequence-label">0</span></label>
                <input type="range" id="sequence-slider" min="0" max="0" value="0">
                <span id="sequence-info">(0 sequences)</span>
            </div>
            
            <div class="control-group">
                <label>Frame: <span id="frame-label">0</span></label>
                <input type="range" id="frame-slider" min="0" max="100" value="0">
            </div>
            
            <h4>Camera Controls</h4>
            <div class="control-group">
                <label>Control Mode:</label>
                <select id="control-mode">
                    <option value="orbit">Orbit (Default)</option>
                    <option value="trackball">Trackball (Free Rotation)</option>
                    <option value="fly">Fly (Flight Mode)</option>
                    <option value="manual">Manual (Manual)</option>
                </select>
            </div>
            
            <div class="control-group">
                <label>Camera Position X: <span id="cam-x-value">0.500</span></label>
                <input type="range" id="cam-x" min="-2" max="2" step="0.01" value="0.5">
            </div>
            
            <div class="control-group">
                <label>Camera Position Y: <span id="cam-y-value">0.500</span></label>
                <input type="range" id="cam-y" min="-2" max="2" step="0.01" value="0.5">
            </div>
            
            <div class="control-group">
                <label>Camera Position Z: <span id="cam-z-value">0.500</span></label>
                <input type="range" id="cam-z" min="-2" max="2" step="0.01" value="0.5">
            </div>
            
            <div class="control-group">
                <button onclick="resetCamera()">Reset Camera</button>
                <button onclick="fitToScene()">Fit Scene</button>
            </div>
            
            <h4>Shortcuts</h4>
            <div class="control-group" style="font-size: 12px;">
                <p><strong>Movement:</strong> W/S(Forward/Back), A/D(Left/Right), Q/E(Up/Down)</p>
                <p><strong>Control:</strong> R(Reset), F(Fit Scene)</p>
                <p><strong>Mode:</strong> 1(Orbit), 2(Trackball), 3(Fly), 4(Manual)</p>
                <p><strong>Mouse:</strong> Left Click(Rotate), Right Click(Pan), Scroll(Zoom)</p>
            </div>
            
            <div class="control-group">
                <label>MANO Scale: <span id="mano-scale-value">1.000</span></label>
                <input type="range" id="mano-scale" min="0.5" max="2.0" step="0.01" value="1.0">
            </div>
            
            <h4>Right Hand Parameters</h4>
            <div class="control-group">
                <label>Right Base X: <span id="right-base-x-value">0.000</span></label>
                <input type="range" id="right-base-x" min="-0.3" max="0.3" step="0.001" value="0.0">
            </div>
            
            <div class="control-group">
                <label>Right Base Y: <span id="right-base-y-value">0.000</span></label>
                <input type="range" id="right-base-y" min="-0.3" max="0.3" step="0.001" value="0.0">
            </div>
            
            <div class="control-group">
                <label>Right Base Z: <span id="right-base-z-value">0.000</span></label>
                <input type="range" id="right-base-z" min="-0.3" max="0.3" step="0.001" value="0.0">
            </div>
            
            <div class="control-group">
                <label>Right Base Roll: <span id="right-base-roll-value">0.000</span></label>
                <input type="range" id="right-base-roll" min="-3.14159" max="3.14159" step="0.01" value="0.0">
            </div>
            
            <div class="control-group">
                <label>Right Base Pitch: <span id="right-base-pitch-value">0.000</span></label>
                <input type="range" id="right-base-pitch" min="-3.14159" max="3.14159" step="0.01" value="0.0">
            </div>
            
            <div class="control-group">
                <label>Right Base Yaw: <span id="right-base-yaw-value">0.000</span></label>
                <input type="range" id="right-base-yaw" min="-3.14159" max="3.14159" step="0.01" value="0.0">
            </div>
            
            <h4>Left Hand Parameters</h4>
            <div class="control-group">
                <label>Left Base X: <span id="left-base-x-value">0.000</span></label>
                <input type="range" id="left-base-x" min="-0.3" max="0.3" step="0.001" value="0.0">
            </div>
            
            <div class="control-group">
                <label>Left Base Y: <span id="left-base-y-value">0.000</span></label>
                <input type="range" id="left-base-y" min="-0.3" max="0.3" step="0.001" value="0.0">
            </div>
            
            <div class="control-group">
                <label>Left Base Z: <span id="left-base-z-value">0.000</span></label>
                <input type="range" id="left-base-z" min="-0.3" max="0.3" step="0.001" value="0.0">
            </div>
            
            <div class="control-group">
                <label>Left Base Roll: <span id="left-base-roll-value">0.000</span></label>
                <input type="range" id="left-base-roll" min="-3.14159" max="3.14159" step="0.01" value="0.0">
            </div>
            
            <div class="control-group">
                <label>Left Base Pitch: <span id="left-base-pitch-value">0.000</span></label>
                <input type="range" id="left-base-pitch" min="-3.14159" max="3.14159" step="0.01" value="0.0">
            </div>
            
            <div class="control-group">
                <label>Left Base Yaw: <span id="left-base-yaw-value">0.000</span></label>
                <input type="range" id="left-base-yaw" min="-3.14159" max="3.14159" step="0.01" value="0.0">
            </div>
            
            <h4>Common Parameters</h4>
            
            <div class="control-group">
                <label>Max Iterations: <span id="max-iter-value">1000</span></label>
                <input type="range" id="max-iter" min="100" max="5000" step="50" value="1000">
            </div>
            
            <div class="control-group">
                <label>Mimic Iterations: <span id="mimic-iter-value">50</span></label>
                <input type="range" id="mimic-iter" min="10" max="200" step="5" value="50">
            </div>
            
            <div class="control-group">
                <label>Mimic Step: <span id="mimic-step-value">20</span></label>
                <input type="range" id="mimic-step" min="5" max="100" step="5" value="20">
            </div>
            
            <div class="control-group">
                <label>Residual Threshold: <span id="res-thresh-value">1e-3</span></label>
                <input type="range" id="res-thresh" min="1e-5" max="1e-1" step="1e-5" value="1e-3">
            </div>
            
            <div class="control-group">
                <label>IK Damping: <span id="ik-damping-value">0.10</span></label>
                <input type="range" id="ik-damping" min="0.01" max="1.0" step="0.01" value="0.1">
            </div>
            
            <button onclick="resetParameters()">Reset Parameters</button>
            
            <div id="status" class="status">Ready</div>
        </div>
        
        <div id="viewer"></div>
    </div>

    <script>
        // Three.js setup
        let scene, camera, renderer, controls;
        let currentFrame = 0;
        let maxFrames = 100;
        let currentSequence = 0;
        let maxSequences = 0;
        let currentDataset = 'H2o';
        let currentHandType = 'Wuji';
        let currentPointcloud = null;  // Cache current point cloud
        let isUpdatingParams = false;  // Flag for parameter updates
        let controlMode = 'orbit';  // Current control mode
        let clock = new THREE.Clock();  // For FlyControls
        
        // Cache original mesh data and current hand meshes
        let handMeshTemplates = {
            'right': { link_meshes: [], other_mesh: null },
            'left': { link_meshes: [], other_mesh: null }
        };
        let currentHandMeshes = {
            'right': [],
            'left': []
        };
        
        function initThreeJS() {
            console.log('Initializing Three.js...');
            scene = new THREE.Scene();
            scene.background = new THREE.Color(0x222222); // Set dark gray background instead of black
            
            camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
            renderer = new THREE.WebGLRenderer({ antialias: true });
            
            const viewerDiv = document.getElementById('viewer');
            renderer.setSize(viewerDiv.clientWidth, viewerDiv.clientHeight);
            renderer.setClearColor(0x222222);
            viewerDiv.appendChild(renderer.domElement);
            
            console.log('Setting up camera and controls...');
            camera.position.set(0.5, 0.5, 0.5);
            camera.lookAt(0, 0, 0);
            
            // Initialize controls
            initializeControls();
            
            // Add lighting
            const ambientLight = new THREE.AmbientLight(0x404040, 0.8);
            scene.add(ambientLight);
            const directionalLight = new THREE.DirectionalLight(0xffffff, 1.0);
            directionalLight.position.set(1, 1, 1);
            scene.add(directionalLight);
            
            animate();
            console.log('Animation started');
        }
        
        function initializeControls() {
            // Clear existing controls
            if (controls) {
                controls.dispose();
            }
            
            const mode = document.getElementById('control-mode').value;
            controlMode = mode;
            
            try {
                switch (mode) {
                    case 'trackball':
                        if (typeof THREE.TrackballControls !== 'undefined') {
                            controls = new THREE.TrackballControls(camera, renderer.domElement);
                            controls.rotateSpeed = 2.0;
                            controls.zoomSpeed = 1.2;
                            controls.panSpeed = 1.0;
                            controls.noZoom = false;
                            controls.noPan = false;
                            controls.staticMoving = true;
                            controls.dynamicDampingFactor = 0.3;
                            console.log('TrackballControls initialized');
                        } else {
                            throw new Error('TrackballControls not available');
                        }
                        break;
                        
                    case 'fly':
                        if (typeof THREE.FlyControls !== 'undefined') {
                            controls = new THREE.FlyControls(camera, renderer.domElement);
                            controls.movementSpeed = 0.5;
                            controls.rollSpeed = Math.PI / 12;
                            controls.autoForward = false;
                            controls.dragToLook = true;
                            console.log('FlyControls initialized');
                        } else {
                            throw new Error('FlyControls not available');
                        }
                        break;
                        
                    case 'manual':
                        // Manual mode does not use Three.js controls
                        controls = null;
                        console.log('Manual controls mode');
                        break;
                        
                    case 'orbit':
                    default:
                        controls = new THREE.OrbitControls(camera, renderer.domElement);
                        controls.enableDamping = true;
                        controls.dampingFactor = 0.1;
                        controls.screenSpacePanning = true;  // Enable screen space panning
                        controls.minDistance = 0.1;
                        controls.maxDistance = 5.0;
                        controls.enableKeys = true;  // Enable keyboard controls
                        controls.keys = {
                            LEFT: 37, // arrow keys
                            UP: 38,
                            RIGHT: 39,
                            BOTTOM: 40
                        };
                        // Enhanced panning functionality
                        controls.mouseButtons = {
                            LEFT: THREE.MOUSE.ROTATE,
                            MIDDLE: THREE.MOUSE.DOLLY,
                            RIGHT: THREE.MOUSE.PAN
                        };
                        controls.touches = {
                            ONE: THREE.TOUCH.ROTATE,
                            TWO: THREE.TOUCH.DOLLY_PAN
                        };
                        console.log('Enhanced OrbitControls initialized');
                        break;
                }
                
                if (controls && controls.update) {
                    controls.update();
                }
                
            } catch (e) {
                console.error(`Failed to initialize ${mode} controls:`, e);
                // Fallback to basic OrbitControls
                controls = new THREE.OrbitControls(camera, renderer.domElement);
                controls.enableDamping = true;
                controls.dampingFactor = 0.1;
                controls.update();
            }
        }
        
        function animate() {
            requestAnimationFrame(animate);
            
            // Update based on control mode
            if (controls) {
                if (controlMode === 'fly') {
                    // FlyControls needs delta time
                    const delta = clock.getDelta();
                    controls.update(delta);
                } else if (controls.update) {
                    controls.update();
                }
            }
            
            // If manual mode, update camera position from sliders
            if (controlMode === 'manual') {
                updateCameraFromSliders();
            }
            
            renderer.render(scene, camera);
        }
        
        function switchDataset(datasetName) {
            console.log('Switching to dataset:', datasetName);
            document.getElementById('status').textContent = `Switching to ${datasetName}...`;
            
            fetch('/api/switch_dataset', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ dataset: datasetName })
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    currentDataset = datasetName;
                    console.log(`Switched to dataset: ${datasetName}`);
                    
                    // Update sequence info
                    updateSequenceInfo();
                    
                    // Reset to first frame
                    currentFrame = 0;
                    currentSequence = 0;
                    document.getElementById('frame-slider').value = 0;
                    document.getElementById('sequence-slider').value = 0;
                    updateSliderLabels();
                    
                    // Reload initial parameters for the new dataset
                    console.log('Reloading initial parameters for new dataset...');
                    loadInitialParameters();
                    
                    // Load first frame of new dataset
                    loadFrameData(0);
                    
                    document.getElementById('status').textContent = `Switched to ${datasetName}`;
                } else {
                    console.error('Error switching dataset:', data.error);
                    document.getElementById('status').textContent = `Error: ${data.error}`;
                }
            })
            .catch(error => {
                console.error('Error switching dataset:', error);
                document.getElementById('status').textContent = `Error: ${error.message}`;
            });
        }
        
        function switchHandType(handType) {
            console.log('Switching to hand type:', handType);
            document.getElementById('status').textContent = `Switching to ${handType}...`;
            
            fetch('/api/switch_hand_type', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ hand_type: handType })
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    currentHandType = handType;
                    console.log(`Switched to hand type: ${handType}`);
                    
                    // Reload initial parameters for the new hand type
                    console.log('Reloading initial parameters for new hand type...');
                    loadInitialParameters();
                    
                    // Reload current frame with new hand type
                    loadFrameData(currentFrame);
                    
                    document.getElementById('status').textContent = `Switched to ${handType}`;
                } else {
                    console.error('Error switching hand type:', data.error);
                    document.getElementById('status').textContent = `Error: ${data.error}`;
                }
            })
            .catch(error => {
                console.error('Error switching hand type:', error);
                document.getElementById('status').textContent = `Error: ${error.message}`;
            });
        }
        
        function switchSequence(sequenceIdx) {
            console.log('Switching to sequence:', sequenceIdx);
            document.getElementById('status').textContent = `Switching to sequence ${sequenceIdx}...`;
            
            fetch('/api/switch_sequence', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ sequence: sequenceIdx })
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    currentSequence = sequenceIdx;
                    console.log(`Switched to sequence: ${sequenceIdx}`);
                    
                    // Update frame count for new sequence
                    updateFrameCount();
                    
                    // Reset to first frame
                    currentFrame = 0;
                    document.getElementById('frame-slider').value = 0;
                    updateSliderLabels();
                    
                    // Load first frame of new sequence
                    loadFrameData(0);
                    
                    document.getElementById('status').textContent = `Switched to sequence ${sequenceIdx}`;
                } else {
                    console.error('Error switching sequence:', data.error);
                    document.getElementById('status').textContent = `Error: ${data.error}`;
                }
            })
            .catch(error => {
                console.error('Error switching sequence:', error);
                document.getElementById('status').textContent = `Error: ${error.message}`;
            });
        }
        
        function updateSequenceInfo() {
            fetch('/api/sequences')
                .then(response => response.json())
                .then(data => {
                    maxSequences = data.count - 1;
                    currentSequence = data.current;
                    
                    document.getElementById('sequence-slider').max = maxSequences;
                    document.getElementById('sequence-slider').value = currentSequence;
                    document.getElementById('sequence-info').textContent = `(${data.count} sequences)`;
                    
                    updateSliderLabels();
                    updateFrameCount();
                })
                .catch(error => {
                    console.error('Error updating sequence info:', error);
                });
        }
        
        function updateCameraFromSliders() {
            const x = parseFloat(document.getElementById('cam-x').value);
            const y = parseFloat(document.getElementById('cam-y').value);
            const z = parseFloat(document.getElementById('cam-z').value);
            camera.position.set(x, y, z);
            camera.lookAt(0, 0, 0);
        }
        
        function updateFrameCount() {
            fetch('/api/frame_count')
                .then(response => response.json())
                .then(data => {
                    maxFrames = data.count - 1;
                    document.getElementById('frame-slider').max = maxFrames;
                    updateSliderLabels();
                })
                .catch(error => {
                    console.error('Error updating frame count:', error);
                });
        }
        
        function resetCamera() {
            camera.position.set(0.5, 0.5, 0.5);
            camera.lookAt(0, 0, 0);
            
            // Update slider values
            document.getElementById('cam-x').value = 0.5;
            document.getElementById('cam-y').value = 0.5;
            document.getElementById('cam-z').value = 0.5;
            updateCameraSliderLabels();
            
            if (controls && controls.reset) {
                controls.reset();
            } else if (controls && controls.update) {
                controls.update();
            }
        }
        
        function fitToScene() {
            // Calculate scene bounding box
            const box = new THREE.Box3();
            scene.traverse(function (object) {
                if (object.isMesh) {
                    box.expandByObject(object);
                }
            });
            
            if (!box.isEmpty()) {
                const center = box.getCenter(new THREE.Vector3());
                const size = box.getSize(new THREE.Vector3());
                const maxDim = Math.max(size.x, size.y, size.z);
                const distance = maxDim * 2;
                
                camera.position.copy(center);
                camera.position.z += distance;
                camera.lookAt(center);
                
                if (controls && controls.target) {
                    controls.target.copy(center);
                    controls.update();
                }
            }
        }
        
        // Ensure re-rendering when controls update
        if (controls) {
            controls.addEventListener('change', function() {
                renderer.render(scene, camera);
            });
        }
        
        // Add window resize event handler
        window.addEventListener('resize', function() {
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
            renderer.render(scene, camera);
        });
        
        function clearScene() {
            console.log('Clearing scene...');
            while(scene.children.length > 0) {
                const child = scene.children[0];
                scene.remove(child);
                if (child.geometry) child.geometry.dispose();
                if (child.material) child.material.dispose();
            }
            // Re-add lighting
            const ambientLight = new THREE.AmbientLight(0x404040, 0.8);
            scene.add(ambientLight);
            const directionalLight = new THREE.DirectionalLight(0xffffff, 1.0);
            directionalLight.position.set(1, 1, 1);
            scene.add(directionalLight);
            // Clear hand mesh cache
            currentHandMeshes = { 'right': [], 'left': [] };
            console.log('Scene cleared and lights restored');
        }
        
        function clearHandsOnly() {
            console.log('Clearing hands only...');
            // Only remove hand meshes, keep point clouds and lighting
            const childrenToRemove = [];
            scene.children.forEach(child => {
                // Remove all objects except point clouds (Points) and lighting
                if (!(child instanceof THREE.Points || child instanceof THREE.AmbientLight || child instanceof THREE.DirectionalLight)) {
                    childrenToRemove.push(child);
                }
            });
            
            childrenToRemove.forEach(child => {
                scene.remove(child);
                if (child.geometry) child.geometry.dispose();
                if (child.material) child.material.dispose();
            });
            
            // Clear hand mesh cache
            currentHandMeshes = { 'right': [], 'left': [] };
            console.log(`Removed ${childrenToRemove.length} hand objects`);
        }
        
        function cacheHandMeshTemplates(handData) {
            console.log('Caching hand mesh templates...');
            ['right', 'left'].forEach(hand => {
                if (handData[hand]) {
                    handMeshTemplates[hand] = {
                        link_meshes: handData[hand].link_meshes || [],
                        other_mesh: handData[hand].other_mesh || null
                    };
                    console.log(`Cached ${hand} hand: ${handMeshTemplates[hand].link_meshes.length} links`);
                }
            });
        }
        
        function createMeshFromCachedData(hand, linkData, transform) {
            if (!linkData.vertices || linkData.vertices.length === 0) {
                return null;
            }
            
            console.log(`Creating mesh for ${hand} hand, link ${linkData.link_idx || 'unknown'}`);
            console.log(`  - Vertices: ${linkData.vertices.length}`);
            console.log(`  - Faces: ${linkData.faces ? linkData.faces.length : 0}`);
            console.log(`  - Transform: ${transform ? 'present' : 'missing'}`);
            
            const geometry = new THREE.BufferGeometry();
            const positions = new Float32Array(linkData.vertices.flat());
            geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
            
            if (linkData.faces && linkData.faces.length > 0) {
                const indices = new Uint32Array(linkData.faces.flat());
                geometry.setIndex(new THREE.BufferAttribute(indices, 1));
            }
            
            if (linkData.colors && linkData.colors.length > 0) {
                const colors = new Float32Array(linkData.colors.flat());
                geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
            }
            
            geometry.computeVertexNormals();
            geometry.computeBoundingSphere();
            
            const material = new THREE.MeshLambertMaterial({
                vertexColors: linkData.colors && linkData.colors.length > 0,
                side: THREE.DoubleSide,
                color: hand === 'right' ? 0xff6666 : 0x6666ff
            });
            
            const mesh = new THREE.Mesh(geometry, material);
            
            // Debug: print mesh bounding box
            geometry.computeBoundingBox();
            console.log(`  - Bounding box: min(${geometry.boundingBox.min.x.toFixed(3)}, ${geometry.boundingBox.min.y.toFixed(3)}, ${geometry.boundingBox.min.z.toFixed(3)}) max(${geometry.boundingBox.max.x.toFixed(3)}, ${geometry.boundingBox.max.y.toFixed(3)}, ${geometry.boundingBox.max.z.toFixed(3)})`);
            console.log(`  - Material color for ${hand}: #${material.color.getHexString()}`);
            console.log(`  - Vertex colors enabled: ${material.vertexColors}`);
            
            // Set transform matrix instead of applying directly to geometry
            if (transform) {
                const matrix = new THREE.Matrix4();
                matrix.fromArray(transform);
                mesh.matrix.copy(matrix);
                mesh.matrixAutoUpdate = false;
                mesh.matrixWorldNeedsUpdate = true;
                console.log(`  - Transform matrix applied to mesh`);
                console.log(`  - Matrix elements (first 4): [${transform.slice(0,4).map(x => x.toFixed(3)).join(', ')}]`);
            } else {
                // If no transform, ensure identity matrix and disable auto update
                mesh.matrix.identity();
                mesh.matrixAutoUpdate = false;
                mesh.matrixWorldNeedsUpdate = true;
                console.log(`  - Using identity matrix`);
            }
            
            // Store link_idx information for future updates
            mesh.userData.link_idx = linkData.link_idx;
            mesh.userData.hand = hand;
            
            return mesh;
        }
        
        function addHandMeshesFromTemplate() {
            console.log('Adding hand meshes from cached templates...');
            console.log('Available templates:', Object.keys(handMeshTemplates));
            let meshesAdded = 0;
            
            ['right', 'left'].forEach(hand => {
                console.log(`Processing ${hand} hand...`);
                console.log(`  - Template exists: ${handMeshTemplates[hand] ? 'yes' : 'no'}`);
                if (handMeshTemplates[hand]) {
                    console.log(`  - Link meshes count: ${handMeshTemplates[hand].link_meshes.length}`);
                    console.log(`  - Other mesh exists: ${handMeshTemplates[hand].other_mesh ? 'yes' : 'no'}`);
                }
                
                if (handMeshTemplates[hand] && handMeshTemplates[hand].link_meshes.length > 0) {
                    console.log(`Adding ${hand} hand from template with ${handMeshTemplates[hand].link_meshes.length} links`);
                    
                    handMeshTemplates[hand].link_meshes.forEach((linkData, index) => {
                        console.log(`  Processing link ${index}, link_idx: ${linkData.link_idx}`);
                        const mesh = createMeshFromCachedData(hand, linkData, linkData.transform);
                        if (mesh) {
                            scene.add(mesh);
                            currentHandMeshes[hand].push(mesh);
                            meshesAdded++;
                            console.log(`Added ${hand} hand link ${linkData.link_idx} - Material color: ${mesh.material.color.getHexString()}`);
                        } else {
                            console.warn(`Failed to create mesh for ${hand} hand link ${linkData.link_idx}`);
                        }
                    });
                    
                    // Add other mesh (target points)
                    if (handMeshTemplates[hand].other_mesh) {
                        const otherMesh = createMeshFromCachedData(hand, handMeshTemplates[hand].other_mesh, null);
                        if (otherMesh) {
                            otherMesh.material.color.setHex(0x00ff00); // Green for target points
                            scene.add(otherMesh);
                            currentHandMeshes[hand].push(otherMesh);
                            meshesAdded++;
                            console.log(`Added ${hand} hand other mesh`);
                        }
                    }
                } else {
                    console.warn(`No template data available for ${hand} hand`);
                }
            });
            
            console.log(`Total meshes added: ${meshesAdded}`);
            console.log('Current scene children count:', scene.children.length);
            return meshesAdded;
        }
        
        function updateHandMeshesWithTransforms(transformData) {
            console.log('Updating hand meshes with new transforms...');
            
            let meshesUpdated = 0;
            
            ['right', 'left'].forEach(hand => {
                if (transformData[hand] && currentHandMeshes[hand] && currentHandMeshes[hand].length > 0) {
                    const handData = transformData[hand];
                    const transforms = handData.transforms || handData; // Handle both old and new format
                    const visible = handData.visible !== undefined ? handData.visible : true;
                    
                    console.log(`Updating ${hand} hand with ${transforms.length} transforms, visible=${visible}`);
                    
                    // Create transform mapping
                    const transformMap = {};
                    transforms.forEach(t => {
                        transformMap[t.link_idx] = t.transform;
                    });
                    
                    // Update transform and visibility of existing meshes
                    currentHandMeshes[hand].forEach((mesh) => {
                        // Update visibility for all meshes of this hand
                        mesh.visible = visible;
                        
                        if (mesh.userData && mesh.userData.link_idx !== undefined) {
                            const linkIdx = mesh.userData.link_idx;
                            if (transformMap[linkIdx]) {
                                // Apply new transform
                                const newMatrix = new THREE.Matrix4();
                                newMatrix.fromArray(transformMap[linkIdx]);
                                mesh.matrix.copy(newMatrix);
                                mesh.matrixAutoUpdate = false;
                                mesh.matrixWorldNeedsUpdate = true;  // Force update world matrix
                                
                                // If object has parent, also need to update parent's world matrix
                                if (mesh.parent) {
                                    mesh.parent.updateMatrixWorld(true);
                                }
                                
                                meshesUpdated++;
                                console.log(`Updated ${hand} hand link ${linkIdx} transform, visible=${visible}`);
                            }
                        }
                    });
                }
            });
            
            // Force scene update
            scene.updateMatrixWorld(true);
            
            console.log(`Total meshes updated: ${meshesUpdated}`);
            return meshesUpdated;
        }
        
        function loadFrameData(frameIdx) {
            console.log(`Loading frame ${frameIdx}...`);
            document.getElementById('status').textContent = `Loading frame ${frameIdx}...`;
            
            fetch(`/api/frame/${frameIdx}`)
                .then(response => {
                    console.log('Response received:', response.status);
                    if (!response.ok) {
                        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                    }
                    return response.json();
                })
                .then(data => {
                    console.log('Frame data received:', data);
                    clearScene();
                    
                    let objectsAdded = 0;
                    
                    // Add point cloud
                    if (data.pointcloud && data.pointcloud.points && data.pointcloud.points.length > 0) {
                        console.log(`Adding pointcloud with ${data.pointcloud.points.length} points`);
                        const geometry = new THREE.BufferGeometry();
                        const positions = new Float32Array(data.pointcloud.points.flat());
                        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
                        
                        if (data.pointcloud.colors && data.pointcloud.colors.length > 0) {
                            const colors = new Float32Array(data.pointcloud.colors.flat());
                            geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
                        }
                        
                        const material = new THREE.PointsMaterial({
                            size: 0.01,
                            vertexColors: data.pointcloud.colors && data.pointcloud.colors.length > 0
                        });
                        
                        const points = new THREE.Points(geometry, material);
                        scene.add(points);
                        currentPointcloud = points;
                        objectsAdded++;
                        console.log('Pointcloud added to scene');
                    } else {
                        console.log('No pointcloud data available');
                        currentPointcloud = null;
                    }
                    
                    // Cache hand mesh template data
                    if (data.hands) {
                        cacheHandMeshTemplates(data.hands);
                        
                        // Add hand meshes
                        const handMeshesAdded = addHandMeshesFromTemplate();
                        objectsAdded += handMeshesAdded;
                    }
                    
                    if (objectsAdded === 0) {
                        console.warn('No objects were added to the scene');
                        // Add a test object to confirm scene is working properly
                        const testGeometry = new THREE.SphereGeometry(0.05);
                        const testMaterial = new THREE.MeshBasicMaterial({ color: 0xff0000 });
                        const testSphere = new THREE.Mesh(testGeometry, testMaterial);
                        testSphere.position.set(0, 0, 0);
                        scene.add(testSphere);
                        console.log('Added test sphere since no data was available');
                    }
                    
                    document.getElementById('status').textContent = `Frame ${frameIdx} loaded (${objectsAdded} objects)`;
                    
                    // Debug: List all scene objects
                    console.log('Scene objects after loading:');
                    scene.traverse((object) => {
                        if (object.isMesh) {
                            const hand = object.userData.hand || 'unknown';
                            const linkIdx = object.userData.link_idx !== undefined ? object.userData.link_idx : 'unknown';
                            const color = object.material.color ? `#${object.material.color.getHexString()}` : 'unknown';
                            console.log(`  - Mesh: ${hand} hand, link ${linkIdx}, color: ${color}, visible: ${object.visible}`);
                        }
                    });
                })
                .catch(error => {
                    console.error('Error loading frame:', error);
                    document.getElementById('status').textContent = `Error: ${error.message}`;
                });
        }
        
        function updateHandTransformsOnly(frameIdx) {
            console.log(`Updating hand transforms only for frame ${frameIdx}...`);
            document.getElementById('status').textContent = 'Updating hand transforms...';
            
            fetch(`/api/transforms_only/${frameIdx}`)
                .then(response => {
                    if (!response.ok) {
                        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                    }
                    return response.json();
                })
                .then(data => {
                    console.log('Transform data received:', data);
                    console.log('Current hand meshes:', currentHandMeshes);
                    
                    if (data.transforms) {
                        // Additional debug information
                        Object.keys(data.transforms).forEach(hand => {
                            console.log(`${hand} hand transforms:`, data.transforms[hand]);
                            if (currentHandMeshes[hand]) {
                                console.log(`${hand} hand current meshes count:`, currentHandMeshes[hand].length);
                            }
                        });
                        
                        const meshesUpdated = updateHandMeshesWithTransforms(data.transforms);
                        document.getElementById('status').textContent = `Hand transforms updated (${meshesUpdated} meshes)`;
                    } else {
                        document.getElementById('status').textContent = 'No transform data received';
                    }
                })
                .catch(error => {
                    console.error('Error updating hand transforms:', error);
                    document.getElementById('status').textContent = `Error: ${error.message}`;
                });
        }
        
        function updateParameters() {
            if (isUpdatingParams) {
                console.log('Parameter update already in progress, skipping...');
                return;
            }
            
            isUpdatingParams = true;
            document.getElementById('status').textContent = 'Updating parameters...';
            
            const params = {
                mano_scale: parseFloat(document.getElementById('mano-scale').value),
                right_base_xyz: [
                    parseFloat(document.getElementById('right-base-x').value),
                    parseFloat(document.getElementById('right-base-y').value),
                    parseFloat(document.getElementById('right-base-z').value)
                ],
                right_base_rpy: [
                    parseFloat(document.getElementById('right-base-roll').value),
                    parseFloat(document.getElementById('right-base-pitch').value),
                    parseFloat(document.getElementById('right-base-yaw').value)
                ],
                left_base_xyz: [
                    parseFloat(document.getElementById('left-base-x').value),
                    parseFloat(document.getElementById('left-base-y').value),
                    parseFloat(document.getElementById('left-base-z').value)
                ],
                left_base_rpy: [
                    parseFloat(document.getElementById('left-base-roll').value),
                    parseFloat(document.getElementById('left-base-pitch').value),
                    parseFloat(document.getElementById('left-base-yaw').value)
                ],
                max_iterations: parseInt(document.getElementById('max-iter').value),
                mimic_iterations: parseInt(document.getElementById('mimic-iter').value),
                mimic_step: parseInt(document.getElementById('mimic-step').value),
                residual_threshold: parseFloat(document.getElementById('res-thresh').value),
                ik_damping: parseFloat(document.getElementById('ik-damping').value)
            };
            
            fetch('/api/update_params', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(params)
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    // After parameter update, only update hand transforms
                    updateHandTransformsOnly(currentFrame);
                } else {
                    document.getElementById('status').textContent = `Error: ${data.error}`;
                }
                isUpdatingParams = false;
            })
            .catch(error => {
                console.error('Error updating parameters:', error);
                document.getElementById('status').textContent = `Error: ${error.message}`;
                isUpdatingParams = false;
            });
        }
        
        function resetParameters() {
            document.getElementById('mano-scale').value = 1.0;
            // Right hand
            document.getElementById('right-base-x').value = 0.0;
            document.getElementById('right-base-y').value = 0.0;
            document.getElementById('right-base-z').value = 0.0;
            document.getElementById('right-base-roll').value = 0.0;
            document.getElementById('right-base-pitch').value = 0.0;
            document.getElementById('right-base-yaw').value = 0.0;
            // Left hand
            document.getElementById('left-base-x').value = 0.0;
            document.getElementById('left-base-y').value = 0.0;
            document.getElementById('left-base-z').value = 0.0;
            document.getElementById('left-base-roll').value = 0.0;
            document.getElementById('left-base-pitch').value = 0.0;
            document.getElementById('left-base-yaw').value = 0.0;
            // Common
            document.getElementById('max-iter').value = 1000;
            document.getElementById('mimic-iter').value = 50;
            document.getElementById('mimic-step').value = 20;
            document.getElementById('res-thresh').value = 1e-3;
            document.getElementById('ik-damping').value = 0.1;
            updateSliderLabels();
            updateParameters();
        }
        
        function loadInitialParameters() {
            console.log('Loading initial parameters from server...');
            fetch('/api/initial_params')
                .then(response => response.json())
                .then(data => {
                    console.log('Initial parameters received:', data);
                    
                    // Handle right hand parameters
                    if (data.right) {
                        if (data.right.base_xyz && data.right.base_xyz.length === 3) {
                            document.getElementById('right-base-x').value = data.right.base_xyz[0];
                            document.getElementById('right-base-y').value = data.right.base_xyz[1];
                            document.getElementById('right-base-z').value = data.right.base_xyz[2];
                        }
                        if (data.right.base_rpy && data.right.base_rpy.length === 3) {
                            document.getElementById('right-base-roll').value = data.right.base_rpy[0];
                            document.getElementById('right-base-pitch').value = data.right.base_rpy[1];
                            document.getElementById('right-base-yaw').value = data.right.base_rpy[2];
                        }
                        if (data.right.mano_scale) {
                            document.getElementById('mano-scale').value = data.right.mano_scale;
                        }
                        if (data.right.max_iterations) {
                            document.getElementById('max-iter').value = data.right.max_iterations;
                        }
                        if (data.right.mimic_iterations) {
                            document.getElementById('mimic-iter').value = data.right.mimic_iterations;
                        }
                        if (data.right.mimic_step) {
                            document.getElementById('mimic-step').value = data.right.mimic_step;
                        }
                        if (data.right.residual_threshold) {
                            document.getElementById('res-thresh').value = data.right.residual_threshold;
                        }
                        if (data.right.ik_damping) {
                            document.getElementById('ik-damping').value = data.right.ik_damping;
                        }
                    }
                    
                    // Handle left hand parameters
                    if (data.left) {
                        if (data.left.base_xyz && data.left.base_xyz.length === 3) {
                            document.getElementById('left-base-x').value = data.left.base_xyz[0];
                            document.getElementById('left-base-y').value = data.left.base_xyz[1];
                            document.getElementById('left-base-z').value = data.left.base_xyz[2];
                        }
                        if (data.left.base_rpy && data.left.base_rpy.length === 3) {
                            document.getElementById('left-base-roll').value = data.left.base_rpy[0];
                            document.getElementById('left-base-pitch').value = data.left.base_rpy[1];
                            document.getElementById('left-base-yaw').value = data.left.base_rpy[2];
                        }
                    }
                    
                    updateSliderLabels();
                    console.log('Initial parameters loaded successfully');
                })
                .catch(error => {
                    console.error('Error loading initial parameters:', error);
                    // If loading fails, use default values
                    updateSliderLabels();
                });
        }

        function updateSliderLabels() {
            document.getElementById('frame-label').textContent = currentFrame;
            document.getElementById('sequence-label').textContent = currentSequence;
            document.getElementById('mano-scale-value').textContent = parseFloat(document.getElementById('mano-scale').value).toFixed(3);
            // Right hand
            document.getElementById('right-base-x-value').textContent = parseFloat(document.getElementById('right-base-x').value).toFixed(3);
            document.getElementById('right-base-y-value').textContent = parseFloat(document.getElementById('right-base-y').value).toFixed(3);
            document.getElementById('right-base-z-value').textContent = parseFloat(document.getElementById('right-base-z').value).toFixed(3);
            document.getElementById('right-base-roll-value').textContent = parseFloat(document.getElementById('right-base-roll').value).toFixed(3);
            document.getElementById('right-base-pitch-value').textContent = parseFloat(document.getElementById('right-base-pitch').value).toFixed(3);
            document.getElementById('right-base-yaw-value').textContent = parseFloat(document.getElementById('right-base-yaw').value).toFixed(3);
            // Left hand
            document.getElementById('left-base-x-value').textContent = parseFloat(document.getElementById('left-base-x').value).toFixed(3);
            document.getElementById('left-base-y-value').textContent = parseFloat(document.getElementById('left-base-y').value).toFixed(3);
            document.getElementById('left-base-z-value').textContent = parseFloat(document.getElementById('left-base-z').value).toFixed(3);
            document.getElementById('left-base-roll-value').textContent = parseFloat(document.getElementById('left-base-roll').value).toFixed(3);
            document.getElementById('left-base-pitch-value').textContent = parseFloat(document.getElementById('left-base-pitch').value).toFixed(3);
            document.getElementById('left-base-yaw-value').textContent = parseFloat(document.getElementById('left-base-yaw').value).toFixed(3);
            // Common
            document.getElementById('max-iter-value').textContent = parseInt(document.getElementById('max-iter').value);
            document.getElementById('mimic-iter-value').textContent = parseInt(document.getElementById('mimic-iter').value);
            document.getElementById('mimic-step-value').textContent = parseInt(document.getElementById('mimic-step').value);
            document.getElementById('res-thresh-value').textContent = parseFloat(document.getElementById('res-thresh').value).toExponential(1);
            document.getElementById('ik-damping-value').textContent = parseFloat(document.getElementById('ik-damping').value).toFixed(2);
            
            // Camera sliders
            updateCameraSliderLabels();
        }
        
        function updateCameraSliderLabels() {
            document.getElementById('cam-x-value').textContent = parseFloat(document.getElementById('cam-x').value).toFixed(3);
            document.getElementById('cam-y-value').textContent = parseFloat(document.getElementById('cam-y').value).toFixed(3);
            document.getElementById('cam-z-value').textContent = parseFloat(document.getElementById('cam-z').value).toFixed(3);
        }
        
        // Initialization
        window.onload = function() {
            console.log('Window loaded, starting initialization...');
            
            // Check if Three.js is properly loaded
            if (typeof THREE === 'undefined') {
                console.error('Three.js failed to load!');
                document.getElementById('status').textContent = 'Error: Three.js failed to load';
                return;
            }
            console.log('Three.js loaded successfully, version:', THREE.REVISION);
            
            initThreeJS();
            
            // First load initial parameters
            console.log('Loading initial parameters...');
            loadInitialParameters();
            
            // Get initial dataset and hand type info
            console.log('Fetching dataset info...');
            Promise.all([
                fetch('/api/datasets').then(r => r.json()),
                fetch('/api/hand_types').then(r => r.json())
            ])
            .then(([datasetData, handTypeData]) => {
                console.log('Dataset info:', datasetData);
                console.log('Hand type info:', handTypeData);
                
                // Set current selections
                currentDataset = datasetData.current;
                currentHandType = handTypeData.current;
                document.getElementById('dataset-selector').value = currentDataset;
                document.getElementById('hand-type-selector').value = currentHandType;
                
                // Get sequence and frame count
                return updateSequenceInfo();
            })
            .then(() => {
                document.getElementById('status').textContent = `Ready (${currentDataset}, ${currentHandType})`;
            })
            .catch(error => {
                console.error('Error fetching initial info:', error);
                document.getElementById('status').textContent = `Error: ${error.message}`;
            });
            
            // Set up event listeners
            document.getElementById('frame-slider').addEventListener('input', function() {
                currentFrame = parseInt(this.value);
                document.getElementById('frame-label').textContent = currentFrame;
                loadFrameData(currentFrame);
            });
            
            // Dataset and hand type selectors
            document.getElementById('dataset-selector').addEventListener('change', function() {
                switchDataset(this.value);
            });
            
            document.getElementById('hand-type-selector').addEventListener('change', function() {
                switchHandType(this.value);
            });
            
            document.getElementById('sequence-slider').addEventListener('input', function() {
                currentSequence = parseInt(this.value);
                document.getElementById('sequence-label').textContent = currentSequence;
                switchSequence(currentSequence);
            });
            
            // Control mode switching
            document.getElementById('control-mode').addEventListener('change', function() {
                console.log('Changing control mode to:', this.value);
                initializeControls();
            });
            
            // Camera position sliders
            ['cam-x', 'cam-y', 'cam-z'].forEach(id => {
                document.getElementById(id).addEventListener('input', function() {
                    updateCameraSliderLabels();
                    if (controlMode === 'manual') {
                        updateCameraFromSliders();
                    }
                });
            });
            
            // Parameter slider event listeners - add debouncing
            let paramUpdateTimeout;
            [
                'mano-scale', 
                'right-base-x', 'right-base-y', 'right-base-z', 'right-base-roll', 'right-base-pitch', 'right-base-yaw',
                'left-base-x', 'left-base-y', 'left-base-z', 'left-base-roll', 'left-base-pitch', 'left-base-yaw',
                'max-iter', 'mimic-iter', 'mimic-step', 'res-thresh', 'ik-damping'
            ].forEach(id => {
                document.getElementById(id).addEventListener('input', function() {
                    updateSliderLabels();
                    
                    // Clear previous timer
                    clearTimeout(paramUpdateTimeout);
                    
                    // Set new timer, execute update after 200ms
                    paramUpdateTimeout = setTimeout(() => {
                        updateParameters();
                    }, 200);
                });
            });
            
            updateSliderLabels();
            
            // Delayed loading of first frame to ensure Three.js is fully initialized
            setTimeout(() => {
                console.log('Loading initial frame...');
                loadFrameData(0);
            }, 1000);
        };
        
        // Window resize handling
        window.addEventListener('resize', function() {
            const viewerDiv = document.getElementById('viewer');
            camera.aspect = viewerDiv.clientWidth / viewerDiv.clientHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(viewerDiv.clientWidth, viewerDiv.clientHeight);
        });
        
        // Add keyboard shortcuts
        document.addEventListener('keydown', function(event) {
            const moveStep = 0.1;
            const rotateStep = 0.1;
            
            switch(event.code) {
                case 'KeyW': // Forward
                    camera.position.z -= moveStep;
                    break;
                case 'KeyS': // Backward
                    camera.position.z += moveStep;
                    break;
                case 'KeyA': // Left
                    camera.position.x -= moveStep;
                    break;
                case 'KeyD': // Right
                    camera.position.x += moveStep;
                    break;
                case 'KeyQ': // Up
                    camera.position.y += moveStep;
                    break;
                case 'KeyE': // Down
                    camera.position.y -= moveStep;
                    break;
                case 'KeyR': // Reset camera
                    resetCamera();
                    break;
                case 'KeyF': // Fit to scene
                    fitToScene();
                    break;
                case 'Digit1': // Switch to Orbit mode
                    document.getElementById('control-mode').value = 'orbit';
                    initializeControls();
                    break;
                case 'Digit2': // Switch to Trackball mode
                    document.getElementById('control-mode').value = 'trackball';
                    initializeControls();
                    break;
                case 'Digit3': // Switch to Fly mode
                    document.getElementById('control-mode').value = 'fly';
                    initializeControls();
                    break;
                case 'Digit4': // Switch to Manual mode
                    document.getElementById('control-mode').value = 'manual';
                    initializeControls();
                    break;
            }
            
            // Update camera slider values
            if (['KeyW', 'KeyS', 'KeyA', 'KeyD', 'KeyQ', 'KeyE'].includes(event.code)) {
                document.getElementById('cam-x').value = camera.position.x;
                document.getElementById('cam-y').value = camera.position.y;
                document.getElementById('cam-z').value = camera.position.z;
                updateCameraSliderLabels();
                event.preventDefault();
            }
        });
    </script>
</body>
</html>
"""

# Inject dynamic hand types into the template
hand_options_html = "\n".join([f'                    <option value="{hand}">{hand} Hand</option>' for hand in HAND_TYPES])
HTML_TEMPLATE = HTML_TEMPLATE.replace('<!-- HAND_OPTIONS_PLACEHOLDER -->', hand_options_html)

# Create template directory and files
@app.route('/templates/<path:filename>')
def send_template(filename):
    return HTML_TEMPLATE

# Set up template
from markupsafe import Markup
app.jinja_env.globals['render_template_string'] = lambda: Markup(HTML_TEMPLATE)

@app.template_global()
def render_index():
    return Markup(HTML_TEMPLATE)

# Override index route to return HTML directly
@app.route('/')
def index():
    return Markup(HTML_TEMPLATE)

@app.route('/api/initial_params')
def get_initial_params():
    """API endpoint to get initial parameters for the visualizer."""
    print(f"Getting initial parameters for {data_manager['current_dataset']} dataset, {data_manager['current_hand_type']} hand type")
    result = {}
    
    if data_manager['right_hand']:
        right_params = data_manager['right_hand'].get_initial_parameters()
        result['right'] = {
            'base_xyz': right_params['base_xyz'],
            'base_rpy': right_params['base_rpy'],
            'mano_scale': right_params['mano_scale'],
            'max_iterations': right_params['max_iterations'],
            'residual_threshold': right_params['residual_threshold'],
            'ik_damping': right_params['ik_damping']
        }
        print(f"Right hand parameters: base_xyz={right_params['base_xyz']}, base_rpy={right_params['base_rpy']}")
    
    if data_manager['left_hand']:
        left_params = data_manager['left_hand'].get_initial_parameters()
        result['left'] = {
            'base_xyz': left_params['base_xyz'],
            'base_rpy': left_params['base_rpy'],
            'mano_scale': left_params['mano_scale'],
            'max_iterations': left_params['max_iterations'],
            'residual_threshold': left_params['residual_threshold'],
            'ik_damping': left_params['ik_damping']
        }
        print(f"Left hand parameters: base_xyz={left_params['base_xyz']}, base_rpy={left_params['base_rpy']}")
    
    if not result:
        return jsonify({'error': 'No hand visualizers initialized'})
    
    return jsonify(result)

@app.route('/favicon.ico')
def favicon():
    """Handle favicon requests to avoid 404 errors"""
    return '', 204  # Return empty response with "No Content" status

if __name__ == '__main__':
    print("Starting Hand Visualizer...")
    print("Initializing data...")
    
    try:
        # Initialize data when the app starts
        initialize_data()
        print("Data initialization completed successfully")
        
        print("Starting Flask server...")
        print("Open your browser and navigate to: http://localhost:5000")
        print("Available datasets:", data_manager['available_datasets'])
        print("Available hand types:", data_manager['available_hand_types'])
        print("Current dataset:", data_manager['current_dataset'])
        print("Current hand type:", data_manager['current_hand_type'])
        
        # Start the Flask development server
        app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
        
    except Exception as e:
        print(f"Error starting application: {e}")
        import traceback
        traceback.print_exc()
    
    print("Hand Visualizer shutting down...")

if __name__ == '__main__':
    initialize_data()
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
