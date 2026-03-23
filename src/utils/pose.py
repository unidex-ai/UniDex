import numpy as np
import torch

def normalize(vec, eps=1e-12):
    if isinstance(vec, torch.Tensor):
        norm = vec.norm(dim=-1, keepdim=True)
        norm = torch.maximum(norm, torch.tensor(eps, device=vec.device))
        out = vec / norm
    else:
        norm = np.linalg.norm(vec, axis=-1)
        norm = np.maximum(norm, eps)
        out = (vec.T / norm).T
    return out

def rot6d_to_mat(d6):
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = normalize(a1)
    if isinstance(d6, torch.Tensor):
        b2 = a2 - torch.sum(b1 * a2, dim=-1, keepdim=True) * b1
        b2 = normalize(b2)
        b3 = torch.cross(b1, b2, dim=-1)
        out = torch.stack((b1, b2, b3), dim=-2)
    else:
        b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
        b2 = normalize(b2)
        b3 = np.cross(b1, b2, axis=-1)
        out = np.stack((b1, b2, b3), axis=-2)
    return out

def mat_to_rot6d(mat):
    if isinstance(mat, torch.Tensor):
        batch_dim = mat.shape[:-2]
        out = mat[..., :2, :].reshape(batch_dim + (6,))
    else:
        batch_dim = mat.shape[:-2]
        out = mat[..., :2, :].copy().reshape(batch_dim + (6,))
    return out

def mat_to_pose9d(mat):
    pos = mat[...,:3,3]
    rotmat = mat[...,:3,:3]
    d6 = mat_to_rot6d(rotmat)
    if isinstance(mat, torch.Tensor):
        d9 = torch.cat([pos, d6], dim=-1)
    else:
        d9 = np.concatenate([pos, d6], axis=-1)
    return d9

def pose9d_to_mat(d9):
    pos = d9[...,:3]
    d6 = d9[...,3:]
    rotmat = rot6d_to_mat(d6)
    if isinstance(d9, torch.Tensor):
        out = torch.zeros(d9.shape[:-1] + (4, 4), dtype=d9.dtype, device=d9.device)
    else:
        out = np.zeros(d9.shape[:-1]+(4,4), dtype=d9.dtype)
    out[...,:3,:3] = rotmat
    out[...,:3,3] = pos
    out[...,3,3] = 1
    return out

def pose7d_to_mat(pose_7d):
    """Convert 7D pose (position + quaternion) to 4x4 transformation matrix
    
    Args:
        pose_7d: Array of shape (..., 7) containing [x, y, z, qx, qy, qz, qw]
        
    Returns:
        Transformation matrix of shape (..., 4, 4)
    """
    pos = pose_7d[..., :3]
    quat = pose_7d[..., 3:]  # [x, y, z, w] quaternion
    
    if isinstance(pose_7d, torch.Tensor):
        # Normalize quaternion
        quat = quat / (torch.norm(quat, dim=-1, keepdim=True) + 1e-8)
        
        # Convert quaternion to rotation matrix
        x, y, z, w = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
        rot_matrix = torch.stack([
            torch.stack([1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w], dim=-1),
            torch.stack([2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w], dim=-1),
            torch.stack([2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y], dim=-1)
        ], dim=-2)
        
        # Create 4x4 transformation matrix
        transform = torch.eye(4, dtype=pose_7d.dtype, device=pose_7d.device).expand(pose_7d.shape[:-1] + (4, 4)).clone()
        transform[..., :3, :3] = rot_matrix
        transform[..., :3, 3] = pos
    else:
        # Normalize quaternion
        quat = quat / (np.linalg.norm(quat, axis=-1, keepdims=True) + 1e-8)
        
        # Convert quaternion to rotation matrix
        x, y, z, w = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
        rot_matrix = np.stack([
            np.stack([1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w], axis=-1),
            np.stack([2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w], axis=-1),
            np.stack([2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y], axis=-1)
        ], axis=-2)
        
        # Create 4x4 transformation matrix
        transform = np.eye(4, dtype=pose_7d.dtype)
        if pose_7d.ndim > 1:
            transform = np.broadcast_to(transform, pose_7d.shape[:-1] + (4, 4)).copy()
        transform[..., :3, :3] = rot_matrix
        transform[..., :3, 3] = pos
    
    return transform