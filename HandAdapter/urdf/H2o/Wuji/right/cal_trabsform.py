#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute static transform between two links in a URDF via fixed joints.

- Only standard library + numpy.
- Parses <joint> origin xyz/rpy and composes along the path between links.
- If a non-fixed joint appears on the path, raises an error (by design for this task).

Usage:
    python urdf_link_tf.py path/to/robot.urdf link_from link_to
"""

import argparse
import math
import xml.etree.ElementTree as ET
from typing import Dict, Tuple, Optional
import numpy as np


def _parse_xyz(text: Optional[str]) -> np.ndarray:
    if not text:
        return np.zeros(3)
    parts = [p for p in text.strip().split()]
    if len(parts) != 3:
        raise ValueError(f"xyz should have 3 numbers, got: {text}")
    return np.array([float(parts[0]), float(parts[1]), float(parts[2])], dtype=float)


def _parse_rpy(text: Optional[str]) -> np.ndarray:
    if not text:
        return np.zeros(3)
    parts = [p for p in text.strip().split()]
    if len(parts) != 3:
        raise ValueError(f"rpy should have 3 numbers, got: {text}")
    return np.array([float(parts[0]), float(parts[1]), float(parts[2])], dtype=float)


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    URDF uses R = Rz(yaw) * Ry(pitch) * Rx(roll).
    """
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    Rx = np.array([[1, 0, 0],
                   [0, cr, -sr],
                   [0, sr,  cr]], dtype=float)
    Ry = np.array([[ cp, 0, sp],
                   [  0, 1,  0],
                   [-sp, 0, cp]], dtype=float)
    Rz = np.array([[cy, -sy, 0],
                   [sy,  cy, 0],
                   [ 0,   0, 1]], dtype=float)

    return Rz @ Ry @ Rx


def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3,  3] = t
    return T


def inv_T(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3,  3]
    Ti = np.eye(4, dtype=float)
    Rt = R.T
    Ti[:3, :3] = Rt
    Ti[:3,  3] = -Rt @ t
    return Ti


def matrix_to_rpy_xyz(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decompose T into (xyz, rpy) with the same Rz*Ry*Rx convention.
    """
    R = T[:3, :3]
    # Guard against numerical drift
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-8):
        # Orthogonalize lightly if needed
        U, _, Vt = np.linalg.svd(R)
        R = U @ Vt

    # For ZYX (yaw-pitch-roll) extraction:
    # pitch = arcsin(-R[2,0])
    pitch = math.asin(-max(-1.0, min(1.0, R[2, 0])))
    # Handle cos(pitch) close to zero (gimbal lock)
    if abs(math.cos(pitch)) > 1e-8:
        roll = math.atan2(R[2, 1], R[2, 2])
        yaw  = math.atan2(R[1, 0], R[0, 0])
    else:
        # Gimbal lock: roll and yaw coupled
        roll = math.atan2(-R[1, 2], R[1, 1])
        yaw  = 0.0

    xyz = T[:3, 3].copy()
    rpy = np.array([roll, pitch, yaw], dtype=float)
    return xyz, rpy


class URDFFixedGraph:
    """
    Build a parent map: child_link -> (parent_link, joint_type, T_parent_to_child)
    from a URDF file. Only uses joint origins; geometry/inertials are irrelevant here.
    """
    def __init__(self, urdf_path: str):
        self.urdf_path = urdf_path
        self.child_to_parent: Dict[str, Tuple[str, str, np.ndarray]] = {}
        self.links_present: Dict[str, bool] = {}
        self._parse()

    def _parse(self):
        tree = ET.parse(self.urdf_path)
        root = tree.getroot()

        # Record links
        for link in root.findall("link"):
            name = link.attrib.get("name")
            if name:
                self.links_present[name] = True

        # Record joints
        for joint in root.findall("joint"):
            jtype = joint.attrib.get("type", "").strip()
            parent_el = joint.find("parent")
            child_el  = joint.find("child")
            if parent_el is None or child_el is None:
                continue
            parent = parent_el.attrib.get("link")
            child  = child_el.attrib.get("link")
            if not parent or not child:
                continue

            origin_el = joint.find("origin")
            xyz = _parse_xyz(origin_el.attrib.get("xyz")) if origin_el is not None else np.zeros(3)
            rpy = _parse_rpy(origin_el.attrib.get("rpy")) if origin_el is not None else np.zeros(3)
            R = rpy_to_matrix(rpy[0], rpy[1], rpy[2])
            T_parent_to_child = make_T(R, xyz)

            # URDF enforces one parent per child; if duplicates occur, last wins
            self.child_to_parent[child] = (parent, jtype, T_parent_to_child)
            # Ensure link presence
            self.links_present[parent] = True
            self.links_present[child]  = True

    def _ascend_chain(self, link: str, require_fixed: bool = True) -> Dict[str, np.ndarray]:
        """
        Walk from 'link' up to root via parents, accumulating transforms.
        Returns a dict: ancestor_link -> T_ancestor_to_link
        """
        if link not in self.links_present:
            raise KeyError(f"Link '{link}' not found in URDF.")
        out = {link: np.eye(4, dtype=float)}  # include self for completeness

        cur = link
        T_cur_to_target = np.eye(4, dtype=float)  # T_cur_to_link
        while cur in self.child_to_parent:
            parent, jtype, T_parent_to_child = self.child_to_parent[cur]
            if require_fixed and jtype != "fixed":
                raise RuntimeError(
                    f"Non-fixed joint on path: '{parent}' -> '{cur}' via joint type '{jtype}'."
                )
            # T_parent_to_link = T_parent_to_child @ T_child_to_link
            T_parent_to_link = T_parent_to_child @ T_cur_to_target
            out[parent] = T_parent_to_link
            cur = parent
            T_cur_to_target = T_parent_to_link

        return out

    def transform_linkA_to_linkB(self, link_from: str, link_to: str, require_fixed: bool = True) -> np.ndarray:
        """
        Compute T_from_to = T(link_from -> link_to)
        """
        if link_from == link_to:
            return np.eye(4, dtype=float)

        anc_A = self._ascend_chain(link_from, require_fixed=require_fixed)
        # ascend from B until find common ancestor
        cur = link_to
        T_cur_to_B = np.eye(4, dtype=float)  # T_cur_to_link_to
        while True:
            if cur in anc_A:
                # cur is LCA
                T_cur_to_A = anc_A[cur]           # T_cur->A
                T_A_to_cur = inv_T(T_cur_to_A)    # invert
                T_cur_to_B_now = T_cur_to_B       # T_cur->B
                T_A_to_B = T_A_to_cur @ T_cur_to_B_now
                return T_A_to_B

            if cur not in self.child_to_parent:
                # hit root and no common ancestor -> disconnected graph
                raise RuntimeError(
                    f"No path found between '{link_from}' and '{link_to}'."
                )

            parent, jtype, T_parent_to_child = self.child_to_parent[cur]
            if require_fixed and jtype != "fixed":
                raise RuntimeError(
                    f"Non-fixed joint on path: '{parent}' -> '{cur}' via joint type '{jtype}'."
                )
            # climb: T_parent_to_B = T_parent_to_child @ T_child_to_B
            T_parent_to_B = T_parent_to_child @ T_cur_to_B
            cur = parent
            T_cur_to_B = T_parent_to_B


def main():
    ap = argparse.ArgumentParser(description="Compute transform between two URDF links via fixed joints.")
    ap.add_argument("urdf", type=str, help="Path to URDF file")
    ap.add_argument("link_from", type=str, help="Source link name (from)")
    ap.add_argument("link_to", type=str, help="Target link name (to)")
    ap.add_argument("--allow-nonfixed", action="store_true",
                    help="If set, do not enforce fixed joints on the path (NOT recommended; angles assumed zero).")
    args = ap.parse_args()

    graph = URDFFixedGraph(args.urdf)
    T = graph.transform_linkA_to_linkB(args.link_from, args.link_to, require_fixed=not args.allow_nonfixed)

    xyz, rpy = matrix_to_rpy_xyz(T)

    np.set_printoptions(precision=6, suppress=True)
    print(f"\nTransform T_{args.link_from}_to_{args.link_to} (4x4):")
    print(T)
    np.save(f"T_{args.link_from}_to_{args.link_to}.npy", T)
    print("\nAs xyz (m) and rpy (rad):")
    print("xyz:", xyz.tolist())
    print("rpy:", rpy.tolist())


if __name__ == "__main__":
    main()
    