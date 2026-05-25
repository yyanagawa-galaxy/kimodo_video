# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generic inverse-kinematics fitter: positions -> SOMA-30 local rotation matrices.

Input-agnostic. Whoever computes the 30 joint positions (MediaPipe adapter,
sam-body4d adapter, anything else) just hands them to this module."""

from __future__ import annotations
from typing import Dict, List, Optional
import torch
from kimodo.skeleton import SOMASkeleton30

CHILDREN_OF: Dict[int, List[int]] = {
    0:  [1, 22, 26],
    1:  [2],
    2:  [3],
    3:  [4, 10, 16],
    4:  [5],
    5:  [6],
    6:  [7, 8, 9],
    7:  [],
    8:  [],
    9:  [],
    10: [11],
    11: [12],
    12: [13],
    13: [14, 15],
    14: [],
    15: [],
    16: [17],
    17: [18],
    18: [19],
    19: [20, 21],
    20: [],
    21: [],
    22: [23],
    23: [24],
    24: [25],
    25: [],
    26: [27],
    27: [28],
    28: [29],
    29: [],
}


def _skew(v: torch.Tensor) -> torch.Tensor:
    """Return the 3x3 skew-symmetric matrix for vector v [3]."""
    x, y, z = v[0], v[1], v[2]
    zero = torch.zeros((), device=v.device, dtype=v.dtype)
    return torch.stack([
        zero,  -z,   y,
        z,    zero, -x,
        -y,    x,   zero,
    ]).reshape(3, 3)


def procrustes_rotation(rest: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
    """rest, current: [N, 3]. Returns R: [3, 3] such that R @ rest[i] ~= current[i] for each row.

    N == 0: identity.
    N == 1: swing-only Rodrigues (twist undefined).
    N >= 2: Kabsch / SVD with reflection guard.
    """
    device, dtype = rest.device, rest.dtype
    eye3 = torch.eye(3, device=device, dtype=dtype)

    N = rest.shape[0]

    if N == 0:
        return eye3

    if N == 1:
        a = rest[0]
        b = current[0]
        a_norm = a.norm()
        b_norm = b.norm()
        if a_norm < 1e-10 or b_norm < 1e-10:
            return eye3
        a = a / a_norm
        b = b / b_norm
        cos_th = torch.clamp(torch.dot(a, b), -1.0, 1.0)
        axis = torch.cross(a, b, dim=0)
        sin_th = axis.norm()
        if sin_th < 1e-8:
            if cos_th > 0.0:
                return eye3
            else:
                # 180 degrees: find a perpendicular axis
                perp = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)
                if abs(a[0].item()) > 0.9:
                    perp = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)
                axis_perp = torch.cross(a, perp, dim=0)
                axis_perp = axis_perp / axis_perp.norm()
                K = _skew(axis_perp)
                return eye3 + 2.0 * (K @ K)
        axis = axis / sin_th
        K = _skew(axis)
        return eye3 + sin_th * K + (1.0 - cos_th) * (K @ K)

    # N >= 2: Kabsch algorithm
    H = rest.T @ current  # [3, 3]
    U, S, Vt = torch.linalg.svd(H)
    d = torch.sign(torch.det(Vt.T @ U.T))
    D = torch.diag(torch.stack([
        torch.ones((), device=device, dtype=dtype),
        torch.ones((), device=device, dtype=dtype),
        d,
    ]))
    R = Vt.T @ D @ U.T
    return R


def fit_soma30_from_positions(
    posed_joints_world: torch.Tensor,
    skeleton: Optional[SOMASkeleton30] = None,
) -> tuple:
    """Run IK per frame. Returns (local_rot_mats [T, 30, 3, 3], root_positions [T, 3]).

    Args:
        posed_joints_world: [T, 30, 3] world positions of SOMA-30 joints per frame.
        skeleton: optional pre-built SOMASkeleton30; default constructs one.
    """
    if skeleton is None:
        skeleton = SOMASkeleton30()

    neutral_joints = skeleton.neutral_joints.to(
        device=posed_joints_world.device, dtype=posed_joints_world.dtype
    )  # [30, 3]
    rest_local_rots = getattr(skeleton, "rest_local_rots", None)
    if rest_local_rots is not None:
        rest_local_rots = rest_local_rots.to(
            device=posed_joints_world.device, dtype=posed_joints_world.dtype
        )

    T = posed_joints_world.shape[0]
    device, dtype = posed_joints_world.device, posed_joints_world.dtype

    local_rot_mats = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).unsqueeze(0).expand(T, 30, 3, 3).clone()
    global_rot_mats = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).unsqueeze(0).expand(T, 30, 3, 3).clone()
    root_positions = posed_joints_world[:, 0, :].clone()  # Hips position

    # BFS order: every joint j > 0 has parent[j] < j, so range(30) respects dependency.
    for t in range(T):
        for j in range(30):
            children = CHILDREN_OF[j]
            parent = int(skeleton.joint_parents[j])

            if len(children) == 0:
                # Leaf joint: local_rot stays identity; propagate global_rot
                if rest_local_rots is not None:
                    eff = rest_local_rots[j]
                else:
                    eff = torch.eye(3, device=device, dtype=dtype)
                if parent >= 0:
                    global_rot_mats[t, j] = global_rot_mats[t, parent] @ eff
                else:
                    global_rot_mats[t, j] = eff
                continue

            # Build child offset vectors in rest and posed space
            rest_offsets = torch.stack(
                [neutral_joints[c] - neutral_joints[j] for c in children]
            )  # [N, 3]
            current_offsets = torch.stack(
                [posed_joints_world[t, c] - posed_joints_world[t, j] for c in children]
            )  # [N, 3]

            if parent < 0:
                # Root: world IS the parent frame
                v_in_parent = current_offsets
            else:
                # Rotate from world into parent's local frame
                v_in_parent = (global_rot_mats[t, parent].T @ current_offsets.T).T  # [N, 3]

            # Solve Procrustes: effective_local_rot maps rest_offsets -> v_in_parent
            effective = procrustes_rotation(rest_offsets, v_in_parent)

            # Store local_rot (accounting for baked rest pose if present)
            if rest_local_rots is not None:
                local_rot_mats[t, j] = rest_local_rots[j].T @ effective
            else:
                local_rot_mats[t, j] = effective

            # Update global_rot for this joint (needed by descendants)
            if parent < 0:
                global_rot_mats[t, j] = effective
            else:
                global_rot_mats[t, j] = global_rot_mats[t, parent] @ effective

    return local_rot_mats, root_positions
