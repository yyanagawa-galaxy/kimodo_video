# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the generic positions->SOMA-30 IK fitter."""
from __future__ import annotations
import pytest
import torch

from kimodo.video._ik_fitter import procrustes_rotation, fit_soma30_from_positions
from kimodo.skeleton import SOMASkeleton30


def test_procrustes_identity_when_inputs_equal():
    rest = torch.tensor([[1., 0, 0], [0, 1, 0], [0, 0, 1]])
    R = procrustes_rotation(rest, rest)
    assert torch.allclose(R, torch.eye(3), atol=1e-6)


def test_procrustes_recovers_known_rotation():
    rest = torch.tensor([[1., 0, 0], [0, 1, 0], [0, 0, 1]])
    # 90 degrees about Y: rotates X -> -Z, Z -> X
    theta = torch.pi / 2
    Ry = torch.tensor([[torch.cos(torch.tensor(theta)), 0, torch.sin(torch.tensor(theta))],
                       [0, 1, 0],
                       [-torch.sin(torch.tensor(theta)), 0, torch.cos(torch.tensor(theta))]])
    current = (Ry @ rest.T).T
    R = procrustes_rotation(rest, current)
    assert torch.allclose(R, Ry, atol=1e-5)


def test_procrustes_single_vector_aligns_direction():
    rest = torch.tensor([[1., 0, 0]])
    current = torch.tensor([[0., 1, 0]])
    R = procrustes_rotation(rest, current)
    out = R @ rest[0]
    assert torch.allclose(out / out.norm(), current[0] / current[0].norm(), atol=1e-5)


def test_procrustes_antiparallel_single_vector_returns_180_rotation():
    rest = torch.tensor([[1., 0, 0]])
    current = torch.tensor([[-1., 0, 0]])
    R = procrustes_rotation(rest, current)
    out = R @ rest[0]
    assert torch.allclose(out, current[0], atol=1e-5)


def test_fit_tpose_returns_near_identity_rotations():
    """If we feed the rest-pose world positions back as input, the IK
    should recover near-identity local rotations (within 1e-3 tolerance)."""
    sk = SOMASkeleton30()
    # Use sk.fk() directly to get the canonical rest-world positions,
    # matching Kimodo's FK convention exactly.
    dtype = sk.neutral_joints.dtype
    local_rots_id = torch.eye(3, dtype=dtype).unsqueeze(0).unsqueeze(0).expand(1, 30, 3, 3)
    root_pos_zero = torch.zeros(1, 3, dtype=dtype)
    _, rest_world, _ = sk.fk(local_rots_id, root_pos_zero)
    # rest_world: [1, 30, 3] in float64

    local_rots, root_pos = fit_soma30_from_positions(rest_world, sk)
    expected_identity = torch.eye(3, dtype=local_rots.dtype).expand(1, 30, 3, 3)
    max_diff = (local_rots - expected_identity).abs().max().item()
    assert max_diff < 1e-3, f"max diff = {max_diff}"


def test_fit_output_shapes():
    sk = SOMASkeleton30()
    # Random reasonable positions (place hips at origin, others nearby)
    pos = torch.randn(5, 30, 3) * 0.1 + sk.neutral_joints.unsqueeze(0).float() / 100.0
    local_rots, root_pos = fit_soma30_from_positions(pos, sk)
    assert local_rots.shape == (5, 30, 3, 3)
    assert root_pos.shape == (5, 3)


def test_fit_single_frame_works():
    sk = SOMASkeleton30()
    pos = sk.neutral_joints.unsqueeze(0).float() / 100.0
    local_rots, root_pos = fit_soma30_from_positions(pos, sk)
    assert local_rots.shape == (1, 30, 3, 3)
