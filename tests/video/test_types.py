# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for kimodo.video motion type dataclasses."""
from __future__ import annotations

import pytest
import torch


def test_soma_motion30_holds_required_fields():
    from kimodo.video.types import SOMAMotion30
    m = SOMAMotion30(
        local_rot_mats=torch.zeros(10, 30, 3, 3),
        root_positions=torch.zeros(10, 3),
        fps=20,
    )
    assert m.local_rot_mats.shape == (10, 30, 3, 3)
    assert m.root_positions.shape == (10, 3)
    assert m.fps == 20


def test_soma_motion30_num_frames_property():
    from kimodo.video.types import SOMAMotion30
    m = SOMAMotion30(
        local_rot_mats=torch.zeros(7, 30, 3, 3),
        root_positions=torch.zeros(7, 3),
        fps=20,
    )
    assert m.num_frames == 7


def test_smplx_motion_holds_required_fields():
    from kimodo.video._types import SMPLXMotion
    m = SMPLXMotion(
        body_pose=torch.zeros(10, 21, 3, 3),
        global_orient=torch.zeros(10, 3, 3),
        transl=torch.zeros(10, 3),
        fps=30,
    )
    assert m.body_pose.shape == (10, 21, 3, 3)
    assert m.fps == 30


def test_smplx_motion_is_not_in_public_namespace():
    """SMPLXMotion is intentionally private — must not be importable from kimodo.video."""
    import kimodo.video
    assert not hasattr(kimodo.video, "SMPLXMotion"), \
        "SMPLXMotion leaked into the public namespace"
