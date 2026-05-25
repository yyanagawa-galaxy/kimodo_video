# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for frame-rate resampling utility."""
from __future__ import annotations

import pytest
import torch

from kimodo.video.resample import resample_to_fps
from kimodo.video.types import SOMAMotion30


def _ramp(t: int, fps: int) -> SOMAMotion30:
    # Use a monotonically increasing root.x to make resampling visible.
    rot = torch.eye(3).expand(t, 30, 3, 3).clone()
    root = torch.zeros(t, 3)
    root[:, 0] = torch.linspace(0, 1, t)
    return SOMAMotion30(local_rot_mats=rot, root_positions=root, fps=fps)


def test_resample_same_fps_is_identity():
    m = _ramp(60, fps=30)
    out = resample_to_fps(m, target_fps=30)
    assert out.fps == 30
    assert out.num_frames == 60
    assert torch.allclose(out.root_positions, m.root_positions)


def test_resample_30_to_20_preserves_duration_within_one_frame():
    # 60 frames @ 30 fps = 2.0 s → at 20 fps that's 40 frames
    m = _ramp(60, fps=30)
    out = resample_to_fps(m, target_fps=20)
    assert out.fps == 20
    assert abs(out.num_frames - 40) <= 1


def test_resample_20_to_30_upsamples():
    m = _ramp(40, fps=20)
    out = resample_to_fps(m, target_fps=30)
    assert out.fps == 30
    assert abs(out.num_frames - 60) <= 1


def test_resample_endpoints_match():
    m = _ramp(60, fps=30)
    out = resample_to_fps(m, target_fps=20)
    assert torch.allclose(out.root_positions[0], m.root_positions[0])
    assert torch.allclose(out.root_positions[-1], m.root_positions[-1], atol=1e-5)
