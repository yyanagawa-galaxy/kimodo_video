# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the GVHMR PoseEstimator adapter.

Real-GVHMR end-to-end coverage requires the upstream package and is left
to manual verification. These tests mock the imports.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

import kimodo.video.estimators  # noqa: F401
from kimodo.video.errors import EstimatorNotInstalledError, NoMotionDetectedError
from kimodo.video.pose_estimator import get_pose_estimator
from kimodo.video.types import SOMAMotion30


def test_gvhmr_is_registered():
    est = get_pose_estimator("gvhmr")
    assert est is not None


def test_gvhmr_raises_install_error_if_dep_missing():
    from kimodo.video.estimators import gvhmr
    with patch.object(gvhmr, "_load_gvhmr", side_effect=ImportError("no module")):
        est = gvhmr.GVHMRAdapter()
        with pytest.raises(EstimatorNotInstalledError, match="gvhmr"):
            est.estimate("any.mp4")


def test_gvhmr_returns_soma_motion30(monkeypatch):
    """When the mocked GVHMR pipeline returns a valid SMPL-X result,
    the adapter retargets it to SOMA-30 internally and returns SOMAMotion30."""
    from kimodo.video.estimators import gvhmr

    fake_gvhmr_module = MagicMock()
    # The adapter expects: run(video_path) -> dict with body_pose, global_orient, transl, fps
    fake_gvhmr_module.run.return_value = {
        "body_pose":     torch.zeros(15, 21, 3, 3),
        "global_orient": torch.zeros(15, 3, 3),
        "transl":        torch.zeros(15, 3),
        "fps":           30,
    }
    monkeypatch.setattr(gvhmr, "_load_gvhmr", lambda: fake_gvhmr_module)

    # Mock the SMPL-X -> SOMA retarget to avoid needing py-soma-x installed.
    # Patch where it's imported in the gvhmr module.
    def _fake_retarget(smplx):
        return SOMAMotion30(
            local_rot_mats=torch.zeros(smplx.num_frames, 30, 3, 3),
            root_positions=torch.zeros(smplx.num_frames, 3),
            fps=smplx.fps,
        )
    monkeypatch.setattr(gvhmr, "smplx_to_soma_retarget", _fake_retarget)

    est = gvhmr.GVHMRAdapter()
    out = est.estimate("any.mp4")
    assert isinstance(out, SOMAMotion30)
    assert out.num_frames == 15
    assert out.fps == 30


def test_gvhmr_raises_when_no_motion_detected(monkeypatch):
    from kimodo.video.estimators import gvhmr

    fake_gvhmr_module = MagicMock()
    fake_gvhmr_module.run.return_value = {
        "body_pose":     torch.zeros(0, 21, 3, 3),
        "global_orient": torch.zeros(0, 3, 3),
        "transl":        torch.zeros(0, 3),
        "fps":           30,
    }
    monkeypatch.setattr(gvhmr, "_load_gvhmr", lambda: fake_gvhmr_module)

    est = gvhmr.GVHMRAdapter()
    with pytest.raises(NoMotionDetectedError):
        est.estimate("any.mp4")
