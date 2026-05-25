# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end test for video_to_motion in SDEdit mode (with mocked estimator + model)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

import kimodo.video.estimators  # noqa: F401 — register estimators
from kimodo.video.pose_estimator import _REGISTRY, register_pose_estimator
from kimodo.video.types import SOMAMotion30


@pytest.fixture(autouse=True)
def isolate_registry():
    snapshot = dict(_REGISTRY)
    yield
    _REGISTRY.clear()
    _REGISTRY.update(snapshot)


def _mock_kimodo_model() -> MagicMock:
    """A MagicMock Kimodo that supports the calls pipeline.py makes."""
    model = MagicMock()
    model.motion_rep.fps = 20
    model.motion_rep.motion_rep_dim = 16
    # motion_rep(rot, root, to_normalize=True) -> encoded [T, D]
    model.motion_rep.side_effect = lambda rot, root, **kw: torch.zeros(rot.shape[0], 16)
    # _generate returns a clean motion of shape [B, T, D]
    model._generate.return_value = torch.zeros(1, 20, 16)
    # motion_rep.inverse -> dict
    model.motion_rep.inverse.return_value = {
        "posed_joints":   torch.zeros(20, 77, 3),
        "local_rot_mats": torch.zeros(20, 30, 3, 3),
    }
    # skeleton.output_to_SOMASkeleton77 -> identity for this stub
    model.skeleton.output_to_SOMASkeleton77.side_effect = lambda d: d
    return model


def test_video_to_motion_sdedit_dispatches_correctly(tmp_path, monkeypatch):
    # Register a fake estimator that returns a fixed SOMAMotion30.
    @register_pose_estimator("fake")
    def _factory():
        class F:
            def estimate(self, video_path, *, person_idx=0):
                return SOMAMotion30(
                    local_rot_mats=torch.eye(3).expand(20, 30, 3, 3).clone(),
                    root_positions=torch.zeros(20, 3),
                    fps=20,
                )
        return F()

    mock_model = _mock_kimodo_model()
    monkeypatch.setattr("kimodo.video.pipeline.load_model", lambda *a, **kw: mock_model)

    from kimodo.video.pipeline import video_to_motion

    result = video_to_motion(
        video_path=str(tmp_path / "fake.mp4"),
        estimator="fake",
        mode="sdedit",
        strength=0.5,
        return_numpy=False,
    )

    # _generate must have been called with init_motion provided (SDEdit path).
    assert mock_model._generate.called
    _args, kwargs = mock_model._generate.call_args
    assert kwargs.get("init_motion") is not None
    assert kwargs.get("strength") == 0.5

    # cfg_weight in SDEdit mode forces the constraint channel to 0.
    assert kwargs.get("cfg_weight")[1] == 0.0

    # The result dict has the expected keys.
    assert "posed_joints" in result
