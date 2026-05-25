# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the pipeline's automatic chunking on long inputs."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

import kimodo.video.estimators  # noqa: F401
from kimodo.video.pose_estimator import _REGISTRY, register_pose_estimator
from kimodo.video.types import SOMAMotion30


@pytest.fixture(autouse=True)
def isolate_registry():
    snapshot = dict(_REGISTRY)
    yield
    _REGISTRY.clear()
    _REGISTRY.update(snapshot)


def _mock_model_with_max_frames(max_frames: int) -> MagicMock:
    model = MagicMock()
    model.motion_rep.fps = 20
    model.motion_rep.motion_rep_dim = 16
    model.max_frames = max_frames
    model.motion_rep.side_effect = lambda rot, root, **kw: torch.zeros(rot.shape[0], 16)
    model._generate.side_effect = lambda *a, **kw: torch.zeros(1, kw["max_frames"], 16)
    model.motion_rep.inverse.return_value = {
        "posed_joints":   torch.zeros(1, 77, 3),
        "local_rot_mats": torch.zeros(1, 30, 3, 3),
    }
    model.skeleton.output_to_SOMASkeleton77.side_effect = lambda d: d
    return model


def test_long_input_triggers_multiple_chunks(tmp_path, monkeypatch):
    @register_pose_estimator("long_fake")
    def _factory():
        class F:
            def estimate(self, video_path, *, person_idx=0):
                return SOMAMotion30(
                    local_rot_mats=torch.eye(3).expand(250, 30, 3, 3).clone(),
                    root_positions=torch.zeros(250, 3),
                    fps=20,
                )
        return F()

    mock_model = _mock_model_with_max_frames(max_frames=100)
    monkeypatch.setattr("kimodo.video.pipeline.load_model", lambda *a, **kw: mock_model)

    from kimodo.video.pipeline import video_to_motion

    video_to_motion(
        video_path=str(tmp_path / "long.mp4"),
        estimator="long_fake",
        mode="sdedit",
        chunk_size=100,
        chunk_overlap=20,
        return_numpy=False,
    )

    # 250 frames with chunk_size=100, overlap=20, stride=80 -> starts 0, 80, 160 -> 3 chunks.
    assert mock_model._generate.call_count == 3
