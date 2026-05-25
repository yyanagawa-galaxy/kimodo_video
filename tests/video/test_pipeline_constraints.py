# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for video_to_motion in constraints mode."""
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


def _mock_kimodo_model() -> MagicMock:
    model = MagicMock()
    model.motion_rep.fps = 20
    model.motion_rep.motion_rep_dim = 16

    # __call__ returns the standard output dict.
    model.return_value = {
        "posed_joints":   torch.zeros(20, 77, 3),
        "local_rot_mats": torch.zeros(20, 30, 3, 3),
    }
    model.skeleton.output_to_SOMASkeleton77.side_effect = lambda d: d
    # FK must return a 2-tuple so _fk() can unpack (posed_joints, global_rot_mats).
    model.skeleton.forward_kinematics.return_value = (
        torch.zeros(20, 30, 3),   # posed_joints
        torch.zeros(20, 30, 3, 3), # global_rot_mats
    )
    return model


def test_constraints_mode_calls_kimodo_with_constraint_lst(tmp_path, monkeypatch):
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
    # Patch constraint classes so we don't need a real skeleton in the test.
    monkeypatch.setattr("kimodo.constraints.FullBodyConstraintSet", lambda *a, **kw: MagicMock())
    monkeypatch.setattr("kimodo.constraints.EndEffectorConstraintSet", lambda *a, **kw: MagicMock())

    from kimodo.video.pipeline import video_to_motion

    result = video_to_motion(
        video_path=str(tmp_path / "fake.mp4"),
        estimator="fake",
        mode="constraints",
        cfg_weight=(2.0, 0.5),
        return_numpy=False,
    )

    # In constraints mode, Kimodo.__call__ is used (not _generate directly).
    assert mock_model.called
    _args, kwargs = mock_model.call_args
    assert "constraint_lst" in kwargs
    assert len(kwargs["constraint_lst"]) > 0  # at least one constraint set built
    assert kwargs.get("cfg_weight") == [2.0, 0.5]

    # Result dict still has the expected schema.
    assert "posed_joints" in result
