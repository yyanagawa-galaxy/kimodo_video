# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the PoseEstimator registry."""
from __future__ import annotations

import pytest
import torch

from kimodo.video.pose_estimator import (
    PoseEstimator,
    _REGISTRY,
    get_pose_estimator,
    register_pose_estimator,
)
from kimodo.video.types import SOMAMotion30


@pytest.fixture(autouse=True)
def isolate_registry():
    """Snapshot and restore the registry around each test."""
    snapshot = dict(_REGISTRY)
    yield
    _REGISTRY.clear()
    _REGISTRY.update(snapshot)


def _make_fake() -> PoseEstimator:
    class Fake:
        def estimate(self, video_path: str, *, person_idx: int = 0) -> SOMAMotion30:
            return SOMAMotion30(
                local_rot_mats=torch.zeros(5, 30, 3, 3),
                root_positions=torch.zeros(5, 3),
                fps=30,
            )
    return Fake()


def test_register_and_get():
    @register_pose_estimator("fake")
    def _factory():
        return _make_fake()

    est = get_pose_estimator("fake")
    motion = est.estimate("nonexistent.mp4")
    assert motion.num_frames == 5
    assert motion.fps == 30


def test_get_unknown_raises_value_error():
    with pytest.raises(ValueError, match="Unknown estimator"):
        get_pose_estimator("does-not-exist")


def test_factory_is_called_lazily_per_get():
    calls = []

    @register_pose_estimator("counting")
    def _factory():
        calls.append(1)
        return _make_fake()

    # Registration should NOT call the factory
    assert calls == []
    get_pose_estimator("counting")
    get_pose_estimator("counting")
    assert len(calls) == 2  # factory runs on every get
