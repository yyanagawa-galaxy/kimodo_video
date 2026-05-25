# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for stub PoseEstimator adapters."""
from __future__ import annotations

import pytest

# Importing the estimators package triggers stub registration.
import kimodo.video.estimators  # noqa: F401
from kimodo.video.pose_estimator import get_pose_estimator


@pytest.mark.parametrize("name", ["gem", "wham", "4dhumans"])
def test_stub_adapters_are_registered(name):
    estimator = get_pose_estimator(name)
    assert estimator is not None


@pytest.mark.parametrize("name", ["gem", "wham", "4dhumans"])
def test_stub_adapters_raise_not_implemented(name):
    estimator = get_pose_estimator(name)
    with pytest.raises(NotImplementedError, match=name):
        estimator.estimate("any.mp4")
