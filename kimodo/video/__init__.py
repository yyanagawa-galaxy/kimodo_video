# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Video-input pipeline for Kimodo.

Public API:
    video_to_motion: drop a video, get clean SOMA-77 motion.
    SOMAMotion30:   public motion type produced by all PoseEstimators.
    PoseEstimator, register_pose_estimator, get_pose_estimator: estimator registry.
    resample_to_fps: frame-rate utility.

This package is opt-in: install with `pip install kimodo[video]`.
"""
from __future__ import annotations

from .pose_estimator import PoseEstimator, get_pose_estimator, register_pose_estimator
from .types import SOMAMotion30

__all__ = [
    "SOMAMotion30",
    "PoseEstimator",
    "register_pose_estimator",
    "get_pose_estimator",
]
