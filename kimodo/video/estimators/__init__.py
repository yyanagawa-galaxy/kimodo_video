# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bundled PoseEstimator adapters.

Importing this package triggers registration of every bundled adapter
(real or stub). After import, all adapter names are visible via
`kimodo.video.get_pose_estimator(name)`.
"""
from __future__ import annotations

from . import fourdhumans, gem, gvhmr, mediapipe, wham  # noqa: F401 -- import triggers registration
