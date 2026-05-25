# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stub adapter for WHAM (world-grounded SMPL-X pose estimator)."""
from __future__ import annotations

from ..pose_estimator import register_pose_estimator
from ..types import SOMAMotion30


class WHAMStub:
    def estimate(self, video_path: str, *, person_idx: int = 0) -> SOMAMotion30:
        raise NotImplementedError(
            "wham adapter is not yet implemented. "
            "Track progress in the project roadmap or contribute via PR."
        )


@register_pose_estimator("wham")
def _factory() -> WHAMStub:
    return WHAMStub()
