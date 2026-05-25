# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stub adapter for NVIDIA GEM (SOMA-native pose estimator).

A real GEM adapter is the recommended commercial-friendly migration path
because GEM outputs SOMA directly, bypassing SMPL-X licensing concerns.
See docs/superpowers/specs/2026-05-25-video-input-pipeline-design.md §15.
"""
from __future__ import annotations

from ..pose_estimator import register_pose_estimator
from ..types import SOMAMotion30


class GEMStub:
    def estimate(self, video_path: str, *, person_idx: int = 0) -> SOMAMotion30:
        raise NotImplementedError(
            "gem adapter is not yet implemented. "
            "Track progress in the project roadmap or contribute via PR."
        )


@register_pose_estimator("gem")
def _factory() -> GEMStub:
    return GEMStub()
