# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pluggable PoseEstimator protocol and registry.

Every estimator adapter returns SOMAMotion30. Body-model intermediates
(SMPL-X etc.) are private to specific adapters and never leak through this
interface.
"""
from __future__ import annotations

from typing import Callable, Dict, Protocol

from .types import SOMAMotion30


class PoseEstimator(Protocol):
    """Adapter that turns a video into SOMA-30 motion."""

    def estimate(self, video_path: str, *, person_idx: int = 0) -> SOMAMotion30:
        ...


_REGISTRY: Dict[str, Callable[[], PoseEstimator]] = {}


def register_pose_estimator(name: str) -> Callable[[Callable[[], PoseEstimator]], Callable[[], PoseEstimator]]:
    """Decorator that adds a factory to the registry under `name`."""
    def deco(factory: Callable[[], PoseEstimator]) -> Callable[[], PoseEstimator]:
        _REGISTRY[name] = factory
        return factory
    return deco


def get_pose_estimator(name: str) -> PoseEstimator:
    """Build a fresh PoseEstimator instance by registered name."""
    if name not in _REGISTRY:
        raise ValueError(f"Unknown estimator {name!r}. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]()
