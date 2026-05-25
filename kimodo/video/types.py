# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public motion types for the video-input pipeline."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SOMAMotion30:
    """Motion on the SOMA-30 skeleton — the pipeline's universal currency.

    Every PoseEstimator returns this type. SMPL-X and other intermediates,
    if used, are private to specific adapters.

    Attributes:
        local_rot_mats: Local (parent-relative) joint rotations, shape [T, 30, 3, 3].
        root_positions: Root joint world positions, shape [T, 3].
        fps:            Frame rate of this motion.
    """
    local_rot_mats: torch.Tensor
    root_positions: torch.Tensor
    fps: int

    @property
    def num_frames(self) -> int:
        return int(self.local_rot_mats.shape[0])
