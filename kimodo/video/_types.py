# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Internal motion types used only inside specific PoseEstimator adapters.

NOT part of the public API. SMPL-X carries a non-commercial license restriction
and is intentionally hidden from the public namespace to avoid implying it is
the pipeline's canonical representation.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SMPLXMotion:
    """Motion in SMPL-X parameters — emitted by GVHMR/WHAM/4DHumans adapters
    internally, immediately retargeted to SOMA-30 before leaving the adapter.

    Attributes:
        body_pose:     Per-frame body joint rotations, shape [T, 21, 3, 3].
        global_orient: Per-frame root rotation, shape [T, 3, 3].
        transl:        Per-frame root translation in world frame, shape [T, 3].
        fps:           Frame rate.
    """
    body_pose: torch.Tensor
    global_orient: torch.Tensor
    transl: torch.Tensor
    fps: int

    @property
    def num_frames(self) -> int:
        return int(self.body_pose.shape[0])
