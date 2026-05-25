# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Map MediaPipe's 33 anatomical landmarks to SOMA-30 joint positions per frame."""
from __future__ import annotations
import torch


def synthesize_soma30_positions(mp_landmarks: torch.Tensor) -> torch.Tensor:
    """Convert MediaPipe landmark tensor to SOMA-30 joint positions.

    Args:
        mp_landmarks: [T, 33, 3] world-space landmark positions from MediaPipe.

    Returns:
        [T, 30, 3] SOMA-30 joint positions.
    """
    # MediaPipe's pose_world_landmarks use image-style axes:
    #   +X: subject's left, +Y: down (head-to-feet), +Z: away from camera
    # SOMA uses Y-up right-handed coordinates:
    #   +X: subject's left (kept), +Y: up, +Z: forward (toward subject's front)
    # To convert: flip Y (down->up) and flip Z (handedness adjustment so
    # left/right limbs aren't mirrored after Y flip).
    lm = mp_landmarks.clone()
    lm[..., 1] = -lm[..., 1]
    lm[..., 2] = -lm[..., 2]

    def mid(a: int, b: int) -> torch.Tensor:
        return 0.5 * (lm[:, a, :] + lm[:, b, :])

    pos = torch.empty(lm.shape[0], 30, 3, device=lm.device, dtype=lm.dtype)

    pos[:, 0]  = mid(23, 24)                                       # Hips
    pos[:, 1]  = 0.75 * mid(23, 24) + 0.25 * mid(11, 12)          # Spine1
    pos[:, 2]  = 0.50 * mid(23, 24) + 0.50 * mid(11, 12)          # Spine2
    pos[:, 3]  = 0.20 * mid(23, 24) + 0.80 * mid(11, 12)          # Chest
    pos[:, 4]  = 0.60 * mid(11, 12) + 0.40 * mid(7, 8)            # Neck1
    pos[:, 5]  = 0.20 * mid(11, 12) + 0.80 * mid(7, 8)            # Neck2
    pos[:, 6]  = mid(7, 8)                                         # Head
    pos[:, 7]  = mid(9, 10)                                        # Jaw
    pos[:, 8]  = lm[:, 2, :]                                       # LeftEye
    pos[:, 9]  = lm[:, 5, :]                                       # RightEye
    pos[:, 10] = lm[:, 11, :]                                      # LeftShoulder
    pos[:, 11] = lm[:, 11, :]                                      # LeftArm
    pos[:, 12] = lm[:, 13, :]                                      # LeftForeArm
    pos[:, 13] = lm[:, 15, :]                                      # LeftHand
    pos[:, 14] = lm[:, 21, :]                                      # LeftHandThumbEnd
    pos[:, 15] = lm[:, 19, :]                                      # LeftHandMiddleEnd
    pos[:, 16] = lm[:, 12, :]                                      # RightShoulder
    pos[:, 17] = lm[:, 12, :]                                      # RightArm
    pos[:, 18] = lm[:, 14, :]                                      # RightForeArm
    pos[:, 19] = lm[:, 16, :]                                      # RightHand
    pos[:, 20] = lm[:, 22, :]                                      # RightHandThumbEnd
    pos[:, 21] = lm[:, 20, :]                                      # RightHandMiddleEnd
    pos[:, 22] = lm[:, 23, :]                                      # LeftLeg
    pos[:, 23] = lm[:, 25, :]                                      # LeftShin
    pos[:, 24] = lm[:, 27, :]                                      # LeftFoot
    pos[:, 25] = lm[:, 31, :]                                      # LeftToeBase
    pos[:, 26] = lm[:, 24, :]                                      # RightLeg
    pos[:, 27] = lm[:, 26, :]                                      # RightShin
    pos[:, 28] = lm[:, 28, :]                                      # RightFoot
    pos[:, 29] = lm[:, 32, :]                                      # RightToeBase

    return pos
