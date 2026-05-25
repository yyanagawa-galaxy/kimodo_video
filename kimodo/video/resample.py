# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Frame-rate resampling for SOMA motion.

Linear interpolation on positions; spherical-linear interpolation (slerp) is
out of scope for v1 — rotations are interpolated as flat 9-vectors and
re-orthogonalized only if downstream needs it (the diffusion prior cleans
up any small drift).
"""
from __future__ import annotations

import torch

from .types import SOMAMotion30


def resample_to_fps(motion: SOMAMotion30, target_fps: int) -> SOMAMotion30:
    """Resample motion from its current fps to target_fps via linear interpolation.

    If `target_fps == motion.fps`, returns the input unchanged.

    Args:
        motion:     Input motion.
        target_fps: Desired output frame rate.

    Returns:
        A new SOMAMotion30 at target_fps. The duration is preserved within
        one-frame rounding.
    """
    if target_fps == motion.fps:
        return motion

    t_in = motion.num_frames
    if t_in < 2:
        # Trivial: a single frame can't be meaningfully resampled.
        return SOMAMotion30(
            local_rot_mats=motion.local_rot_mats,
            root_positions=motion.root_positions,
            fps=target_fps,
        )

    duration_s = (t_in - 1) / motion.fps
    t_out = max(1, round(duration_s * target_fps) + 1)

    src_times = torch.linspace(0, duration_s, t_in)
    tgt_times = torch.linspace(0, duration_s, t_out)

    # Linear interp on root positions.
    root = _interp_1d(motion.root_positions, src_times, tgt_times)

    # Linear interp on flattened rotation matrices.
    rot_flat = motion.local_rot_mats.reshape(t_in, -1)
    rot_out_flat = _interp_1d(rot_flat, src_times, tgt_times)
    rot_out = rot_out_flat.reshape(t_out, 30, 3, 3)

    return SOMAMotion30(local_rot_mats=rot_out, root_positions=root, fps=target_fps)


def _interp_1d(values: torch.Tensor, src_t: torch.Tensor, tgt_t: torch.Tensor) -> torch.Tensor:
    """Linear interp along axis 0. `values` has shape [T_in, ...]."""
    t_in = src_t.shape[0]
    # For each target time, find the bracketing source indices.
    # src_t is monotonic over [0, duration_s], so we can use searchsorted.
    idx_hi = torch.searchsorted(src_t, tgt_t, right=False).clamp(1, t_in - 1)
    idx_lo = idx_hi - 1
    t_lo = src_t[idx_lo]
    t_hi = src_t[idx_hi]
    w = ((tgt_t - t_lo) / (t_hi - t_lo).clamp(min=1e-12)).clamp(0, 1)
    # Broadcast w across the trailing dims of values.
    while w.ndim < values.ndim:
        w = w.unsqueeze(-1)
    return (1 - w) * values[idx_lo] + w * values[idx_hi]
