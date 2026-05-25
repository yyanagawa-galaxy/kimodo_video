# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Overlap-blend chunking for long-video pipelines.

Splits motion into overlapping windows for independent denoising, then
stitches them back together with raised-cosine weighting on the overlaps.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

import torch


@dataclass
class Window:
    """A single chunk of motion.

    Attributes:
        start:  Inclusive start frame index in the full motion.
        end:    Exclusive end frame index in the full motion.
        motion: Tensor of shape [end - start, D].
    """
    start: int
    end: int
    motion: torch.Tensor


def chunk_motion(motion: torch.Tensor, chunk_size: int, overlap: int) -> List[Window]:
    """Split [T, D] motion into overlapping [chunk_size, D] windows.

    If T <= chunk_size, returns a single window covering [0, T).

    Args:
        motion:     Input motion, shape [T, D].
        chunk_size: Max length of each window in frames.
        overlap:    Overlap between consecutive windows in frames.

    Returns:
        List of Window objects whose `motion` tensors cover the input.
    """
    t_total = motion.shape[0]
    if t_total <= chunk_size:
        return [Window(start=0, end=t_total, motion=motion)]
    if overlap >= chunk_size:
        raise ValueError(f"overlap ({overlap}) must be strictly less than chunk_size ({chunk_size})")

    stride = chunk_size - overlap
    windows: List[Window] = []
    start = 0
    while start < t_total:
        end = min(start + chunk_size, t_total)
        windows.append(Window(start=start, end=end, motion=motion[start:end]))
        if end == t_total:
            break
        start += stride
    return windows


def stitch_chunks(windows: List[Window], total_T: int) -> torch.Tensor:
    """Blend cleaned windows back into a [T, D] tensor with raised-cosine alpha.

    Args:
        windows: List of Window objects (cleaned). All must share the same
                 trailing shape (D) and contiguous coverage of [0, total_T).
        total_T: Total length of the output.

    Returns:
        Tensor of shape [total_T, D], where overlap regions are weighted
        averages of contributing windows.
    """
    if not windows:
        raise ValueError("stitch_chunks requires at least one window")

    d = windows[0].motion.shape[-1]
    device = windows[0].motion.device
    dtype = windows[0].motion.dtype
    out = torch.zeros(total_T, d, device=device, dtype=dtype)
    weight = torch.zeros(total_T, 1, device=device, dtype=dtype)

    for w in windows:
        length = w.end - w.start
        alpha = _raised_cosine(length, device=device, dtype=dtype).unsqueeze(-1)
        out[w.start : w.end] += w.motion * alpha
        weight[w.start : w.end] += alpha

    return out / weight.clamp(min=1e-12)


def _raised_cosine(length: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """A raised-cosine window of `length` frames in [0, 1]. Symmetric, ends at small nonzero."""
    if length == 1:
        return torch.ones(1, device=device, dtype=dtype)
    n = torch.arange(length, device=device, dtype=dtype)
    # Half-period cosine that avoids zero: 0.5 - 0.5*cos gives 0→1, but we shift to avoid 0
    # Use (1 + cos(pi + pi*n/(length-1))) / 2 = (1 - cos(pi*n/(length-1))) / 2
    # which goes from 0 to 1. To make it stay positive, use (2 + cos(pi*n/(length-1))) / 3
    # which goes from 1 to 1/3. Normalize to [0.1, 1] range: scale and shift
    base = 0.5 - 0.5 * torch.cos(math.pi * n / (length - 1))  # [0, 1]
    # Shift upward so minimum is 0.1: 0.1 + 0.9 * base
    return 0.1 + 0.9 * base
