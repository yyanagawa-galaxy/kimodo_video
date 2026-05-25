# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for overlap-blend chunking."""
from __future__ import annotations

import pytest
import torch

from kimodo.video.chunking import Window, chunk_motion, stitch_chunks


def test_chunk_short_input_returns_single_window():
    motion = torch.randn(50, 8)
    windows = chunk_motion(motion, chunk_size=100, overlap=20)
    assert len(windows) == 1
    assert windows[0].start == 0
    assert windows[0].end == 50
    assert torch.equal(windows[0].motion, motion)


def test_chunk_returns_overlapping_windows_for_long_input():
    motion = torch.randn(250, 8)  # 250 frames, chunk_size=100, overlap=20
    windows = chunk_motion(motion, chunk_size=100, overlap=20)
    # stride = 80, so windows start at 0, 80, 160 → 240 (last truncated)
    assert windows[0].start == 0 and windows[0].end == 100
    assert windows[1].start == 80 and windows[1].end == 180
    assert windows[2].start == 160 and windows[2].end == 250  # last window truncated to length


def test_stitch_preserves_length_and_smooth_signal():
    # A smooth signal should pass through chunk + stitch unchanged (within float tol).
    t_total = 250
    motion = torch.linspace(0, 1, t_total).unsqueeze(-1).expand(-1, 4).contiguous()

    windows = chunk_motion(motion, chunk_size=100, overlap=20)
    cleaned = [Window(start=w.start, end=w.end, motion=w.motion.clone()) for w in windows]
    out = stitch_chunks(cleaned, t_total)

    assert out.shape == motion.shape
    assert torch.allclose(out, motion, atol=1e-5)


def test_stitch_blends_overlap():
    # Two windows of constant values 0 and 1, with overlap → middle ramps 0→1.
    t_total = 30
    w1_motion = torch.zeros(20, 1)
    w2_motion = torch.ones(20, 1)
    windows = [
        Window(start=0, end=20, motion=w1_motion),
        Window(start=10, end=30, motion=w2_motion),
    ]
    out = stitch_chunks(windows, t_total)
    assert out.shape == (30, 1)
    assert out[0].item() == 0.0
    assert out[-1].item() == 1.0
    # In the overlap region (frames 10..19), the output should be between 0 and 1.
    overlap = out[10:20, 0]
    assert (overlap > 0).all() and (overlap < 1).all()
