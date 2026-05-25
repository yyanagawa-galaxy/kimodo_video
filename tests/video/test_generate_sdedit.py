# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the new SDEdit init path in Kimodo._generate."""
from __future__ import annotations

from unittest.mock import patch

import pytest
import torch


# Reuse the fake_kimodo fixture from the regression test by importing it.
from tests.video.test_regression_existing_path import fake_kimodo  # noqa: F401


def test_generate_calls_q_sample_when_init_motion_provided(fake_kimodo):
    init_motion = torch.zeros(1, 5, 16)
    with patch.object(fake_kimodo.diffusion, "q_sample",
                      wraps=fake_kimodo.diffusion.q_sample) as mock_q:
        fake_kimodo._generate(
            texts=[""],
            max_frames=5,
            num_denoising_steps=10,
            pad_mask=torch.ones(1, 5, dtype=torch.bool),
            first_heading_angle=torch.tensor([0.0]),
            motion_mask=None,
            observed_motion=None,
            cfg_weight=[2.0, 0.0],
            progress_bar=lambda x: x,
            init_motion=init_motion,
            strength=0.5,
        )
    assert mock_q.called, "q_sample must be called when init_motion is provided"


def test_generate_truncates_indices_to_strength_when_init_motion_provided(fake_kimodo):
    init_motion = torch.zeros(1, 5, 16)
    fake_kimodo._generate(
        texts=[""],
        max_frames=5,
        num_denoising_steps=10,
        pad_mask=torch.ones(1, 5, dtype=torch.bool),
        first_heading_angle=torch.tensor([0.0]),
        motion_mask=None,
        observed_motion=None,
        cfg_weight=[2.0, 0.0],
        progress_bar=lambda x: x,
        init_motion=init_motion,
        strength=0.5,
    )
    # strength=0.5 with 10 steps → k=5 → indices [4, 3, 2, 1, 0] → 5 denoising_step calls
    assert fake_kimodo.denoising_step.call_count == 5
    t_values = [call.args[4][0].item() for call in fake_kimodo.denoising_step.call_args_list]
    assert t_values == list(range(4, -1, -1))


def test_generate_does_not_call_torch_randn_when_init_motion_provided(fake_kimodo):
    init_motion = torch.zeros(1, 5, 16)
    with patch("kimodo.model.kimodo_model.torch.randn") as mock_randn:
        fake_kimodo._generate(
            texts=[""],
            max_frames=5,
            num_denoising_steps=10,
            pad_mask=torch.ones(1, 5, dtype=torch.bool),
            first_heading_angle=torch.tensor([0.0]),
            motion_mask=None,
            observed_motion=None,
            cfg_weight=[2.0, 0.0],
            progress_bar=lambda x: x,
            init_motion=init_motion,
            strength=0.5,
        )
    mock_randn.assert_not_called()
