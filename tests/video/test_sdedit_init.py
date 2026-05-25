# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for SDEdit init helper."""
from __future__ import annotations

import pytest
import torch

from kimodo.model.diffusion import Diffusion
from kimodo.video.errors import InvalidStrengthError
from kimodo.video.sdedit import sdedit_init


@pytest.fixture
def diffusion():
    return Diffusion(num_base_steps=100)


def test_strength_zero_returns_input_with_step_zero(diffusion):
    x = torch.randn(2, 30, 64)
    noisy, k = sdedit_init(x, strength=0.0, num_denoising_steps=100, diffusion=diffusion)
    assert k == 1  # we clamp k to at least 1 so the loop runs once
    # At step 0 (clamped to 1), q_sample adds minimal noise. Tensor stays close to input.
    assert noisy.shape == x.shape


def test_strength_one_returns_pure_noise_like_shape(diffusion):
    x = torch.zeros(2, 30, 64)
    noisy, k = sdedit_init(x, strength=1.0, num_denoising_steps=100, diffusion=diffusion)
    assert k == 100
    assert noisy.shape == x.shape
    # With zero input and full noise, the output should NOT be all zeros.
    assert noisy.abs().sum() > 0


def test_strength_half_uses_midpoint(diffusion):
    x = torch.randn(2, 30, 64)
    _, k = sdedit_init(x, strength=0.5, num_denoising_steps=100, diffusion=diffusion)
    assert k == 50


def test_invalid_strength_below_zero_raises(diffusion):
    with pytest.raises(InvalidStrengthError):
        sdedit_init(torch.zeros(1, 1, 1), strength=-0.1, num_denoising_steps=100, diffusion=diffusion)


def test_invalid_strength_above_one_raises(diffusion):
    with pytest.raises(InvalidStrengthError):
        sdedit_init(torch.zeros(1, 1, 1), strength=1.1, num_denoising_steps=100, diffusion=diffusion)
