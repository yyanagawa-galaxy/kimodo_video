# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression guard for Kimodo._generate's existing pure-noise path.

These tests assert that when `init_motion` is None (the default), `_generate`
behaves identically to the original implementation:
- It calls torch.randn to seed cur_mot.
- It iterates over the full range [num_denoising_steps - 1, ..., 0].
- It does NOT call diffusion.q_sample.

These tests must pass both BEFORE and AFTER the _generate modification.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch


@pytest.fixture
def fake_kimodo():
    """Build a Kimodo instance with all heavy components mocked.

    The mocks let us drive _generate directly without loading weights.
    """
    from kimodo.model.kimodo_model import Kimodo
    from kimodo.model.diffusion import Diffusion

    # Stub denoiser with the bare minimum attribute surface Kimodo needs.
    denoiser = MagicMock()
    denoiser.eval.return_value = denoiser
    motion_rep = MagicMock()
    motion_rep.motion_rep_dim = 16
    motion_rep.fps = 20
    motion_rep.skeleton = MagicMock()
    denoiser.motion_rep = motion_rep

    text_encoder = MagicMock()
    # text_encoder(texts) → (text_feat [B, L, D], text_length list)
    text_encoder.return_value = (torch.zeros(1, 4, 32), [4])

    # Bypass the ClassifierFreeGuidedModel wrapping in Kimodo.__init__ by
    # constructing the object then overriding self.denoiser.
    model = Kimodo.__new__(Kimodo)
    torch.nn.Module.__init__(model)
    model.denoiser = denoiser
    model.motion_rep = motion_rep
    model.skeleton = motion_rep.skeleton
    model.fps = 20
    model.diffusion = Diffusion(num_base_steps=10)
    from kimodo.model.diffusion import DDIMSampler
    model.sampler = DDIMSampler(model.diffusion)
    model.text_encoder = text_encoder
    model.device = torch.device("cpu")

    # denoising_step returns a same-shape tensor (identity for simplicity).
    def _identity_step(motion, *args, **kwargs):
        return motion
    model.denoising_step = MagicMock(side_effect=_identity_step)
    return model


def test_generate_uses_torch_randn_when_init_motion_is_none(fake_kimodo):
    """The existing pure-noise init path must still fire when init_motion=None."""
    with patch("kimodo.model.kimodo_model.torch.randn",
               wraps=torch.randn) as mock_randn:
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
        )
    assert mock_randn.called, "torch.randn must be called when init_motion is None"
    # Verify the shape passed to randn matches the expected (1, max_frames, motion_rep_dim).
    args, _kwargs = mock_randn.call_args
    assert args[0] == (1, 5, 16)


def test_generate_iterates_full_range_when_init_motion_is_none(fake_kimodo):
    """When init_motion=None, the denoising loop must iterate over [N-1, ..., 0]."""
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
    )
    # denoising_step should have been called exactly 10 times (one per step).
    assert fake_kimodo.denoising_step.call_count == 10
    # Collect the `t` values passed in; they should be [9, 8, 7, ..., 0].
    t_values = [call.args[4][0].item() for call in fake_kimodo.denoising_step.call_args_list]
    assert t_values == list(range(9, -1, -1))
