# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SDEdit-style initial noise computation.

Encodes a noisy motion observation through the forward diffusion process
to a specified noise level, so the denoising loop can start from there
rather than from pure Gaussian noise.
"""
from __future__ import annotations

from typing import Tuple

import torch

from kimodo.model.diffusion import Diffusion

from .errors import InvalidStrengthError


def sdedit_init(
    motion: torch.Tensor,
    strength: float,
    num_denoising_steps: int,
    diffusion: Diffusion,
) -> Tuple[torch.Tensor, int]:
    """Compute the SDEdit starting tensor and starting step index.

    Args:
        motion:               Normalized motion features, shape [B, T, D].
        strength:             SDEdit strength in [0, 1]. 0 = no noise added,
                              1 = full diffusion noise (equivalent to pure-noise init).
        num_denoising_steps:  Total number of DDIM steps the sampler will run.
        diffusion:            A Kimodo Diffusion instance.

    Returns:
        A tuple `(noisy_motion, start_step)`:
        - noisy_motion: motion noised to step (start_step - 1).
        - start_step:   the diffusion step index from which to begin denoising
                        (the loop runs from start_step - 1 down to 0).

    Raises:
        InvalidStrengthError: if strength is outside [0, 1].
    """
    if not 0.0 <= strength <= 1.0:
        raise InvalidStrengthError(strength)

    # Ensure diffusion vars match the requested schedule length.
    use_timesteps, _ = diffusion.space_timesteps(num_denoising_steps)
    diffusion.calc_diffusion_vars(use_timesteps)

    k = max(1, int(strength * num_denoising_steps))
    batch_size = motion.shape[0]
    t_init = torch.full((batch_size,), k - 1, dtype=torch.long, device=motion.device)
    noisy = diffusion.q_sample(motion, t_init)
    return noisy, k
