# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pipeline orchestrator for video -> clean SOMA motion.

This is the entry point for the public `video_to_motion` function.
Operates exclusively in SOMA: SMPL-X (if used by an estimator adapter) is
already retargeted by the time the pipeline sees it.
"""
from __future__ import annotations

from typing import Literal, Optional, Tuple

import torch

from kimodo import load_model
from kimodo.tools import to_numpy

import kimodo.video.estimators  # noqa: F401 — register estimators on import

from .errors import InvalidStrengthError, VideoTooShortError
from .pose_estimator import get_pose_estimator
from .resample import resample_to_fps
from .sdedit import sdedit_init
from .types import SOMAMotion30

_MIN_FRAMES = 10


def video_to_motion(
    video_path: str,
    *,
    output_path: Optional[str] = None,
    model_name: str = "Kimodo-SOMA-RP-v1.1",
    prompt: str = "",
    strength: float = 0.5,
    mode: Literal["sdedit", "constraints"] = "sdedit",
    estimator: str = "gvhmr",
    num_denoising_steps: int = 100,
    num_samples: int = 1,
    cfg_weight: Tuple[float, float] = (2.0, 0.5),
    post_processing: bool = True,
    seed: Optional[int] = None,
    person_idx: int = 0,
    save_intermediates: bool = False,
    device: Optional[str] = None,
    return_numpy: bool = True,
) -> dict:
    """Run the video -> clean SOMA motion pipeline.

    See docs/superpowers/specs/2026-05-25-video-input-pipeline-design.md for the full API.
    """
    if not 0.0 <= strength <= 1.0:
        raise InvalidStrengthError(strength)
    if seed is not None:
        torch.manual_seed(seed)

    # 1. Pose estimation -> SOMAMotion30
    est = get_pose_estimator(estimator)
    soma_in = est.estimate(video_path, person_idx=person_idx)

    # 2. Load the model so we know its target fps and motion_rep
    model = load_model(model_name, device=device)

    # 3. Resample to the model's fps
    soma_in = resample_to_fps(soma_in, target_fps=model.motion_rep.fps)
    if soma_in.num_frames < _MIN_FRAMES:
        raise VideoTooShortError(soma_in.num_frames, _MIN_FRAMES)

    # 4. Dispatch by mode
    if mode == "sdedit":
        result = _run_sdedit(
            model=model, soma_in=soma_in, prompt=prompt, strength=strength,
            num_denoising_steps=num_denoising_steps, num_samples=num_samples,
            cfg_weight=cfg_weight, post_processing=post_processing,
        )
    elif mode == "constraints":
        result = _run_constraints(
            model=model, soma_in=soma_in, prompt=prompt,
            num_denoising_steps=num_denoising_steps, num_samples=num_samples,
            cfg_weight=cfg_weight, post_processing=post_processing,
        )
    else:
        raise ValueError(f"Unknown mode {mode!r}; expected 'sdedit' or 'constraints'")

    # 5. Convert SOMA-30 output dict to SOMA-77 for the public API
    if hasattr(model.skeleton, "output_to_SOMASkeleton77"):
        result = model.skeleton.output_to_SOMASkeleton77(result)

    if return_numpy:
        result = to_numpy(result)

    if output_path is not None:
        _save_output(result, output_path)

    return result


def _run_sdedit(
    *,
    model,
    soma_in: SOMAMotion30,
    prompt: str,
    strength: float,
    num_denoising_steps: int,
    num_samples: int,
    cfg_weight: Tuple[float, float],
    post_processing: bool,
) -> dict:
    # Encode SOMA motion to Kimodo's normalized feature space.
    encoded = model.motion_rep(
        soma_in.local_rot_mats,
        soma_in.root_positions,
        to_normalize=True,
    )
    if encoded.ndim == 2:
        encoded = encoded.unsqueeze(0)  # add batch dim -> [1, T, D]
    # Replicate across batch for num_samples.
    init_motion = encoded.expand(num_samples, -1, -1).contiguous()

    # In SDEdit mode, no observed_motion -> force constraint CFG to 0.
    cfg = (cfg_weight[0], 0.0)

    # Build the pad_mask, heading, and texts the way Kimodo.__call__ does.
    T = init_motion.shape[1]
    pad_mask = torch.ones(num_samples, T, dtype=torch.bool, device=init_motion.device)
    first_heading_angle = torch.zeros(num_samples, device=init_motion.device)
    texts = [prompt] * num_samples

    clean = model._generate(
        texts=texts,
        max_frames=T,
        num_denoising_steps=num_denoising_steps,
        pad_mask=pad_mask,
        first_heading_angle=first_heading_angle,
        motion_mask=None,
        observed_motion=None,
        cfg_weight=list(cfg),
        init_motion=init_motion,
        strength=strength,
    )

    # Decode back to joint positions and rotations.
    output = model.motion_rep.inverse(clean, is_normalized=True, return_numpy=False)

    if post_processing and "root_positions" in output and "foot_contacts" in output:
        from kimodo.postprocess import post_process_motion
        corrected = post_process_motion(
            output["local_rot_mats"],
            output["root_positions"],
            output["foot_contacts"],
            model.skeleton,
            [],  # no constraints in SDEdit mode
        )
        output.update(corrected)

    return output


def _run_constraints(
    *,
    model,
    soma_in: SOMAMotion30,
    prompt: str,
    num_denoising_steps: int,
    num_samples: int,
    cfg_weight: Tuple[float, float],
    post_processing: bool,
) -> dict:
    """Constraints-mode pipeline: feed retargeted SOMA motion as soft full-body constraints."""
    from kimodo.constraints import EndEffectorConstraintSet, FullBodyConstraintSet

    T = soma_in.num_frames
    constraint_lst = []
    for _ in range(num_samples):
        # Compute posed joints from local rotations + root positions for the constraint.
        # Forward-kinematics path: delegate to the model's skeleton's FK if available.
        posed_joints, global_rot_mats = _fk(model.skeleton, soma_in.local_rot_mats, soma_in.root_positions)
        smooth_root_2d = soma_in.root_positions[..., [0, 2]]
        full_body = FullBodyConstraintSet(
            model.skeleton,
            torch.arange(T),
            posed_joints,
            global_rot_mats,
            smooth_root_2d,
        )
        end_eff = EndEffectorConstraintSet(
            model.skeleton,
            torch.arange(T),
            posed_joints,
            global_rot_mats,
            smooth_root_2d,
            joint_names=["LeftHand", "RightHand", "LeftFoot", "RightFoot"],
        )
        constraint_lst.append([full_body, end_eff])

    return model(
        prompts=[prompt] * num_samples,
        num_frames=[T] * num_samples,
        num_denoising_steps=num_denoising_steps,
        constraint_lst=constraint_lst,
        cfg_weight=list(cfg_weight),
        num_samples=num_samples,
        post_processing=post_processing,
        return_numpy=False,
    )


def _fk(skeleton, local_rot_mats, root_positions):
    """Forward-kinematics helper. Delegates to the skeleton's FK if available;
    otherwise falls back to a minimal implementation."""
    if hasattr(skeleton, "forward_kinematics"):
        posed, global_rots = skeleton.forward_kinematics(local_rot_mats, root_positions)
        return posed, global_rots
    raise NotImplementedError(
        "skeleton.forward_kinematics is required for constraints mode. "
        "If your Kimodo skeleton does not expose FK, implement it or use mode='sdedit'."
    )


def _save_output(result: dict, output_path: str) -> None:
    """Write the motion dict to NPZ. Mirrors kimodo.scripts.generate output format."""
    from kimodo.exports.motion_io import save_kimodo_npz
    save_kimodo_npz(result, output_path)
