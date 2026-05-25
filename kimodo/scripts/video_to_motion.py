# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CLI: kimodo_video_gen — drop a video, get clean SOMA motion."""
from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from kimodo.video import video_to_motion


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="kimodo_video_gen",
        description="Generate clean SOMA motion from a video using Kimodo's diffusion prior.",
    )
    p.add_argument("video_path", help="Path to the input video file")
    p.add_argument("--output", default="video_output",
                   help='Output NPZ stem (default: "video_output")')
    p.add_argument("--prompt", default="",
                   help="Optional text steering (default: empty)")
    p.add_argument("--strength", type=float, default=0.5,
                   help="SDEdit cleanup strength in [0,1] (default: 0.5)")
    p.add_argument("--mode", choices=["sdedit", "constraints"], default="sdedit",
                   help="Pipeline mode (default: sdedit)")
    p.add_argument("--model", default="Kimodo-SOMA-RP-v1.1",
                   help="Kimodo model name (default: Kimodo-SOMA-RP-v1.1)")
    p.add_argument("--estimator", default="mediapipe",
                   help="Pose estimator (default: mediapipe)")
    p.add_argument("--diffusion_steps", type=int, default=100,
                   help="DDIM steps (default: 100)")
    p.add_argument("--num_samples", type=int, default=1,
                   help="Number of samples (default: 1)")
    p.add_argument("--cfg_weight", nargs=2, type=float, default=[2.0, 0.5],
                   metavar=("TEXT", "CONSTRAINT"),
                   help="CFG weights; constraint channel only honored in constraints mode (default: 2.0 0.5)")
    p.add_argument("--no-postprocess", dest="post_processing", action="store_false",
                   default=True, help="Disable foot-skate cleanup")
    p.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    p.add_argument("--save_intermediates", action="store_true",
                   help="Save adapter intermediates alongside output")
    p.add_argument("--person_idx", type=int, default=0,
                   help="If multiple people detected, pick this index (default: 0)")
    p.add_argument("--chunk_size", type=int, default=None,
                   help="Override default chunk size (default: model.max_frames)")
    p.add_argument("--chunk_overlap", type=int, default=None,
                   help="Override default overlap in frames (default: 1.0s at model.fps)")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    video_to_motion(
        video_path=args.video_path,
        output_path=args.output,
        model_name=args.model,
        prompt=args.prompt,
        strength=args.strength,
        mode=args.mode,
        estimator=args.estimator,
        num_denoising_steps=args.diffusion_steps,
        num_samples=args.num_samples,
        cfg_weight=(args.cfg_weight[0], args.cfg_weight[1]),
        post_processing=args.post_processing,
        seed=args.seed,
        person_idx=args.person_idx,
        save_intermediates=args.save_intermediates,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
