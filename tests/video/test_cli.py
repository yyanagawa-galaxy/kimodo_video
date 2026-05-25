# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the kimodo_video_gen CLI."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest


def test_cli_parses_required_video_path_arg():
    from kimodo.scripts.video_to_motion import parse_args
    args = parse_args(["my_video.mp4"])
    assert args.video_path == "my_video.mp4"


def test_cli_defaults_match_python_api():
    from kimodo.scripts.video_to_motion import parse_args
    args = parse_args(["my_video.mp4"])
    assert args.strength == 0.5
    assert args.mode == "sdedit"
    assert args.estimator == "mediapipe"
    assert args.num_samples == 1
    assert args.diffusion_steps == 100
    assert args.prompt == ""


def test_cli_accepts_strength_and_mode_args():
    from kimodo.scripts.video_to_motion import parse_args
    args = parse_args(["my_video.mp4", "--strength", "0.7", "--mode", "constraints"])
    assert args.strength == 0.7
    assert args.mode == "constraints"


def test_cli_main_dispatches_to_video_to_motion(monkeypatch):
    fake_v2m = MagicMock(return_value={"posed_joints": object()})
    monkeypatch.setattr("kimodo.scripts.video_to_motion.video_to_motion", fake_v2m)

    from kimodo.scripts.video_to_motion import main
    main(["my_video.mp4", "--strength", "0.3"])
    assert fake_v2m.called
    _args, kwargs = fake_v2m.call_args
    assert kwargs["video_path"] == "my_video.mp4"
    assert kwargs["strength"] == 0.3
