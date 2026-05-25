# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Typed exceptions for the video-input pipeline.

All errors inherit from PipelineError so callers can catch the umbrella class.
"""
from __future__ import annotations


class PipelineError(Exception):
    """Base class for all kimodo.video pipeline errors."""


class EstimatorNotInstalledError(PipelineError):
    """Raised when a PoseEstimator adapter's upstream dependency cannot be imported."""

    def __init__(self, estimator_name: str, install_url: str):
        super().__init__(
            f"Pose estimator {estimator_name!r} is not installed. "
            f"Install it from {install_url} and try again."
        )
        self.estimator_name = estimator_name
        self.install_url = install_url


class NoMotionDetectedError(PipelineError):
    """Raised when the pose estimator returns zero usable frames."""

    def __init__(self, video_path: str):
        super().__init__(
            f"No person detected in {video_path!r}. "
            f"Check video quality, ensure a single person is clearly visible, "
            f"or try --person_idx to select a specific detection."
        )
        self.video_path = video_path


class MultiplePersonsDetectedError(PipelineError):
    """Raised when multiple people are detected and person_idx is ambiguous."""

    def __init__(self, num_persons: int):
        super().__init__(
            f"{num_persons} people detected. Re-run with --person_idx <0..{num_persons - 1}> "
            f"to select which one to track."
        )
        self.num_persons = num_persons


class VideoTooShortError(PipelineError):
    """Raised when the video has fewer frames than the pipeline minimum."""

    def __init__(self, num_frames: int, min_frames: int):
        super().__init__(
            f"Video too short: {num_frames} frames at target fps, "
            f"need at least {min_frames}."
        )
        self.num_frames = num_frames
        self.min_frames = min_frames


class RetargetingError(PipelineError):
    """Raised when the SMPL-X → SOMA retargeter fails."""


class InvalidStrengthError(PipelineError):
    """Raised when SDEdit strength is out of [0, 1]."""

    def __init__(self, value: float):
        super().__init__(f"Invalid strength: {value}. Must be in [0, 1].")
        self.value = value
