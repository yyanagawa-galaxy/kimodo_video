# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for typed pipeline exceptions."""
from __future__ import annotations

import pytest

from kimodo.video.errors import (
    EstimatorNotInstalledError,
    InvalidStrengthError,
    MultiplePersonsDetectedError,
    NoMotionDetectedError,
    PipelineError,
    RetargetingError,
    VideoTooShortError,
)


def test_all_inherit_from_pipeline_error():
    for cls in [
        EstimatorNotInstalledError, NoMotionDetectedError,
        MultiplePersonsDetectedError, VideoTooShortError,
        RetargetingError, InvalidStrengthError,
    ]:
        assert issubclass(cls, PipelineError)


def test_estimator_not_installed_carries_install_hint():
    err = EstimatorNotInstalledError("gvhmr", install_url="https://github.com/example/gvhmr")
    assert "gvhmr" in str(err)
    assert "https://github.com/example/gvhmr" in str(err)


def test_invalid_strength_carries_actual_value():
    err = InvalidStrengthError(1.5)
    assert "1.5" in str(err)
    assert "[0, 1]" in str(err)


def test_video_too_short_carries_actual_and_minimum():
    err = VideoTooShortError(num_frames=3, min_frames=10)
    assert "3" in str(err)
    assert "10" in str(err)
