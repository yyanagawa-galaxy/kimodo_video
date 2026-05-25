# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the MediaPipePoseEstimator adapter.

Real-mediapipe end-to-end is left to manual verification; these tests mock PyAV + mediapipe.
"""
from __future__ import annotations
from unittest.mock import MagicMock, patch
import pytest
import torch
import numpy as np

import kimodo.video.estimators  # noqa: F401 -- register estimators
from kimodo.video.errors import EstimatorNotInstalledError, NoMotionDetectedError
from kimodo.video.pose_estimator import get_pose_estimator
from kimodo.video.types import SOMAMotion30


def _make_fake_av(num_frames: int):
    """Build a MagicMock of the PyAV API surface the adapter uses."""
    fake_av = MagicMock()
    container = MagicMock()
    stream = MagicMock()
    stream.average_rate = 30.0
    container.streams.video = [stream]

    # The adapter does: for frame in container.decode(stream):
    # Each frame has .to_ndarray(format='rgb24')
    frames = []
    for _ in range(num_frames):
        fr = MagicMock()
        fr.to_ndarray.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
        frames.append(fr)
    container.decode.return_value = iter(frames)

    fake_av.open.return_value = container
    return fake_av


def test_mediapipe_is_registered():
    est = get_pose_estimator("mediapipe")
    assert est is not None


def test_mediapipe_raises_install_error_if_dep_missing():
    from kimodo.video.estimators import mediapipe as mp_adapter
    with patch.object(mp_adapter, "_load_mediapipe", side_effect=ImportError("no module")):
        est = mp_adapter.MediaPipePoseEstimator()
        with pytest.raises(EstimatorNotInstalledError, match="mediapipe"):
            est.estimate("any.mp4")


def test_mediapipe_raises_no_motion_when_no_landmarks(monkeypatch):
    from kimodo.video.estimators import mediapipe as mp_adapter

    monkeypatch.setattr(mp_adapter, "_load_av", lambda: _make_fake_av(num_frames=1))

    # Mock mediapipe so process() returns no landmarks
    fake_mp = MagicMock()
    pose_obj = MagicMock()
    result_no_landmarks = MagicMock()
    result_no_landmarks.pose_world_landmarks = None
    pose_obj.process.return_value = result_no_landmarks
    fake_mp.solutions.pose.Pose.return_value = pose_obj
    monkeypatch.setattr(mp_adapter, "_load_mediapipe", lambda: fake_mp)

    est = mp_adapter.MediaPipePoseEstimator()
    with pytest.raises(NoMotionDetectedError):
        est.estimate("any.mp4")


def test_mediapipe_returns_soma_motion30_with_correct_shapes(monkeypatch):
    """End-to-end with mocked PyAV + MediaPipe outputs."""
    from kimodo.video.estimators import mediapipe as mp_adapter

    monkeypatch.setattr(mp_adapter, "_load_av", lambda: _make_fake_av(num_frames=3))

    def fake_landmarks():
        result = MagicMock()
        landmark_list = []
        for i in range(33):
            lm = MagicMock()
            lm.x, lm.y, lm.z = float(i) * 0.01, 0.0, 0.0
            lm.visibility = 1.0
            landmark_list.append(lm)
        result.pose_world_landmarks = MagicMock()
        result.pose_world_landmarks.landmark = landmark_list
        return result

    fake_mp = MagicMock()
    pose_obj = MagicMock()
    pose_obj.process.side_effect = lambda f: fake_landmarks()
    fake_mp.solutions.pose.Pose.return_value = pose_obj
    monkeypatch.setattr(mp_adapter, "_load_mediapipe", lambda: fake_mp)

    est = mp_adapter.MediaPipePoseEstimator()
    out = est.estimate("any.mp4")
    assert isinstance(out, SOMAMotion30)
    assert out.local_rot_mats.shape == (3, 30, 3, 3)
    assert out.root_positions.shape == (3, 3)
    assert out.fps == 30
