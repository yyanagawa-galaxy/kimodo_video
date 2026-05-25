# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the internal SMPL-X → SOMA-30 retarget helper.

These tests mock `py-soma-x` so they run without the SOMA-X package installed.
A real-integration test (marked @pytest.mark.slow) covers the live retargeter.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from kimodo.video._types import SMPLXMotion


def _fake_smplx(t: int = 10) -> SMPLXMotion:
    return SMPLXMotion(
        body_pose=torch.zeros(t, 21, 3, 3),
        global_orient=torch.zeros(t, 3, 3),
        transl=torch.zeros(t, 3),
        fps=30,
    )


def test_retarget_returns_soma_motion30_with_correct_shape():
    from kimodo.video import _smplx_to_soma

    # py-soma-x is mocked: pretend it returns the correct SOMA-30 tensors.
    fake_module = MagicMock()
    fake_module.retarget_smplx.return_value = {
        "local_rot_mats": torch.zeros(10, 30, 3, 3),
        "root_positions": torch.zeros(10, 3),
    }
    with patch.object(_smplx_to_soma, "_load_soma_x", return_value=fake_module):
        out = _smplx_to_soma.retarget(_fake_smplx(t=10))

    assert out.local_rot_mats.shape == (10, 30, 3, 3)
    assert out.root_positions.shape == (10, 3)
    assert out.fps == 30  # fps preserved from input


def test_retarget_wraps_soma_x_errors_in_retargeting_error():
    from kimodo.video import _smplx_to_soma
    from kimodo.video.errors import RetargetingError

    fake_module = MagicMock()
    fake_module.retarget_smplx.side_effect = RuntimeError("boom")
    with patch.object(_smplx_to_soma, "_load_soma_x", return_value=fake_module):
        with pytest.raises(RetargetingError, match="boom"):
            _smplx_to_soma.retarget(_fake_smplx())


def test_retarget_raises_when_soma_x_not_installed():
    from kimodo.video import _smplx_to_soma
    from kimodo.video.errors import EstimatorNotInstalledError

    with patch.object(_smplx_to_soma, "_load_soma_x", side_effect=ImportError("no module")):
        with pytest.raises(EstimatorNotInstalledError, match="py-soma-x"):
            _smplx_to_soma.retarget(_fake_smplx())
