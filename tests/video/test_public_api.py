# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verify the public API surface of kimodo.video matches the spec.

Also verifies SMPL-X internals are NOT importable from the public namespace.
"""
from __future__ import annotations

import pytest


def test_public_names_are_importable():
    from kimodo.video import (  # noqa: F401
        video_to_motion,
        SOMAMotion30,
        PoseEstimator,
        register_pose_estimator,
        get_pose_estimator,
        resample_to_fps,
    )


def test_smplx_motion_is_not_public():
    import kimodo.video
    assert not hasattr(kimodo.video, "SMPLXMotion"), \
        "SMPLXMotion must not appear in the public namespace"


def test_retarget_smplx_to_soma_is_not_public():
    import kimodo.video
    assert not hasattr(kimodo.video, "retarget_smplx_to_soma"), \
        "retarget_smplx_to_soma must not appear in the public namespace"


def test_kimodo_gen_cli_still_importable():
    """Smoke test: the existing CLI module must still import without the video extra installed."""
    from kimodo.scripts import generate  # noqa: F401


def test_kimodo_base_import_does_not_load_video():
    """Importing kimodo (the package) must not transitively import kimodo.video,
    which would defeat the lazy-import design and pull in optional deps."""
    import sys
    # Clear modules so the test is independent of import order in this session.
    for mod in list(sys.modules):
        if mod.startswith("kimodo"):
            del sys.modules[mod]
    import kimodo  # noqa: F401
    assert "kimodo.video" not in sys.modules, \
        "kimodo.video was imported transitively from `import kimodo` (lazy-import violated)"
