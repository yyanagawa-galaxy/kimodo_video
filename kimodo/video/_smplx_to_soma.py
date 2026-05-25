# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Private SMPL-X → SOMA-30 retarget helper.

Used by SMPL-X-based PoseEstimator adapters (GVHMR, WHAM, 4DHumans).
Wraps the `py-soma-x` package (installed via `pip install kimodo[soma]`).

NOT part of the public API. Callers should always go through a PoseEstimator
adapter, which returns SOMAMotion30 directly.
"""
from __future__ import annotations

from types import ModuleType

import torch

from ._types import SMPLXMotion
from .errors import EstimatorNotInstalledError, RetargetingError
from .types import SOMAMotion30


def _load_soma_x() -> ModuleType:
    """Lazy import of py-soma-x. Wrapped for monkeypatching in tests."""
    import py_soma_x  # type: ignore[import-not-found]
    return py_soma_x


def retarget(smplx: SMPLXMotion) -> SOMAMotion30:
    """Retarget SMPL-X motion to SOMA-30.

    Raises:
        EstimatorNotInstalledError: if py-soma-x is not installed.
        RetargetingError: if py-soma-x fails for any reason.
    """
    try:
        soma_x = _load_soma_x()
    except ImportError as exc:
        raise EstimatorNotInstalledError(
            "py-soma-x",
            install_url="https://github.com/NVlabs/SOMA-X (install via `pip install kimodo[soma]`)",
        ) from exc

    try:
        out = soma_x.retarget_smplx(
            body_pose=smplx.body_pose,
            global_orient=smplx.global_orient,
            transl=smplx.transl,
        )
    except Exception as exc:
        raise RetargetingError(str(exc)) from exc

    return SOMAMotion30(
        local_rot_mats=out["local_rot_mats"],
        root_positions=out["root_positions"],
        fps=smplx.fps,
    )
