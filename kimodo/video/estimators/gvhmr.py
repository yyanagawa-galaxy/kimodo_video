# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GVHMR PoseEstimator adapter.

Uses SMPL-X internally and retargets to SOMA before returning. SMPL-X never
leaves this adapter.

GVHMR is research code; install it separately per its README:
    https://github.com/zju3dv/GVHMR

The first call to estimate() triggers `import gvhmr`. If it fails, the
adapter raises EstimatorNotInstalledError with the install URL.
"""
from __future__ import annotations

from types import ModuleType

from .._smplx_to_soma import retarget as smplx_to_soma_retarget
from .._types import SMPLXMotion
from ..errors import EstimatorNotInstalledError, NoMotionDetectedError
from ..pose_estimator import register_pose_estimator
from ..types import SOMAMotion30

_GVHMR_INSTALL_URL = "https://github.com/zju3dv/GVHMR"


def _load_gvhmr() -> ModuleType:
    """Lazy import of GVHMR. Wrapped for monkeypatching in tests."""
    import gvhmr  # type: ignore[import-not-found]
    return gvhmr


class GVHMRAdapter:
    """Adapter around the upstream GVHMR package."""

    def estimate(self, video_path: str, *, person_idx: int = 0) -> SOMAMotion30:
        try:
            gvhmr = _load_gvhmr()
        except ImportError as exc:
            raise EstimatorNotInstalledError(
                "gvhmr", install_url=_GVHMR_INSTALL_URL
            ) from exc

        # Note: the actual GVHMR API may differ; the adapter assumes
        # gvhmr.run(video_path, person_idx=...) -> dict with SMPL-X tensors.
        # Adjust to match the live API as needed.
        result = gvhmr.run(video_path, person_idx=person_idx)

        smplx = SMPLXMotion(
            body_pose=result["body_pose"],
            global_orient=result["global_orient"],
            transl=result["transl"],
            fps=int(result["fps"]),
        )

        if smplx.num_frames == 0:
            raise NoMotionDetectedError(video_path)

        return smplx_to_soma_retarget(smplx)


@register_pose_estimator("gvhmr")
def _factory() -> GVHMRAdapter:
    return GVHMRAdapter()
