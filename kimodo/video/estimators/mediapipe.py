# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MediaPipe Pose Landmarker adapter -- commercial-clean pose estimator path.

No SMPL involvement. Outputs SOMA-30 directly via inverse kinematics.
"""
from __future__ import annotations
from types import ModuleType
from typing import List
import torch

from .._ik_fitter import fit_soma30_from_positions
from .._mediapipe_correspondence import synthesize_soma30_positions
from ..errors import EstimatorNotInstalledError, NoMotionDetectedError
from ..pose_estimator import register_pose_estimator
from ..types import SOMAMotion30

_MEDIAPIPE_INSTALL_URL = "https://pypi.org/project/mediapipe/  (pip install mediapipe)"


def _load_mediapipe() -> ModuleType:
    """Lazy import. Wrapped so tests can mock."""
    import mediapipe as mp
    return mp


def _load_av() -> ModuleType:
    """Lazy import of PyAV (bundled with kimodo). Wrapped so tests can mock."""
    import av
    return av


class MediaPipePoseEstimator:
    """Pose estimator using MediaPipe's built-in Pose solution.

    Uses world-space landmarks (metric units, root at hips) and maps them
    to SOMA-30 local rotation matrices via IK.
    """

    def estimate(self, video_path: str, *, person_idx: int = 0) -> SOMAMotion30:
        """Estimate SOMA-30 motion from a video file.

        Args:
            video_path: Path to the input video file.
            person_idx: Ignored (MediaPipe single-person; kept for API compatibility).

        Returns:
            SOMAMotion30 with local_rot_mats [T, 30, 3, 3], root_positions [T, 3], fps.

        Raises:
            EstimatorNotInstalledError: If mediapipe or cv2 is not installed.
            NoMotionDetectedError: If no pose landmarks are detected in the video.
        """
        try:
            mp = _load_mediapipe()
            av = _load_av()
        except ImportError as exc:
            raise EstimatorNotInstalledError("mediapipe", install_url=_MEDIAPIPE_INSTALL_URL) from exc

        container = av.open(video_path)
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else 30.0
        try:
            pose = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                enable_segmentation=False,
            )
            world_landmarks_per_frame: List[List[List[float]]] = []
            for frame in container.decode(stream):
                frame_rgb = frame.to_ndarray(format="rgb24")
                result = pose.process(frame_rgb)
                if result.pose_world_landmarks is None:
                    continue
                world_landmarks_per_frame.append(
                    [[lm.x, lm.y, lm.z] for lm in result.pose_world_landmarks.landmark]
                )
            pose.close()
        finally:
            container.close()

        if not world_landmarks_per_frame:
            raise NoMotionDetectedError(video_path)

        lm_tensor = torch.tensor(world_landmarks_per_frame, dtype=torch.float32)  # [T, 33, 3]
        soma_positions = synthesize_soma30_positions(lm_tensor)                   # [T, 30, 3]
        local_rot_mats, root_positions = fit_soma30_from_positions(soma_positions)

        return SOMAMotion30(
            local_rot_mats=local_rot_mats,
            root_positions=root_positions,
            fps=int(round(fps)),
        )


@register_pose_estimator("mediapipe")
def _factory() -> MediaPipePoseEstimator:
    return MediaPipePoseEstimator()
