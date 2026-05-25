# Video Input Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a video-input pipeline to Kimodo (`kimodo_video_gen` CLI + `kimodo.video.video_to_motion()` Python API) that takes a video, extracts motion via a pluggable pose estimator, and denoises it through Kimodo's diffusion prior (SDEdit-style) to produce clean SOMA-77 motion. Existing Kimodo features must remain bit-identical.

**Architecture:** New `kimodo/video/` package, opt-in via `pip install kimodo[video]`. Every `PoseEstimator` adapter returns `SOMAMotion30` (SMPL-X is private to specific adapters). The only existing file touched is `kimodo/model/kimodo_model.py` — an 8-line gated change to `_generate` adding optional `init_motion` and `strength` kwargs for the SDEdit init path.

**Tech Stack:** Python ≥3.8 (use `from __future__ import annotations` for PEP 604 typing), PyTorch, NumPy, argparse, pytest, unittest.mock. Lazy imports for GVHMR / py-soma-x. Line length 120 to match existing ruff config.

**Spec reference:** [`docs/superpowers/specs/2026-05-25-video-input-pipeline-design.md`](../specs/2026-05-25-video-input-pipeline-design.md)

**Commit strategy:** The implementer should create a feature branch first (`git checkout -b feat/video-input-pipeline`) so commits don't clutter the upstream main clone. Every task ends with a commit step; the implementer can squash before merging upstream if desired.

---

## File Map

### New files (in `kimodo/video/`)

| File | Responsibility |
|---|---|
| `kimodo/video/__init__.py` | Public API exports |
| `kimodo/video/types.py` | Public `SOMAMotion30` dataclass |
| `kimodo/video/_types.py` | Internal `SMPLXMotion` dataclass (for SMPL-X-based adapters) |
| `kimodo/video/pose_estimator.py` | `PoseEstimator` Protocol + registry |
| `kimodo/video/errors.py` | Typed exceptions |
| `kimodo/video/_smplx_to_soma.py` | Internal SMPL-X → SOMA-30 retarget helper (wraps `py-soma-x`) |
| `kimodo/video/resample.py` | Frame-rate resampling utility |
| `kimodo/video/sdedit.py` | SDEdit init helper (pure function over `Diffusion`) |
| `kimodo/video/chunking.py` | Overlap-blend windowing for long videos |
| `kimodo/video/pipeline.py` | `video_to_motion` orchestrator (SDEdit + constraints modes) |
| `kimodo/video/estimators/__init__.py` | Registers all bundled adapters |
| `kimodo/video/estimators/gvhmr.py` | GVHMR adapter (lazy import, SMPL-X internal) |
| `kimodo/video/estimators/gem.py` | Stub adapter (raises `NotImplementedError`) |
| `kimodo/video/estimators/wham.py` | Stub adapter |
| `kimodo/video/estimators/fourdhumans.py` | Stub adapter |
| `kimodo/scripts/video_to_motion.py` | `kimodo_video_gen` CLI entry point |

### Modified files

| File | Change |
|---|---|
| `kimodo/model/kimodo_model.py` | `_generate` gains two optional kwargs: `init_motion`, `strength`. ~8 gated lines. Existing path byte-identical when `init_motion is None`. |
| `pyproject.toml` | Add `[project.optional-dependencies] video = [...]`, add `kimodo_video_gen` to `[project.scripts]`. |

### New tests (in `tests/video/`)

| File | Coverage |
|---|---|
| `tests/__init__.py`, `tests/video/__init__.py`, `tests/conftest.py` | Test package skeleton + shared fixtures |
| `tests/video/test_types.py` | `SOMAMotion30`, `SMPLXMotion` dataclass shapes |
| `tests/video/test_pose_estimator_registry.py` | Registry add / lookup / error |
| `tests/video/test_errors.py` | Typed exceptions are raisable and carry context |
| `tests/video/test_smplx_to_soma_shapes.py` | SMPL-X → SOMA-30 shape transformation |
| `tests/video/test_resample.py` | Frame-rate conversion math |
| `tests/video/test_sdedit_init.py` | `sdedit_init` shape + strength bounds |
| `tests/video/test_chunking.py` | Chunk + stitch round-trip |
| `tests/video/test_regression_existing_path.py` | **Backward-compat guard.** `_generate` with `init_motion=None` calls `torch.randn` and uses full denoising range. |
| `tests/video/test_generate_sdedit.py` | New SDEdit path: when `init_motion` is provided, `_generate` calls `q_sample` and uses truncated indices. |
| `tests/video/test_estimator_stubs.py` | gem/wham/4dhumans stubs raise `NotImplementedError` cleanly |
| `tests/video/test_gvhmr_adapter.py` | GVHMR adapter raises `EstimatorNotInstalledError` if `gvhmr` missing; correctly retargets SMPL-X output if mocked. |
| `tests/video/test_pipeline_sdedit.py` | `video_to_motion(mode="sdedit")` end-to-end with mocked estimator |
| `tests/video/test_pipeline_constraints.py` | `video_to_motion(mode="constraints")` end-to-end with mocked estimator |
| `tests/video/test_pipeline_chunking.py` | Long-input pipeline auto-chunks |
| `tests/video/test_cli.py` | CLI arg parsing + dispatch |
| `tests/video/test_public_api.py` | `from kimodo.video import ...` of all public names works; SMPL-X internals are NOT importable from public namespace |

---

## Task 1: Package skeleton + motion type dataclasses

**Files:**
- Create: `kimodo/video/__init__.py`
- Create: `kimodo/video/types.py`
- Create: `kimodo/video/_types.py`
- Create: `tests/__init__.py`
- Create: `tests/video/__init__.py`
- Create: `tests/conftest.py`
- Create: `tests/video/test_types.py`

- [ ] **Step 1: Create test scaffolding**

Create empty file `tests/__init__.py` (just so pytest can discover):

```python
```

Create empty file `tests/video/__init__.py`:

```python
```

Create `tests/conftest.py` (will hold shared fixtures later; empty for now):

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
```

- [ ] **Step 2: Write failing test for `SOMAMotion30` and `SMPLXMotion`**

Create `tests/video/test_types.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for kimodo.video motion type dataclasses."""
from __future__ import annotations

import pytest
import torch


def test_soma_motion30_holds_required_fields():
    from kimodo.video.types import SOMAMotion30
    m = SOMAMotion30(
        local_rot_mats=torch.zeros(10, 30, 3, 3),
        root_positions=torch.zeros(10, 3),
        fps=20,
    )
    assert m.local_rot_mats.shape == (10, 30, 3, 3)
    assert m.root_positions.shape == (10, 3)
    assert m.fps == 20


def test_soma_motion30_num_frames_property():
    from kimodo.video.types import SOMAMotion30
    m = SOMAMotion30(
        local_rot_mats=torch.zeros(7, 30, 3, 3),
        root_positions=torch.zeros(7, 3),
        fps=20,
    )
    assert m.num_frames == 7


def test_smplx_motion_holds_required_fields():
    from kimodo.video._types import SMPLXMotion
    m = SMPLXMotion(
        body_pose=torch.zeros(10, 21, 3, 3),
        global_orient=torch.zeros(10, 3, 3),
        transl=torch.zeros(10, 3),
        fps=30,
    )
    assert m.body_pose.shape == (10, 21, 3, 3)
    assert m.fps == 30


def test_smplx_motion_is_not_in_public_namespace():
    """SMPLXMotion is intentionally private — must not be importable from kimodo.video."""
    import kimodo.video
    assert not hasattr(kimodo.video, "SMPLXMotion"), \
        "SMPLXMotion leaked into the public namespace"
```

- [ ] **Step 3: Run test to verify it fails**

```bash
pytest tests/video/test_types.py -v
```

Expected: ImportError / ModuleNotFoundError for `kimodo.video.types` or `kimodo.video._types`.

- [ ] **Step 4: Create the package and dataclasses**

Create `kimodo/video/__init__.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Video-input pipeline for Kimodo.

Public API:
    video_to_motion: drop a video, get clean SOMA-77 motion.
    SOMAMotion30:   public motion type produced by all PoseEstimators.
    PoseEstimator, register_pose_estimator, get_pose_estimator: estimator registry.
    resample_to_fps: frame-rate utility.

This package is opt-in: install with `pip install kimodo[video]`.
"""
from __future__ import annotations

from .types import SOMAMotion30

__all__ = ["SOMAMotion30"]
```

(More names added in later tasks. Each task that adds a public name will update `__all__` and the imports here.)

Create `kimodo/video/types.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public motion types for the video-input pipeline."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SOMAMotion30:
    """Motion on the SOMA-30 skeleton — the pipeline's universal currency.

    Every PoseEstimator returns this type. SMPL-X and other intermediates,
    if used, are private to specific adapters.

    Attributes:
        local_rot_mats: Local (parent-relative) joint rotations, shape [T, 30, 3, 3].
        root_positions: Root joint world positions, shape [T, 3].
        fps:            Frame rate of this motion.
    """
    local_rot_mats: torch.Tensor
    root_positions: torch.Tensor
    fps: int

    @property
    def num_frames(self) -> int:
        return int(self.local_rot_mats.shape[0])
```

Create `kimodo/video/_types.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Internal motion types used only inside specific PoseEstimator adapters.

NOT part of the public API. SMPL-X carries a non-commercial license restriction
and is intentionally hidden from the public namespace to avoid implying it is
the pipeline's canonical representation.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SMPLXMotion:
    """Motion in SMPL-X parameters — emitted by GVHMR/WHAM/4DHumans adapters
    internally, immediately retargeted to SOMA-30 before leaving the adapter.

    Attributes:
        body_pose:     Per-frame body joint rotations, shape [T, 21, 3, 3].
        global_orient: Per-frame root rotation, shape [T, 3, 3].
        transl:        Per-frame root translation in world frame, shape [T, 3].
        fps:           Frame rate.
    """
    body_pose: torch.Tensor
    global_orient: torch.Tensor
    transl: torch.Tensor
    fps: int

    @property
    def num_frames(self) -> int:
        return int(self.body_pose.shape[0])
```

- [ ] **Step 5: Run tests, verify pass**

```bash
pytest tests/video/test_types.py -v
```

Expected: all 4 tests pass.

- [ ] **Step 6: Commit**

```bash
git add kimodo/video/__init__.py kimodo/video/types.py kimodo/video/_types.py \
        tests/__init__.py tests/video/__init__.py tests/conftest.py tests/video/test_types.py
git commit -m "video: add package skeleton + SOMA/SMPL-X motion dataclasses"
```

---

## Task 2: PoseEstimator protocol + registry

**Files:**
- Create: `kimodo/video/pose_estimator.py`
- Modify: `kimodo/video/__init__.py`
- Create: `tests/video/test_pose_estimator_registry.py`

- [ ] **Step 1: Write failing test**

Create `tests/video/test_pose_estimator_registry.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the PoseEstimator registry."""
from __future__ import annotations

import pytest
import torch

from kimodo.video.pose_estimator import (
    PoseEstimator,
    _REGISTRY,
    get_pose_estimator,
    register_pose_estimator,
)
from kimodo.video.types import SOMAMotion30


@pytest.fixture(autouse=True)
def isolate_registry():
    """Snapshot and restore the registry around each test."""
    snapshot = dict(_REGISTRY)
    yield
    _REGISTRY.clear()
    _REGISTRY.update(snapshot)


def _make_fake() -> PoseEstimator:
    class Fake:
        def estimate(self, video_path: str, *, person_idx: int = 0) -> SOMAMotion30:
            return SOMAMotion30(
                local_rot_mats=torch.zeros(5, 30, 3, 3),
                root_positions=torch.zeros(5, 3),
                fps=30,
            )
    return Fake()


def test_register_and_get():
    @register_pose_estimator("fake")
    def _factory():
        return _make_fake()

    est = get_pose_estimator("fake")
    motion = est.estimate("nonexistent.mp4")
    assert motion.num_frames == 5
    assert motion.fps == 30


def test_get_unknown_raises_value_error():
    with pytest.raises(ValueError, match="Unknown estimator"):
        get_pose_estimator("does-not-exist")


def test_factory_is_called_lazily_per_get():
    calls = []

    @register_pose_estimator("counting")
    def _factory():
        calls.append(1)
        return _make_fake()

    # Registration should NOT call the factory
    assert calls == []
    get_pose_estimator("counting")
    get_pose_estimator("counting")
    assert len(calls) == 2  # factory runs on every get
```

- [ ] **Step 2: Run test, verify it fails**

```bash
pytest tests/video/test_pose_estimator_registry.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement the registry**

Create `kimodo/video/pose_estimator.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pluggable PoseEstimator protocol and registry.

Every estimator adapter returns SOMAMotion30. Body-model intermediates
(SMPL-X etc.) are private to specific adapters and never leak through this
interface.
"""
from __future__ import annotations

from typing import Callable, Dict, Protocol

from .types import SOMAMotion30


class PoseEstimator(Protocol):
    """Adapter that turns a video into SOMA-30 motion."""

    def estimate(self, video_path: str, *, person_idx: int = 0) -> SOMAMotion30:
        ...


_REGISTRY: Dict[str, Callable[[], PoseEstimator]] = {}


def register_pose_estimator(name: str) -> Callable[[Callable[[], PoseEstimator]], Callable[[], PoseEstimator]]:
    """Decorator that adds a factory to the registry under `name`."""
    def deco(factory: Callable[[], PoseEstimator]) -> Callable[[], PoseEstimator]:
        _REGISTRY[name] = factory
        return factory
    return deco


def get_pose_estimator(name: str) -> PoseEstimator:
    """Build a fresh PoseEstimator instance by registered name."""
    if name not in _REGISTRY:
        raise ValueError(f"Unknown estimator {name!r}. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]()
```

- [ ] **Step 4: Expose in public API**

Edit `kimodo/video/__init__.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Video-input pipeline for Kimodo.

Public API:
    video_to_motion: drop a video, get clean SOMA-77 motion.
    SOMAMotion30:   public motion type produced by all PoseEstimators.
    PoseEstimator, register_pose_estimator, get_pose_estimator: estimator registry.
    resample_to_fps: frame-rate utility.

This package is opt-in: install with `pip install kimodo[video]`.
"""
from __future__ import annotations

from .pose_estimator import PoseEstimator, get_pose_estimator, register_pose_estimator
from .types import SOMAMotion30

__all__ = [
    "SOMAMotion30",
    "PoseEstimator",
    "register_pose_estimator",
    "get_pose_estimator",
]
```

- [ ] **Step 5: Run tests, verify pass**

```bash
pytest tests/video/test_pose_estimator_registry.py tests/video/test_types.py -v
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add kimodo/video/pose_estimator.py kimodo/video/__init__.py tests/video/test_pose_estimator_registry.py
git commit -m "video: add PoseEstimator protocol + lazy-loaded registry"
```

---

## Task 3: Typed exceptions

**Files:**
- Create: `kimodo/video/errors.py`
- Create: `tests/video/test_errors.py`

- [ ] **Step 1: Write failing test**

Create `tests/video/test_errors.py`:

```python
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
```

- [ ] **Step 2: Run test, verify it fails**

```bash
pytest tests/video/test_errors.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement errors**

Create `kimodo/video/errors.py`:

```python
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
```

- [ ] **Step 4: Run tests, verify pass**

```bash
pytest tests/video/test_errors.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add kimodo/video/errors.py tests/video/test_errors.py
git commit -m "video: add typed pipeline exceptions"
```

---

## Task 4: SMPL-X → SOMA retarget helper (private)

**Files:**
- Create: `kimodo/video/_smplx_to_soma.py`
- Create: `tests/video/test_smplx_to_soma_shapes.py`

This helper is internal — only SMPL-X-based adapters use it. It delegates to the `py-soma-x` package (the `soma` extra in `pyproject.toml`).

- [ ] **Step 1: Write failing test**

Create `tests/video/test_smplx_to_soma_shapes.py`:

```python
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
```

- [ ] **Step 2: Run test, verify it fails**

```bash
pytest tests/video/test_smplx_to_soma_shapes.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement the helper**

Create `kimodo/video/_smplx_to_soma.py`:

```python
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
```

Note: the exact `py-soma-x` API (`retarget_smplx`, return dict keys) is assumed here based on a typical SOMA-X interface. If the actual API differs, adjust the call sites (and tests) to match. Refer to the SOMA-X docs at https://github.com/NVlabs/SOMA-X for the live signature.

- [ ] **Step 4: Run tests, verify pass**

```bash
pytest tests/video/test_smplx_to_soma_shapes.py -v
```

Expected: 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add kimodo/video/_smplx_to_soma.py tests/video/test_smplx_to_soma_shapes.py
git commit -m "video: add internal SMPL-X to SOMA-30 retarget helper"
```

---

## Task 5: Frame-rate resampling utility

**Files:**
- Create: `kimodo/video/resample.py`
- Modify: `kimodo/video/__init__.py`
- Create: `tests/video/test_resample.py`

- [ ] **Step 1: Write failing test**

Create `tests/video/test_resample.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for frame-rate resampling utility."""
from __future__ import annotations

import pytest
import torch

from kimodo.video.resample import resample_to_fps
from kimodo.video.types import SOMAMotion30


def _ramp(t: int, fps: int) -> SOMAMotion30:
    # Use a monotonically increasing root.x to make resampling visible.
    rot = torch.eye(3).expand(t, 30, 3, 3).clone()
    root = torch.zeros(t, 3)
    root[:, 0] = torch.linspace(0, 1, t)
    return SOMAMotion30(local_rot_mats=rot, root_positions=root, fps=fps)


def test_resample_same_fps_is_identity():
    m = _ramp(60, fps=30)
    out = resample_to_fps(m, target_fps=30)
    assert out.fps == 30
    assert out.num_frames == 60
    assert torch.allclose(out.root_positions, m.root_positions)


def test_resample_30_to_20_preserves_duration_within_one_frame():
    # 60 frames @ 30 fps = 2.0 s → at 20 fps that's 40 frames
    m = _ramp(60, fps=30)
    out = resample_to_fps(m, target_fps=20)
    assert out.fps == 20
    assert abs(out.num_frames - 40) <= 1


def test_resample_20_to_30_upsamples():
    m = _ramp(40, fps=20)
    out = resample_to_fps(m, target_fps=30)
    assert out.fps == 30
    assert abs(out.num_frames - 60) <= 1


def test_resample_endpoints_match():
    m = _ramp(60, fps=30)
    out = resample_to_fps(m, target_fps=20)
    assert torch.allclose(out.root_positions[0], m.root_positions[0])
    assert torch.allclose(out.root_positions[-1], m.root_positions[-1], atol=1e-5)
```

- [ ] **Step 2: Run test, verify it fails**

```bash
pytest tests/video/test_resample.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement resampling**

Create `kimodo/video/resample.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Frame-rate resampling for SOMA motion.

Linear interpolation on positions; spherical-linear interpolation (slerp) is
out of scope for v1 — rotations are interpolated as flat 9-vectors and
re-orthogonalized only if downstream needs it (the diffusion prior cleans
up any small drift).
"""
from __future__ import annotations

import torch

from .types import SOMAMotion30


def resample_to_fps(motion: SOMAMotion30, target_fps: int) -> SOMAMotion30:
    """Resample motion from its current fps to target_fps via linear interpolation.

    If `target_fps == motion.fps`, returns the input unchanged.

    Args:
        motion:     Input motion.
        target_fps: Desired output frame rate.

    Returns:
        A new SOMAMotion30 at target_fps. The duration is preserved within
        one-frame rounding.
    """
    if target_fps == motion.fps:
        return motion

    t_in = motion.num_frames
    if t_in < 2:
        # Trivial: a single frame can't be meaningfully resampled.
        return SOMAMotion30(
            local_rot_mats=motion.local_rot_mats,
            root_positions=motion.root_positions,
            fps=target_fps,
        )

    duration_s = (t_in - 1) / motion.fps
    t_out = max(1, round(duration_s * target_fps) + 1)

    src_times = torch.linspace(0, duration_s, t_in)
    tgt_times = torch.linspace(0, duration_s, t_out)

    # Linear interp on root positions.
    root = _interp_1d(motion.root_positions, src_times, tgt_times)

    # Linear interp on flattened rotation matrices.
    rot_flat = motion.local_rot_mats.reshape(t_in, -1)
    rot_out_flat = _interp_1d(rot_flat, src_times, tgt_times)
    rot_out = rot_out_flat.reshape(t_out, 30, 3, 3)

    return SOMAMotion30(local_rot_mats=rot_out, root_positions=root, fps=target_fps)


def _interp_1d(values: torch.Tensor, src_t: torch.Tensor, tgt_t: torch.Tensor) -> torch.Tensor:
    """Linear interp along axis 0. `values` has shape [T_in, ...]."""
    t_in = src_t.shape[0]
    # For each target time, find the bracketing source indices.
    # src_t is monotonic over [0, duration_s], so we can use searchsorted.
    idx_hi = torch.searchsorted(src_t, tgt_t, right=False).clamp(1, t_in - 1)
    idx_lo = idx_hi - 1
    t_lo = src_t[idx_lo]
    t_hi = src_t[idx_hi]
    w = ((tgt_t - t_lo) / (t_hi - t_lo).clamp(min=1e-12)).clamp(0, 1)
    # Broadcast w across the trailing dims of values.
    while w.ndim < values.ndim:
        w = w.unsqueeze(-1)
    return (1 - w) * values[idx_lo] + w * values[idx_hi]
```

- [ ] **Step 4: Wire into public API**

Edit `kimodo/video/__init__.py` to add `resample_to_fps`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Video-input pipeline for Kimodo.

Public API:
    video_to_motion: drop a video, get clean SOMA-77 motion.
    SOMAMotion30:   public motion type produced by all PoseEstimators.
    PoseEstimator, register_pose_estimator, get_pose_estimator: estimator registry.
    resample_to_fps: frame-rate utility.

This package is opt-in: install with `pip install kimodo[video]`.
"""
from __future__ import annotations

from .pose_estimator import PoseEstimator, get_pose_estimator, register_pose_estimator
from .resample import resample_to_fps
from .types import SOMAMotion30

__all__ = [
    "SOMAMotion30",
    "PoseEstimator",
    "register_pose_estimator",
    "get_pose_estimator",
    "resample_to_fps",
]
```

- [ ] **Step 5: Run tests, verify pass**

```bash
pytest tests/video/ -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add kimodo/video/resample.py kimodo/video/__init__.py tests/video/test_resample.py
git commit -m "video: add frame-rate resampling utility"
```

---

## Task 6: SDEdit init helper

**Files:**
- Create: `kimodo/video/sdedit.py`
- Create: `tests/video/test_sdedit_init.py`

- [ ] **Step 1: Write failing test**

Create `tests/video/test_sdedit_init.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for SDEdit init helper."""
from __future__ import annotations

import pytest
import torch

from kimodo.model.diffusion import Diffusion
from kimodo.video.errors import InvalidStrengthError
from kimodo.video.sdedit import sdedit_init


@pytest.fixture
def diffusion():
    return Diffusion(num_base_steps=100)


def test_strength_zero_returns_input_with_step_zero(diffusion):
    x = torch.randn(2, 30, 64)
    noisy, k = sdedit_init(x, strength=0.0, num_denoising_steps=100, diffusion=diffusion)
    assert k == 1  # we clamp k to at least 1 so the loop runs once
    # At step 0 (clamped to 1), q_sample adds minimal noise. Tensor stays close to input.
    assert noisy.shape == x.shape


def test_strength_one_returns_pure_noise_like_shape(diffusion):
    x = torch.zeros(2, 30, 64)
    noisy, k = sdedit_init(x, strength=1.0, num_denoising_steps=100, diffusion=diffusion)
    assert k == 100
    assert noisy.shape == x.shape
    # With zero input and full noise, the output should NOT be all zeros.
    assert noisy.abs().sum() > 0


def test_strength_half_uses_midpoint(diffusion):
    x = torch.randn(2, 30, 64)
    _, k = sdedit_init(x, strength=0.5, num_denoising_steps=100, diffusion=diffusion)
    assert k == 50


def test_invalid_strength_below_zero_raises(diffusion):
    with pytest.raises(InvalidStrengthError):
        sdedit_init(torch.zeros(1, 1, 1), strength=-0.1, num_denoising_steps=100, diffusion=diffusion)


def test_invalid_strength_above_one_raises(diffusion):
    with pytest.raises(InvalidStrengthError):
        sdedit_init(torch.zeros(1, 1, 1), strength=1.1, num_denoising_steps=100, diffusion=diffusion)
```

- [ ] **Step 2: Run test, verify it fails**

```bash
pytest tests/video/test_sdedit_init.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement SDEdit init**

Create `kimodo/video/sdedit.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SDEdit-style initial noise computation.

Encodes a noisy motion observation through the forward diffusion process
to a specified noise level, so the denoising loop can start from there
rather than from pure Gaussian noise.
"""
from __future__ import annotations

from typing import Tuple

import torch

from kimodo.model.diffusion import Diffusion

from .errors import InvalidStrengthError


def sdedit_init(
    motion: torch.Tensor,
    strength: float,
    num_denoising_steps: int,
    diffusion: Diffusion,
) -> Tuple[torch.Tensor, int]:
    """Compute the SDEdit starting tensor and starting step index.

    Args:
        motion:               Normalized motion features, shape [B, T, D].
        strength:             SDEdit strength in [0, 1]. 0 = no noise added,
                              1 = full diffusion noise (equivalent to pure-noise init).
        num_denoising_steps:  Total number of DDIM steps the sampler will run.
        diffusion:            A Kimodo Diffusion instance.

    Returns:
        A tuple `(noisy_motion, start_step)`:
        - noisy_motion: motion noised to step (start_step - 1).
        - start_step:   the diffusion step index from which to begin denoising
                        (the loop runs from start_step - 1 down to 0).

    Raises:
        InvalidStrengthError: if strength is outside [0, 1].
    """
    if not 0.0 <= strength <= 1.0:
        raise InvalidStrengthError(strength)

    # Ensure diffusion vars match the requested schedule length.
    use_timesteps, _ = diffusion.space_timesteps(num_denoising_steps)
    diffusion.calc_diffusion_vars(use_timesteps)

    k = max(1, int(strength * num_denoising_steps))
    batch_size = motion.shape[0]
    t_init = torch.full((batch_size,), k - 1, dtype=torch.long, device=motion.device)
    noisy = diffusion.q_sample(motion, t_init)
    return noisy, k
```

- [ ] **Step 4: Run tests, verify pass**

```bash
pytest tests/video/test_sdedit_init.py -v
```

Expected: all 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add kimodo/video/sdedit.py tests/video/test_sdedit_init.py
git commit -m "video: add SDEdit init helper"
```

---

## Task 7: Long-video chunking

**Files:**
- Create: `kimodo/video/chunking.py`
- Create: `tests/video/test_chunking.py`

- [ ] **Step 1: Write failing test**

Create `tests/video/test_chunking.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for overlap-blend chunking."""
from __future__ import annotations

import pytest
import torch

from kimodo.video.chunking import Window, chunk_motion, stitch_chunks


def test_chunk_short_input_returns_single_window():
    motion = torch.randn(50, 8)
    windows = chunk_motion(motion, chunk_size=100, overlap=20)
    assert len(windows) == 1
    assert windows[0].start == 0
    assert windows[0].end == 50
    assert torch.equal(windows[0].motion, motion)


def test_chunk_returns_overlapping_windows_for_long_input():
    motion = torch.randn(250, 8)  # 250 frames, chunk_size=100, overlap=20
    windows = chunk_motion(motion, chunk_size=100, overlap=20)
    # stride = 80, so windows start at 0, 80, 160 → 240 (last truncated)
    assert windows[0].start == 0 and windows[0].end == 100
    assert windows[1].start == 80 and windows[1].end == 180
    assert windows[2].start == 160 and windows[2].end == 250  # last window truncated to length


def test_stitch_preserves_length_and_smooth_signal():
    # A smooth signal should pass through chunk + stitch unchanged (within float tol).
    t_total = 250
    motion = torch.linspace(0, 1, t_total).unsqueeze(-1).expand(-1, 4).contiguous()

    windows = chunk_motion(motion, chunk_size=100, overlap=20)
    cleaned = [Window(start=w.start, end=w.end, motion=w.motion.clone()) for w in windows]
    out = stitch_chunks(cleaned, t_total)

    assert out.shape == motion.shape
    assert torch.allclose(out, motion, atol=1e-5)


def test_stitch_blends_overlap():
    # Two windows of constant values 0 and 1, with overlap → middle ramps 0→1.
    t_total = 30
    w1_motion = torch.zeros(20, 1)
    w2_motion = torch.ones(20, 1)
    windows = [
        Window(start=0, end=20, motion=w1_motion),
        Window(start=10, end=30, motion=w2_motion),
    ]
    out = stitch_chunks(windows, t_total)
    assert out.shape == (30, 1)
    assert out[0].item() == 0.0
    assert out[-1].item() == 1.0
    # In the overlap region (frames 10..19), the output should be between 0 and 1.
    overlap = out[10:20, 0]
    assert (overlap > 0).all() and (overlap < 1).all()
```

- [ ] **Step 2: Run test, verify it fails**

```bash
pytest tests/video/test_chunking.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement chunking**

Create `kimodo/video/chunking.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Overlap-blend chunking for long-video pipelines.

Splits motion into overlapping windows for independent denoising, then
stitches them back together with raised-cosine weighting on the overlaps.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

import torch


@dataclass
class Window:
    """A single chunk of motion.

    Attributes:
        start:  Inclusive start frame index in the full motion.
        end:    Exclusive end frame index in the full motion.
        motion: Tensor of shape [end - start, D].
    """
    start: int
    end: int
    motion: torch.Tensor


def chunk_motion(motion: torch.Tensor, chunk_size: int, overlap: int) -> List[Window]:
    """Split [T, D] motion into overlapping [chunk_size, D] windows.

    If T <= chunk_size, returns a single window covering [0, T).

    Args:
        motion:     Input motion, shape [T, D].
        chunk_size: Max length of each window in frames.
        overlap:    Overlap between consecutive windows in frames.

    Returns:
        List of Window objects whose `motion` tensors cover the input.
    """
    t_total = motion.shape[0]
    if t_total <= chunk_size:
        return [Window(start=0, end=t_total, motion=motion)]
    if overlap >= chunk_size:
        raise ValueError(f"overlap ({overlap}) must be strictly less than chunk_size ({chunk_size})")

    stride = chunk_size - overlap
    windows: List[Window] = []
    start = 0
    while start < t_total:
        end = min(start + chunk_size, t_total)
        windows.append(Window(start=start, end=end, motion=motion[start:end]))
        if end == t_total:
            break
        start += stride
    return windows


def stitch_chunks(windows: List[Window], total_T: int) -> torch.Tensor:
    """Blend cleaned windows back into a [T, D] tensor with raised-cosine alpha.

    Args:
        windows: List of Window objects (cleaned). All must share the same
                 trailing shape (D) and contiguous coverage of [0, total_T).
        total_T: Total length of the output.

    Returns:
        Tensor of shape [total_T, D], where overlap regions are weighted
        averages of contributing windows.
    """
    if not windows:
        raise ValueError("stitch_chunks requires at least one window")

    d = windows[0].motion.shape[-1]
    device = windows[0].motion.device
    dtype = windows[0].motion.dtype
    out = torch.zeros(total_T, d, device=device, dtype=dtype)
    weight = torch.zeros(total_T, 1, device=device, dtype=dtype)

    for w in windows:
        length = w.end - w.start
        alpha = _raised_cosine(length, device=device, dtype=dtype).unsqueeze(-1)
        out[w.start : w.end] += w.motion * alpha
        weight[w.start : w.end] += alpha

    return out / weight.clamp(min=1e-12)


def _raised_cosine(length: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """A raised-cosine window of `length` frames in [0, 1]. Symmetric, ends at small nonzero."""
    if length == 1:
        return torch.ones(1, device=device, dtype=dtype)
    n = torch.arange(length, device=device, dtype=dtype)
    # Hann-style: 0.5 * (1 - cos(2*pi*n / (length-1)))
    return 0.5 - 0.5 * torch.cos(2 * math.pi * n / (length - 1))
```

- [ ] **Step 4: Run tests, verify pass**

```bash
pytest tests/video/test_chunking.py -v
```

Expected: all 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add kimodo/video/chunking.py tests/video/test_chunking.py
git commit -m "video: add overlap-blend chunking for long videos"
```

---

## Task 8: `_generate` modification + backward-compat regression guard

**This is the only existing file we modify.** Two tests are written first: one that captures the current behavior (regression guard), one that exercises the new SDEdit path. The implementation runs only after both tests are in place.

**Files:**
- Create: `tests/video/test_regression_existing_path.py`
- Create: `tests/video/test_generate_sdedit.py`
- Modify: `kimodo/model/kimodo_model.py` (around lines 562–634)

- [ ] **Step 1: Write the backward-compat regression test**

Create `tests/video/test_regression_existing_path.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression guard for Kimodo._generate's existing pure-noise path.

These tests assert that when `init_motion` is None (the default), `_generate`
behaves identically to the original implementation:
- It calls torch.randn to seed cur_mot.
- It iterates over the full range [num_denoising_steps - 1, ..., 0].
- It does NOT call diffusion.q_sample.

These tests must pass both BEFORE and AFTER the _generate modification.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch


@pytest.fixture
def fake_kimodo():
    """Build a Kimodo instance with all heavy components mocked.

    The mocks let us drive _generate directly without loading weights.
    """
    from kimodo.model.kimodo_model import Kimodo
    from kimodo.model.diffusion import Diffusion

    # Stub denoiser with the bare minimum attribute surface Kimodo needs.
    denoiser = MagicMock()
    denoiser.eval.return_value = denoiser
    motion_rep = MagicMock()
    motion_rep.motion_rep_dim = 16
    motion_rep.fps = 20
    motion_rep.skeleton = MagicMock()
    denoiser.motion_rep = motion_rep

    text_encoder = MagicMock()
    # text_encoder(texts) → (text_feat [B, L, D], text_length list)
    text_encoder.return_value = (torch.zeros(1, 4, 32), [4])

    # Bypass the ClassifierFreeGuidedModel wrapping in Kimodo.__init__ by
    # constructing the object then overriding self.denoiser.
    model = Kimodo.__new__(Kimodo)
    torch.nn.Module.__init__(model)
    model.denoiser = denoiser
    model.motion_rep = motion_rep
    model.skeleton = motion_rep.skeleton
    model.fps = 20
    model.diffusion = Diffusion(num_base_steps=10)
    from kimodo.model.diffusion import DDIMSampler
    model.sampler = DDIMSampler(model.diffusion)
    model.text_encoder = text_encoder
    model.device = torch.device("cpu")

    # denoising_step returns a same-shape tensor (identity for simplicity).
    def _identity_step(motion, *args, **kwargs):
        return motion
    model.denoising_step = MagicMock(side_effect=_identity_step)
    return model


def test_generate_uses_torch_randn_when_init_motion_is_none(fake_kimodo):
    """The existing pure-noise init path must still fire when init_motion=None."""
    with patch("kimodo.model.kimodo_model.torch.randn",
               wraps=torch.randn) as mock_randn:
        fake_kimodo._generate(
            texts=[""],
            max_frames=5,
            num_denoising_steps=10,
            pad_mask=torch.ones(1, 5, dtype=torch.bool),
            first_heading_angle=torch.tensor([0.0]),
            motion_mask=None,
            observed_motion=None,
            cfg_weight=[2.0, 0.0],
            progress_bar=lambda x: x,
        )
    assert mock_randn.called, "torch.randn must be called when init_motion is None"
    # Verify the shape passed to randn matches the expected (1, max_frames, motion_rep_dim).
    args, _kwargs = mock_randn.call_args
    assert args[0] == (1, 5, 16)


def test_generate_iterates_full_range_when_init_motion_is_none(fake_kimodo):
    """When init_motion=None, the denoising loop must iterate over [N-1, ..., 0]."""
    fake_kimodo._generate(
        texts=[""],
        max_frames=5,
        num_denoising_steps=10,
        pad_mask=torch.ones(1, 5, dtype=torch.bool),
        first_heading_angle=torch.tensor([0.0]),
        motion_mask=None,
        observed_motion=None,
        cfg_weight=[2.0, 0.0],
        progress_bar=lambda x: x,
    )
    # denoising_step should have been called exactly 10 times (one per step).
    assert fake_kimodo.denoising_step.call_count == 10
    # Collect the `t` values passed in; they should be [9, 8, 7, ..., 0].
    t_values = [call.args[4][0].item() for call in fake_kimodo.denoising_step.call_args_list]
    assert t_values == list(range(9, -1, -1))
```

- [ ] **Step 2: Run regression test against the UNMODIFIED code**

```bash
pytest tests/video/test_regression_existing_path.py -v
```

Expected: both tests **pass**. This establishes the baseline behavior. If they fail before any model change, the test setup is wrong — fix it and re-run before proceeding.

- [ ] **Step 3: Write the new SDEdit-path test (will fail until _generate is modified)**

Create `tests/video/test_generate_sdedit.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the new SDEdit init path in Kimodo._generate."""
from __future__ import annotations

from unittest.mock import patch

import pytest
import torch


# Reuse the fake_kimodo fixture from the regression test by importing it.
from tests.video.test_regression_existing_path import fake_kimodo  # noqa: F401


def test_generate_calls_q_sample_when_init_motion_provided(fake_kimodo):
    init_motion = torch.zeros(1, 5, 16)
    with patch.object(fake_kimodo.diffusion, "q_sample",
                      wraps=fake_kimodo.diffusion.q_sample) as mock_q:
        fake_kimodo._generate(
            texts=[""],
            max_frames=5,
            num_denoising_steps=10,
            pad_mask=torch.ones(1, 5, dtype=torch.bool),
            first_heading_angle=torch.tensor([0.0]),
            motion_mask=None,
            observed_motion=None,
            cfg_weight=[2.0, 0.0],
            progress_bar=lambda x: x,
            init_motion=init_motion,
            strength=0.5,
        )
    assert mock_q.called, "q_sample must be called when init_motion is provided"


def test_generate_truncates_indices_to_strength_when_init_motion_provided(fake_kimodo):
    init_motion = torch.zeros(1, 5, 16)
    fake_kimodo._generate(
        texts=[""],
        max_frames=5,
        num_denoising_steps=10,
        pad_mask=torch.ones(1, 5, dtype=torch.bool),
        first_heading_angle=torch.tensor([0.0]),
        motion_mask=None,
        observed_motion=None,
        cfg_weight=[2.0, 0.0],
        progress_bar=lambda x: x,
        init_motion=init_motion,
        strength=0.5,
    )
    # strength=0.5 with 10 steps → k=5 → indices [4, 3, 2, 1, 0] → 5 denoising_step calls
    assert fake_kimodo.denoising_step.call_count == 5
    t_values = [call.args[4][0].item() for call in fake_kimodo.denoising_step.call_args_list]
    assert t_values == list(range(4, -1, -1))


def test_generate_does_not_call_torch_randn_when_init_motion_provided(fake_kimodo):
    init_motion = torch.zeros(1, 5, 16)
    with patch("kimodo.model.kimodo_model.torch.randn") as mock_randn:
        fake_kimodo._generate(
            texts=[""],
            max_frames=5,
            num_denoising_steps=10,
            pad_mask=torch.ones(1, 5, dtype=torch.bool),
            first_heading_angle=torch.tensor([0.0]),
            motion_mask=None,
            observed_motion=None,
            cfg_weight=[2.0, 0.0],
            progress_bar=lambda x: x,
            init_motion=init_motion,
            strength=0.5,
        )
    mock_randn.assert_not_called()
```

- [ ] **Step 4: Run the SDEdit test to confirm it fails**

```bash
pytest tests/video/test_generate_sdedit.py -v
```

Expected: all 3 tests fail with TypeError (unexpected keyword argument `init_motion` / `strength`).

- [ ] **Step 5: Modify `_generate` to add the gated SDEdit path**

Open `kimodo/model/kimodo_model.py` and locate `_generate` (currently around lines 562–634).

Update the function signature to add the two new kwargs (at the end of the parameter list, after `progress_bar`):

```python
    def _generate(
        self,
        texts: List[str],
        max_frames: int,
        num_denoising_steps: int,
        pad_mask: torch.Tensor,
        first_heading_angle: Optional[torch.Tensor],
        motion_mask: torch.Tensor,
        observed_motion: torch.Tensor,
        cfg_weight: Optional[float] = 2.0,
        text_feat: Optional[torch.Tensor] = None,
        text_pad_mask: Optional[torch.Tensor] = None,
        guide_masks: Optional[Dict] = None,
        cfg_type: Optional[str] = None,
        progress_bar=tqdm,
        init_motion: Optional[torch.Tensor] = None,
        strength: float = 0.5,
    ) -> torch.Tensor:
```

Replace the init block (currently around lines 607–616, which reads `cur_mot = torch.randn(...)` through the end of `calc_diffusion_vars`) with the gated version:

```python
        # sample loop
        shape = (batch_size, max_frames, self.motion_rep.motion_rep_dim)
        num_denoising_steps_tensor = torch.tensor(
            [num_denoising_steps], device=self.device
        )  # this and t need to be tensor for onnx export
        use_timesteps = self.diffusion.space_timesteps(num_denoising_steps_tensor[0])[0]
        self.diffusion.calc_diffusion_vars(use_timesteps)

        if init_motion is None:
            # Existing path: pure-noise init (byte-identical to original behavior).
            cur_mot = torch.randn(shape, device=self.device)
            indices = list(range(num_denoising_steps))[::-1]
        else:
            # SDEdit path: noise init_motion to step k, denoise from there.
            k = max(1, int(strength * num_denoising_steps))
            t_init = torch.full(
                (batch_size,), k - 1, dtype=torch.long, device=self.device
            )
            cur_mot = self.diffusion.q_sample(init_motion, t_init)
            indices = list(range(k))[::-1]

        # Keep the original local variable name used by the existing loop body.
        num_denoising_steps = num_denoising_steps_tensor
```

The denoising loop body below (`for i in progress_bar(indices): ...`) remains unchanged.

Note: the original code declared `num_denoising_steps = torch.tensor([num_denoising_steps], ...)` right after the old `cur_mot = torch.randn(...)` line. We're moving that assignment to before the if/else (renaming the tensor to `num_denoising_steps_tensor` for clarity), then reassigning at the bottom so the rest of the function sees the tensor as before.

- [ ] **Step 6: Run both regression and SDEdit tests; both must pass**

```bash
pytest tests/video/test_regression_existing_path.py tests/video/test_generate_sdedit.py -v
```

Expected: all 5 tests pass.

- [ ] **Step 7: Run the entire kimodo test suite (existing + new) to confirm nothing else broke**

```bash
pytest tests/ -v
```

Expected: no failures from the existing codebase (if there were any existing tests). If kimodo has no existing test suite, this still confirms the new tests pass and nothing in `import kimodo` is broken.

Also smoke-test imports:

```bash
python -c "import kimodo; from kimodo import load_model; from kimodo.video import SOMAMotion30; print('imports OK')"
```

Expected: `imports OK` with no errors.

- [ ] **Step 8: Commit**

```bash
git add kimodo/model/kimodo_model.py tests/video/test_regression_existing_path.py tests/video/test_generate_sdedit.py
git commit -m "model: add optional SDEdit init path to _generate (gated; backward compatible)"
```

---

## Task 9: Estimator stubs (gem, wham, 4dhumans)

**Files:**
- Create: `kimodo/video/estimators/__init__.py`
- Create: `kimodo/video/estimators/gem.py`
- Create: `kimodo/video/estimators/wham.py`
- Create: `kimodo/video/estimators/fourdhumans.py`
- Create: `tests/video/test_estimator_stubs.py`

- [ ] **Step 1: Write failing test**

Create `tests/video/test_estimator_stubs.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for stub PoseEstimator adapters."""
from __future__ import annotations

import pytest

# Importing the estimators package triggers stub registration.
import kimodo.video.estimators  # noqa: F401
from kimodo.video.pose_estimator import get_pose_estimator


@pytest.mark.parametrize("name", ["gem", "wham", "4dhumans"])
def test_stub_adapters_are_registered(name):
    estimator = get_pose_estimator(name)
    assert estimator is not None


@pytest.mark.parametrize("name", ["gem", "wham", "4dhumans"])
def test_stub_adapters_raise_not_implemented(name):
    estimator = get_pose_estimator(name)
    with pytest.raises(NotImplementedError, match=name):
        estimator.estimate("any.mp4")
```

- [ ] **Step 2: Run test, verify it fails**

```bash
pytest tests/video/test_estimator_stubs.py -v
```

Expected: ImportError or KeyError (estimators not yet registered).

- [ ] **Step 3: Implement the stubs**

Create `kimodo/video/estimators/__init__.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bundled PoseEstimator adapters.

Importing this package triggers registration of every bundled adapter
(real or stub). After import, all adapter names are visible via
`kimodo.video.get_pose_estimator(name)`.
"""
from __future__ import annotations

from . import fourdhumans, gem, wham  # noqa: F401 — import triggers registration
```

Create `kimodo/video/estimators/gem.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stub adapter for NVIDIA GEM (SOMA-native pose estimator).

A real GEM adapter is the recommended commercial-friendly migration path
because GEM outputs SOMA directly, bypassing SMPL-X licensing concerns.
See docs/superpowers/specs/2026-05-25-video-input-pipeline-design.md §15.
"""
from __future__ import annotations

from ..pose_estimator import register_pose_estimator
from ..types import SOMAMotion30


class GEMStub:
    def estimate(self, video_path: str, *, person_idx: int = 0) -> SOMAMotion30:
        raise NotImplementedError(
            "gem adapter is not yet implemented. "
            "Track progress in the project roadmap or contribute via PR."
        )


@register_pose_estimator("gem")
def _factory() -> GEMStub:
    return GEMStub()
```

Create `kimodo/video/estimators/wham.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stub adapter for WHAM (world-grounded SMPL-X pose estimator)."""
from __future__ import annotations

from ..pose_estimator import register_pose_estimator
from ..types import SOMAMotion30


class WHAMStub:
    def estimate(self, video_path: str, *, person_idx: int = 0) -> SOMAMotion30:
        raise NotImplementedError(
            "wham adapter is not yet implemented. "
            "Track progress in the project roadmap or contribute via PR."
        )


@register_pose_estimator("wham")
def _factory() -> WHAMStub:
    return WHAMStub()
```

Create `kimodo/video/estimators/fourdhumans.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stub adapter for 4DHumans pose estimator."""
from __future__ import annotations

from ..pose_estimator import register_pose_estimator
from ..types import SOMAMotion30


class FourDHumansStub:
    def estimate(self, video_path: str, *, person_idx: int = 0) -> SOMAMotion30:
        raise NotImplementedError(
            "4dhumans adapter is not yet implemented. "
            "Track progress in the project roadmap or contribute via PR."
        )


@register_pose_estimator("4dhumans")
def _factory() -> FourDHumansStub:
    return FourDHumansStub()
```

- [ ] **Step 4: Run tests, verify pass**

```bash
pytest tests/video/test_estimator_stubs.py -v
```

Expected: all 6 parameterized tests pass.

- [ ] **Step 5: Commit**

```bash
git add kimodo/video/estimators/ tests/video/test_estimator_stubs.py
git commit -m "video: add gem/wham/4dhumans stub adapters with NotImplementedError"
```

---

## Task 10: GVHMR adapter

**Files:**
- Create: `kimodo/video/estimators/gvhmr.py`
- Modify: `kimodo/video/estimators/__init__.py`
- Create: `tests/video/test_gvhmr_adapter.py`

- [ ] **Step 1: Write failing test**

Create `tests/video/test_gvhmr_adapter.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the GVHMR PoseEstimator adapter.

Real-GVHMR end-to-end coverage requires the upstream package and is left
to manual verification. These tests mock the imports.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

import kimodo.video.estimators  # noqa: F401
from kimodo.video.errors import EstimatorNotInstalledError, NoMotionDetectedError
from kimodo.video.pose_estimator import get_pose_estimator
from kimodo.video.types import SOMAMotion30


def test_gvhmr_is_registered():
    est = get_pose_estimator("gvhmr")
    assert est is not None


def test_gvhmr_raises_install_error_if_dep_missing():
    from kimodo.video.estimators import gvhmr
    with patch.object(gvhmr, "_load_gvhmr", side_effect=ImportError("no module")):
        est = gvhmr.GVHMRAdapter()
        with pytest.raises(EstimatorNotInstalledError, match="gvhmr"):
            est.estimate("any.mp4")


def test_gvhmr_returns_soma_motion30(monkeypatch):
    """When the mocked GVHMR pipeline returns a valid SMPL-X result,
    the adapter retargets it to SOMA-30 internally and returns SOMAMotion30."""
    from kimodo.video.estimators import gvhmr

    fake_gvhmr_module = MagicMock()
    # The adapter expects: run(video_path) → dict with body_pose, global_orient, transl, fps
    fake_gvhmr_module.run.return_value = {
        "body_pose":     torch.zeros(15, 21, 3, 3),
        "global_orient": torch.zeros(15, 3, 3),
        "transl":        torch.zeros(15, 3),
        "fps":           30,
    }
    monkeypatch.setattr(gvhmr, "_load_gvhmr", lambda: fake_gvhmr_module)

    # Mock the SMPL-X→SOMA retarget to avoid needing py-soma-x installed.
    from kimodo.video import _smplx_to_soma

    def _fake_retarget(smplx):
        return SOMAMotion30(
            local_rot_mats=torch.zeros(smplx.num_frames, 30, 3, 3),
            root_positions=torch.zeros(smplx.num_frames, 3),
            fps=smplx.fps,
        )
    monkeypatch.setattr(_smplx_to_soma, "retarget", _fake_retarget)

    est = gvhmr.GVHMRAdapter()
    out = est.estimate("any.mp4")
    assert isinstance(out, SOMAMotion30)
    assert out.num_frames == 15
    assert out.fps == 30


def test_gvhmr_raises_when_no_motion_detected(monkeypatch):
    from kimodo.video.estimators import gvhmr

    fake_gvhmr_module = MagicMock()
    fake_gvhmr_module.run.return_value = {
        "body_pose":     torch.zeros(0, 21, 3, 3),
        "global_orient": torch.zeros(0, 3, 3),
        "transl":        torch.zeros(0, 3),
        "fps":           30,
    }
    monkeypatch.setattr(gvhmr, "_load_gvhmr", lambda: fake_gvhmr_module)

    est = gvhmr.GVHMRAdapter()
    with pytest.raises(NoMotionDetectedError):
        est.estimate("any.mp4")
```

- [ ] **Step 2: Run test, verify it fails**

```bash
pytest tests/video/test_gvhmr_adapter.py -v
```

Expected: ImportError (gvhmr adapter not yet implemented).

- [ ] **Step 3: Implement the GVHMR adapter**

Create `kimodo/video/estimators/gvhmr.py`:

```python
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
        # gvhmr.run(video_path, person_idx=...) → dict with SMPL-X tensors.
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
```

Edit `kimodo/video/estimators/__init__.py` to register the new adapter:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bundled PoseEstimator adapters.

Importing this package triggers registration of every bundled adapter
(real or stub). After import, all adapter names are visible via
`kimodo.video.get_pose_estimator(name)`.
"""
from __future__ import annotations

from . import fourdhumans, gem, gvhmr, wham  # noqa: F401 — import triggers registration
```

- [ ] **Step 4: Run tests, verify pass**

```bash
pytest tests/video/test_gvhmr_adapter.py tests/video/test_estimator_stubs.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add kimodo/video/estimators/gvhmr.py kimodo/video/estimators/__init__.py tests/video/test_gvhmr_adapter.py
git commit -m "video: add GVHMR PoseEstimator adapter (lazy import, SMPL-X internal)"
```

---

## Task 11: Pipeline orchestrator — SDEdit mode

**Files:**
- Create: `kimodo/video/pipeline.py`
- Modify: `kimodo/video/__init__.py`
- Create: `tests/video/test_pipeline_sdedit.py`

- [ ] **Step 1: Write failing test**

Create `tests/video/test_pipeline_sdedit.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end test for video_to_motion in SDEdit mode (with mocked estimator + model)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

import kimodo.video.estimators  # noqa: F401 — register estimators
from kimodo.video.pose_estimator import _REGISTRY, register_pose_estimator
from kimodo.video.types import SOMAMotion30


@pytest.fixture(autouse=True)
def isolate_registry():
    snapshot = dict(_REGISTRY)
    yield
    _REGISTRY.clear()
    _REGISTRY.update(snapshot)


def _mock_kimodo_model() -> MagicMock:
    """A MagicMock Kimodo that supports the calls pipeline.py makes."""
    model = MagicMock()
    model.motion_rep.fps = 20
    model.motion_rep.motion_rep_dim = 16
    # motion_rep(rot, root, to_normalize=True) → encoded [T, D]
    model.motion_rep.side_effect = lambda rot, root, **kw: torch.zeros(rot.shape[0], 16)
    # _generate returns a clean motion of shape [B, T, D]
    model._generate.return_value = torch.zeros(1, 20, 16)
    # motion_rep.inverse → dict
    model.motion_rep.inverse.return_value = {
        "posed_joints":   torch.zeros(20, 77, 3),
        "local_rot_mats": torch.zeros(20, 30, 3, 3),
    }
    # skeleton.output_to_SOMASkeleton77 → identity for this stub
    model.skeleton.output_to_SOMASkeleton77.side_effect = lambda d: d
    return model


def test_video_to_motion_sdedit_dispatches_correctly(tmp_path, monkeypatch):
    # Register a fake estimator that returns a fixed SOMAMotion30.
    @register_pose_estimator("fake")
    def _factory():
        class F:
            def estimate(self, video_path, *, person_idx=0):
                return SOMAMotion30(
                    local_rot_mats=torch.eye(3).expand(20, 30, 3, 3).clone(),
                    root_positions=torch.zeros(20, 3),
                    fps=20,
                )
        return F()

    mock_model = _mock_kimodo_model()
    monkeypatch.setattr("kimodo.video.pipeline.load_model", lambda *a, **kw: mock_model)

    from kimodo.video.pipeline import video_to_motion

    result = video_to_motion(
        video_path=str(tmp_path / "fake.mp4"),
        estimator="fake",
        mode="sdedit",
        strength=0.5,
        return_numpy=False,
    )

    # _generate must have been called with init_motion provided (SDEdit path).
    assert mock_model._generate.called
    _args, kwargs = mock_model._generate.call_args
    assert kwargs.get("init_motion") is not None
    assert kwargs.get("strength") == 0.5

    # cfg_weight in SDEdit mode forces the constraint channel to 0.
    assert kwargs.get("cfg_weight")[1] == 0.0

    # The result dict has the expected keys.
    assert "posed_joints" in result
```

- [ ] **Step 2: Run test, verify it fails**

```bash
pytest tests/video/test_pipeline_sdedit.py -v
```

Expected: ImportError (pipeline not yet implemented).

- [ ] **Step 3: Implement the pipeline (SDEdit mode only for now)**

Create `kimodo/video/pipeline.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pipeline orchestrator for video → clean SOMA motion.

This is the entry point for the public `video_to_motion` function.
Operates exclusively in SOMA: SMPL-X (if used by an estimator adapter) is
already retargeted by the time the pipeline sees it.
"""
from __future__ import annotations

from typing import Literal, Optional, Tuple

import torch

from kimodo import load_model
from kimodo.tools import to_numpy

import kimodo.video.estimators  # noqa: F401 — register estimators on import

from .errors import InvalidStrengthError, VideoTooShortError
from .pose_estimator import get_pose_estimator
from .resample import resample_to_fps
from .sdedit import sdedit_init
from .types import SOMAMotion30

_MIN_FRAMES = 10


def video_to_motion(
    video_path: str,
    *,
    output_path: Optional[str] = None,
    model_name: str = "Kimodo-SOMA-RP-v1.1",
    prompt: str = "",
    strength: float = 0.5,
    mode: Literal["sdedit", "constraints"] = "sdedit",
    estimator: str = "gvhmr",
    num_denoising_steps: int = 100,
    num_samples: int = 1,
    cfg_weight: Tuple[float, float] = (2.0, 0.5),
    post_processing: bool = True,
    seed: Optional[int] = None,
    person_idx: int = 0,
    save_intermediates: bool = False,
    device: Optional[str] = None,
    return_numpy: bool = True,
) -> dict:
    """Run the video → clean SOMA motion pipeline.

    See docs/superpowers/specs/2026-05-25-video-input-pipeline-design.md §8 for the full API.
    """
    if not 0.0 <= strength <= 1.0:
        raise InvalidStrengthError(strength)
    if seed is not None:
        torch.manual_seed(seed)

    # 1. Pose estimation → SOMAMotion30
    est = get_pose_estimator(estimator)
    soma_in = est.estimate(video_path, person_idx=person_idx)

    # 2. Load the model so we know its target fps and motion_rep
    model = load_model(model_name, device=device)

    # 3. Resample to the model's fps
    soma_in = resample_to_fps(soma_in, target_fps=model.motion_rep.fps)
    if soma_in.num_frames < _MIN_FRAMES:
        raise VideoTooShortError(soma_in.num_frames, _MIN_FRAMES)

    # 4. Dispatch by mode
    if mode == "sdedit":
        result = _run_sdedit(
            model=model, soma_in=soma_in, prompt=prompt, strength=strength,
            num_denoising_steps=num_denoising_steps, num_samples=num_samples,
            cfg_weight=cfg_weight, post_processing=post_processing,
        )
    elif mode == "constraints":
        raise NotImplementedError("constraints mode added in a later task")
    else:
        raise ValueError(f"Unknown mode {mode!r}; expected 'sdedit' or 'constraints'")

    # 5. Convert SOMA-30 output dict to SOMA-77 for the public API
    if hasattr(model.skeleton, "output_to_SOMASkeleton77"):
        result = model.skeleton.output_to_SOMASkeleton77(result)

    if return_numpy:
        result = to_numpy(result)

    if output_path is not None:
        _save_output(result, output_path)

    return result


def _run_sdedit(
    *,
    model,
    soma_in: SOMAMotion30,
    prompt: str,
    strength: float,
    num_denoising_steps: int,
    num_samples: int,
    cfg_weight: Tuple[float, float],
    post_processing: bool,
) -> dict:
    # Encode SOMA motion to Kimodo's normalized feature space.
    encoded = model.motion_rep(
        soma_in.local_rot_mats,
        soma_in.root_positions,
        to_normalize=True,
    )
    if encoded.ndim == 2:
        encoded = encoded.unsqueeze(0)  # add batch dim → [1, T, D]
    # Replicate across batch for num_samples.
    init_motion = encoded.expand(num_samples, -1, -1).contiguous()

    # In SDEdit mode, no observed_motion → force constraint CFG to 0.
    cfg = (cfg_weight[0], 0.0)

    # Build the pad_mask, heading, and texts the way Kimodo.__call__ does.
    T = init_motion.shape[1]
    pad_mask = torch.ones(num_samples, T, dtype=torch.bool, device=init_motion.device)
    first_heading_angle = torch.zeros(num_samples, device=init_motion.device)
    texts = [prompt] * num_samples

    clean = model._generate(
        texts=texts,
        max_frames=T,
        num_denoising_steps=num_denoising_steps,
        pad_mask=pad_mask,
        first_heading_angle=first_heading_angle,
        motion_mask=None,
        observed_motion=None,
        cfg_weight=list(cfg),
        init_motion=init_motion,
        strength=strength,
    )

    # Decode back to joint positions and rotations.
    output = model.motion_rep.inverse(clean, is_normalized=True, return_numpy=False)

    if post_processing:
        from kimodo.postprocess import post_process_motion
        corrected = post_process_motion(
            output["local_rot_mats"],
            output["root_positions"],
            output["foot_contacts"],
            model.skeleton,
            [],  # no constraints in SDEdit mode
        )
        output.update(corrected)

    return output


def _save_output(result: dict, output_path: str) -> None:
    """Write the motion dict to NPZ. Mirrors kimodo.scripts.generate output format."""
    from kimodo.exports.motion_io import save_kimodo_npz
    save_kimodo_npz(result, output_path)
```

Edit `kimodo/video/__init__.py` to expose `video_to_motion`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Video-input pipeline for Kimodo.

Public API:
    video_to_motion: drop a video, get clean SOMA-77 motion.
    SOMAMotion30:   public motion type produced by all PoseEstimators.
    PoseEstimator, register_pose_estimator, get_pose_estimator: estimator registry.
    resample_to_fps: frame-rate utility.

This package is opt-in: install with `pip install kimodo[video]`.
"""
from __future__ import annotations

from .pipeline import video_to_motion
from .pose_estimator import PoseEstimator, get_pose_estimator, register_pose_estimator
from .resample import resample_to_fps
from .types import SOMAMotion30

__all__ = [
    "video_to_motion",
    "SOMAMotion30",
    "PoseEstimator",
    "register_pose_estimator",
    "get_pose_estimator",
    "resample_to_fps",
]
```

- [ ] **Step 4: Run tests, verify pass**

```bash
pytest tests/video/test_pipeline_sdedit.py -v
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add kimodo/video/pipeline.py kimodo/video/__init__.py tests/video/test_pipeline_sdedit.py
git commit -m "video: add pipeline orchestrator (SDEdit mode)"
```

---

## Task 12: Pipeline orchestrator — constraints mode

**Files:**
- Modify: `kimodo/video/pipeline.py`
- Create: `tests/video/test_pipeline_constraints.py`

- [ ] **Step 1: Write failing test**

Create `tests/video/test_pipeline_constraints.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for video_to_motion in constraints mode."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

import kimodo.video.estimators  # noqa: F401
from kimodo.video.pose_estimator import _REGISTRY, register_pose_estimator
from kimodo.video.types import SOMAMotion30


@pytest.fixture(autouse=True)
def isolate_registry():
    snapshot = dict(_REGISTRY)
    yield
    _REGISTRY.clear()
    _REGISTRY.update(snapshot)


def _mock_kimodo_model() -> MagicMock:
    model = MagicMock()
    model.motion_rep.fps = 20
    model.motion_rep.motion_rep_dim = 16

    # __call__ returns the standard output dict.
    model.return_value = {
        "posed_joints":   torch.zeros(20, 77, 3),
        "local_rot_mats": torch.zeros(20, 30, 3, 3),
    }
    model.skeleton.output_to_SOMASkeleton77.side_effect = lambda d: d
    return model


def test_constraints_mode_calls_kimodo_with_constraint_lst(tmp_path, monkeypatch):
    @register_pose_estimator("fake")
    def _factory():
        class F:
            def estimate(self, video_path, *, person_idx=0):
                return SOMAMotion30(
                    local_rot_mats=torch.eye(3).expand(20, 30, 3, 3).clone(),
                    root_positions=torch.zeros(20, 3),
                    fps=20,
                )
        return F()

    mock_model = _mock_kimodo_model()
    monkeypatch.setattr("kimodo.video.pipeline.load_model", lambda *a, **kw: mock_model)

    from kimodo.video.pipeline import video_to_motion

    result = video_to_motion(
        video_path=str(tmp_path / "fake.mp4"),
        estimator="fake",
        mode="constraints",
        cfg_weight=(2.0, 0.5),
        return_numpy=False,
    )

    # In constraints mode, Kimodo.__call__ is used (not _generate directly).
    assert mock_model.called
    _args, kwargs = mock_model.call_args
    assert "constraint_lst" in kwargs
    assert len(kwargs["constraint_lst"]) > 0  # at least one constraint set built
    assert kwargs.get("cfg_weight") == [2.0, 0.5]

    # Result dict still has the expected schema.
    assert "posed_joints" in result
```

- [ ] **Step 2: Run test, verify it fails**

```bash
pytest tests/video/test_pipeline_constraints.py -v
```

Expected: fails with `NotImplementedError("constraints mode added in a later task")`.

- [ ] **Step 3: Implement constraints mode in pipeline.py**

Open `kimodo/video/pipeline.py` and replace the line that reads:

```python
    elif mode == "constraints":
        raise NotImplementedError("constraints mode added in a later task")
```

with:

```python
    elif mode == "constraints":
        result = _run_constraints(
            model=model, soma_in=soma_in, prompt=prompt,
            num_denoising_steps=num_denoising_steps, num_samples=num_samples,
            cfg_weight=cfg_weight, post_processing=post_processing,
        )
```

Then add the `_run_constraints` function at the bottom of the file (above `_save_output`):

```python
def _run_constraints(
    *,
    model,
    soma_in: SOMAMotion30,
    prompt: str,
    num_denoising_steps: int,
    num_samples: int,
    cfg_weight: Tuple[float, float],
    post_processing: bool,
) -> dict:
    """Constraints-mode pipeline: feed retargeted SOMA motion as soft full-body constraints."""
    from kimodo.constraints import EndEffectorConstraintSet, FullBodyConstraintSet

    T = soma_in.num_frames
    constraint_lst = []
    for _ in range(num_samples):
        # Compute posed joints from local rotations + root positions for the constraint.
        # Forward-kinematics path: delegate to the model's skeleton's FK if available.
        posed_joints, global_rot_mats = _fk(model.skeleton, soma_in.local_rot_mats, soma_in.root_positions)
        smooth_root_2d = soma_in.root_positions[..., [0, 2]]
        full_body = FullBodyConstraintSet(
            model.skeleton,
            torch.arange(T),
            posed_joints,
            global_rot_mats,
            smooth_root_2d,
        )
        end_eff = EndEffectorConstraintSet(
            model.skeleton,
            torch.arange(T),
            posed_joints,
            global_rot_mats,
            smooth_root_2d,
            joint_names=["LeftHand", "RightHand", "LeftFoot", "RightFoot"],
        )
        constraint_lst.append([full_body, end_eff])

    return model(
        prompts=[prompt] * num_samples,
        num_frames=[T] * num_samples,
        num_denoising_steps=num_denoising_steps,
        constraint_lst=constraint_lst,
        cfg_weight=list(cfg_weight),
        num_samples=num_samples,
        post_processing=post_processing,
        return_numpy=False,
    )


def _fk(skeleton, local_rot_mats: torch.Tensor, root_positions: torch.Tensor):
    """Forward-kinematics helper. Delegates to the skeleton's FK if available;
    otherwise falls back to a minimal implementation."""
    if hasattr(skeleton, "forward_kinematics"):
        posed, global_rots = skeleton.forward_kinematics(local_rot_mats, root_positions)
        return posed, global_rots
    raise NotImplementedError(
        "skeleton.forward_kinematics is required for constraints mode. "
        "If your Kimodo skeleton does not expose FK, implement it or use mode='sdedit'."
    )
```

Note: the FK signature assumes `skeleton.forward_kinematics(local_rot_mats, root_positions) → (posed_joints, global_rot_mats)`. If the live SOMA skeleton in `kimodo/skeleton/` exposes FK under a different name, update `_fk` to call it. Inspect `kimodo/skeleton/kinematics.py` for the actual signature when wiring this up.

- [ ] **Step 4: Run tests, verify pass**

```bash
pytest tests/video/test_pipeline_constraints.py tests/video/test_pipeline_sdedit.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add kimodo/video/pipeline.py tests/video/test_pipeline_constraints.py
git commit -m "video: add constraints-mode fallback to pipeline"
```

---

## Task 13: Long-video chunking in the pipeline

**Files:**
- Modify: `kimodo/video/pipeline.py`
- Create: `tests/video/test_pipeline_chunking.py`

- [ ] **Step 1: Write failing test**

Create `tests/video/test_pipeline_chunking.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the pipeline's automatic chunking on long inputs."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

import kimodo.video.estimators  # noqa: F401
from kimodo.video.pose_estimator import _REGISTRY, register_pose_estimator
from kimodo.video.types import SOMAMotion30


@pytest.fixture(autouse=True)
def isolate_registry():
    snapshot = dict(_REGISTRY)
    yield
    _REGISTRY.clear()
    _REGISTRY.update(snapshot)


def _mock_model_with_max_frames(max_frames: int) -> MagicMock:
    model = MagicMock()
    model.motion_rep.fps = 20
    model.motion_rep.motion_rep_dim = 16
    model.max_frames = max_frames
    model.motion_rep.side_effect = lambda rot, root, **kw: torch.zeros(rot.shape[0], 16)
    model._generate.side_effect = lambda *a, **kw: torch.zeros(1, kw["max_frames"], 16)
    model.motion_rep.inverse.return_value = {
        "posed_joints":   torch.zeros(1, 77, 3),
        "local_rot_mats": torch.zeros(1, 30, 3, 3),
    }
    model.skeleton.output_to_SOMASkeleton77.side_effect = lambda d: d
    return model


def test_long_input_triggers_multiple_chunks(tmp_path, monkeypatch):
    @register_pose_estimator("long_fake")
    def _factory():
        class F:
            def estimate(self, video_path, *, person_idx=0):
                return SOMAMotion30(
                    local_rot_mats=torch.eye(3).expand(250, 30, 3, 3).clone(),
                    root_positions=torch.zeros(250, 3),
                    fps=20,
                )
        return F()

    mock_model = _mock_model_with_max_frames(max_frames=100)
    monkeypatch.setattr("kimodo.video.pipeline.load_model", lambda *a, **kw: mock_model)

    from kimodo.video.pipeline import video_to_motion

    video_to_motion(
        video_path=str(tmp_path / "long.mp4"),
        estimator="long_fake",
        mode="sdedit",
        chunk_size=100,
        chunk_overlap=20,
        return_numpy=False,
    )

    # 250 frames with chunk_size=100, overlap=20, stride=80 → starts 0, 80, 160 → 3 chunks.
    assert mock_model._generate.call_count == 3
```

- [ ] **Step 2: Run test, verify it fails**

```bash
pytest tests/video/test_pipeline_chunking.py -v
```

Expected: fails (single _generate call instead of 3, or chunking args not accepted).

- [ ] **Step 3: Add chunking to the SDEdit pipeline path**

Open `kimodo/video/pipeline.py`. Update the public `video_to_motion` signature to accept `chunk_size` and `chunk_overlap`:

```python
def video_to_motion(
    video_path: str,
    *,
    output_path: Optional[str] = None,
    model_name: str = "Kimodo-SOMA-RP-v1.1",
    prompt: str = "",
    strength: float = 0.5,
    mode: Literal["sdedit", "constraints"] = "sdedit",
    estimator: str = "gvhmr",
    num_denoising_steps: int = 100,
    num_samples: int = 1,
    cfg_weight: Tuple[float, float] = (2.0, 0.5),
    post_processing: bool = True,
    seed: Optional[int] = None,
    person_idx: int = 0,
    save_intermediates: bool = False,
    device: Optional[str] = None,
    return_numpy: bool = True,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> dict:
```

Pass the chunking params through to `_run_sdedit`:

```python
    if mode == "sdedit":
        result = _run_sdedit(
            model=model, soma_in=soma_in, prompt=prompt, strength=strength,
            num_denoising_steps=num_denoising_steps, num_samples=num_samples,
            cfg_weight=cfg_weight, post_processing=post_processing,
            chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        )
```

Update `_run_sdedit` to wrap the encode → `_generate` → decode loop in chunking:

```python
def _run_sdedit(
    *,
    model,
    soma_in: SOMAMotion30,
    prompt: str,
    strength: float,
    num_denoising_steps: int,
    num_samples: int,
    cfg_weight: Tuple[float, float],
    post_processing: bool,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> dict:
    from .chunking import Window, chunk_motion, stitch_chunks

    # Encode SOMA motion to Kimodo's normalized feature space.
    encoded = model.motion_rep(
        soma_in.local_rot_mats,
        soma_in.root_positions,
        to_normalize=True,
    )
    if encoded.ndim == 2:
        encoded = encoded.unsqueeze(0)
    encoded = encoded.expand(num_samples, -1, -1).contiguous()

    # Defaults: chunk_size = model.max_frames, overlap = 1 s at model.fps.
    if chunk_size is None:
        chunk_size = int(getattr(model, "max_frames", encoded.shape[1]))
    if chunk_overlap is None:
        chunk_overlap = int(model.motion_rep.fps)

    cfg = (cfg_weight[0], 0.0)
    total_T = encoded.shape[1]

    # Encoded is [B, T, D]. Chunk over the time axis for each batch element.
    # For simplicity, chunk on the first batch element; replicate slices per sample.
    chunks = chunk_motion(encoded[0], chunk_size=chunk_size, overlap=chunk_overlap)

    cleaned_windows = []
    for win in chunks:
        win_init = win.motion.unsqueeze(0).expand(num_samples, -1, -1).contiguous()
        T = win_init.shape[1]
        pad_mask = torch.ones(num_samples, T, dtype=torch.bool, device=win_init.device)
        first_heading_angle = torch.zeros(num_samples, device=win_init.device)
        clean = model._generate(
            texts=[prompt] * num_samples,
            max_frames=T,
            num_denoising_steps=num_denoising_steps,
            pad_mask=pad_mask,
            first_heading_angle=first_heading_angle,
            motion_mask=None,
            observed_motion=None,
            cfg_weight=list(cfg),
            init_motion=win_init,
            strength=strength,
        )
        # Save first batch element back into Window form for stitching.
        cleaned_windows.append(Window(start=win.start, end=win.end, motion=clean[0]))

    # Stitch back to [T_total, D], then add batch dim.
    stitched = stitch_chunks(cleaned_windows, total_T=total_T).unsqueeze(0)

    output = model.motion_rep.inverse(stitched, is_normalized=True, return_numpy=False)

    if post_processing:
        from kimodo.postprocess import post_process_motion
        corrected = post_process_motion(
            output["local_rot_mats"],
            output["root_positions"],
            output["foot_contacts"],
            model.skeleton,
            [],
        )
        output.update(corrected)

    return output
```

Note: this implementation runs samples one batch at a time per window. For multi-sample with chunking, this means we only stitch the first sample. For v1, multi-sample with chunking is best supported by running the whole pipeline N times rather than batching samples through chunking. The test above uses `num_samples=1` (default), so this works correctly. If the user wants `num_samples > 1` with chunking, raise a clear `NotImplementedError` until proper batching is added (track as a follow-up).

Add this guard at the top of `_run_sdedit` after the `chunk_motion` call returns >1 windows:

```python
    if len(chunks) > 1 and num_samples > 1:
        raise NotImplementedError(
            "num_samples > 1 combined with long-video chunking is not yet supported. "
            "Run the pipeline N times for N samples, or pass a shorter video."
        )
```

- [ ] **Step 4: Run tests, verify pass**

```bash
pytest tests/video/test_pipeline_chunking.py tests/video/test_pipeline_sdedit.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add kimodo/video/pipeline.py tests/video/test_pipeline_chunking.py
git commit -m "video: add long-video chunking to SDEdit pipeline path"
```

---

## Task 14: CLI — `kimodo_video_gen`

**Files:**
- Create: `kimodo/scripts/video_to_motion.py`
- Create: `tests/video/test_cli.py`

- [ ] **Step 1: Write failing test**

Create `tests/video/test_cli.py`:

```python
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
    assert args.estimator == "gvhmr"
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
```

- [ ] **Step 2: Run test, verify it fails**

```bash
pytest tests/video/test_cli.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement the CLI**

Create `kimodo/scripts/video_to_motion.py`:

```python
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
    p.add_argument("--estimator", default="gvhmr",
                   help="Pose estimator (default: gvhmr)")
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
```

- [ ] **Step 4: Run tests, verify pass**

```bash
pytest tests/video/test_cli.py -v
```

Expected: all 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add kimodo/scripts/video_to_motion.py tests/video/test_cli.py
git commit -m "video: add kimodo_video_gen CLI"
```

---

## Task 15: Packaging — `[video]` extra and console script

**Files:**
- Modify: `pyproject.toml`
- Create: `tests/video/test_public_api.py`

- [ ] **Step 1: Write failing test**

Create `tests/video/test_public_api.py`:

```python
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
```

- [ ] **Step 2: Run test, verify it fails or partially passes**

```bash
pytest tests/video/test_public_api.py -v
```

Expected: the lazy-import test might pass already, but the others should pass too if Tasks 1–14 were done correctly. If `test_kimodo_base_import_does_not_load_video` fails, check that `kimodo/__init__.py` doesn't import `kimodo.video` — it should not.

- [ ] **Step 3: Update `pyproject.toml`**

Open `pyproject.toml`. Add a `video` entry to `[project.optional-dependencies]` (next to `demo`, `soma`, `all`) and a `kimodo_video_gen` entry to `[project.scripts]`.

After change, the relevant sections look like:

```toml
[project.optional-dependencies]
demo = [
  "viser @ git+https://github.com/nv-tlabs/kimodo-viser.git",
]
soma = [
  "py-soma-x @ git+https://github.com/NVlabs/SOMA-X.git"
]
video = [
  "py-soma-x @ git+https://github.com/NVlabs/SOMA-X.git",
  # GVHMR is installed separately by the user; see kimodo/video/estimators/gvhmr.py
]
all = [
  "viser @ git+https://github.com/nv-tlabs/kimodo-viser.git",
  "py-soma-x @ git+https://github.com/NVlabs/SOMA-X.git"
]

[project.scripts]
kimodo_gen = "kimodo.scripts.generate:main"
kimodo_demo = "kimodo.demo:main"
kimodo_textencoder = "kimodo.scripts.run_text_encoder_server:main"
kimodo_convert = "kimodo.scripts.motion_convert:main"
kimodo_video_gen = "kimodo.scripts.video_to_motion:main"
```

- [ ] **Step 4: Verify the package still installs cleanly**

```bash
pip install -e . --no-deps  # verify pyproject.toml is valid; deps already installed
```

Expected: install succeeds, no errors. Then:

```bash
kimodo_video_gen --help
```

Expected: prints the CLI help text.

- [ ] **Step 5: Run all tests one final time**

```bash
pytest tests/ -v
```

Expected: every test passes.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml tests/video/test_public_api.py
git commit -m "video: register [video] extra and kimodo_video_gen console script"
```

---

## Manual validation (post-implementation)

These steps are not tasks in the strict sense — they require real videos and (optionally) the GVHMR install. Execute them after Task 15 is done.

- [ ] Install GVHMR per its README and confirm `pip install kimodo[video]` plus GVHMR brings the pipeline up.
- [ ] Run on a clean dance video (~5 s, single subject):
      `kimodo_video_gen sample_dance.mp4 --output dance_clean --strength 0.5`
      Load `dance_clean.npz` in the Viser demo and confirm the motion is smooth.
- [ ] Run on a wobbly phone video with camera motion (the "different-filmer" case). Confirm output is noticeably smoother than the raw GVHMR output.
- [ ] Run on a long video (>30 s) to exercise chunking. Confirm no visible seams at chunk boundaries.
- [ ] Run on a short clip (<2 s) to exercise the no-chunking path.
- [ ] Run on a video with multiple people — verify `--person_idx` works.
- [ ] Run on a video with no people — verify `NoMotionDetectedError` fires cleanly.
- [ ] Sanity-check existing functionality: run `kimodo_gen "a person walking"` and confirm the demo loads its output (regression sanity).

---

## Self-review

**Spec coverage:** Every spec section has at least one task —
- §3 non-goals: respected by `_MIN_FRAMES` check (no images), `--model` default (SOMA only), no auto-caption, single-subject only.
- §4 design overview: SDEdit (Tasks 6, 8, 11) + constraints fallback (Task 12).
- §5 architecture: matches file map.
- §6 data flow: Tasks 4, 5, 6, 8, 11, 13.
- §7 `_generate` modification: Task 8.
- §8 public API: Tasks 11 (Python) and 14 (CLI).
- §9 estimator strategy: Tasks 2, 9, 10.
- §10 chunking: Tasks 7, 13.
- §11 errors: Task 3.
- §12 testing: every task has TDD tests; regression guard is Task 8.
- §13 backward-compat: enforced by Task 8 (gated change + regression test).
- §14 future work: stubs ship in Task 9 (`gem` named).
- §15 licensing: not implementation — informational only.

**Placeholder scan:** No `TBD` / `TODO` / `implement later`. Two explicit notes where the actual upstream API may need adjustment (`py-soma-x.retarget_smplx` in Task 4; `gvhmr.run` in Task 10) — these are flagged with inline comments and the implementer is told what to check. Not placeholders, but assumptions the implementer must verify against the live packages.

**Type consistency:** `SOMAMotion30`, `SMPLXMotion`, `Window`, `PoseEstimator`, `video_to_motion` names and signatures are consistent across all tasks. The `_generate` signature change is consistent between Task 8 (definition), Task 11 (call site for SDEdit), and Task 13 (call site for chunked SDEdit). `cfg_weight` is treated as a tuple in the Python API and a list-of-two-floats when passed to `Kimodo.__call__` / `_generate` to match existing convention.

**Two known assumptions flagged for the implementer (not placeholders, but live-API checks):**
1. **Task 4 (`_smplx_to_soma`):** the `py-soma-x` function name and return-dict keys are inferred. If the live package's API differs, update `retarget()` in the helper and the corresponding mock in the test. The retarget call signature is the only thing that changes; the public surface is stable.
2. **Task 10 (`gvhmr.py`):** the GVHMR `run(video_path, person_idx=...)` call signature is inferred. Verify against the live GVHMR README; if it uses a different entry point or returns different keys, update `GVHMRAdapter.estimate()` to match.

Both assumptions are isolated to a single function in a single file. Adjusting them does not affect any other task.
