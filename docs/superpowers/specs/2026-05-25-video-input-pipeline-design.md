# Video Input Pipeline for Kimodo — Design Spec

**Date:** 2026-05-25
**Status:** Draft — pending user review
**Owner:** Yusuke Yanagawa

---

## 1. Problem

Today Kimodo generates motion from text prompts (and optional kinematic constraints authored in the interactive demo). Users with a video of a human performing a motion have no first-class way to bring that motion into Kimodo. Modern world-grounded 3D pose estimators (GVHMR, WHAM, 4DHumans) can recover full-body SMPL-X motion from monocular video, but their outputs have two recurring problems:

1. **Per-frame wiggle / jitter.** Estimator outputs are temporally noisy — foot skate, joint twitching, micro-jitter on otherwise still limbs.
2. **Filmer-dependent variation.** The same physical motion, shot by different people with different camera trajectories and angles, produces visibly different estimator outputs. There is no canonical, normalized representation.

A pose estimator's output is already "an animation" in the literal sense, but it is not usable as-is for downstream pipelines that need clean, canonical, retargeted motion (ProtoMotions training data, robotics, animation authoring).

## 2. Goal

Let users drop a video in and get clean, canonical motion on Kimodo's SOMA-77 skeleton out — with Kimodo's diffusion prior acting as both a denoiser and a retargeter.

**Primary value props:**
- **Stability / cleanup:** Kimodo's motion prior removes wiggle and jitter from noisy estimator output.
- **Canonical skeleton:** Output is always on SOMA-77, regardless of input source variation.

**Secondary properties:**
- Optional text prompt for steering when video is ambiguous.
- Output format identical to `kimodo_gen` (NPZ), so downstream tooling (demo, ProtoMotions, GMR) works unchanged.
- Existing Kimodo functionality is preserved bit-for-bit. The new pipeline is purely additive.

## 3. Non-goals

- **Motion imitation / high-fidelity reproduction.** If the user wants to faithfully replicate the exact motion in the video, they should use the pose estimator directly. This pipeline trades fidelity for cleanliness.
- **Multi-person scene reconstruction.** Pipeline operates on a single subject (selected via `--person_idx` if multiple detected).
- **Real-time / interactive operation.** Offline batch processing only in v1.
- **Bundled estimator weights.** The pose estimator is installed and configured by the user (Section 8).
- **G1 robot / SMPL-X output.** SOMA-77 only in v1. (Other skeletons trivially supported later by changing `model_name`.)
- **Auto-captioning the video.** The text prompt is user-provided or empty.

## 4. Design overview

The pipeline shape is:

```
video.mp4
  → pose estimator adapter (pluggable; returns SOMA-30 directly)
       — GVHMR adapter: SMPL-X internally, retargets to SOMA before returning
       — GEM adapter (future): SOMA natively, no SMPL-X involvement
  → Kimodo motion_rep encoding + normalization
  → SDEdit-style partial-noise denoising (the key technique)
  → Kimodo motion_rep decoding
  → optional post-processing (foot-skate cleanup)
  → SOMA-30 → SOMA-77 (existing path)
  → NPZ output
```

**SOMA is the pipeline's only public motion type.** SMPL-X (or any other intermediate representation) is a private implementation detail of specific estimator adapters. Swapping to a SOMA-native estimator (e.g. NVIDIA's GEM, when an adapter ships) requires no changes to the orchestrator, the public API, the CLI, or `_generate`. This also keeps SMPL-X's license burden encapsulated inside adapters that use it — relevant for commercial deployment (see §15).

The novel ingredient is **SDEdit-style denoising**: instead of starting the diffusion sampling loop from pure Gaussian noise, we encode the noisy motion into Kimodo's feature space, add diffusion noise at an intermediate timestep `k = strength × num_denoising_steps`, and let Kimodo denoise from step `k` down to `0`. The diffusion prior pulls the wiggly observation toward the manifold of plausible motion. A single `strength ∈ [0, 1]` knob controls the cleanup-vs-fidelity tradeoff:

- `strength = 0.3` — light cleanup, output closely tracks input minus jitter
- `strength = 0.5` — balanced (default)
- `strength = 0.7` — heavy cleanup, only rough motion shape preserved
- `strength = 1.0` — equivalent to vanilla Kimodo (ignore the video)

A secondary **constraints mode** (`--mode constraints`) is provided as a fallback: it routes the retargeted motion through Kimodo's existing constraint pathway (`FullBodyConstraintSet` + `EndEffectorConstraintSet`) with `cfg_weight=(2.0, 0.5)`. This mode produces an editable `constraints.json` compatible with the Viser demo, at the cost of weaker denoising than SDEdit.

## 5. Architecture & module layout

```
kimodo/video/                          # NEW package; opt-in via pip install kimodo[video]
    __init__.py                        # public: video_to_motion, SOMAMotion30, registry
    pipeline.py                        # orchestrator (video → motion); operates in SOMA only
    pose_estimator.py                  # PoseEstimator protocol + registry
    estimators/
        __init__.py
        gvhmr.py                       # GVHMR adapter (uses SMPL-X internally → SOMA out)
        gem.py                         # stub for NVIDIA GEM (SOMA-native, commercial-friendly)
        wham.py                        # stub, raises NotImplementedError
        fourdhumans.py                 # stub, raises NotImplementedError
    _smplx_to_soma.py                  # private helper used by SMPL-X-based adapters; wraps SOMA-X
    _types.py                          # internal SMPLXMotion (for adapter use only)
    sdedit.py                          # SDEdit init helper (pure function)
    chunking.py                        # overlap-blend windowing for long videos
    types.py                           # public: SOMAMotion30 dataclass

kimodo/model/kimodo_model.py           # MODIFIED: _generate gains 2 optional kwargs
                                       #          (gated; existing path unchanged)

kimodo/scripts/video_to_motion.py      # NEW CLI: `kimodo_video_gen`

tests/video/                           # NEW test directory
    test_chunking.py
    test_pose_estimator_registry.py
    test_resample.py
    test_smplx_to_soma_shapes.py       # internal helper (used by SMPL-X-based adapters)
    test_sdedit_init.py
    test_constraints_mode.py
    test_errors.py
    test_regression_existing_path.py   # safety net for backward compat
    fixtures/
        short_clip.mp4
        reference_outputs/
            <model>_<prompt_hash>.pt
```

**Module boundaries:**

- `PoseEstimator.estimate(video_path, *, person_idx=0) → SOMAMotion30` — pluggable; every adapter returns SOMA, full stop. Any intermediate representation (SMPL-X, GHUM, raw keypoints) is private to the adapter.
- `kimodo.video._smplx_to_soma.retarget(SMPLXMotion) → SOMAMotion30` — internal helper used by SMPL-X-based adapters (GVHMR, future WHAM/4DHumans). Not part of the public API.
- `sdedit_init(motion, strength, diffusion) → (noisy_init, start_step)` — pure function over the `Diffusion` object.
- `chunk_motion(motion, chunk_size, overlap) → list[Window]` + `stitch_chunks(windows) → motion` — pure functions.
- `Kimodo._generate(..., init_motion=None, strength=0.5)` — one new code path, gated on `init_motion is not None`.

The pose estimator and its intermediate-representation deps (SMPL-X libs, etc.) are imported lazily inside the adapter module, so `import kimodo` and `import kimodo.video` both stay cheap and have no new transitive dependencies.

## 6. Pipeline data flow

```
video.mp4
   │  PoseEstimator.estimate(video_path, person_idx=0)
   │    ┌─ inside the adapter (private):
   │    │    GVHMR/WHAM/4DHumans:  video → SMPL-X → _smplx_to_soma.retarget → SOMA-30
   │    │    GEM (future):         video → SOMA-30 (no SMPL-X)
   │    └─
   ↓
SOMAMotion30 {
    local_rot_mats:  [T_in, 30, 3, 3]
    root_positions:  [T_in, 3]
    fps:             int (estimator-native; typ. 30)
}
   │  resample_to_fps(motion, target=model.fps)
   ↓
SOMAMotion30 @ model.fps
   │  motion_rep(local_rot_mats, root_positions, to_normalize=True)
   ↓
encoded_motion: [T, motion_rep_dim]
   │  optional: chunk_motion(...) if T > model.max_frames
   ↓
   │  Kimodo._generate(
   │       texts=[prompt or ""] * num_samples,
   │       init_motion=encoded_motion.unsqueeze(0).repeat(num_samples, 1, 1),
   │       strength=0.5,
   │       num_denoising_steps=100,
   │       cfg_weight=[2.0, 0.0],   # text on, constraint guidance off (no observed_motion in SDEdit mode)
   │       ...
   │  )
   ↓
clean_motion: [B, T, motion_rep_dim]
   │  optional: stitch_chunks(...) if chunked
   │  motion_rep.inverse(clean_motion, is_normalized=True)
   │  optional: post_process_motion(...)
   │  SOMASkeleton30.output_to_SOMASkeleton77(...)  (existing)
   ↓
output dict: {posed_joints, global_rot_mats, local_rot_mats,
              foot_contacts, smooth_root_pos, root_positions, global_root_heading}
   │  save_kimodo_npz(output, output_path)
   ↓
clean.npz
```

In constraints mode, the pipeline branches after resample: instead of encoding to `motion_rep` and going through SDEdit, the SOMA-30 motion is wrapped in a `FullBodyConstraintSet` + `EndEffectorConstraintSet` and passed to the unmodified `Kimodo.__call__(..., constraint_lst=...)` with `cfg_weight=(2.0, 0.5)`.

## 7. The `_generate` modification

The only change to `kimodo/model/kimodo_model.py`. The init block (currently around line 608) becomes:

```python
shape = (batch_size, max_frames, self.motion_rep.motion_rep_dim)
num_denoising_steps_tensor = torch.tensor([num_denoising_steps], device=self.device)
use_timesteps = self.diffusion.space_timesteps(num_denoising_steps_tensor[0])[0]
self.diffusion.calc_diffusion_vars(use_timesteps)

if init_motion is None:
    # Existing path: pure-noise init (byte-identical to current behavior)
    cur_mot = torch.randn(shape, device=self.device)
    indices = list(range(num_denoising_steps))[::-1]
else:
    # SDEdit path: noise init_motion to step k, denoise from there
    k = max(1, int(strength * num_denoising_steps))
    t_init = torch.full((batch_size,), k - 1, dtype=torch.long, device=self.device)
    cur_mot = self.diffusion.q_sample(init_motion, t_init)
    indices = list(range(k))[::-1]

# Denoising loop body unchanged
for i in progress_bar(indices):
    t = torch.tensor([i] * cur_mot.size(0), device=self.device)
    with torch.inference_mode():
        cur_mot = self.denoising_step(...)
return cur_mot
```

Two new optional kwargs in `_generate`:

- `init_motion: Optional[Tensor] = None` — pre-encoded, normalized motion features of shape `[B, T, motion_rep_dim]`. When `None`, behavior is identical to today.
- `strength: float = 0.5` — SDEdit strength in `[0, 1]`. Only consulted when `init_motion is not None`.

`Kimodo.__call__` is **not modified**. The new pipeline calls `model._generate(...)` directly from `kimodo/video/pipeline.py`.

## 8. Public API

### CLI: `kimodo_video_gen`

```
kimodo_video_gen path/to/video.mp4 [OPTIONS]

Options:
  --output STEM            Output NPZ stem (default: "video_output")
  --prompt TEXT            Optional text steering (default: "")
  --strength FLOAT         SDEdit cleanup strength in [0,1] (default: 0.5)
  --mode {sdedit,constraints}
                           Pipeline mode (default: sdedit)
  --model NAME             Kimodo model (default: Kimodo-SOMA-RP-v1.1)
  --estimator NAME         Pose estimator (default: gvhmr)
  --diffusion_steps INT    DDIM steps (default: 100)
  --num_samples INT        Number of samples (default: 1)
  --cfg_weight T C         Text/constraint CFG; constraints mode only (default: 2.0 0.5)
  --no-postprocess         Disable foot-skate cleanup
  --seed INT               Random seed
  --save_intermediates     Save adapter-specific intermediates + SOMA output to <stem>_intermediates/
                           (e.g. raw SMPL-X for GVHMR; nothing extra for SOMA-native adapters)
  --person_idx INT         If multiple people detected, pick this index (default: 0)
  --chunk_size INT         Override default chunk size (default: model.max_frames)
  --chunk_overlap INT      Override default overlap in frames (default: 1.0s at model.fps)
```

Output naming mirrors `kimodo_gen`: single sample writes `<stem>.npz`; multiple samples write `<stem>/<stem>_NN.npz`. Constraints mode additionally writes `<stem>_constraints.json` (demo-loadable).

### Python: `kimodo.video.video_to_motion`

```python
from kimodo.video import video_to_motion

result = video_to_motion(
    video_path: str,
    output_path: Optional[str] = None,        # if set, also writes NPZ
    *,
    model_name: str = "Kimodo-SOMA-RP-v1.1",
    prompt: str = "",
    strength: float = 0.5,
    mode: Literal["sdedit", "constraints"] = "sdedit",
    estimator: str = "gvhmr",
    num_denoising_steps: int = 100,
    num_samples: int = 1,
    cfg_weight: tuple[float, float] = (2.0, 0.5),  # honored only in mode="constraints"
    post_processing: bool = True,
    seed: Optional[int] = None,
    person_idx: int = 0,
    save_intermediates: bool = False,
    device: Optional[str] = None,
    return_numpy: bool = True,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> dict  # same schema as Kimodo.__call__ output
```

**`cfg_weight` semantics:** In `mode="sdedit"`, no `observed_motion` is passed to the denoiser, so the constraint CFG weight has nothing to guide; the pipeline forces it to `(cfg_weight[0], 0.0)` regardless of the user's value. In `mode="constraints"`, the user's `cfg_weight` is passed through to `Kimodo.__call__` unchanged.

Lower-level building blocks are also importable:

```python
from kimodo.video import (
    PoseEstimator, get_pose_estimator, register_pose_estimator,
    resample_to_fps,
    SOMAMotion30,
)
```

Note: `SMPLXMotion` and the SMPL-X → SOMA retargeter are intentionally *not* public — they live in `kimodo.video._types` and `kimodo.video._smplx_to_soma` as adapter-internal implementation details. The public motion type is SOMA, always.

## 9. Pose estimator strategy

A `PoseEstimator` protocol + lazy-loaded registry. Adapters live in `kimodo/video/estimators/` and are imported only when their factory is called. **Every adapter returns SOMA**; the body-model details an adapter uses internally are not visible to the rest of the pipeline.

```python
class PoseEstimator(Protocol):
    def estimate(self, video_path: str, *, person_idx: int = 0) -> SOMAMotion30: ...

_REGISTRY: dict[str, Callable[[], PoseEstimator]] = {}

def register_pose_estimator(name: str):
    def deco(factory):
        _REGISTRY[name] = factory
        return factory
    return deco

def get_pose_estimator(name: str) -> PoseEstimator:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown estimator '{name}'. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]()
```

**Default: GVHMR** (current SOTA for world-grounded SMPL-X). It decouples subject motion from camera motion, directly addressing the "filmer-dependent variation" problem. The GVHMR adapter encapsulates SMPL-X internally and uses `kimodo.video._smplx_to_soma.retarget` to return SOMA-30 to the pipeline. SMPL-X never escapes the adapter.

**Distribution:** GVHMR is research code, not a clean pip package. We use **lazy import** — `pip install kimodo[video]` installs core deps; the user installs GVHMR separately per its README; our adapter imports it on first use and raises `EstimatorNotInstalledError` with the install URL if missing.

**Stubs ship for `gem`, `wham`, and `4dhumans`** (registered, raise `NotImplementedError` on use) so they appear in the registry as discoverable extension points. The `gem` stub is intentionally surfaced as the commercial-friendly migration target: a future GEM adapter outputs SOMA natively (no SMPL-X anywhere), which is the clean path for commercial deployment (see §15).

## 10. Long-video chunking

```python
def chunk_motion(motion: Tensor, chunk_size: int, overlap: int) -> list[Window]:
    """Split [T, D] motion into overlapping [chunk_size, D] windows."""

def stitch_chunks(windows: list[CleanedWindow], total_T: int) -> Tensor:
    """Blend cleaned windows back into [T, D] with raised-cosine alpha on overlaps."""
```

- **Defaults:** `chunk_size = model.max_frames`, `overlap = 1.0 s at model.fps` (~20 frames at 20 fps).
- **Strategy:** Each window is SDEdit-denoised independently (can be batched together for GPU efficiency); overlapping regions are blended with a raised-cosine weighting so seams are smooth.
- **Edge cases:**
  - `T ≤ chunk_size` → one window, no blending.
  - Last window shorter than `chunk_size` → padded for SDEdit, truncated on output.
  - `T_total < 2 × overlap` → falls back to single window.

This is not entangled with Kimodo's existing `multi_prompt` mode, which is designed for *generating* fresh segments with transitions — bolting SDEdit onto it would be invasive. Overlap-blend chunking is a small, self-contained module.

## 11. Error handling

Typed exceptions with actionable messages. No silent failures.

| Exception | Trigger | Message includes |
|---|---|---|
| `EstimatorNotInstalledError` | Adapter import fails | Install instructions URL for that estimator |
| `NoMotionDetectedError` | Estimator returns 0 frames | "No person detected — try `--person_idx`, check video quality" |
| `MultiplePersonsDetectedError` | N > 1 and `person_idx` unspecified | List of detected person IDs with bbox previews |
| `VideoTooShortError` | T < 10 frames at target fps | Min duration |
| `RetargetingError` | SOMA-X retargeter raises | Underlying error wrapped |
| `InvalidStrengthError` | `strength` not in `[0, 1]` | Valid range |

Errors during the pipeline never leave the Kimodo model in a corrupt state — all mutations in `_generate` happen on local tensors.

## 12. Testing strategy

**Layer 1 — unit tests (fast, no GPU, no estimator deps):**
- Registry add/lookup
- Frame-rate resampling math (30 fps → 20 fps preserves duration ±1 frame)
- Chunk + stitch round-trip on synthetic motion
- SDEdit init shape + properties (`strength=0` ≈ input, `strength=1` ≈ pure noise)
- Constraint-mode pre-processing builds expected `FullBodyConstraintSet` / `EndEffectorConstraintSet`
- Each typed exception fires when expected

**Layer 2 — regression guard (the backward-compat safety net):**
- `test_existing_call_unchanged`: with a fixed seed, calls `Kimodo.__call__` with the standard arg shape used by `kimodo_gen` and `_multiprompt`, asserts the output tensor matches a checked-in reference. Any drift in the existing code path trips this test.
- `test_base_install_clean`: confirms `pip install kimodo` (no `[video]`) imports cleanly and `kimodo_gen --help` runs without pulling in `kimodo/video/`.

**Layer 3 — integration tests (`@pytest.mark.slow`, GPU):**
- Synthetic wiggly motion → SDEdit → assert velocity variance drops monotonically as strength rises.
- Short test video (≤ 2 s, in `tests/fixtures/`) → full `video_to_motion(...)` → assert output schema matches Kimodo's NPZ spec.
- Same E2E on `--mode constraints` → assert constraints file is demo-loadable.

**Manual validation checklist (pre-merge):**
- Clean dance video
- Wobbly phone video with camera motion (the "different filmer" case)
- Long video (> 30 s) — exercises chunking
- Short clip (< 2 s) — exercises no-chunking path
- Multi-person video — exercises `--person_idx`
- Video with no people — exercises `NoMotionDetectedError`
- All outputs loaded in the Viser demo for visual confirmation

## 13. Backward-compatibility commitments

This is a load-bearing constraint, called out explicitly:

| Surface | Change |
|---|---|
| `Kimodo.__call__` signature & behavior | **None.** |
| Existing constraint pathway | **None in the model.** |
| Default `pip install kimodo` deps | **None.** Video deps live behind `[video]` extra. |
| Existing CLI (`kimodo_gen`) | **None.** New entry point is `kimodo_video_gen`. |
| Existing tests | **None.** Must continue to pass unmodified. |
| Interactive demo | **None.** Demo does not import `kimodo/video/`. |
| `kimodo/model/kimodo_model.py` | The only existing file touched. Eight lines added, gated on `init_motion is not None`. The existing path is byte-identical. |

The Layer 2 regression test (`test_existing_call_unchanged`) is the enforcement mechanism for this guarantee.

## 14. Future work (out of scope for v1)

- **GEM adapter (commercial-friendly path).** Stub ships in v1 (raises `NotImplementedError`); a real GEM adapter would output SOMA directly with no SMPL-X involvement, clearing the SMPL-X commercial-license blocker. Highest-leverage follow-up if commercial use is a real goal.
- Additional pose estimator adapters (WHAM, 4DHumans) — stubs ship; full implementations in follow-up PRs.
- HTTP-service estimator adapter (`gvhmr_api`) for users who would rather not install GVHMR locally.
- G1 and SMPL-X output paths (trivially supported by passing a different `model_name`).
- Image input (single frame). Currently the pipeline requires video; image is conceptually a one-frame video but the `video_to_motion` API will reject T=1 with `VideoTooShortError`. Could be re-enabled with a pose-conditioned generation mode (different from SDEdit) in a future iteration.
- Auto-caption integration for the text prompt.
- Multi-person scene reconstruction.
- Real-time / streaming operation.

## 15. Licensing notes (commercial use)

This is a heads-up, not a blocker. The pipeline's design is license-agnostic; these are run-time considerations the owner should verify before any commercial deployment:

- **Kimodo code (this repo):** Apache 2.0 — commercial use permitted.
- **Kimodo model weights:** Most variants ship under the NVIDIA Open Model License (commercial use generally permitted, terms apply). The **Kimodo-SMPLX-RP-v1** variant is research-only (NVIDIA R&D Model License). The default `Kimodo-SOMA-RP-v1.1` is the Open Model variant.
- **GVHMR (default pose estimator):** Verify the upstream repo's license; many pose-estimation research projects are non-commercial. If commercial use is blocked, swap the estimator (the registry is built for this).
- **SMPL / SMPL-X body model:** Licensed by MPI for non-commercial research by default. In the v1 pipeline, SMPL-X is used **only inside SMPL-X-based adapters** (GVHMR, future WHAM/4DHumans) — it never appears in the public API, the orchestrator, or any output. Commercial deployment options: (a) license SMPL-X commercially from MPI, (b) swap to a SOMA-native estimator adapter (GEM, when its adapter lands), or (c) build a custom adapter that bypasses SMPL-X.
- **SOMA / SOMA-X retargeter:** NVIDIA NVlabs project — verify upstream license; expected to be permissive.
- **Training data:** Affects model weights, not user code. Bones Rigplay 1 is proprietary; BONES-SEED is publicly available.

**The recommended commercial path is the GEM adapter** (currently a stub; see §14). Because SMPL-X is encapsulated inside specific adapters and not part of the pipeline's public motion type, swapping to a SOMA-native estimator removes the SMPL-X license dependency without touching `_generate`, the CLI, the orchestrator, or any other adapter.
