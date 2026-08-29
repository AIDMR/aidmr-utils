# Migrating the AID-MR repos onto aidmr-utils

**Status: phases 1-3 are done and live on an `aidmr-utils` branch in all seven
repos** (AMP, CMRQ, AIFS, BPF, LGEP, AIDMR-FIRE, AIDMR-dashboard). Phases 4-5
remain. The per-repo notes below are kept as the record of what changed and why.

Nothing is pushed: the branches resolve aidmr-utils through a
`[tool.uv.sources]` path entry. Swap those for a pinned git tag once
`github.com/AIDMR/aidmr-utils` exists, and change the FIRE Dockerfiles'
`AIDMR_UTILS_REF` ARG from `main` to that tag.

Ordered by risk reduced per unit of effort, not by module size. Each phase is
independently shippable; nothing below depends on a later phase.

**Delete the vendored copy in the same PR that adopts the package.** No
deprecation window, no compatibility shim, no re-export of the old name. A
half-migrated repo is harder to reason about than either end state, and the
schema itself already sets this precedent ("the version-1 mirror lists are
**gone**, not deprecated").

---

## Phase 0 — stop the drift (do this first, it is an hour)

Nothing here changes behaviour. It stops the situation getting worse while the
real work happens.

1. Add `docs/check_vendored.py` to each model repo, or a single script in
   AIDMR-FIRE, that hashes the still-vendored files against
   `AIDMR-dashboard/schema/` and fails on mismatch — **ignoring line endings**,
   since that is one of the two ways FIRE's copy had already drifted.
2. Pin the FIRE Dockerfile's clones to tags rather than `-b main`. A FIRE image
   is currently not reproducible: rebuilding picks up whatever was on main that
   morning, for five repos at once.

---

## Phase 1 — `result`  (DONE)

Six copies, five of them byte-identical. Nothing to reconcile.

```python
# before
from .aidmr import AIDMRResult, verdict, resolved_by      # or ..aidmr in AMP
# after
from aidmr_utils.result import AIDMRResult, verdict, resolved_by
```

| Repo | Delete | Notes |
|---|---|---|
| CMRQ | `src/lib/aidmr.py` | identical |
| AIFS | `src/lib/aidmr.py` | identical |
| BPF | `src/lib/aidmr.py` | identical |
| AMP | `src/lib/inference/aidmr.py` | identical |
| AIDMR-FIRE | `aidmr.py` | **drifted** — keep the `from notifications import manager` line in `server.py`, not in the schema module |
| dashboard | `schema/aidmr.py` | keep `schema/README.md`; it documents the payload for the TypeScript side |

Also delete all five `verify_aidmr_result.py` — `tests/test_result.py` replaces
them. Port anything they assert that the new suite does not; they existed in
three mutually different versions, so read all three before deleting.

**FIRE and every program must move together.** `from_dict` now refuses a payload
written by a newer aidmr-utils than the host runs, which makes a staggered
rollout fail loudly at the boundary instead of silently dropping fields.

---

## Phase 2 — `windowing` + `dicom`  (DONE)

This is the phase that fixes the Philips bug.

```python
# before (CMRQ, AIFS)
img = read_image_from_dicom(dcm, use_dicom_level, norm_min, norm_max)
# before (BPF, LGEP)
img = read_image_from_dicom(dcm, window_method)
# after — everywhere
from aidmr_utils.dicom import read_image, read_series
img    = read_image(dcm, cfg.data.windowing)      # '5_95', '0-100', 'dicom'
frames = read_series(dcms, cfg.data.windowing)    # per-series percentile
```

Per repo:

- **CMRQ** — `read_image_from_dicom` returned a 3-channel stack; the package
  returns single-channel. Add the `np.stack((img,)*3, -1)` at the call site in
  `datasets.py` and `predict_on_dicoms.py`, where it belongs. Gains the rescale.
- **AIFS** — **fix the truthiness bug as part of this.** All four call sites pass
  `cfg.data.windowing` into a `use_dicom_level` boolean, so configs 137 and 140
  ask for `0-100` and get DICOM WC/WW. `read_image` raises on a bare bool, so a
  missed call site cannot survive the migration silently. Delete the unused
  `dicom_level_norm_min_max_from_dicoms`. **Re-check whether the deployed models
  were trained on what their configs claim** before assuming this is cosmetic.
- **BPF** — closest to the package already; `read_series_from_dicoms` becomes
  `read_series`, `MixedFrameSizesError` moves. `DEPLOYABLE_WINDOWINGS` and the
  YAML octal guard came from here and are preserved.
- **LGEP** — already applies rescale (cfg `064` is literally "the dicom windowing
  fix for philips/GE images"). Straight swap; gains MONOCHROME1 handling.

`'5-95'` and `'5_95'` both parse, so **no config file has to change**.

Verify with `docs/verify_against_existing.py`: percentile windowing is currently
byte-identical to BPF across 165 real series including 24 Philips.

---

## Phase 3 — `mrd` + `pixel`  (DONE)

Built from **BPF's** `mrd.py` (newest, true colour, the only one with the
2026-08-29 layout fix), plus AMP's parity coercion and dimension validation.
`pixel` came with it because `mrd` depends on `assert_square_pixels`.

```python
# after
from aidmr_utils.mrd import (np_float_to_mrd, standard_image_meta,
                             report_image_to_mrd, padded_square_geometry,
                             native_pixel_spacing_mm)
```

What each repo has to change:

| | change |
|---|---|
| **AMP** | `get_standard_ismrmrd_meta_for_uint16_image` -> `standard_image_meta`, keyword-only, `seres_desc` typo gone. **Pass `row_dir`/`col_dir`** — AMP currently declares no layout at all. Drop `title=` and `output_shape_yx=`: do them at the call site (`np_float_to_mrd(put_text_on_img(img, t), ...)`). Its outputs now hit `assert_square_pixels`; a failure there is a real finding, not a regression. |
| **CMRQ** | same rename. **Delete the commented-out block** — it writes `ImageColDir`, the wrong key — and pass `row_dir`/`col_dir` instead. Gains unit-normalised direction vectors and loses the square-matrix assert. `WindowCenter`/`WindowWidth` now only default when there is no `base_meta`, matching AMP and BPF. |
| **BPF** | closest already; `get_standard_ismrmrd_meta` -> `standard_image_meta`, and pass `program='bpf'` to keep `ImageProcessingHistory` as it was. |
| **AIFS** | only `parse_h5_into_fire_arguments` — its `mrd.py` is 34 lines. Note the config is now an argument rather than AMP's hard-coded planning parameters. |

Consolidating surfaced a latent bug in **all three** originals: the greyscale
path never set `head.channels`, so a template header carrying `channels=0` gave a
zero-size `.data` array. Verified against BPF's — it returns `shape=(0,1,64,64)`.
Production headers carry 1, which is why nobody hit it.

---

## Phase 4 — AMP's sagittal, and AMP's outgoing layout  (REMAINING)

The orientation adoption and the cross-repo test deletion happened in phase 3's
sweep — CMRQ and BPF both use `aidmr_utils.orientation` and
`BPF/docs/verify_orientation.py` is gone. What is left is AMP-specific, and the
two halves belong together:

- **AMP still does not declare `ImageRowDir` / `ImageColumnDir` on outgoing
  images.** BPF fixed this on 2026-08-29 and CMRQ now does it too, but AMP's
  previews are slices of the patient with real geometry rather than pages, so
  the pair has to come from AMP's own geometry, not the report identity triple.
- Fix AMP's `get_default_right_down_unit_vectors_for_freq_phase` sagittal branch
  (`vec_right = -v_phase_xyz` -> `+v_phase_xyz`). Low urgency: its call site notes
  the vectors do not affect slice planning, so the only symptom is a mirrored 2ch
  preview. Delete the now-false "the two repos are deliberately inconsistent"
  notes in CMRQ and BPF.
- AMP's `Direction` enum has `RIGHT = auto()  # -v`, which should read `-x`.

---

## Phase 5 — the rest

`onnx.load_model_onnx` (4 copies; take AMP's, it has session options and a CPU
fallback — and delete AMP_LOC's, which is legacy), `imaging` (`put_text_on_img`,
`pad_to_square`, the draw primitives from AMP's `utils.py`), and the FIRE glue
(`ReplayableConnection`, the `process()` contract).

Do **not** move: `lightning.py`, `datasets.py`, `models.py`, `losses.py`,
`optimizers.py`, `transforms.py`, `callbacks.py`. They look duplicated but each
is tuned to its own model, they change independently, and none is on the
deployment path. Sharing them buys coupling with no safety payoff.

`CMRQ/src/lib/inference_utils.load_ordered_dicoms_from_dir` has an unconditional
`raise ValueError` mid-function with unreachable code after it. It is dead —
delete it rather than migrating it.

---

## The FIRE Dockerfile, after

Five `git clone` + `cp -rf` blocks for shared code become one pinned install:

```dockerfile
RUN pip3 install --no-cache-dir \
    "aidmr-utils[mrd,onnx] @ git+https://github.com/AIDMR/aidmr-utils@v0.3.0"
```

Note the extras: the container needs `mrd` and `onnx` but **not** `dicom` — it
never reads a DICOM. That is why `windowing` is dependency-free and `dicom` is an
extra.

This also removes the ordering hazard where BPF's `fire_inference` imports
`amp_lib.inference.models`, so BPF silently depends on AMP's clone having run
first.

---

## Notes

- Python floor is `>=3.11` because CMRQ, AIFS and LGEP are. Raising it to 3.12
  (AMP and BPF already are) means moving those three first.
- The container installs its own pinned numpy/scipy/torch and reads none of the
  `pyproject.toml` files — a fourth environment that nothing describes. The
  package's extras are the first declared account of what the deployed code
  actually needs.
- `AMP`'s git remote has a GitHub PAT embedded in the URL in `.git/config`. Local
  only and not committed, but worth rotating and switching to a credential
  helper. (`AIDMR-FIRE/*.secret` is correctly gitignored.)
- AMPLOC is described as legacy but is still downloaded in the Dockerfile, loaded
  in `server.py`, and ONNX-warmed on every boot. Deleting it is free.
