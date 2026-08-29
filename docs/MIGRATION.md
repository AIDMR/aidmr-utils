# Migrating the AID-MR repos onto aidmr-utils

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

## Phase 1 — `result`  (highest confidence, zero merge risk)

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

## Phase 2 — `windowing` + `dicom`  (highest correctness payoff)

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

## Phase 3 — `mrd`  (closes the live orientation bug)

Not yet in the package. Take **BPF's** `mrd.py` as the base: it is the newest,
it has true-colour support, and it is the only one carrying the 2026-08-29
`ImageRowDir`/`ImageColumnDir` fix.

Reconcile on the way in:

| | CMRQ | AMP | BPF |
|---|---|---|---|
| `np_float_to_mrd` | stale subset; **does not unit-normalise** the direction vectors; asserts square matrix | superset: `title`, parity coercion, dim validation | superset of AMP + real colour + square-pixel assert |
| meta builder name | `get_standard_ismrmrd_meta_for_uint16_image` | same (`seres_desc` typo) | `get_standard_ismrmrd_meta` |
| `ImageRowDir` out | **commented out, and writes `ImageColDir` — the wrong key** | absent | correct |
| WC/WW | set unconditionally | only when `base_meta is None` | only when `base_meta is None` |

Settle on one name (`standard_image_meta`), one signature
(`(series_desc, *, base_meta=None, row_dir=None, col_dir=None)`), and fix the
`seres_desc` typo. Back-porting the row/col dir fix to AMP and CMRQ is the point
of this phase — both currently send images back without declaring a layout.

---

## Phase 4 — `pixel`, and AMP's geometry

`orientation` and `geometry` already shipped; the sagittal question that blocked
them is settled (see the README).

- Move BPF's `geometry.py` (mm-per-pixel, anisotropy, square-pixel asserts) in as
  `aidmr_utils.pixel`. **Do not call it `geometry`** — AMP's `geometry.py` is
  cardiac plane geometry and has no overlap with it. Same filename, two subjects,
  which is how the estate got here.
- Adopt `aidmr_utils.orientation` in CMRQ and BPF; delete both copies and
  `BPF/docs/verify_orientation.py`, whose whole purpose was checking the two
  against each other.
- Fix AMP's `get_default_right_down_unit_vectors_for_freq_phase` sagittal branch
  (`vec_right = -v_phase_xyz` → `+v_phase_xyz`). Low urgency: its call site notes
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
