# aidmr-utils

One implementation of the things AMP, CMRQ, AIFS, BPF and LGEP all need, so a fix
made once is a fix everywhere.

```bash
pip install "aidmr-utils[dicom] @ git+https://github.com/AIDMR/aidmr-utils@v0.1.1"
```

Pin to a **tag**, never a branch. The FIRE container currently clones every model
repo at `-b main`, which means a rebuild silently picks up whatever happened to be
on main that morning; this package should not add to that.

## Why this exists

Five repos had grown their own copy of the same code. Not as a tidiness problem —
the duplication had already produced defects that reached the scanner:

| What was duplicated | State when this package was created |
|---|---|
| `aidmr.py` | 6 copies. Five byte-identical; FIRE's had drifted, despite a docstring saying drift causes silent data loss |
| `verify_aidmr_result.py` | 5 copies in **3 mutually different versions** — the drift detector had itself drifted |
| `np_float_to_mrd` | 3 copies: CMRQ ⊂ AMP ⊂ BPF. CMRQ's skips unit-normalising the direction vectors |
| `orientation.py` | 2 copies, kept in step by a hand-written cross-repo test (`BPF/docs/verify_orientation.py`) |
| `closest_plane` | 3 copies (algorithmically identical, at least) |
| `load_model_onnx` | 4 copies |
| `read_image_from_dicom` | 4 copies, **2 incompatible signatures**, 3 windowing string formats |

Two of those had live consequences:

**Outgoing image orientation.** BPF fixed a 180°-rotated measurements table on
2026-08-29 by declaring `ImageRowDir` / `ImageColumnDir` on outgoing images. AMP
never got the fix. CMRQ has it commented out — and the commented code writes
`ImageColDir`, the wrong key, so uncommenting it would not have worked either.

**The Philips rescale.** `read_image_from_dicom` in CMRQ and AIFS ignores
`RescaleSlope` / `RescaleIntercept`. Harmless under a percentile window (which is
exactly invariant to an affine rescale) and catastrophic under DICOM `WC/WW`,
where the tags are quoted in rescaled units. Measured on 11,458 scanner-written
DICOMs: **127/127 Philips images carry a non-identity rescale**, against almost no
Siemens and no GE. On a real Philips LGE series the un-rescaled window admits
**0.0%** of pixels — a uniform white frame.

## What is in it

| Module | Contents | Extra needed |
|---|---|---|
| `result` | `AIDMRResult`, `verdict`, `resolved_by` — the dashboard payload | — |
| `windowing` | `parse_window_spec`, `window`, `window_series` | — (numpy only) |
| `dicom` | `load_pixels`, `read_image`, `read_series` | `[dicom]` |
| `geometry` | `Plane`, `closest_plane`, `CANONICAL_RIGHT_DOWN` | — |
| `orientation` | `canonicalise_cine`, the transform group, `inverse_transform` | — |
| `pixel` | `mm_per_pixel_for_side`, `assert_square_pixels`, `anisotropy` | — |
| `mrd` | `np_float_to_mrd`, `standard_image_meta`, `report_image_to_mrd`, geometry readers | `[mrd]` |

`windowing` is deliberately dependency-free. Both sides of every deployment
import it — training reads DICOMs, FIRE receives MRD — and the two must apply the
identical mapping or the model sees intensities it was never trained on.

`dicom` and `mrd` are not re-exported from the package root, so importing
`aidmr_utils` works with neither pydicom nor ismrmrd installed. Import those two
by module.

Still to move: `onnx`, `imaging`, and the FIRE glue.

### The outgoing-layout fix

`mrd` carries BPF's 2026-08-29 fix, which AMP never got and CMRQ has commented
out. Outgoing images must declare `ImageRowDir` / `ImageColumnDir` in the
MetaAttributes — those, not `read_dir` / `phase_dir`, are what become DICOM's
`ImageOrientationPatient` and decide the displayed layout. Omitting them let the
scanner reorient a measurements table 180 degrees. `standard_image_meta` takes
`row_dir` / `col_dir`, and CMRQ's commented-out version wrote `ImageColDir`,
which is not the key anything reads.

`np_float_to_mrd` takes `require_square_pixels` (default `True`). BPF echoes
acquired geometry onto a padded square, so anisotropic pixels there mean the
geometry was got wrong and a Siemens reconstruction refuses the result — it needs
the assert. AMP *prescribes* slices with a deliberately rectangular field of view,
and its phase axis is not always the row axis, so it opts out. Making the check
unconditional imposed one program's invariant on another and broke 36 of AMP's
tests; that is what the parameter exists to prevent.

Consolidating also surfaced a latent bug present in **all three** originals: the
greyscale path never set `head.channels`, so a template header carrying
`channels=0` (a fresh `ImageHeader`, or one from a replayed capture) produced a
zero-size `.data` array. It never bit in production only because the headers FIRE
forwards happen to carry 1.

## The DICOM API, and why it is shaped like this

Getting pixels out of a DICOM is two jobs. The old code conflated them into one
function that grew two incompatible signatures:

```python
load_pixels(dcm)        # true stored values. ONE right answer, no arguments.
window(pixels, spec)    # a display range. A policy, so it takes a spec.
```

There is no code path that reaches pixel data without rescaling it, so the
Philips failure is not something you can forget any more.

```python
from aidmr_utils.dicom import read_series

frames = read_series(dcms, '0_100')     # per-series percentile, as the scanner does
frames = read_series(dcms, 'dicom')     # this image's own WC/WW
```

`read_image` **refuses a bare bool**. The old signature was
`read_image_from_dicom(dcm, use_dicom_level, ...)`, and AIFS calls it with
`cfg.data.windowing` — a non-empty string, therefore always truthy — so its
deployed configs 137 and 140 ask for `0-100` and silently get DICOM `WC/WW`
instead. An unmigrated call site now fails loudly rather than being
reinterpreted. Both `'5_95'` and `'5-95'` parse, so no config has to change.

## DICOM norm

When the scanner writes a slice out it has already put the pixel matrix into a
conventional display orientation, decided by which cardinal plane the slice is
closest to:

| Plane | screen right | screen down | reads as |
|---|---|---|---|
| axial | +x (patient left) | +y (posterior) | front at top, patient's **right on the left** |
| coronal | +x (patient left) | −z (foot) | head at top, patient's **right on the left** |
| sagittal | **+y (posterior)** | −z (foot) | head at top, patient's **front on the left** |

**Sagittal handedness is settled — do not re-litigate it.** It used to be the open
question here: CMRQ picked anterior-on-left from the look of its training set and
hedged that the convention "differs between sites"; BPF inherited that and
recorded it had never been re-checked; AMP asserts the opposite. Measured over
scanner-written DICOMs from Siemens, GE and Philips:

```
sagittal   screen-right = +y (posterior)     7150 / 7150   100.0%
coronal    screen-right = +x (patient left)  1459 / 1459   100.0%
axial      screen-right = +x (patient left)  1082 / 1084    99.8%
```

Unanimous, not a site preference. Slices too rotated for the sign of the in-plane
component to be meaningful were excluded rather than allowed to vote — for a
90°-out slice that sign is noise.

AMP's `get_default_right_down_unit_vectors_for_freq_phase` is therefore wrong on
sagittal, but inert: its call site notes those vectors "do NOT get used at ALL for
slice planning — they are purely to generate the preview and then calculate the
FOV". The symptom is a mirrored 2ch preview, not a mis-planned slice.

### Which models need canonicalising

Only those not trained to be orientation-invariant. Check the augmentation:

| | augmentation | needs `canonicalise_cine` |
|---|---|---|
| CMRQ | none | yes |
| BPF | `hflip/vflip/shift_scale_rotate: false` | yes |
| AIFS | `hflip, vflip, rotate: 90` | **no** — invariant by design |

## Versioning replaces byte-identity

`aidmr.py` used to carry the instruction that its six copies be byte-identical,
because AIDMR-FIRE rehydrates a program's dict with its own copy — so a program
writing a field its host cannot read is silent data loss. That rule was
unenforceable and was already broken.

`to_dict` now stamps `utilsVersion`, and `from_dict` refuses a payload written by
a **newer** aidmr-utils than the host runs. The asymmetry is deliberate: a newer
writer may carry fields this host would drop on write; an older one cannot. FIRE,
the dashboard and the programs still deploy together — what changed is that
"compatible" is asserted rather than assumed.

## Development

```bash
python -m venv .venv && .venv/Scripts/pip install -e ".[dev]"
.venv/Scripts/pytest
```

The suite needs no patient data: DICOM fixtures are synthesised in
`tests/conftest.py`, and the geometry vectors are real ones copied from
`AIDMR-FIRE/logs/mrd` captures and from scanner-written DICOMs.
