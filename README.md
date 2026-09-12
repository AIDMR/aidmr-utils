# aidmr-utils

One implementation of the things AMP, CMRQ, AIFS, BPF and LGEP all need, so a fix
made once is a fix everywhere.

```bash
pip install "aidmr-utils[dicom,mrd] @ git+https://github.com/AIDMR/aidmr-utils@v0.1.4"
```

Pin to a **tag**, never a branch.

| extra | pulls in | for |
|---|---|---|
| *(core)* | numpy, loguru, imageio | `result`, `windowing`, `geometry`, `plane`, `orientation`, `pixel` |
| `dicom` | pydicom | reading DICOMs — training side |
| `mrd` | ismrmrd | building MRD images — scanner side |
| `onnx` | onnxruntime | loading exported models |
| `imaging` | opencv, scikit-image | drawing on previews |
| `config` | pyyaml, easydict | YAML experiment configs |

Import `dicom`, `mrd`, `onnx`, `imaging`, `config` and `fire` by module — they are
not re-exported from the package root, so `import aidmr_utils` works in an
environment that has none of their dependencies.

---

# Capabilities

## `result` — the dashboard payload

The JSON each program writes and AIDMR-FIRE files into the DB.

- **`AIDMRResult`** — one measurement's result: images, waveforms, ECG and any
  number of renderable sections. `add_quality_data()`, `add_biplanar_data()`,
  `add_rwma_data()`, `add_flow_data()` and `add_planning_data()` add a section of
  each kind; `write_to_disk()` files it under `db/<study>/<tab>/<measurement>.json`.
- **`verdict()`** — one interpretation entry. Give it a `severity` and it also
  becomes an issue in the study-wide panel, so the two can never disagree.
- **`resolved_by()`** — how a radiographer clears an issue, optionally naming the
  later measurement that would satisfy it.
- **`utc_now_iso()`** — the sortable timestamp the dashboard orders by. Inference
  time, not header time, because an anonymised study has neither.

## `windowing` — mapping stored values onto a display range

Dependency-free on purpose: training and the scanner both import it, and the two
must apply the identical mapping or the model sees intensities it never saw.

- **`parse_window_spec()`** — `'5_95'`, `'0-100'` or `'dicom'` into a spec. Both
  spellings parse, so no config has to change.
- **`window()`** — window one array using its own extremes.
- **`window_series()`** — window a whole cine under one set of bounds, which is
  what the scanner side does and what stops the cine flickering.
- **`apply_window()`** — map a known `[low, high]` onto 0–1 or 0–255.
- **`check_window_method()`** — guards the YAML trap where an unquoted `0_100` is
  read as octal 64.
- **`PercentileWindow` / `DicomWindow`** — the two policies. `DEPLOYABLE_WINDOWINGS`
  lists the ones the scanner can reproduce, since MRD carries no WC/WW tags.

## `dicom` — reading DICOM pixel data *(extra: `dicom`)*

Split in two on purpose: getting the true values has one right answer, choosing a
display range is a policy.

- **`load_pixels()`** — stored data in real output units. **Always** applies
  RescaleSlope/Intercept, with no argument to skip it. That is what stops the
  Philips failure below.
- **`read_image()`** — one DICOM to one windowed image, taking a window spec.
  Refuses a bare bool, so an unmigrated `use_dicom_level` call site fails loudly.
- **`read_series()`** — a whole cine, windowed together, in one call.
- **`sorted_by_instance_number()`** — cine frames in acquisition order, keeping
  untagged frames rather than dropping them.
- **`is_monochrome1()`**, **`window_spec_for()`**, **`series_frame_shapes()`** —
  photometric check, resolving `'dicom'` against one image's tags, and frame sizes
  read from headers without decoding pixels.
- **`MixedFrameSizesError`** — raised for a series whose frames differ in size,
  which is almost never one cine.

## `plane` — one slice, three sources, compared without guessing

`ImagePlane` is a slice as centre + two in-plane unit axes with their extents + thickness,
built `from_dicom(ds)` (ImagePositionPatient is the centre of the FIRST pixel, so the plane
centre is half a matrix in along each axis; PixelSpacing is rows-then-columns),
`from_mrd(image)` (centre from the `SlicePosLightMarker` attribute, which is the slice centre
in the exam frame the scanner plans in; the header `position` is in the ICE frame and was
24 mm away on a real exam) and `from_exam_memory(slice_entry, fov_entry)` (AMP-style
`SiemensExamMemory_wip_070_fire_ICEOut_<view>_Slice` / `_FOV`). Comparisons match axes by
direction, never by the words row/column/readout/phase, and normals up to sign:
`centre_distance_mm`, `normal_angle_deg`, `in_plane_rotation_deg`, `extents_along`.
`SliceStack` groups parallel slices (a SAX stack from 350 DICOMs, or an `n_slices > 1`
instruction) and reports n, spacing, centre and normal. `planned_views(meta)` decodes every
view in a returned image. Needs only numpy: pass pydicom / ismrmrd objects in, nothing is
imported.

## `geometry` — cardinal planes and DICOM norm

- **`closest_plane()`** — which cardinal plane a slice normal is nearest.
- **`plane_of()`**, **`normal_from_right_down()`** — the same from an in-plane
  (right, down) pair.
- **`canonical_right_down()`** — the (right, down) DICOM norm puts that plane into.
- **`normal_for_named_plane()`**, **`unit()`**, **`Plane`** — supporting bits.

## `orientation` — reorienting into DICOM norm, and back

- **`canonicalise_cine()`** — rotate/flip a cine into the orientation the model was
  trained on. Only needed by models without rotation augmentation.
- **`choose_transform()`** — which of the eight flips/rotations best matches, and
  how well (2.00 is exact).
- **`apply_transform()`** / **`inverse_transform()`** — apply one, or undo it to
  put an overlay back into the acquired layout before sending it.
- **`image_axes_from_mrd()`** / **`image_axes_from_dicom()`** — the (right, down)
  pair, read from the MetaAttributes or from `ImageOrientationPatient`.
- **`image_row_col_dirs()`** — the same for the send path, with a header fallback.

## `pixel` — millimetre scale

- **`mm_per_pixel_for_side()`** — mm per pixel after pad-to-square then resize.
  Training and the scanner both call it, so the scale a deployed model is handed
  is the same arithmetic it was trained against.
- **`assert_square_pixels()`** — row and column spacing must agree. Also catches a
  FOV paired with the wrong matrix axis.
- **`anisotropy()`**, **`validate_transform_pipeline()`**, **`log_scale_summary()`**
  — the ratio, a guard that a pipeline has one well-defined scale, and a one-shot
  report of the scales seen.

## `mrd` — MRD images for the scanner *(extra: `mrd`)*

- **`np_float_to_mrd()`** — a float image in [0,1] to an `ismrmrd.Image`, greyscale
  or RGB, with the header geometry filled in and the pixels checked square.
- **`standard_image_meta()`** — the MetaAttributes for an outgoing image, including
  the `ImageRowDir`/`ImageColumnDir` pair that decides its displayed layout.
- **`report_image_to_mrd()`** — for an image that is a page of text rather than a
  slice: no acquired geometry, 1 mm square pixels, values scrapeable from the meta.
- **`padded_square_geometry()`** — the geometry for an output that is the acquired
  matrix padded to square. Not the same as echoing the acquired header.
- **`image_geometry()`**, **`native_pixel_spacing_mm()`**, **`slice_normal()`**,
  **`get_fov_and_pixel_size()`** — reading an incoming image's geometry.
- **`parse_h5_to_images()`** / **`parse_h5_into_fire_arguments()`** — replay a
  captured `.h5`, reading every image group rather than only the first.
- **`ecg_from_waveforms()`**, **`sort_mrd_images_by_phase()`**,
  **`update_img_header_from_raw()`** — the ECG as (5, n), cine frames in cardiac
  order, and populating an ImageHeader from an AcquisitionHeader.
- **`validate_row_col_parity()`**, **`validate_existing_mrd_image_dims()`**,
  **`direction_meta()`**, **`meta_string_list_to_array()`** — the checks and the
  string/vector conversions MetaAttributes need.

## `onnx` — loading an exported model *(extra: `onnx`)*

- **`load_model_onnx()`** — a session plus its `.json` sidecar, because a session
  without its preprocessing contract cannot be fed correctly. CUDA with a CPU
  fallback that re-raises rather than looping.
- **`load_models_onnx()`** — several at once as parallel tuples, the shape a
  two-model `process()` wants.
- **`warm_up()`** — one dummy batch so the first real request does not pay the
  compile, feeding every input so a multi-input model is not skipped.

## `imaging` — drawing on previews *(extra: `imaging`)*

- **`put_text_on_img()`** — a centred caption, float in and float out.
- **`add_cross_to_array()`** / **`add_dashed_cross_to_array()`** — keypoint markers,
  safe when the point lands off the image.
- **`add_line_to_array()`** / **`add_dotted_line_to_array()`** — measurement lines,
  clipped to the array.

## `fire` — the connection *(extra: `mrd`)*

- **`ReplayableConnection`** — lets several configured programs each read the same
  incoming data. Note: a program that stops early truncates the replay for
  everything after it.
- **`save_and_return()`** — the no-model config, for capturing what a scanner sent.

## `config` — YAML experiment configs *(extra: `config`)*

- **`load_config()`** — read a config into an EasyDict, stamp its filename, and
  check its `desc` matches so a copied config cannot keep the wrong description.
- **`update_config()`** — fold a sweep's flat `a.b.c` overrides into the nested
  config without dropping their siblings.
- **`dot_check()`** — refuse a key containing a dot anywhere in the tree, since it
  would collide with those dotted paths.
- **`unflatten_dot()`**, **`update()`** — the flattening and the recursive merge.

## `testing` — proving a change did not move the scanner instructions

- **`diff()`** — compare two recordings and report any difference in the
  `SiemensExamMemory_*` payload: position, normal, thickness, phase FOV, readout
  FOV, in-plane rotation, slice count.
- **`describe()`** — one outgoing image's header geometry and full MetaAttributes.
- Used as a pytest plugin, it records every image `process()` returns:

```bash
git checkout main       && pytest src/tests -p aidmr_utils.testing --instr-out before.json
git checkout my-change  && pytest src/tests -p aidmr_utils.testing --instr-out after.json
python -m aidmr_utils.testing before.json after.json
```

---

# Why this exists

Five repos had grown their own copy of the same code, and the duplication had
already produced defects that reached the scanner.

| duplicated | state when this package was created |
|---|---|
| `aidmr.py` | 6 copies; five byte-identical, FIRE's had drifted — under a docstring saying drift causes silent data loss |
| `verify_aidmr_result.py` | 5 copies in **3 mutually different versions** — the drift detector had itself drifted |
| `np_float_to_mrd` | 3 copies: CMRQ ⊂ AMP ⊂ BPF |
| `read_image_from_dicom` | 4 copies, **2 incompatible signatures**, 3 windowing string formats |
| `metadata_window` | 6 copies, three with an integer `// 2` half-width |
| `load_model_onnx` | 4 copies |
| `config.py` | 5 copies; `dot_check` and `update_config` byte-identical in all five |
| `orientation.py` | 2 copies, kept in step by a hand-written cross-repo test |

**Outgoing image orientation.** BPF fixed a 180°-rotated measurements table by
declaring `ImageRowDir`/`ImageColumnDir`. AMP never got the fix; CMRQ had it
commented out, writing `ImageColDir` — the wrong key, so uncommenting would not
have worked either.

**The Philips rescale.** `read_image_from_dicom` in CMRQ and AIFS ignored
RescaleSlope/Intercept. Harmless under a percentile window, which is exactly
invariant to an affine rescale, and catastrophic under DICOM WC/WW, whose tags are
quoted in rescaled units. Measured over 11,458 scanner-written DICOMs: **127/127
Philips images carry a non-identity rescale**, against almost no Siemens and no GE.
On a real Philips LGE series the un-rescaled window admits **0.0%** of pixels.

**`dot_check` had never run.** In all five copies it was defined inside
`load_config` and never called — and it recursed on `d` rather than `v`, so any
nested dict was infinite recursion.

## DICOM norm

When the scanner writes a slice out it has already put the matrix into a
conventional display orientation, decided by the nearest cardinal plane:

| plane | screen right | screen down | reads as |
|---|---|---|---|
| axial | +x (patient left) | +y (posterior) | front at top, patient's **right on the left** |
| coronal | +x (patient left) | −z (foot) | head at top, patient's **right on the left** |
| sagittal | **+y (posterior)** | −z (foot) | head at top, patient's **front on the left** |

**Sagittal handedness is settled.** It was the open question here: CMRQ picked
anterior-on-left from the look of its training set and hedged that it "differs
between sites"; BPF inherited that and recorded it had never been re-checked; AMP
asserted the opposite. Measured over scanner-written DICOMs from Siemens, GE and
Philips:

```
sagittal   screen-right = +y (posterior)     7150 / 7150   100.0%
coronal    screen-right = +x (patient left)  1459 / 1459   100.0%
axial      screen-right = +x (patient left)  1082 / 1084    99.8%
```

Unanimous, not a site preference. Slices too rotated for the sign of the in-plane
component to be meaningful were excluded rather than allowed to vote.

Only models **not** trained to be orientation-invariant need `canonicalise_cine`:
CMRQ and BPF do (no flip/rotate augmentation); AIFS does not (`hflip`, `vflip`,
`rotate: 90`).

## Square pixels

`np_float_to_mrd` takes `require_square_pixels`, defaulting to **on** — an image
whose pixels are not square is nearly always a mistake. A rectangular *field of
view* is fine; it needs a matrix in the same proportion.

Turning it on found AMP emitting non-square pixels on **94.8%** of its outgoing
images (529 of 558), up to 2:1, from three causes: slice previews padded to square
without extending the FOV, a keypoint mosaic whose FOV tuple was transposed, and
shim previews squashed rather than padded. All three are fixed and AMP now
measures 0%.

`phase_is_rows` (default `True`) says which matrix axis the phase encoding runs
along — nothing in the MRD header states it. It matters only for a **non-square
matrix**: a 480×512 matrix at 340 × 318.8 mm is 0.664 mm square one way round and
12.9% anisotropic the other.

## Versioning replaces byte-identity

`aidmr.py` used to carry an instruction that its six copies be byte-identical,
because AIDMR-FIRE rehydrates a program's dict with its own copy. That rule was
unenforceable and was already broken. `to_dict` now stamps `utilsVersion` and
`from_dict` refuses a payload written by a **newer** aidmr-utils than the host
runs. FIRE, the dashboard and the programs still deploy together; what changed is
that "compatible" is asserted rather than assumed.

## Development

```bash
python -m venv .venv && .venv/Scripts/pip install -e ".[dev]"
.venv/Scripts/pytest
```

The suite needs no patient data: DICOM fixtures are synthesised in
`tests/conftest.py`, and the geometry vectors are real ones copied from FIRE
captures and from scanner-written DICOMs.
