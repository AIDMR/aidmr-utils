"""MRD <-> numpy for the FIRE deployment.

Unified from three implementations that had diverged: BPF's (newest, true colour,
and the only one carrying the outgoing-layout fix), AMP's (parity coercion and
dimension validation), and CMRQ's (a stale subset of AMP's that had lost the
unit-normalisation of the direction vectors).

TWO PLACES DECLARE ORIENTATION, AND THE META IS THE ONE THAT WINS
-----------------------------------------------------------------
`head.read_dir` / `head.phase_dir` describe the *encoding* axes. The layout of
the stored pixel matrix is declared separately, in the MetaAttributes, as
`ImageRowDir` / `ImageColumnDir` — and downstream those are what become DICOM's
`ImageOrientationPatient` (upstream `mrd2dicom.py` prefers the meta over the
header when both are present). `aidmr_utils.orientation` already relies on this
on the way IN: it reads the meta rather than the header precisely because the two
disagree on the images FIRE sends us.

Until 2026-08-29 the outgoing meta declared neither, so the scanner had only the
acquired `read_dir` / `phase_dir` to lay the returned matrix out by — a different
pair of vectors from the ones the pixels are actually indexed along — and
reoriented every returned image into DICOM norm accordingly. On a heart that
looks plausible; on a measurements table it came back rotated 180 degrees and
unreadable.

**BPF fixed that. AMP never got the fix, and CMRQ has it commented out — writing
`ImageColDir`, which is not the key, so uncommenting it would not have worked
either.** Every returned image must declare the pair:

  * overlay cines carry the INCOMING image's pair, because the caller runs
    `orientation.inverse_transform` to put the pixels back into the incoming
    matrix layout before sending them;
  * a report page carries the identity pair, because it is text drawn in screen
    order and was never in the patient's frame at all — see `report_image_to_mrd`.

Setting the meta cannot conflict with the square-pixel rule below: direction
cosines are unit vectors and carry no scale. `Keep_image_geometry` is still set,
but it only stops the geometry being *reversed* — it does not supply a layout,
which is why it never fixed this.

OUTGOING GEOMETRY IS NOT INCOMING GEOMETRY
------------------------------------------
What goes back is the acquired matrix padded to square and resized, so it spans
the longer of the two acquired field-of-view axes in *both* directions — see
`padded_square_geometry`. Echoing the acquired `field_of_view` onto it instead
declares anisotropic pixels, which a Siemens scanner cannot represent and refuses
the reconstruction over. `np_float_to_mrd` asserts the pixels it writes are
square, so this cannot be got wrong silently again.

RGB IN MRD IS FIDDLY
--------------------
  * pixels must be uint16 holding values 0-255, NOT 0-65535;
  * the array handed to `Image.from_array` is (cha, z, y, x), so the colour axis
    moves to the front and a singleton z is inserted;
  * `image_type` must be 6 (RGB) and `channels` 3 — `from_array` sets channels
    but the header is overwritten from the template, so it has to be set again;
  * `matrix_size` is taken from the spatial axes only.
Not every scanner renders RGB, which is why overlays are usually also sent as
greyscale.

DELIBERATELY NOT HERE
---------------------
AMP's `np_float_to_mrd` also took `title=` (drawing text on the image) and
`output_shape_yx=` (resizing it). Both are presentation, and both dragged cv2 and
scikit-image into a module the FIRE container needs to import cheaply. Do them at
the call site — `np_float_to_mrd(put_text_on_img(img, title), ...)` — which gives
the identical result and keeps "convert to MRD" from accreting a third job, which
is how the pre-package versions got into three shapes.
"""

import ctypes
from copy import deepcopy
from typing import Tuple

import numpy as np
from loguru import logger

from .pixel import SQUARE_PIXEL_TOLERANCE, anisotropy, assert_square_pixels

try:
    import ismrmrd
except ImportError:  # pragma: no cover - exercised by the extras, not the suite
    ismrmrd = None


def _require_ismrmrd():
    if ismrmrd is None:
        raise ImportError(
            "MRD support needs the 'mrd' extra: pip install 'aidmr-utils[mrd]'")


INT16_MAX = 2 ** 15 - 1

#: In-plane direction cosines for an image that is a page of text rather than a
#: slice of the patient: increasing column runs to the patient's left, increasing
#: row runs posteriorly. Not an arbitrary "identity" — it is exactly
#: `geometry.CANONICAL_RIGHT_DOWN[Plane.AXIAL]`, the conventional axial display
#: orientation, and therefore a fixed point of the reorientation into DICOM norm:
#: hand this pair to `orientation.choose_transform` and it scores 2.00/2.00 and
#: returns 'identity'. That is the whole reason a report image declared this way
#: comes back the way it was drawn. Upstream `report.py` sets the same triple.
REPORT_ROW_DIR = (1.0, 0.0, 0.0)
REPORT_COL_DIR = (0.0, 1.0, 0.0)


# ---------------------------------------------------------------------------
# Meta helpers
# ---------------------------------------------------------------------------

def meta_string_list_to_array(meta_string) -> np.ndarray:
    """MetaAttribute values arrive as strings; get a float vector back."""
    return np.array([float(x) for x in meta_string])


def array_to_meta_string_list(array) -> list:
    return [str(x) for x in array]


def direction_meta(vector) -> list:
    """A direction cosine as MRD MetaAttributes carries it: three strings.

    18 decimal places and string-typed, matching upstream
    `python-ismrmrd-server` exactly — the values are parsed back with `float()`
    by anything that reads them, so the format only has to round-trip, but there
    is no reason to differ from the reference implementation.
    """
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector)
    assert norm > 0, f"zero-length direction vector {tuple(vector)}"
    return ["{:.18f}".format(v) for v in vector / norm]


def standard_image_meta(series_desc: str, *, base_meta=None,
                        row_dir=None, col_dir=None, program: str | None = None):
    """MetaAttributes for one outgoing image.

    `row_dir` / `col_dir` are the LPS directions of increasing COLUMN and
    increasing ROW index in the stored matrix — the screen's (right, down), the
    same pair `orientation.image_axes_from_mrd` reads on the way in. They go out
    as `ImageRowDir` / `ImageColumnDir`, which is what actually decides the
    displayed layout; see the module docstring for why omitting them rotated a
    measurements table 180 degrees.

    Both are optional and an existing value in `base_meta` wins, matching the
    `if tmpMeta.get('ImageRowDir') is None` guard every module in
    `python-ismrmrd-server` uses: if the incoming image already declared a pair,
    it is authoritative and must not be second-guessed from the header.

    Replaces `get_standard_ismrmrd_meta_for_uint16_image` (AMP, CMRQ — note the
    `seres_desc` typo in both) and `get_standard_ismrmrd_meta` (BPF). One name,
    one signature, keyword-only after `series_desc` so the three old positional
    orders cannot be silently mixed up.
    """
    _require_ismrmrd()
    meta = base_meta.copy() if base_meta is not None else ismrmrd.Meta()
    meta['DataRole'] = 'Image'
    meta['ImageProcessingHistory'] = (
        ['FIRE', 'PYTHON', program.upper()] if program else ['FIRE', 'PYTHON'])
    if row_dir is not None and meta.get('ImageRowDir') is None:
        meta['ImageRowDir'] = direction_meta(row_dir)
    if col_dir is not None and meta.get('ImageColumnDir') is None:
        meta['ImageColumnDir'] = direction_meta(col_dir)
    if base_meta is None:
        # A window covering the full int16 range the greyscale path writes into.
        # Only when there is no base_meta: CMRQ set these unconditionally, which
        # overwrote a window the incoming image had already chosen.
        meta['WindowCenter'] = str(2 ** 15 / 2)
        meta['WindowWidth'] = str(2 ** 15)
        meta['SequenceDescription'] = series_desc
        meta['SequenceDescriptionAdditional'] = (
            f"AID-MR {program.upper()}" if program else "AID-MR")
        # Stops the geometry being reversed. It does NOT supply a layout - see
        # the module docstring.
        meta['Keep_image_geometry'] = 1
    return meta


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _has_matching_row_col_parity(rows: int, cols: int) -> bool:
    return (rows % 2) == (cols % 2)


def validate_row_col_parity(*, rows: int, cols: int) -> None:
    """Scanner images need both dimensions odd or both even."""
    if not _has_matching_row_col_parity(rows, cols):
        raise ValueError(
            f"Invalid row/col parity: rows={rows}, cols={cols}. "
            "Scanner images must have both dimensions odd or both dimensions even.")


def validate_existing_mrd_image_dims(mrd_image) -> None:
    """The data shape and the header's matrix_size must agree."""
    rows, cols = [int(v) for v in mrd_image.data.shape[-2:]]
    matrix_cols, matrix_rows = [int(v) for v in mrd_image.getHead().matrix_size[:2]]
    if (rows, cols) != (matrix_rows, matrix_cols):
        raise ValueError(
            "MRD image data shape and header matrix_size disagree: "
            f"data rows/cols={rows}x{cols}, header rows/cols={matrix_rows}x{matrix_cols}")
    validate_row_col_parity(rows=rows, cols=cols)


def _coerce_row_col_parity(img, position_xyz, fov_freq_phase_slice,
                           freq_encoding_dir_xyz, phase_encoding_dir_xyz,
                           *, coerce_odd_even_dims: bool):
    rows, cols = int(img.shape[0]), int(img.shape[1])
    if _has_matching_row_col_parity(rows, cols):
        return img, position_xyz, fov_freq_phase_slice
    if not coerce_odd_even_dims:
        raise ValueError(
            f"Invalid row/col parity: rows={rows}, cols={cols}. Scanner images "
            f"must have both dimensions odd or both dimensions even. Pass "
            f"coerce_odd_even_dims=True to pad one row/column instead.")

    position_xyz = np.asarray(position_xyz, dtype=float)
    fov_freq_phase_slice = np.asarray(fov_freq_phase_slice, dtype=float)
    pad = ((0, 1), (0, 0)) if rows % 2 == 1 else ((0, 0), (0, 1))
    if img.ndim == 3:
        pad = pad + ((0, 0),)

    if rows % 2 == 1:
        row_pixel_size = fov_freq_phase_slice[1] / rows
        img = np.pad(img, pad, mode='constant', constant_values=0)
        position_xyz = position_xyz + (
            np.asarray(phase_encoding_dir_xyz, float) * (row_pixel_size / 2.0))
        fov_freq_phase_slice[1] += row_pixel_size
        which = 'row on the positive/down side'
    else:
        col_pixel_size = fov_freq_phase_slice[0] / cols
        img = np.pad(img, pad, mode='constant', constant_values=0)
        position_xyz = position_xyz + (
            np.asarray(freq_encoding_dir_xyz, float) * (col_pixel_size / 2.0))
        fov_freq_phase_slice[0] += col_pixel_size
        which = 'column on the positive/right side'

    logger.warning(f"Coerced mismatched dims by padding one {which} "
                   f"({rows}x{cols} -> {img.shape[0]}x{img.shape[1]}).")
    validate_row_col_parity(rows=int(img.shape[0]), cols=int(img.shape[1]))
    return img, tuple(position_xyz.tolist()), tuple(fov_freq_phase_slice.tolist())


# ---------------------------------------------------------------------------
# Building outgoing images
# ---------------------------------------------------------------------------

def np_float_to_mrd(img,
                    position_xyz,
                    fov_freq_phase_slice,
                    phase_encoding_dir_xyz,
                    freq_encoding_dir_xyz,
                    template_head,
                    image_index: int,
                    series_index: int,
                    attribute_string: str,
                    use_table_position: bool = True,
                    use_rgb: bool = False,
                    coerce_odd_even_dims: bool = False,
                    require_square_pixels: bool = True):
    """One float image in [0, 1] -> one ismrmrd.Image.

    `img` is (H, W) for greyscale, or (H, W, 3) with `use_rgb` — a 2-D array with
    `use_rgb=True` is replicated across the three channels.

    The direction vectors are unit-normalised here. CMRQ's copy did not, which
    left whatever the caller passed in stamped on the header verbatim.

    To draw a title on the image or resize it, do that before calling; see the
    module docstring.
    """
    _require_ismrmrd()
    img = np.asarray(img)
    assert img.ndim in (2, 3), f"expected (H, W) or (H, W, 3), got {img.shape}"
    assert img.min() >= 0 and img.max() <= 1.01, \
        f"expected [0, 1], got {img.min()}..{img.max()}"
    assert len(fov_freq_phase_slice) == 3, \
        f"field of view must be (freq, phase, slice), got {fov_freq_phase_slice}"

    freq_encoding_dir_xyz = np.asarray(freq_encoding_dir_xyz, dtype=float)
    freq_encoding_dir_xyz = freq_encoding_dir_xyz / np.linalg.norm(freq_encoding_dir_xyz)
    phase_encoding_dir_xyz = np.asarray(phase_encoding_dir_xyz, dtype=float)
    phase_encoding_dir_xyz = phase_encoding_dir_xyz / np.linalg.norm(phase_encoding_dir_xyz)

    img, position_xyz, fov_freq_phase_slice = _coerce_row_col_parity(
        img, position_xyz, fov_freq_phase_slice,
        freq_encoding_dir_xyz, phase_encoding_dir_xyz,
        coerce_odd_even_dims=coerce_odd_even_dims)

    head = deepcopy(template_head) if template_head is not None else ismrmrd.ImageHeader()

    if use_rgb:
        rgb = img if img.ndim == 3 else np.stack((img,) * 3, axis=-1)   # y, x, c
        assert rgb.shape[-1] == 3, f"expected 3 colour channels, got {rgb.shape[-1]}"
        data = np.around(rgb * 255).astype(np.uint16)   # 0-255 in a uint16
        data = np.transpose(data, (2, 1, 0))            # c, x, y
        data = np.expand_dims(data, axis=1)             # c, z, y, x
        head.image_type = 6                             # ismrmrd.IMTYPE_RGB
        head.channels = 3
    else:
        assert img.ndim == 2, \
            "greyscale output needs a 2-D image; pass use_rgb=True for colour"
        data = np.around(img * INT16_MAX).astype(np.int16)
        # `from_array` sets channels=1, but the header is then overwritten from
        # the template, so it has to be set again here — exactly as the RGB path
        # sets 3. None of the three implementations this replaces did, which left
        # `.data` reshaping to zero size whenever the template header carried
        # channels=0 (a fresh ImageHeader, or a replayed capture). It never bit in
        # production only because the images FIRE forwards happen to carry 1.
        head.channels = 1

    mrd_image = ismrmrd.Image.from_array(data, transpose=False)

    if use_table_position:
        table_position = tuple(head.patient_table_position)
        logger.debug(f"Adding table position {table_position} to image position "
                     f"{tuple(position_xyz)}")
        position_xyz = tuple(a + b for a, b in zip(position_xyz, table_position))

    head.data_type = mrd_image.data_type
    head.position = tuple(ctypes.c_float(v) for v in position_xyz)
    head.read_dir = tuple(ctypes.c_float(v) for v in freq_encoding_dir_xyz)
    head.phase_dir = tuple(ctypes.c_float(v) for v in phase_encoding_dir_xyz)
    head.slice_dir = tuple(np.cross(freq_encoding_dir_xyz, phase_encoding_dir_xyz))
    head.field_of_view = tuple(ctypes.c_float(v) for v in fov_freq_phase_slice)
    if use_rgb:
        head.matrix_size = tuple([ctypes.c_ushort(v) for v in data.shape[2:]]
                                 + [ctypes.c_ushort(1)])
    else:
        # matrix_size is (x, y, z) i.e. (cols, rows, 1) - the reverse of numpy's
        head.matrix_size = tuple([ctypes.c_ushort(v) for v in data.shape[::-1]]
                                 + [ctypes.c_ushort(1)])

    # The same check `native_pixel_spacing_mm` applies to images coming IN, now
    # applied to the header going OUT.
    #
    # A parameter, not a law - but the default is on, because an image whose
    # pixels are not square is nearly always a mistake. A rectangular FIELD OF
    # VIEW is fine and normal; it just needs a matrix in the same proportion.
    #
    # Turning this on across the estate found that AMP emitted non-square pixels
    # on 94.8% of its outgoing images (529 of 558 measured over its test suite),
    # up to 2:1. Two causes, both since fixed in AMP: previews padded to square
    # without extending the FOV - exactly the bug padded_square_geometry exists
    # for - and a keypoint mosaic whose FOV tuple was transposed.
    #
    # What legitimately wants this off is an image that is deliberately squashed:
    # AMP's shim previews resize a rectangular region into a fixed 192x192, so
    # their anisotropic FOV is an honest description of the render rather than an
    # error. Those are the only ones left.
    #
    # The pairing below assumes phase runs along rows and frequency along
    # columns, which is what field_of_view being (freq, phase) against a
    # (cols, rows) matrix means. A caller whose phase axis is the column axis
    # must pass its FOV in that order rather than turn this off - AMP's mosaic
    # had them the wrong way round and this is what caught it.
    spacing = (float(fov_freq_phase_slice[1]) / float(head.matrix_size[1]),
               float(fov_freq_phase_slice[0]) / float(head.matrix_size[0]))
    context = (f"outgoing MRD image (matrix {head.matrix_size[0]}x{head.matrix_size[1]}, "
               f"FOV {float(fov_freq_phase_slice[0]):.1f}x"
               f"{float(fov_freq_phase_slice[1]):.1f} mm). Build the geometry with "
               f"padded_square_geometry rather than echoing the acquired field_of_view")
    if require_square_pixels:
        assert_square_pixels(spacing, context=context)
    elif anisotropy(spacing) > SQUARE_PIXEL_TOLERANCE:
        logger.debug(f"Non-square outgoing pixels ({anisotropy(spacing):.1%}) for "
                     f"{context} - not checked, require_square_pixels=False")
    head.image_index = image_index
    head.image_series_index = series_index
    mrd_image.setHead(head)
    mrd_image.attribute_string = attribute_string

    validate_existing_mrd_image_dims(mrd_image)
    return mrd_image


def report_image_to_mrd(img, source_image, image_index: int, series_index: int,
                        series_desc: str, report_values: dict | None = None,
                        program: str | None = None):
    """One float report page in [0, 1] -> one ismrmrd.Image, in screen orientation.

    For an image that is TEXT, not anatomy. A port of upstream
    `python-ismrmrd-server/report.py`, which exists for exactly this case, and it
    differs from `np_float_to_mrd` in the one way that matters: it inherits no
    acquired geometry at all.

    `np_float_to_mrd` + `padded_square_geometry` stamps the acquired read/phase
    directions on its output, which is right for an overlay — that image really is
    a picture of that slice of the patient and has to line up with it. It is wrong
    for a table. A table is drawn in screen order (row 0 is the top of the page)
    and has no orientation in the patient's frame, so declaring a 4-chamber's
    oblique direction cosines on it invites the scanner to rotate it into DICOM
    norm, which is what made it unreadable.

    So instead, as report.py does:

      * `read_dir` / `phase_dir` / `slice_dir` are the identity triple, and
        `ImageRowDir` / `ImageColumnDir` say the same thing in the meta. Per
        `REPORT_ROW_DIR` that pair IS the canonical axial display orientation, so
        the reorientation is a no-op rather than something being suppressed.
      * `field_of_view` equals `matrix_size`, i.e. 1 mm square pixels.
      * `position` is left at the origin, and only `patient_table_position` is
        taken from `source_image`: position is optional but relative to the table
        position, and setting it keeps the report from landing a long way from the
        rest of the study.
      * `acquisition_time_stamp` is carried over so the series sorts sensibly.

    Departures from report.py, both deliberate: the pixels are int16 scaled by
    `INT16_MAX` rather than float32, so the window in `standard_image_meta`
    describes them and the table windows like every other series; and the header
    is built by property assignment on the fresh image rather than copied from a
    template, because there is no template.

    `report_values` is folded into the MetaAttributes verbatim, which is
    report.py's other good idea: the numbers on the page can then be scraped off
    the returned image instead of OCR'd. Values must be strings.
    """
    _require_ismrmrd()
    img = np.asarray(img)
    assert img.ndim == 2, f"expected a 2-D report page (H, W), got {img.shape}"
    assert img.min() >= 0 and img.max() <= 1.01, \
        f"expected [0, 1], got {img.min()}..{img.max()}"

    data = np.around(img * INT16_MAX).astype(np.int16)
    mrd_image = ismrmrd.Image.from_array(data, transpose=False)

    # `getHead().matrix_size` is (x, y, z) == (cols, rows, slices). Read it off the
    # head, NOT via the `mrd.matrix_size` convenience property, which returns a
    # transposed result and warns about it (ismrmrd-python#54).
    matrix = mrd_image.getHead().matrix_size
    mrd_image.field_of_view = (ctypes.c_float(float(matrix[0])),   # 1 mm square px
                               ctypes.c_float(float(matrix[1])),
                               ctypes.c_float(float(matrix[2])))
    # Read back off the head rather than asserting on the values just passed in,
    # which would be a tautology while FOV is set from matrix_size. This way the
    # check still bites if that ever stops being true.
    written = mrd_image.getHead()
    assert_square_pixels(
        (float(written.field_of_view[1]) / float(written.matrix_size[1]),
         float(written.field_of_view[0]) / float(written.matrix_size[0])),
        context=f"report image (matrix {matrix[0]}x{matrix[1]})")

    mrd_image.read_dir = tuple(ctypes.c_float(v) for v in REPORT_ROW_DIR)
    mrd_image.phase_dir = tuple(ctypes.c_float(v) for v in REPORT_COL_DIR)
    mrd_image.slice_dir = tuple(ctypes.c_float(v)
                                for v in np.cross(REPORT_ROW_DIR, REPORT_COL_DIR))

    if source_image is not None:
        source_head = source_image.getHead()
        mrd_image.patient_table_position = tuple(source_head.patient_table_position)
        mrd_image.acquisition_time_stamp = source_head.acquisition_time_stamp

    mrd_image.image_type = ismrmrd.IMTYPE_MAGNITUDE   # the default, 0, is invalid
    mrd_image.image_index = image_index
    mrd_image.image_series_index = series_index

    meta = standard_image_meta(series_desc, row_dir=REPORT_ROW_DIR,
                               col_dir=REPORT_COL_DIR, program=program)
    for key, value in (report_values or {}).items():
        meta[key] = value
    mrd_image.attribute_string = meta.serialize()
    return mrd_image


# ---------------------------------------------------------------------------
# Reading incoming geometry
# ---------------------------------------------------------------------------

def image_geometry(mrd_image) -> dict:
    """Position/direction/FOV of an incoming image, verbatim.

    NOT what an output image wants — see `padded_square_geometry`. Kept because
    reading the acquired geometry back out is a reasonable thing to want on its
    own, and naming the difference is the point.
    """
    head = mrd_image.getHead()
    return {
        'position_xyz': tuple(head.position),
        'fov_freq_phase_slice': tuple(head.field_of_view),
        'freq_encoding_dir_xyz': tuple(head.read_dir),
        'phase_encoding_dir_xyz': tuple(head.phase_dir),
    }


def native_pixel_spacing_mm(mrd_image) -> Tuple[float, float]:
    """(row, col) mm per pixel of the incoming image, from FOV / matrix size.

    `field_of_view` is taken as (freq, phase, slice) against `matrix_size`
    (x, y, z). That pairing is a convention, not something the header states, so
    the result is checked for square pixels — on a non-square acquisition a wrong
    pairing shows up immediately. See `pixel.assert_square_pixels`.
    """
    head = mrd_image.getHead()
    fov = np.asarray(head.field_of_view, dtype=float)
    matrix = np.asarray(head.matrix_size, dtype=float)
    assert matrix[0] > 0 and matrix[1] > 0, f"bad matrix_size {tuple(matrix)}"
    spacing = (float(fov[1] / matrix[1]), float(fov[0] / matrix[0]))
    assert_square_pixels(
        spacing,
        context=f"incoming MRD image (matrix {matrix[0]:.0f}x{matrix[1]:.0f}, "
                f"FOV {fov[0]:.1f}x{fov[1]:.1f} mm)")
    return spacing


def get_fov_and_pixel_size(mrd_image):
    """(fov_mm, pixel_size_mm) as (x, y, z) triples, straight from the header.

    Unchecked, and in header axis order rather than (row, col) — prefer
    `native_pixel_spacing_mm` for in-plane spacing.
    """
    fov_mm = np.array(mrd_image.field_of_view)
    matrix_size = np.array(mrd_image.getHead().matrix_size)
    return fov_mm, fov_mm / matrix_size


def slice_normal(mrd_image) -> np.ndarray:
    """Unit slice normal in patient coordinates (x = L-R, y = A-P, z = H-F)."""
    head = mrd_image.getHead()
    normal = np.cross(np.asarray(head.read_dir, dtype=float),
                      np.asarray(head.phase_dir, dtype=float))
    norm = np.linalg.norm(normal)
    assert norm > 0, "degenerate read/phase directions - cannot form a slice normal"
    return normal / norm


def padded_square_geometry(mrd_image) -> dict:
    """Geometry for an output that is `mrd_image` centre-padded to square.

    Anything that pads the acquired matrix to `max(rows, cols)` with zeros and
    then resizes that square needs this rather than `image_geometry`. Two things
    follow, and both were wrong when this echoed the acquired geometry.

    **The field of view is square.** Padding to the longer axis means the output
    covers `max(fov_freq, fov_phase)` in both directions, and resizing a square to
    a square cannot change that. For a 224x180 acquisition at 340 x 273.2 mm the
    output spans 340 x 340 mm, giving 340/192 = 1.7708 mm/px — exactly the
    mm-per-pixel handed to the output scaler, so the header and the volumes agree
    by construction. Echoing 340 x 273.2 onto the square matrix instead declared
    1.771 x 1.423 mm pixels, which Siemens reconstruction refuses.

    **The position needs no table offset.** `head.position` is already in the frame
    the scanner delivered it in, so `np_float_to_mrd` must not add
    `patient_table_position` on top — hence `use_table_position: False` in the
    returned dict, which is there so a caller cannot forget it. AMP adds the
    offset because its positions are computed in light-marker coordinates; an
    echoed header is a different case.

    The only correction position does need is for asymmetric padding. MRD's
    `position` is the image *centre*, and padding floor-divides the shortfall, so
    an odd shortfall puts the extra row (or column) on the far side and moves the
    centre by half a pixel. Zero for the common even case.

    Computed from the acquired header, not from the reoriented array, so it is
    independent of whether `canonicalise_cine` transposed the cine:
    `transpose(pad(transpose(x)))` and `pad(x)` pad the same edges by the same
    amounts, for both parities of the shortfall.
    """
    head = mrd_image.getHead()
    fov = np.asarray(head.field_of_view, dtype=float)
    read_dir = np.asarray(head.read_dir, dtype=float)
    phase_dir = np.asarray(head.phase_dir, dtype=float)

    # matrix_size is (x, y, z) == (cols, rows, slices); the data is (rows, cols).
    # Increasing row index runs along phase_dir, increasing column along read_dir.
    cols, rows = int(head.matrix_size[0]), int(head.matrix_size[1])
    row_spacing, col_spacing = native_pixel_spacing_mm(mrd_image)

    side_px = max(rows, cols)
    pad_top = (side_px - rows) // 2
    pad_bottom = (side_px - rows) - pad_top
    pad_left = (side_px - cols) // 2
    pad_right = (side_px - cols) - pad_left

    centre_shift = ((pad_bottom - pad_top) / 2.0 * row_spacing * phase_dir
                    + (pad_right - pad_left) / 2.0 * col_spacing * read_dir)
    if np.any(centre_shift):
        logger.debug(
            f"Odd pad-to-square shortfall ({rows}x{cols} -> {side_px}x{side_px}): "
            f"shifting the output centre by {np.linalg.norm(centre_shift):.3f} mm")

    side_mm = max(float(fov[0]), float(fov[1]))
    position = np.asarray(head.position, dtype=float) + centre_shift
    return {
        'position_xyz': tuple(float(v) for v in position),
        'fov_freq_phase_slice': (side_mm, side_mm, float(fov[2])),
        'freq_encoding_dir_xyz': tuple(read_dir),
        'phase_encoding_dir_xyz': tuple(phase_dir),
        'use_table_position': False,
    }


# ---------------------------------------------------------------------------
# Reading .h5 captures
# ---------------------------------------------------------------------------

def parse_h5_to_images(h5_path: str) -> list:
    """Every image in an ISMRMRD .h5 capture, from every image group.

    Groups are named `images_0` on newer captures and `image_0` on older ones,
    and a capture can hold SEVERAL - one per series sent down the connection.
    AMP's copy of this tried `images_0` then `image_0` and returned the first it
    found, which silently dropped every later group; AIFS's took all of them.
    AIFS's behaviour is the correct one and is what this does, sorted so the
    order is deterministic.
    """
    _require_ismrmrd()
    ds = ismrmrd.Dataset(h5_path, '/dataset', False)
    groups = sorted(k for k in ds.list() if 'image' in k)
    if not groups:
        raise LookupError(f"no image group in {h5_path}")
    return [ds.read_image(g, i) for g in groups for i in range(ds.number_of_images(g))]


def parse_h5_into_fire_arguments(h5_path: str, config=None):
    """Replay a capture as if it were a live FIRE connection.

    :param config: the additional-config dict the server would have received.
        AMP's copy hard-coded its own planning parameters here, which is why this
        takes it as an argument: a capture is not the property of one program,
        and BPF or CMRQ replaying the same file need their own.

        It defaults to None and is returned unchanged, so a caller whose
        `process()` reads `config['parameters']` will fail downstream on a
        NoneType rather than here. That is deliberate - this function has no way
        to know what any given program needs - but it means each program should
        wrap this with its own default, as AMP does in
        `lib/inference/mrd.parse_h5_into_fire_arguments`.
    :return: (iterator over waveforms then images, config, metadata)
    """
    _require_ismrmrd()
    ds = ismrmrd.Dataset(h5_path, '/dataset', False)
    images = parse_h5_to_images(h5_path)
    try:
        waveforms = [ds.read_waveform(i) for i in range(ds.number_of_waveforms())]
    except LookupError:
        # Dataset raises LookupError from number_of_waveforms if there are none
        waveforms = []
    metadata = ismrmrd.xsd.CreateFromDocument(ds.read_xml_header())
    return iter(waveforms + images), config, metadata


def update_img_header_from_raw(head_img, head_raw):
    """Populate an ImageHeader from an AcquisitionHeader.

    From upstream `python-ismrmrd-server/mrdhelper.py`. `data_type`,
    `matrix_size` and `channels` are deliberately not copied — `from_array` fills
    them — and `field_of_view` must come from the XML header, not from here.
    """
    _require_ismrmrd()
    if head_raw is None:
        return head_img

    head_img.version = head_raw.version
    head_img.flags = head_raw.flags
    head_img.measurement_uid = head_raw.measurement_uid

    head_img.position = head_raw.position
    head_img.read_dir = head_raw.read_dir
    head_img.phase_dir = head_raw.phase_dir
    head_img.slice_dir = head_raw.slice_dir
    head_img.patient_table_position = head_raw.patient_table_position

    if hasattr(head_raw, 'idx'):
        head_img.average = head_raw.idx.average
        head_img.slice = head_raw.idx.slice
        head_img.contrast = head_raw.idx.contrast
        head_img.phase = head_raw.idx.phase
        head_img.repetition = head_raw.idx.repetition
        head_img.set = head_raw.idx.set
    else:
        logger.warning("Header passed to update_img_header_from_raw has no 'idx'")

    head_img.acquisition_time_stamp = head_raw.acquisition_time_stamp
    head_img.physiology_time_stamp = head_raw.physiology_time_stamp

    # Defaults, to be updated by the caller
    head_img.image_type = ismrmrd.IMTYPE_MAGNITUDE
    head_img.image_index = 1
    head_img.image_series_index = 0

    head_img.user_float = head_raw.user_float
    head_img.user_int = head_raw.user_int
    return head_img


def ecg_from_waveforms(waveforms: list):
    """The 5-channel ECG from a capture's waveforms, as (5, n_samples).

    Returns None when there are no waveforms — a fully anonymised or
    retrospectively saved study often has none.
    """
    if not waveforms:
        return None
    five_channel = np.array([w.data for w in waveforms if w.channels == 5])
    if not len(five_channel):
        return None
    return five_channel.transpose(1, 0, 2).reshape(5, -1)


def sort_mrd_images_by_phase(images: list) -> list:
    """Cine frames in cardiac-phase order."""
    return sorted(images, key=lambda im: im.getHead().phase)
