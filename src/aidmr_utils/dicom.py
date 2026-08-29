"""Reading DICOM pixel data correctly, once.

THE SHAPE OF THIS MODULE IS THE POINT
-------------------------------------
Getting pixels out of a DICOM is two jobs, and the five pre-package copies of
this code conflated them into one `read_image_from_dicom` that grew two mutually
incompatible signatures:

    load_pixels(dcm)          -> the true stored values. ONE right answer.
                                 No policy, no arguments, nothing to forget.
    window(pixels, spec)      -> a display range. A policy, so it takes a spec.

Keeping them apart is what stops the Philips failure below, because there is no
longer any code path that reaches pixel data without rescaling it.

THE PHILIPS FAILURE
-------------------
`RescaleSlope` / `RescaleIntercept` convert stored values to the real output
units, and `WindowCenter` / `WindowWidth` are quoted in those output units.
Skip the rescale and the window no longer refers to the same domain.

Measured across 11,458 scanner-written cardiac DICOMs:

    Philips   127/127  non-identity rescale (slopes 1.6-4.3, intercepts to -5587)
    Siemens    30/11k  non-identity - almost always absent or 1/0
    GE          0/303  non-identity

So the bug is invisible on Siemens and universal on Philips. On a real Philips
LGE series with `RescaleSlope=2.73, RescaleIntercept=-5587`, applying the image's
own WC/WW to un-rescaled values puts 0.0% of pixels inside the window: the
result is a uniform white frame carrying no information at all.

A percentile window, by contrast, is exactly invariant to the rescale — which is
why this went unnoticed for so long in the repos that happened to use one.

MONOCHROME1
-----------
Inversion happens AFTER windowing, in display space, where it is `1 - x`. In
stored-value space it would be `max - x`, which needs a range nobody has yet.
The photometric interpretation is not encoded in WC/WW, so the order is always
rescale -> window -> invert.
"""

import numpy as np
from loguru import logger

from .windowing import (DicomWindow, WindowSpec, _DicomWindowPlaceholder,
                        apply_window, parse_window_spec)

try:
    import pydicom
except ImportError:  # pragma: no cover - exercised by the extras, not the suite
    pydicom = None


def _require_pydicom():
    if pydicom is None:
        raise ImportError(
            "reading DICOMs needs the 'dicom' extra: "
            "pip install 'aidmr-utils[dicom]'")


def _as_dataset(dcm):
    """Accept a path or an already-read dataset."""
    if isinstance(dcm, (str, bytes)) or hasattr(dcm, '__fspath__'):
        _require_pydicom()
        return pydicom.dcmread(dcm)
    return dcm


def _first(value):
    """WC/WW may be a single value or a MultiValue; take the first."""
    if value is None:
        return None
    if hasattr(value, '__len__') and not isinstance(value, (str, bytes)):
        return float(value[0]) if len(value) else None
    return float(value)


def load_pixels(dcm) -> np.ndarray:
    """Stored pixel data in real output units, as float64.

    Applies RescaleSlope and RescaleIntercept. This is not a policy and there is
    no argument to switch it off: a caller who wants raw stored values wants
    `dcm.pixel_array`, and should say so in those words so it is obvious in
    review that the rescale was skipped on purpose.

    float64, not float32, and that is load-bearing. A slope of 0.0030518 applied
    in float32 moves ~1% of pixels (94% for small slopes) across a uint8 rounding
    boundary relative to the same arithmetic in float64 — a max difference of one
    display level, harmless in itself but enough to stop the migration being
    verifiable byte for byte against the implementation this replaces. The
    normalisation in `apply_window` drops back to float32, where the dynamic
    range is small and the precision is ample.

    Does NOT apply the photometric inversion - see the module docstring.
    """
    dcm = _as_dataset(dcm)
    img = np.asarray(dcm.pixel_array, dtype=np.float64)
    slope = getattr(dcm, 'RescaleSlope', None)
    intercept = getattr(dcm, 'RescaleIntercept', None)
    slope = 1.0 if slope is None else float(slope)
    intercept = 0.0 if intercept is None else float(intercept)
    if slope != 1.0 or intercept != 0.0:
        img = img * slope + intercept
    return img


def is_monochrome1(dcm) -> bool:
    """Whether minimum stored value means white for this image."""
    return getattr(_as_dataset(dcm), 'PhotometricInterpretation', '') == 'MONOCHROME1'


def _invert_if_monochrome1(dcm, img):
    if not is_monochrome1(dcm):
        return img
    return 255 - img if img.dtype == np.uint8 else 1.0 - img


def window_spec_for(dcm, spec) -> WindowSpec:
    """Resolve a config-level spec against one image.

    Only `'dicom'` needs resolving: it is a placeholder until an image supplies
    the WindowCenter / WindowWidth it refers to. Everything else passes through.
    """
    if isinstance(spec, str):
        spec = parse_window_spec(spec)
    if not isinstance(spec, _DicomWindowPlaceholder):
        return spec
    dcm = _as_dataset(dcm)
    center, width = _first(getattr(dcm, 'WindowCenter', None)), _first(
        getattr(dcm, 'WindowWidth', None))
    if center is None or width is None or width <= 0:
        raise ValueError(
            f"'dicom' windowing was asked for but this image has "
            f"WindowCenter={getattr(dcm, 'WindowCenter', None)!r} "
            f"WindowWidth={getattr(dcm, 'WindowWidth', None)!r}. Use a "
            f"percentile window, which needs no tags and is what the scanner "
            f"side can reproduce.")
    return DicomWindow(center, width)


def read_image(dcm, spec, as_uint8: bool = True) -> np.ndarray:
    """One DICOM -> one windowed image.

    `spec` is a WindowSpec or a config string ('5_95', '0_100', 'dicom').

    NOTE the second argument is a *spec*, not a boolean. The pre-package
    signature was `read_image_from_dicom(dcm, use_dicom_level, ...)`, and two
    repos called it with `cfg.data.windowing` — a non-empty string, therefore
    always truthy — so a config asking for '0-100' silently got DICOM WC/WW
    instead. Passing a bare bool here raises rather than being reinterpreted.

    For a percentile spec this uses THIS FRAME's own extremes. Prefer
    `read_series` for cines; see its docstring.
    """
    if isinstance(spec, bool):
        raise TypeError(
            "read_image takes a window spec, not a boolean. Pass '5_95', "
            "'0_100', 'dicom', or a WindowSpec.")
    dcm = _as_dataset(dcm)
    resolved = window_spec_for(dcm, spec)
    pixels = load_pixels(dcm)
    low, high = resolved.bounds(pixels)
    return _invert_if_monochrome1(dcm, apply_window(pixels, low, high,
                                                    as_uint8=as_uint8))


class MixedFrameSizesError(ValueError):
    """A series whose frames are not all the same size.

    Almost certainly not one cine: two acquisitions sharing a series number, or a
    localiser grouped in with the real images. Such a series is mishandled by
    everything downstream — augmentation scales each frame to the model input
    independently, so the anatomy changes size across the cine, and pixel
    geometry read from the first instance is wrong for the rest. Better skipped
    than exported.
    """

    def __init__(self, shapes, n_frames, series_uid=None):
        self.shapes = sorted(shapes)
        self.n_frames = n_frames
        self.series_uid = series_uid
        super().__init__(
            f"{n_frames} frames with {len(self.shapes)} different sizes "
            f"{self.shapes} (series {series_uid or '?'})"
        )


def series_frame_shapes(dcms) -> set:
    """(Rows, Columns) of each frame, from the header — no pixel decode."""
    return {(int(d.Rows), int(d.Columns)) for d in (_as_dataset(x) for x in dcms)}


def read_series(dcms, spec, as_uint8: bool = True) -> list:
    """A whole cine -> windowed frames, windowed TOGETHER.

    A percentile window is computed ONCE over every frame, because that is what
    the scanner side does to an MRD cine and because per-frame min-max makes the
    cine flicker. See `windowing.window_series`.

    A 'dicom' window is per-image by construction and so delegates frame by
    frame.

    Raises MixedFrameSizesError if the frames are not all the same size, checked
    from the headers before any pixel data is touched.
    """
    dcms = [_as_dataset(d) for d in dcms]
    if not dcms:
        return []

    shapes = series_frame_shapes(dcms)
    if len(shapes) > 1:
        raise MixedFrameSizesError(
            shapes, len(dcms), getattr(dcms[0], 'SeriesInstanceUID', None))

    if isinstance(spec, str):
        spec = parse_window_spec(spec)
    if isinstance(spec, _DicomWindowPlaceholder):
        return [read_image(d, spec, as_uint8=as_uint8) for d in dcms]

    pixels = [load_pixels(d) for d in dcms]
    low, high = spec.bounds(np.stack(pixels))
    return [_invert_if_monochrome1(d, apply_window(p, low, high, as_uint8=as_uint8))
            for d, p in zip(dcms, pixels)]


def sorted_by_instance_number(dcms) -> list:
    """Cine frames in acquisition order.

    Instances missing InstanceNumber sort last rather than raising: a series with
    one untagged frame is still usable, and dropping it silently would be worse.
    """
    dcms = [_as_dataset(d) for d in dcms]
    missing = [d for d in dcms if getattr(d, 'InstanceNumber', None) is None]
    if missing:
        logger.warning(f"{len(missing)}/{len(dcms)} instances have no "
                       f"InstanceNumber; sorting those last")
    return sorted(dcms, key=lambda d: (getattr(d, 'InstanceNumber', None) is None,
                                       getattr(d, 'InstanceNumber', 0)))
