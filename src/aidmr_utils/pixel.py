"""How a millimetre scale survives the transform pipeline.

Named `pixel`, not `geometry`. In the repos this came from, `geometry.py` meant
two unrelated things — cardiac plane geometry in AMP, pixel scale in BPF — which
is one of the ways the estate got into the state it did. `aidmr_utils.geometry`
is planes; this is scale.

Single source of truth, so dataset building, any inference path, and anything
converting a pixel-space measurement back to physical units all agree.

The chain a frame goes through is:

    native DICOM (rows x cols, PixelSpacing mm)
      -> LongestMaxSize(max_size=side)   uniform scale k = side / max(rows, cols)
      -> PadIfNeeded(side, side)         zero pad, no scale change
      -> Resize(side, side)              no-op once the image is already side x side

so one pixel of the tensor the network sees spans

    mm_per_pixel = PixelSpacing / k = PixelSpacing * max(rows, cols) / side

This is the whole reason padding to square matters: `PadIfNeeded(min_height,
min_width)` squashes the aspect ratio instead of padding, and then there is no
single scale factor to speak of.
"""

import numpy as np
from loguru import logger

#: In-plane spacing is square on Siemens, so anything beyond rounding is wrong.
SQUARE_PIXEL_TOLERANCE = 0.01


def mm_per_pixel_for_side(side, rows, cols, pixel_spacing) -> float:
    """mm spanned by one pixel after pad-to-square then resize to `side`.

    The scanner path calls this too, so the millimetre scale a deployed model is
    handed is computed by exactly the same arithmetic as the one it was trained
    against. Do not reimplement it anywhere else.

    `pixel_spacing` is the DICOM (row_spacing, col_spacing) in mm. The mean of the
    two axes is returned; in-plane CMR spacing is almost always isotropic, and
    `anisotropy` lets the caller check rather than assume.
    """
    scale_y, scale_x = float(pixel_spacing[0]), float(pixel_spacing[1])
    k = float(side) / max(int(rows), int(cols))
    return (scale_y + scale_x) / 2.0 / k


def anisotropy(pixel_spacing) -> float:
    """|sy - sx| / mean(sy, sx). Zero for isotropic in-plane spacing."""
    scale_y, scale_x = float(pixel_spacing[0]), float(pixel_spacing[1])
    mean = (scale_y + scale_x) / 2.0
    return abs(scale_y - scale_x) / mean if mean else 0.0


def assert_square_pixels(pixel_spacing, context: str = '',
                         tolerance: float = SQUARE_PIXEL_TOLERANCE) -> float:
    """Row and column spacing must agree. Returns the anisotropy.

    Worth doing for its own sake — `mm_per_pixel_for_side` averages the two axes,
    which is only exact when they are equal — but on the scanner it does a second,
    more valuable job. There, spacing is derived as FOV / matrix_size, and the
    pairing of the FOV axes to the matrix axes is a convention this code assumes
    rather than reads. If that pairing were wrong, a non-square acquisition
    (208 x 256, say) would produce two spacings differing by (256/208)^2 — a 50%
    discrepancy this catches instantly. It cannot detect the error on a square
    matrix, where the mis-pairing is harmless anyway.
    """
    ratio = anisotropy(pixel_spacing)
    assert ratio <= tolerance, (
        f"non-square pixels{f' for {context}' if context else ''}: row spacing "
        f"{float(pixel_spacing[0]):.6f} mm vs column {float(pixel_spacing[1]):.6f} mm "
        f"({ratio:.2%} apart, tolerance {tolerance:.2%}).\nEither the acquisition "
        f"is genuinely anisotropic — unexpected on Siemens — or, on the scanner "
        f"path, field_of_view has been paired with the wrong matrix_size axis in "
        f"mrd.native_pixel_spacing_mm, which would scale every volume wrongly."
    )
    return ratio


def validate_transform_pipeline(*, resize, pad_to_square: bool,
                                split_flags: dict | None = None) -> None:
    """Assert the pipeline is one where a single mm-per-pixel is well defined.

    Takes explicit arguments rather than a config object: the five repos spell
    their configs differently, and this needs to be callable from all of them.

    :param resize:       (h, w) the pipeline resizes to; must be square
    :param pad_to_square: whether padding, not squashing, reaches that square
    :param split_flags:  {'train': {'random_resized_crop': bool,
                                    'shift_scale_rotate': bool}, ...}
    """
    h, w = resize
    assert h == w, (
        f"pixel-scaled targets assume a square resize (the pad-to-square path); "
        f"got {h}x{w}")
    assert pad_to_square, (
        "pixel-scaled targets require pad_to_square — without it Resize squashes "
        "the aspect ratio and there is no single scale factor")
    for split, flags in (split_flags or {}).items():
        for flag in ('random_resized_crop', 'shift_scale_rotate'):
            assert not (flags or {}).get(flag, False), (
                f"transforms.{split}.{flag} rescales the image per sample, and the "
                f"replay parameters are not threaded through to the target "
                f"transform, so mm-per-pixel becomes unknowable. Turn it off, or "
                f"recover the true scale by pushing reference keypoints at (0,0), "
                f"(100,0), (0,100) through the same ReplayCompose and measuring "
                f"their output separation.")


def log_scale_summary(scales, anisotropies, within_study_spread) -> None:
    """One-shot report so the assumptions above are visible, not implicit."""
    scales = np.asarray(scales, dtype=float)
    if len(scales):
        logger.info(
            f"[pixel scale] {len(scales)} studies: median {np.median(scales):.4f} "
            f"mm/px (p5 {np.percentile(scales, 5):.4f}, "
            f"p95 {np.percentile(scales, 95):.4f})")
    if len(anisotropies):
        worst = float(np.max(anisotropies))
        n_bad = int((np.asarray(anisotropies) > SQUARE_PIXEL_TOLERANCE).sum())
        logger.info(
            f"[pixel scale] in-plane anisotropy: worst {worst * 100:.2f}%, "
            f"{n_bad} series over 1% (mean of the two axes is used)")
    if len(within_study_spread):
        spread = np.asarray(within_study_spread, dtype=float)
        n_bad = int((spread > 0.02).sum())
        logger.info(
            f"[pixel scale] 2c-vs-4c scale disagreement within a study: median "
            f"{np.median(spread) * 100:.2f}%, {n_bad} studies over 2%")
        if n_bad:
            logger.warning(
                f"{n_bad} studies have 2c and 4c acquired at materially different "
                f"scales; a single per-study mm/px is an approximation for those.")
