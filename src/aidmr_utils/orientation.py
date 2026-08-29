"""Reorienting an image into DICOM norm, and back again.

WHY THIS EXISTS
---------------
The AID-MR models are trained on DICOM pixel arrays, and by the time the scanner
writes a DICOM it has already applied its display convention (see
`aidmr_utils.geometry`). The training set is therefore consistent: a 2ch is
upright, a 4ch is level.

MRD images arriving over FIRE have had no such treatment. The pixel matrix is
whatever fell out of the reconstruction, and when several differently-oriented
slices share one protocol and matrix — a combined "2ch 4ch" cine is exactly this
— the readout and phase axes get reassigned per slice, so one view can arrive 90
degrees out relative to the other. A network trained without rotation
augmentation then produces meaningless output that still looks like output.

WHICH MODELS NEED THIS
----------------------
Only the ones not trained to be orientation-invariant. Check the augmentation
before assuming:

    CMRQ   no flip/rotate augmentation      -> needs canonicalising
    BPF    hflip/vflip/shift_scale_rotate false -> needs canonicalising
    AIFS   hflip, vflip, rotate:90          -> invariant by design, needs none

Canonicalising an already-invariant model's input is harmless but pointless;
skipping it for a non-invariant one is silent corruption.

USE THE META, NOT read_dir / phase_dir
--------------------------------------
For an MRD image the stored matrix layout is declared in the MetaAttributes as
`ImageRowDir` / `ImageColumnDir`, and those are what become DICOM's
`ImageOrientationPatient` downstream. `head.read_dir` / `head.phase_dir` describe
the *encoding* axes, which is a different pair of vectors.

This cost a scanner run to find out. An earlier version read the header on the
assumption that python-ismrmrd-server fills the meta from it — true for images
the server creates, not for images arriving from the scanner. On the 2026-08-12
`amplax2sax__cmrq__bpf` run the two disagreed by a flip on the 4ch and a
90-degree rotation on the 2ch, so BPF and CMRQ reoriented the same two cines
differently: BPF's 4ch reached the network mirrored and its 2ch rotated, while
CMRQ's were right. Both repos then had to be checked against each other by hand.
That cross-repo check is why this module is here rather than there.

Only images captured AFTER the python-ismrmrd-server `fixTransposed` fix (first
seen in the AIDMR-FIRE logs on 2025-07-26) follow this convention; older saved
.h5 files are transposed relative to it.
"""

from typing import Callable, Tuple

import numpy as np
from loguru import logger

from .geometry import CANONICAL_RIGHT_DOWN, Plane, closest_plane, unit


def _rot90(a: np.ndarray, k: int) -> np.ndarray:
    return np.rot90(a, k, axes=(-2, -1))


# The dihedral group of the square: every way of flipping/rotating a 2D image.
# Each entry is (name, array op, vector op) where the vector op maps the current
# (right, down) direction vectors to the ones the transformed image would have.
#
# For np.rot90(a, 1) the old top-right corner ends up top-left, so the old "right"
# direction becomes the new "up": right' = down, down' = -right.
TRANSFORMS: list[Tuple[str, Callable[[np.ndarray], np.ndarray], Callable]] = [
    ('identity',       lambda a: a,                            lambda r, d: (r, d)),
    ('rot90',          lambda a: _rot90(a, 1),                 lambda r, d: (d, -r)),
    ('rot180',         lambda a: _rot90(a, 2),                 lambda r, d: (-r, -d)),
    ('rot270',         lambda a: _rot90(a, 3),                 lambda r, d: (-d, r)),
    ('fliplr',         lambda a: np.flip(a, axis=-1),          lambda r, d: (-r, d)),
    ('flipud',         lambda a: np.flip(a, axis=-2),          lambda r, d: (r, -d)),
    ('transpose',      lambda a: np.swapaxes(a, -2, -1),       lambda r, d: (d, r)),
    ('anti_transpose', lambda a: np.swapaxes(_rot90(a, 2), -2, -1), lambda r, d: (-d, -r)),
]

TRANSFORMS_BY_NAME = {name: (arr_op, vec_op) for name, arr_op, vec_op in TRANSFORMS}

#: The inverse of each transform, needed because overlays are sent back to the
#: scanner. The model works in canonical orientation, but the images pushed back
#: carry the *acquired* image's position and direction vectors, so they have to be
#: returned to the acquired layout or they would display rotated relative to the
#: source cine. Every rotation is its own inverse except the 90/270 pair; both
#: mirrors and both diagonal reflections are self-inverse. The test suite checks
#: each pair by composition rather than trusting this table.
INVERSE_TRANSFORMS: dict[str, str] = {
    'identity': 'identity',
    'rot90': 'rot270',
    'rot180': 'rot180',
    'rot270': 'rot90',
    'fliplr': 'fliplr',
    'flipud': 'flipud',
    'transpose': 'transpose',
    'anti_transpose': 'anti_transpose',
}


def _strict_unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    if not np.linalg.norm(v):
        raise ValueError("Zero-length direction vector")
    return unit(v)


# ---------------------------------------------------------------------------
# Reading geometry off an image
# ---------------------------------------------------------------------------

def _meta_vector(mrd, key: str) -> np.ndarray:
    """Pull a 3-vector out of an ismrmrd.Image's meta (values arrive as strings)."""
    return _strict_unit([float(x) for x in mrd.meta[key]])


def image_axes_from_mrd(mrd) -> Tuple[np.ndarray, np.ndarray]:
    """LPS unit vectors for an MRD image's (right, down) screen directions.

    That is, the directions of increasing column index and increasing row index
    in `np.squeeze(mrd.data)`. Deliberately the meta, not the header — see the
    module docstring.
    """
    right = _meta_vector(mrd, 'ImageRowDir')      # along a row -> increasing column
    down = _meta_vector(mrd, 'ImageColumnDir')    # down a column -> increasing row
    return right, down


def image_axes_from_dicom(dcm) -> Tuple[np.ndarray, np.ndarray]:
    """The same (right, down) pair, from a DICOM's ImageOrientationPatient.

    Present so a training pipeline can ask the same question of its inputs that
    the deployed pipeline asks of MRD images — for instance to confirm that a
    training corpus really is in DICOM norm before relying on it.
    """
    iop = [float(x) for x in dcm.ImageOrientationPatient]
    if len(iop) != 6:
        raise ValueError(f"ImageOrientationPatient must have 6 values, got {len(iop)}")
    return _strict_unit(iop[:3]), _strict_unit(iop[3:])


def image_row_col_dirs(mrd) -> Tuple[np.ndarray, np.ndarray]:
    """The (right, down) pair to declare on an output built from this image.

    `image_axes_from_mrd` with a fallback, for the send path rather than the
    receive path. A missing tag has to produce *something* here, since the
    alternative is declaring no layout at all — which is the bug this exists to
    fix.

    The fallback is the header's `read_dir` / `phase_dir`. `canonicalise_cine`
    refuses that substitution and this makes it, and both are right: there, the
    wrong vectors silently rotate the cine the network sees and corrupt the
    measurement; here they are the best available guess at the layout and are
    what the scanner would have assumed anyway, so the worst case is no worse
    than declaring nothing. It is also what every module in
    `python-ismrmrd-server` does when the incoming image carries no pair.

    In practice FIRE always sends the pair, so this warns when it fires.
    """
    try:
        return image_axes_from_mrd(mrd)
    except (KeyError, AttributeError, TypeError, ValueError) as e:
        head = mrd.getHead()
        logger.warning(f"No usable ImageRowDir/ImageColumnDir on incoming image "
                       f"({e!r}) - falling back to the header's read_dir/phase_dir "
                       f"for the layout declared on the images sent back, which "
                       f"may leave them rotated relative to the source cine")
        return (_strict_unit(np.asarray(head.read_dir, dtype=float)),
                _strict_unit(np.asarray(head.phase_dir, dtype=float)))


# ---------------------------------------------------------------------------
# Choosing and applying a transform
# ---------------------------------------------------------------------------

def choose_transform(right_xyz, down_xyz,
                     plane: Plane | None = None) -> Tuple[str, float, Plane]:
    """Pick the flip/rotate that best matches the conventional display orientation.

    Scoring is `dot(right', right_target) + dot(down', down_target)`, so a perfect
    match scores 2.0. Mirror transforms are included on purpose: the display
    convention fixes the side the slice is viewed from, so a slice acquired with
    the opposite normal legitimately needs mirroring, which is what any DICOM
    viewer does.

    :return: (transform name, score, plane used)
    """
    right_xyz, down_xyz = _strict_unit(right_xyz), _strict_unit(down_xyz)
    if plane is None:
        plane = closest_plane(np.cross(right_xyz, down_xyz))

    right_target, down_target = CANONICAL_RIGHT_DOWN[plane]

    best_name, best_score = 'identity', -np.inf
    for name, _arr_op, vec_op in TRANSFORMS:
        r_new, d_new = vec_op(right_xyz, down_xyz)
        score = float(np.dot(r_new, right_target) + np.dot(d_new, down_target))
        if score > best_score:
            best_name, best_score = name, score

    return best_name, best_score, plane


def apply_transform(video: np.ndarray, name: str) -> np.ndarray:
    """Apply a named transform to the last two axes of an array.

    The last two axes must be (row, col), so this takes (H, W) heatmaps and
    (T, H, W) cines but NOT a (T, H, W, 3) colour overlay. Build overlays from
    already-transformed greyscale inputs rather than transforming the colour
    result.
    """
    return np.ascontiguousarray(TRANSFORMS_BY_NAME[name][0](video))


def inverse_transform(name: str) -> str:
    """The transform that undoes `name`."""
    return INVERSE_TRANSFORMS[name]


def canonicalise_cine(video: np.ndarray, mrd) -> Tuple[np.ndarray, str, Plane]:
    """Reorient a cine into the conventional DICOM display orientation.

    Falls back to the identity (with a warning) if the image carries no usable
    geometry metadata, so a missing tag degrades to the pre-fix behaviour rather
    than killing the measurement. It does NOT fall back to the header's
    read_dir/phase_dir: those are a different pair of vectors, and using them
    silently produces a mirrored or rotated cine, which is worse than not
    reorienting at all because it looks plausible.

    :param video: array whose last two axes are (row, col)
    :param mrd:   an ismrmrd.Image from this cine (all frames share a geometry)
    :return:      (reoriented video, transform name, plane)
    """
    try:
        right, down = image_axes_from_mrd(mrd)
    except (KeyError, AttributeError, TypeError, ValueError) as e:
        logger.warning(f"No usable ImageRowDir/ImageColumnDir on incoming image "
                       f"({e!r}) - leaving orientation untouched, so this cine "
                       f"reaches the network as the recon produced it")
        return video, 'identity', Plane.AXIAL

    name, score, plane = choose_transform(right, down)
    logger.info(f"Slice is closest to {plane.value}; applying '{name}' "
                f"(alignment {score:.2f}/2.00) to reach DICOM display orientation")
    if score < 1.0:
        logger.warning(f"Best orientation match is only {score:.2f}/2.00 - this "
                       f"slice is very oblique, so the canonical orientation is "
                       f"approximate")

    return apply_transform(video, name), name, plane
