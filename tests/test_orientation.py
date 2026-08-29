"""Reorienting into DICOM norm, and the inverse used to send images back."""

import numpy as np
import pytest

from aidmr_utils.geometry import Plane
from aidmr_utils.orientation import (INVERSE_TRANSFORMS, TRANSFORMS,
                                     TRANSFORMS_BY_NAME, apply_transform,
                                     canonicalise_cine, choose_transform,
                                     image_axes_from_dicom, image_axes_from_mrd,
                                     image_row_col_dirs, inverse_transform)

from test_geometry import (CINE_2CH_ROTATED, CINE_3CH_CORONALISH,
                           CINE_3CH_SAGITTALISH, CINE_4CH, DICOM_AXIAL,
                           DICOM_CORONAL, DICOM_SAGITTAL, TRUE_AXIAL)

NAMES = [name for name, _, _ in TRANSFORMS]


class FakeMRD:
    """Only `meta` is needed for orientation."""

    def __init__(self, image_row_dir, image_column_dir):
        self.meta = {
            'ImageRowDir': [str(v) for v in image_row_dir],
            'ImageColumnDir': [str(v) for v in image_column_dir],
        }


class MetalessMRD:
    meta = {}

    class _Head:
        read_dir = (1.0, 0.0, 0.0)
        phase_dir = (0.0, 1.0, 0.0)

    def getHead(self):
        return self._Head()


# ---------------------------------------------------------------------------
# The transform group
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('name', NAMES)
def test_vector_op_matches_array_op(name):
    """The vector bookkeeping must agree with what the array op actually does.

    Encode the right/down directions as gradients: pixel value increases with
    column index along `right` and with row index along `down`. After
    transforming, the gradients must run along the transformed vectors.
    """
    rows, cols = 5, 7
    r, c = np.mgrid[0:rows, 0:cols]
    right, down = np.array([1.0, 0, 0]), np.array([0, 1.0, 0])
    arr_op, vec_op = TRANSFORMS_BY_NAME[name]
    r_new, d_new = vec_op(right, down)

    out_x, out_y = arr_op(c.astype(float)), arr_op(r.astype(float))

    def gradient(img):
        return float(img[0, 1] - img[0, 0]), float(img[1, 0] - img[0, 0])

    assert gradient(out_x) == pytest.approx((r_new[0], d_new[0]))
    assert gradient(out_y) == pytest.approx((r_new[1], d_new[1]))


@pytest.mark.parametrize('name', NAMES)
def test_inverse_undoes_the_transform(name):
    """Checked by composition rather than by trusting INVERSE_TRANSFORMS."""
    img = np.arange(35, dtype=float).reshape(5, 7)
    there = apply_transform(img, name)
    back = apply_transform(there, inverse_transform(name))
    assert np.array_equal(back, img)


def test_inverse_table_covers_every_transform():
    assert set(INVERSE_TRANSFORMS) == set(NAMES)


def test_transforms_are_distinct():
    """Eight members of the dihedral group, no accidental duplicates."""
    img = np.arange(16, dtype=float).reshape(4, 4)
    seen = {apply_transform(img, n).tobytes() for n in NAMES}
    assert len(seen) == len(NAMES)


def test_apply_transform_touches_only_the_last_two_axes():
    cine = np.random.default_rng(0).random((9, 4, 6))
    out = apply_transform(cine, 'transpose')
    assert out.shape == (9, 6, 4)
    assert np.allclose(out[3], cine[3].T)


# ---------------------------------------------------------------------------
# Choosing a transform on real acquisitions
# ---------------------------------------------------------------------------

def test_true_axial_needs_no_transform():
    name, score, plane = choose_transform(*[np.array(v) for v in TRUE_AXIAL])
    assert (name, plane) == ('identity', Plane.AXIAL)
    assert score == pytest.approx(2.0)


def test_4ch_arrives_correctly_oriented():
    name, score, _ = choose_transform(*[np.array(v) for v in CINE_4CH])
    assert name == 'identity' and score > 1.5


def test_2ch_sharing_a_matrix_with_4ch_is_rotated():
    """The bug that started all this: a 2ch acquired alongside a 4ch comes in 90 out."""
    name, score, plane = choose_transform(*[np.array(v) for v in CINE_2CH_ROTATED])
    assert (name, plane) == ('transpose', Plane.SAGITTAL)
    assert score > 1.5


@pytest.mark.parametrize('geom', [CINE_3CH_SAGITTALISH, CINE_3CH_CORONALISH])
def test_3ch_arrives_correctly_oriented(geom):
    name, score, _ = choose_transform(*[np.array(v) for v in geom])
    assert name == 'identity' and score > 1.5


@pytest.mark.parametrize('geom', [DICOM_SAGITTAL, DICOM_CORONAL, DICOM_AXIAL])
def test_scanner_written_dicoms_are_already_canonical(geom):
    """DICOM norm is what the scanner already applied, so this must be a no-op.

    If this fails, CANONICAL_RIGHT_DOWN disagrees with the scanner - which is
    exactly the sagittal question this package settled.
    """
    name, score, _ = choose_transform(*[np.array(v) for v in geom])
    assert name == 'identity', f"scanner DICOM would be re-oriented by '{name}'"
    assert score > 1.0


def test_a_mirrored_sagittal_is_corrected():
    """A sagittal acquired with anterior on the right must be flipped back."""
    right, down = np.array([0.0, -1.0, 0.0]), np.array([0.0, 0.0, -1.0])
    name, score, plane = choose_transform(right, down)
    assert plane == Plane.SAGITTAL
    assert name == 'fliplr'
    assert score == pytest.approx(2.0)


def test_score_is_two_for_an_exactly_canonical_slice():
    from aidmr_utils.geometry import canonical_right_down
    for plane in Plane:
        name, score, found = choose_transform(*canonical_right_down(plane))
        assert (name, found) == ('identity', plane)
        assert score == pytest.approx(2.0)


def test_zero_length_vector_is_rejected():
    with pytest.raises(ValueError, match='Zero-length'):
        choose_transform(np.zeros(3), np.array([0.0, 0.0, -1.0]))


# ---------------------------------------------------------------------------
# Reading geometry off images
# ---------------------------------------------------------------------------

def test_image_axes_read_from_meta_not_the_header():
    mrd = FakeMRD(*CINE_4CH)
    right, down = image_axes_from_mrd(mrd)
    assert np.allclose(right, np.array(CINE_4CH[0]) / np.linalg.norm(CINE_4CH[0]))
    assert np.allclose(down, np.array(CINE_4CH[1]) / np.linalg.norm(CINE_4CH[1]))


def test_image_axes_from_dicom_matches_the_mrd_reading():
    """The same pair, read from ImageOrientationPatient."""
    class FakeDCM:
        ImageOrientationPatient = list(DICOM_SAGITTAL[0]) + list(DICOM_SAGITTAL[1])

    right, down = image_axes_from_dicom(FakeDCM())
    m_right, m_down = image_axes_from_mrd(FakeMRD(*DICOM_SAGITTAL))
    assert np.allclose(right, m_right) and np.allclose(down, m_down)


def test_image_axes_from_dicom_rejects_a_short_iop():
    class FakeDCM:
        ImageOrientationPatient = [1.0, 0.0, 0.0]

    with pytest.raises(ValueError, match='6 values'):
        image_axes_from_dicom(FakeDCM())


def test_row_col_dirs_falls_back_to_the_header_for_the_send_path():
    right, down = image_row_col_dirs(MetalessMRD())
    assert np.allclose(right, [1, 0, 0]) and np.allclose(down, [0, 1, 0])


# ---------------------------------------------------------------------------
# canonicalise_cine
# ---------------------------------------------------------------------------

def test_canonicalise_cine_rotates_the_2ch():
    raw = np.random.default_rng(0).random((12, 8, 6))
    out, name, plane = canonicalise_cine(raw, FakeMRD(*CINE_2CH_ROTATED))
    assert name == 'transpose' and plane == Plane.SAGITTAL
    assert out.shape == (12, 6, 8)
    assert np.array_equal(out, apply_transform(raw, 'transpose'))


def test_canonicalise_cine_leaves_a_level_4ch_alone():
    raw = np.random.default_rng(1).random((12, 8, 8))
    out, name, _ = canonicalise_cine(raw, FakeMRD(*CINE_4CH))
    assert name == 'identity' and np.array_equal(out, raw)


def test_canonicalise_cine_does_NOT_fall_back_to_the_header():
    """Refusing the substitution is the point: the header is a different pair.

    Using it silently produces a mirrored or rotated cine, which is worse than
    not reorienting at all because it still looks plausible.
    """
    raw = np.random.default_rng(2).random((4, 5, 5))
    out, name, _ = canonicalise_cine(raw, MetalessMRD())
    assert name == 'identity'
    assert np.array_equal(out, raw)


def test_canonicalise_then_invert_returns_the_acquired_layout():
    """What the send path relies on for overlays pushed back to the scanner."""
    raw = np.random.default_rng(3).random((6, 8, 5))
    canon, name, _ = canonicalise_cine(raw, FakeMRD(*CINE_2CH_ROTATED))
    assert np.array_equal(apply_transform(canon, inverse_transform(name)), raw)
