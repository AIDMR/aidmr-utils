"""Cardinal planes and the DICOM norm table.

The geometries here are real, from AIDMR-FIRE/logs/mrd captures and from
scanner-written DICOMs in the LGEP corpus.
"""

import numpy as np
import pytest

from aidmr_utils.geometry import (CANONICAL_RIGHT_DOWN, CONVENTION_EVIDENCE,
                                  Plane, canonical_right_down, closest_plane,
                                  normal_for_named_plane, plane_of, unit)


# --- real geometries, (ImageRowDir, ImageColumnDir) = (right, down) ----------
TRUE_AXIAL = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
CINE_4CH = ((0.996, 0.089, 0.0), (0.06, -0.676, -0.734))
CINE_2CH_ROTATED = ((0.168, -0.088, -0.982), (-0.609, 0.774, -0.173))
CINE_3CH_SAGITTALISH = ((0.095, 0.995, 0.0), (0.637, -0.061, -0.768))
CINE_3CH_CORONALISH = ((0.996, 0.086, 0.0), (-0.006, 0.075, -0.997))

# Scanner-written DICOMs (Siemens LGE LVSA), already in DICOM norm
DICOM_SAGITTAL = ((0.196, 0.975, -0.106), (-0.475, 0.0, -0.88))
DICOM_CORONAL = ((0.682, 0.68, -0.268), (-0.366, 0.0, -0.931))
DICOM_AXIAL = ((0.799, 0.0, 0.602), (0.275, 0.89, -0.365))


def test_closest_plane_on_cardinal_normals():
    assert closest_plane([0, 0, 1.0]) == Plane.AXIAL
    assert closest_plane([1.0, 0, 0]) == Plane.SAGITTAL
    assert closest_plane([0, 1.0, 0]) == Plane.CORONAL


def test_closest_plane_ignores_the_sign_of_the_normal():
    """A slice and the same slice acquired from the other side are one plane."""
    for v in ([0, 0, 1.0], [1.0, 0, 0], [0, 1.0, 0], [0.065, -0.731, 0.679]):
        assert closest_plane(v) == closest_plane(np.negative(v))


def test_closest_plane_does_not_need_a_unit_vector():
    assert closest_plane([0, 0, 7.3]) == Plane.AXIAL


@pytest.mark.parametrize('geom,expected', [
    (TRUE_AXIAL, Plane.AXIAL),
    (CINE_4CH, Plane.CORONAL),
    (CINE_2CH_ROTATED, Plane.SAGITTAL),
    (CINE_3CH_SAGITTALISH, Plane.SAGITTAL),
    (CINE_3CH_CORONALISH, Plane.CORONAL),
    (DICOM_SAGITTAL, Plane.SAGITTAL),
    (DICOM_CORONAL, Plane.CORONAL),
    (DICOM_AXIAL, Plane.AXIAL),
])
def test_plane_of_real_acquisitions(geom, expected):
    assert plane_of(*geom) == expected


# ---------------------------------------------------------------------------
# The convention itself
# ---------------------------------------------------------------------------

def test_sagittal_is_anterior_on_the_left():
    """Measured 7150/7150 on scanner-written DICOMs. See the module docstring.

    screen-right = +y = posterior, therefore anterior is on the screen LEFT.
    This is the OPPOSITE of AMP's planning geometry, which is wrong (though
    inert, since those vectors only drive its preview and FOV).
    """
    right, down = canonical_right_down(Plane.SAGITTAL)
    assert np.allclose(right, [0, 1, 0]), "sagittal screen-right must be posterior"
    assert np.allclose(down, [0, 0, -1]), "sagittal screen-down must be footward"


def test_axial_and_coronal_put_patient_right_on_the_screen_left():
    for plane in (Plane.AXIAL, Plane.CORONAL):
        right, _ = canonical_right_down(plane)
        assert np.allclose(right, [1, 0, 0])


def test_axial_looks_up_from_the_feet():
    _, down = canonical_right_down(Plane.AXIAL)
    assert np.allclose(down, [0, 1, 0]), "anterior must be at the top"


def test_canonical_pairs_are_orthonormal():
    for plane, (right, down) in CANONICAL_RIGHT_DOWN.items():
        assert np.isclose(np.linalg.norm(right), 1.0)
        assert np.isclose(np.linalg.norm(down), 1.0)
        assert np.isclose(np.dot(right, down), 0.0), f"{plane} pair is not orthogonal"


def test_each_canonical_pair_really_lies_in_its_own_plane():
    """A self-consistency check on the table: right x down must classify back."""
    for plane, (right, down) in CANONICAL_RIGHT_DOWN.items():
        assert plane_of(right, down) == plane


def test_canonical_right_down_returns_copies():
    """The table is mutable; a caller normalising in place must not corrupt it."""
    right, _ = canonical_right_down(Plane.SAGITTAL)
    right *= 5
    assert np.allclose(CANONICAL_RIGHT_DOWN[Plane.SAGITTAL][0], [0, 1, 0])


def test_convention_evidence_is_recorded():
    """The counts behind the table, so a future reader can weigh them."""
    for plane, (matching, total) in CONVENTION_EVIDENCE.items():
        assert Plane(plane)
        assert matching / total > 0.99


def test_normal_for_named_plane_round_trips():
    for plane in Plane:
        assert closest_plane(normal_for_named_plane(plane)) == plane


def test_plane_serialises_as_its_name():
    import json
    assert json.dumps({'p': Plane.SAGITTAL}) == '{"p": "sagittal"}'


def test_unit_leaves_a_zero_vector_alone():
    assert np.allclose(unit([0, 0, 0]), [0, 0, 0])
