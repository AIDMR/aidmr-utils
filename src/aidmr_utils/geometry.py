"""Cardinal planes and the display orientation the scanner writes into DICOM.

All vectors are LPS (DICOM patient coordinates):

    +x = patient LEFT       +y = POSTERIOR      +z = HEAD

An image's in-plane geometry is a (right, down) pair: the LPS directions of
increasing COLUMN index and increasing ROW index respectively. In a DICOM these
are `ImageOrientationPatient[0:3]` and `[3:6]`; in an MRD image they are the
`ImageRowDir` and `ImageColumnDir` MetaAttributes.

DICOM NORM
----------
When the scanner writes a slice out as a DICOM it has already put the pixel
matrix into a conventional display orientation ("DICOM norm"), which depends only
on which cardinal plane the slice is closest to:

    axial     down = posterior (+y),  right = patient left (+x)
        -> patient's front at the top, patient's RIGHT on the LEFT of screen
    coronal   down = foot      (-z),  right = patient left (+x)
        -> patient's head at the top, patient's RIGHT on the LEFT of screen
    sagittal  down = foot      (-z),  right = posterior    (+y)
        -> patient's head at the top, patient's FRONT on the LEFT of screen

THE SAGITTAL CONVENTION IS SETTLED; DO NOT RE-LITIGATE IT
---------------------------------------------------------
Sagittal handedness used to be the open question in this codebase. CMRQ chose
"anterior on the left" from the look of its 2ch training set and hedged that the
convention "differs between sites"; BPF inherited that and recorded that it had
never been re-checked; AMP's planning geometry asserts the OPPOSITE ("right =
anterior"). Three repos, two answers, no evidence.

It has now been measured, over 11,458 scanner-written cardiac DICOMs from four
Manufacturer strings (Siemens, GE, Philips):

    sagittal   screen-right = +y (posterior)     7150 / 7150     100.0%
    coronal    screen-right = +x (patient left)  1459 / 1459     100.0%
    axial      screen-right = +x (patient left)  1082 / 1084      99.8%

Anterior-on-the-left is not a site preference; it is unanimous. Slices too far
rotated for the sign of the in-plane component to be meaningful were excluded
from the counts rather than being allowed to vote — for a 90-degrees-out slice
that sign is noise, which is what made an earlier survey of MRD captures read
75% instead of 100%.

AMP's `get_default_right_down_unit_vectors_for_freq_phase` is therefore wrong on
sagittal. It is also inert: its own call site notes the vectors "do NOT get used
at ALL for slice planning - they are purely to generate the preview and then
calculate the FOV", so the symptom is a mirrored sagittal-ish preview (a 2ch),
not a mis-planned slice.
"""

from enum import Enum
from typing import Tuple

import numpy as np

#: Number of DICOMs behind CANONICAL_RIGHT_DOWN, quoted so a future reader can
#: judge the evidence rather than trusting the table.
CONVENTION_EVIDENCE = {
    'sagittal': (7150, 7150),
    'coronal': (1459, 1459),
    'axial': (1082, 1084),
}


class Plane(str, Enum):
    """A cardinal plane. `str` so it serialises into JSON as its own name."""
    AXIAL = 'axial'
    SAGITTAL = 'sagittal'
    CORONAL = 'coronal'


#: (right, down) unit vectors in LPS for the conventional display of each plane.
CANONICAL_RIGHT_DOWN: dict[Plane, Tuple[np.ndarray, np.ndarray]] = {
    Plane.AXIAL:    (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])),
    Plane.CORONAL:  (np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, -1.0])),
    Plane.SAGITTAL: (np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, -1.0])),
}


def unit(v) -> np.ndarray:
    """Normalise, leaving a zero vector alone rather than dividing by zero."""
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n else v


def closest_plane(normal_xyz) -> Plane:
    """Classify a slice by whichever component of its normal is largest.

    The sign of the normal must not matter: a slice and the same slice acquired
    from the other side are the same plane.
    """
    i = int(np.argmax(np.abs(unit(normal_xyz))))
    return (Plane.SAGITTAL, Plane.CORONAL, Plane.AXIAL)[i]


def normal_from_right_down(right_xyz, down_xyz) -> np.ndarray:
    """The slice normal implied by an in-plane (right, down) pair."""
    return np.cross(unit(right_xyz), unit(down_xyz))


def plane_of(right_xyz, down_xyz) -> Plane:
    """Which cardinal plane an in-plane (right, down) pair is closest to."""
    return closest_plane(normal_from_right_down(right_xyz, down_xyz))


def canonical_right_down(plane: Plane) -> Tuple[np.ndarray, np.ndarray]:
    """The (right, down) DICOM norm puts a slice of this plane into.

    Returns copies: the module-level table is mutable and a caller normalising
    in place would silently corrupt every later classification.
    """
    right, down = CANONICAL_RIGHT_DOWN[Plane(plane)]
    return right.copy(), down.copy()


def normal_for_named_plane(plane) -> np.ndarray:
    """The LPS normal of a named cardinal plane."""
    plane = Plane(plane)
    return {
        Plane.AXIAL: np.array([0.0, 0.0, 1.0]),
        Plane.SAGITTAL: np.array([1.0, 0.0, 0.0]),
        Plane.CORONAL: np.array([0.0, 1.0, 0.0]),
    }[plane]
