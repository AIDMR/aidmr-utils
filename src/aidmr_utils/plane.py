"""Slice planes in the scanner's exam frame, from three sources, and how to compare them.

The three sources are the ones the AID-MR replay tests have to reconcile:

* a **DICOM** the scanner wrote (an acquired cine or stack slice);
* an **MRD image** FIRE sent (an acquired cine, seen by a program);
* an **exam-memory instruction** a program sent back (what it *planned*).

FRAMES
------
All vectors are LPS, as in `geometry.py`. A DICOM's ImagePositionPatient /
ImageOrientationPatient are in the patient frame the scanner plans in, which is
also the frame the exam-memory positions are in. An MRD image from FIRE carries
TWO positions: the header's `position` (the ICE frame, offset from the exam
frame by the table / light-marker shift) and the `SlicePosLightMarker`
MetaAttribute, which is the slice centre in the exam frame. Measured on a
patient exam (2026-09-12): the acquired 2ch and 4ch cines' SlicePosLightMarker
equalled the planned centre to 0.01 mm while `position` was 24 mm away, and the
DICOM centres agreed with the plan within one pixel. AMP itself moves its plan
into the exam frame with `SlicePosLightMarker - stack centre`, so this module is
the same logic read the other way: an acquired MRD image's exam-frame centre IS
its SlicePosLightMarker. Nothing here is fitted.

A DICOM's ImagePositionPatient is the CENTRE OF THE FIRST (top-left) PIXEL, not
the corner of the image and not the image centre. The plane centre is therefore
IPP + right * spacing_col * (cols - 1) / 2 + down * spacing_row * (rows - 1) / 2,
and PixelSpacing is (between rows, between columns), i.e. (along down, along
right). Siemens positions on an even matrix carry a half-pixel convention, which
is why a DICOM should be compared with a plan to one pixel, not to 0.1 mm.

AXES
----
A plane's in-plane axes are stored as an unordered pair (u, v) with an extent
along each: for a DICOM or MRD image u is `right` (increasing column index) and
v is `down` (increasing row index); for a plan u is the readout (frequency)
direction and v the phase direction. Which of those coincides with which is a
display convention (on the XA60 exam above the readout ran DOWN the DICOM, with
InPlanePhaseEncodingDirection = ROW), so comparisons match axes by direction,
never by the words row/column/phase/readout. Normals are compared up to sign: a
slice and the same slice acquired from the other side are the same plane.

EXAM MEMORY LAYOUT (what AMP's fire_utils writes)
--------------------------------------------------
    <prefix>_<name>_Slice = [slicegroup, name, cx, cy, cz, nx, ny, nz, thickness,
                             distance_factor, phase_fov_prop, readout_fov_mm,
                             in_plane_rot_rad, n_slices]
    <prefix>_<name>_FOV   = [slicefov, 1, cx, cy, cz, readout_mm, fx, fy, fz,
                             phase_mm, px, py, pz, 0, 0, 0, name]
with prefix `SiemensExamMemory_wip_070_fire_ICEOut`. The f and p vectors carry
direction AND magnitude (|f| = readout FOV). A stack (n_slices > 1) is centred
on (cx, cy, cz) with slices `thickness * (1 + distance_factor)` apart along the
normal.

Needs nothing beyond numpy. Pass it a pydicom Dataset, an ismrmrd.Image, or the
plain records h5py hands back; it reads attributes, it does not import either
library.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np

from .geometry import unit

EXAM_MEMORY_PREFIX = 'SiemensExamMemory_wip_070_fire_ICEOut_'
LIGHT_MARKER_KEY = 'SlicePosLightMarker'


def _vec(v) -> np.ndarray:
    return np.asarray([float(x) for x in v], dtype=float)


def _get(obj, name, default=None):
    """Attribute or item access, so pydicom Datasets, ismrmrd headers and numpy
    structured records all work."""
    if hasattr(obj, name):
        return getattr(obj, name)
    try:
        return obj[name]
    except (KeyError, IndexError, TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# The plane
# ---------------------------------------------------------------------------

@dataclass
class ImagePlane:
    """One slice: centre, in-plane axes with their extents, thickness.

    `u_xyz` / `v_xyz` are unit vectors spanning the plane; `extent_u_mm` is the
    full field of view along u. See the module docstring for which is which
    per source. `normal` is u x v, so for a DICOM (right x down) it points the
    way DICOM's own convention does.
    """
    centre_xyz: np.ndarray
    u_xyz: np.ndarray
    v_xyz: np.ndarray
    extent_u_mm: float
    extent_v_mm: float
    thickness_mm: Optional[float] = None
    name: str = ''
    source: str = ''
    #: where the axes came from, for messages: 'right/down' or 'readout/phase'
    axes: str = ''
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        self.centre_xyz = _vec(self.centre_xyz)
        self.u_xyz = unit(_vec(self.u_xyz))
        self.v_xyz = unit(_vec(self.v_xyz))
        if abs(float(np.dot(self.u_xyz, self.v_xyz))) > 1e-3:
            raise ValueError(f"{self.name or self.source}: in-plane axes are not orthogonal "
                             f"(u.v = {float(np.dot(self.u_xyz, self.v_xyz)):.4f})")

    @property
    def normal_xyz(self) -> np.ndarray:
        return unit(np.cross(self.u_xyz, self.v_xyz))

    @property
    def pixel_mm(self) -> Optional[float]:
        """In-plane pixel size if the source had a matrix (DICOM / MRD), else None."""
        return self.extra.get('pixel_mm')

    # -- constructors -------------------------------------------------------

    @classmethod
    def from_dicom(cls, ds, name: str = '') -> 'ImagePlane':
        """From ImagePositionPatient, ImageOrientationPatient, PixelSpacing, Rows, Columns.

        Works on a pydicom Dataset or anything with those attributes.
        """
        ipp = _vec(_get(ds, 'ImagePositionPatient'))
        iop = _vec(_get(ds, 'ImageOrientationPatient'))
        if iop.shape != (6,):
            raise ValueError(f"ImageOrientationPatient must have 6 values, got {iop.shape}")
        right, down = unit(iop[:3]), unit(iop[3:])
        spacing = _vec(_get(ds, 'PixelSpacing'))          # (between rows, between columns)
        sp_row, sp_col = float(spacing[0]), float(spacing[1])
        rows, cols = int(_get(ds, 'Rows')), int(_get(ds, 'Columns'))
        centre = ipp + right * sp_col * (cols - 1) / 2.0 + down * sp_row * (rows - 1) / 2.0
        thickness = _get(ds, 'SliceThickness', None)
        return cls(centre_xyz=centre, u_xyz=right, v_xyz=down,
                   extent_u_mm=sp_col * cols, extent_v_mm=sp_row * rows,
                   thickness_mm=float(thickness) if thickness is not None else None,
                   name=name or str(_get(ds, 'SeriesDescription', '') or '').strip(),
                   source='dicom', axes='right/down',
                   extra={'pixel_mm': (sp_row + sp_col) / 2.0, 'rows': rows, 'cols': cols,
                          'ipp_xyz': ipp,
                          'phase_encoding_direction': _get(ds, 'InPlanePhaseEncodingDirection', None)})

    @classmethod
    def from_mrd(cls, image=None, *, head=None, meta: Optional[Mapping] = None,
                 frame: str = 'exam', name: str = '') -> 'ImagePlane':
        """From an MRD image: an `ismrmrd.Image`, or the header record and meta dict.

        frame='exam' (default) takes the centre from the SlicePosLightMarker
        attribute, the exam frame the plan is in, and raises if it is absent
        rather than silently using the header position, which is in another
        frame. frame='header' takes `head.position` as is.

        In-plane axes come from the ImageRowDir / ImageColumnDir attributes when
        present (the layout the scanner declared), else from the header's
        read_dir / phase_dir, which is the acquisition frame and may be rotated
        relative to the stored matrix (see aidmr_utils.orientation).
        """
        if image is not None:
            head = image.getHead() if hasattr(image, 'getHead') else head
            if meta is None:
                meta = getattr(image, 'meta', None) or {}
        if head is None:
            raise ValueError("from_mrd needs an image or a header")
        meta = meta or {}

        if frame == 'exam':
            if LIGHT_MARKER_KEY not in meta:
                raise KeyError(f"{LIGHT_MARKER_KEY} missing from the image meta: the exam-frame "
                               f"centre is not known (pass frame='header' for the ICE-frame position)")
            centre = _vec(meta[LIGHT_MARKER_KEY])
        elif frame == 'header':
            centre = _vec(_get(head, 'position'))
        else:
            raise ValueError(f"frame must be 'exam' or 'header', got {frame!r}")

        if 'ImageRowDir' in meta and 'ImageColumnDir' in meta:
            right, down = unit(_vec(meta['ImageRowDir'])), unit(_vec(meta['ImageColumnDir']))
            axes = 'right/down'
        else:
            right, down = unit(_vec(_get(head, 'read_dir'))), unit(_vec(_get(head, 'phase_dir')))
            axes = 'read/phase (header fallback)'

        matrix = _vec(_get(head, 'matrix_size'))
        fov = _vec(_get(head, 'field_of_view'))            # (x, y, z) mm against matrix (x, y, z)
        extent_u, extent_v = float(fov[0]), float(fov[1])    # x = columns = along a row = u
        thickness = float(fov[2]) if len(fov) > 2 and fov[2] else None
        desc = meta.get('SequenceDescription', '')
        return cls(centre_xyz=centre, u_xyz=right, v_xyz=down,
                   extent_u_mm=extent_u, extent_v_mm=extent_v, thickness_mm=thickness,
                   name=name or str(desc), source='mrd', axes=axes,
                   extra={'pixel_mm': float((fov[0] / matrix[0] + fov[1] / matrix[1]) / 2.0),
                          'header_position_xyz': _vec(_get(head, 'position')),
                          'frame': frame})

    @classmethod
    def from_exam_memory(cls, slice_entry: Sequence, fov_entry: Optional[Sequence] = None,
                         name: str = '') -> 'ImagePlane':
        """From a `_Slice` entry and, for the in-plane axes, its `_FOV` entry.

        Without the FOV entry only the centre, normal and thickness are known;
        the axes are then built from the normal and the in-plane rotation is
        NOT applied, so compare such planes by centre and normal only.
        """
        p = parse_slice_entry(slice_entry)
        if fov_entry is not None:
            f = parse_fov_entry(fov_entry)
            u, v = unit(f['freq_vec']), unit(f['phase_vec'])
            # Make v exactly orthogonal to u (the scanner's vectors are, to float precision)
            v = unit(v - np.dot(v, u) * u)
            plane = cls(centre_xyz=p['centre_xyz'], u_xyz=u, v_xyz=v,
                        extent_u_mm=f['readout_mm'], extent_v_mm=f['phase_mm'],
                        thickness_mm=p['thickness_mm'], name=name or p['name'],
                        source='plan', axes='readout/phase', extra={**p, **f})
            # The FOV entry's freq x phase may point the other way from the Slice
            # entry's normal; both describe the same plane. Record the Slice normal.
            plane.extra['slice_normal_xyz'] = p['normal_xyz']
            return plane
        n = unit(p['normal_xyz'])
        # any orthonormal pair in the plane; rotation unknown
        helper = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        u = unit(np.cross(n, helper))
        v = unit(np.cross(n, u))
        return cls(centre_xyz=p['centre_xyz'], u_xyz=u, v_xyz=v,
                   extent_u_mm=p['readout_fov_mm'], extent_v_mm=p['readout_fov_mm'] * p['phase_fov_prop'],
                   thickness_mm=p['thickness_mm'], name=name or p['name'], source='plan',
                   axes='unknown (no FOV entry)', extra={**p, 'slice_normal_xyz': p['normal_xyz']})

    # -- comparisons --------------------------------------------------------

    def centre_offset_mm(self, other: 'ImagePlane') -> np.ndarray:
        """other.centre - self.centre, in mm."""
        return other.centre_xyz - self.centre_xyz

    def centre_distance_mm(self, other: 'ImagePlane') -> float:
        return float(np.linalg.norm(self.centre_offset_mm(other)))

    def centre_offset_split_mm(self, other: 'ImagePlane') -> tuple[float, float]:
        """(along this plane's normal, within this plane) components of the centre offset."""
        d = self.centre_offset_mm(other)
        along = float(np.dot(d, self.normal_xyz))
        within = float(np.sqrt(max(0.0, float(np.dot(d, d)) - along * along)))
        return along, within

    def normal_angle_deg(self, other: 'ImagePlane') -> float:
        """Angle between the two slice normals, sign-insensitive."""
        return angle_between_lines_deg(self.normal_xyz, other.normal_xyz)

    def match_axes(self, other: 'ImagePlane') -> dict:
        """Pair each of this plane's axes with the other's axis it is most parallel to.

        Returns {'u': (other_axis_name, angle_deg, other_extent_mm),
                 'v': (...)} where angle is sign-insensitive. If both of this
        plane's axes prefer the same axis of the other, the planes are rotated
        near 45 degrees and the pairing is reported with a flag.
        """
        out = {}
        for mine_name, mine in (('u', self.u_xyz), ('v', self.v_xyz)):
            best = min((('u', other.u_xyz, other.extent_u_mm), ('v', other.v_xyz, other.extent_v_mm)),
                       key=lambda t: angle_between_lines_deg(mine, t[1]))
            out[mine_name] = (best[0], angle_between_lines_deg(mine, best[1]), best[2])
        out['ambiguous'] = out['u'][0] == out['v'][0]
        return out

    def in_plane_rotation_deg(self, other: 'ImagePlane') -> float:
        """Rotation between the two in-plane axis frames, sign-insensitive, 0..90.

        The projection of the other's u onto this plane, against this plane's
        u and v: the smaller angle to either axis. Independent of which of the
        other's axes is called u.
        """
        n = self.normal_xyz
        proj = other.u_xyz - np.dot(other.u_xyz, n) * n
        if np.linalg.norm(proj) < 1e-9:
            return float('nan')
        proj = unit(proj)
        return min(angle_between_lines_deg(proj, self.u_xyz), angle_between_lines_deg(proj, self.v_xyz))

    def extents_along(self, other: 'ImagePlane') -> dict:
        """This plane's extents measured along the OTHER plane's axes.

        {'u': (self extent along other.u, other.extent_u_mm), 'v': (...)}, so a
        plan (readout/phase) can be checked against a DICOM (right/down)
        without caring which stored axis the readout ran along.
        """
        out = {}
        for oname, oaxis, oext in (('u', other.u_xyz, other.extent_u_mm), ('v', other.v_xyz, other.extent_v_mm)):
            mine = min((('u', self.u_xyz, self.extent_u_mm), ('v', self.v_xyz, self.extent_v_mm)),
                       key=lambda t: angle_between_lines_deg(oaxis, t[1]))
            out[oname] = (mine[2], oext, mine[0])
        return out

    def describe(self) -> str:
        c = ', '.join(f'{x:.1f}' for x in self.centre_xyz)
        n = ', '.join(f'{x:.3f}' for x in self.normal_xyz)
        t = f' thick={self.thickness_mm:g}' if self.thickness_mm else ''
        return (f"{self.name or self.source}: centre=[{c}] normal=[{n}] "
                f"extents u={self.extent_u_mm:.1f} v={self.extent_v_mm:.1f} ({self.axes}){t}")


def angle_between_lines_deg(a, b) -> float:
    """Angle between two directions ignoring sign, in degrees, 0..90."""
    c = abs(float(np.dot(unit(a), unit(b))))
    return math.degrees(math.acos(min(1.0, max(-1.0, c))))


# ---------------------------------------------------------------------------
# Exam memory
# ---------------------------------------------------------------------------

def parse_slice_entry(entry: Sequence) -> dict:
    """Decode a `<prefix>_<name>_Slice` list. Values arrive as strings or numbers."""
    if len(entry) != 14:
        raise ValueError(f"a _Slice entry has 14 values, got {len(entry)}: {entry!r}")
    (tag, name, cx, cy, cz, nx, ny, nz, thick, df, pfov, rfov, rot, n) = entry
    return {
        'tag': str(tag), 'name': str(name),
        'centre_xyz': _vec((cx, cy, cz)),
        'normal_xyz': unit(_vec((nx, ny, nz))),
        'thickness_mm': float(thick),
        'distance_factor': float(df),
        'phase_fov_prop': float(pfov),
        'readout_fov_mm': float(rfov),
        'in_plane_rot_rad': float(rot),
        'n_slices': int(float(n)),
    }


def parse_fov_entry(entry: Sequence) -> dict:
    """Decode a `<prefix>_<name>_FOV` list."""
    if len(entry) != 17:
        raise ValueError(f"a _FOV entry has 17 values, got {len(entry)}: {entry!r}")
    return {
        'centre_xyz': _vec(entry[2:5]),
        'readout_mm': float(entry[5]),
        'freq_vec': _vec(entry[6:9]),
        'phase_mm': float(entry[9]),
        'phase_vec': _vec(entry[10:13]),
        'fov_name': str(entry[16]),
    }


def exam_memory_entries(meta: Mapping) -> dict:
    """All planned views in a MetaAttributes mapping: {name: (slice_entry, fov_entry or None)}."""
    out = {}
    for key, value in meta.items():
        if not key.startswith(EXAM_MEMORY_PREFIX) or not key.endswith('_Slice'):
            continue
        name = key[len(EXAM_MEMORY_PREFIX):-len('_Slice')]
        fov = meta.get(f"{EXAM_MEMORY_PREFIX}{name}_FOV")
        out[name] = (list(value), list(fov) if fov is not None else None)
    return out


def planned_views(meta: Mapping) -> dict:
    """{name: ImagePlane} for every planned view in a MetaAttributes mapping."""
    return {name: ImagePlane.from_exam_memory(s, f, name=name)
            for name, (s, f) in exam_memory_entries(meta).items()}


# ---------------------------------------------------------------------------
# Stacks
# ---------------------------------------------------------------------------

@dataclass
class SliceStack:
    """Parallel slices: what a SAX stack or a 3-slice valve stack is."""
    slices: list                      # ImagePlane, sorted along the normal
    normal_xyz: np.ndarray
    centre_xyz: np.ndarray            # mean of the slice centres
    spacing_mm: Optional[float]       # centre-to-centre along the normal (None for one slice)
    name: str = ''
    source: str = ''

    @property
    def n_slices(self) -> int:
        return len(self.slices)

    @property
    def thickness_mm(self) -> Optional[float]:
        return self.slices[0].thickness_mm if self.slices else None

    @property
    def extent_along_normal_mm(self) -> float:
        """First to last slice centre."""
        if len(self.slices) < 2:
            return 0.0
        n = self.normal_xyz
        s = [float(np.dot(p.centre_xyz, n)) for p in self.slices]
        return max(s) - min(s)

    @classmethod
    def from_planes(cls, planes: Iterable[ImagePlane], name: str = '', source: str = '') -> 'SliceStack':
        planes = list(planes)
        if not planes:
            raise ValueError("no slices")
        n = planes[0].normal_xyz
        for p in planes[1:]:
            if angle_between_lines_deg(n, p.normal_xyz) > 0.5:
                raise ValueError(f"slices are not parallel ({angle_between_lines_deg(n, p.normal_xyz):.2f} deg apart)")
        ordered = sorted(planes, key=lambda p: float(np.dot(p.centre_xyz, n)))
        s = [float(np.dot(p.centre_xyz, n)) for p in ordered]
        steps = np.diff(s)
        spacing = float(np.median(np.abs(steps))) if len(steps) else None
        centre = np.mean([p.centre_xyz for p in ordered], axis=0)
        return cls(slices=ordered, normal_xyz=n, centre_xyz=centre, spacing_mm=spacing,
                   name=name or ordered[0].name, source=source or ordered[0].source)

    @classmethod
    def from_exam_memory(cls, slice_entry: Sequence, fov_entry: Optional[Sequence] = None,
                         name: str = '') -> 'SliceStack':
        """The slices a `_Slice` instruction with n_slices describes: centred on the
        entry's position, `thickness * (1 + distance_factor)` apart along its normal."""
        p = parse_slice_entry(slice_entry)
        template = ImagePlane.from_exam_memory(slice_entry, fov_entry, name=name)
        n = p['n_slices']
        spacing = p['thickness_mm'] * (1.0 + p['distance_factor'])
        normal = p['normal_xyz']
        planes = []
        for i in range(n):
            offset = (i - (n - 1) / 2.0) * spacing
            planes.append(ImagePlane(centre_xyz=p['centre_xyz'] + offset * normal,
                                     u_xyz=template.u_xyz, v_xyz=template.v_xyz,
                                     extent_u_mm=template.extent_u_mm, extent_v_mm=template.extent_v_mm,
                                     thickness_mm=p['thickness_mm'], name=f"{template.name}[{i}]",
                                     source='plan', axes=template.axes))
        return cls(slices=planes, normal_xyz=unit(normal), centre_xyz=p['centre_xyz'],
                   spacing_mm=spacing if n > 1 else None, name=name or p['name'], source='plan')

    def describe(self) -> str:
        c = ', '.join(f'{x:.1f}' for x in self.centre_xyz)
        n = ', '.join(f'{x:.3f}' for x in self.normal_xyz)
        sp = f' spacing={self.spacing_mm:.2f}' if self.spacing_mm else ''
        return f"{self.name or self.source}: {self.n_slices} slices centre=[{c}] normal=[{n}]{sp}"


def group_dicoms_into_planes(datasets: Iterable, name: str = '') -> list:
    """One ImagePlane per distinct slice position in a series of DICOMs (frames of a
    cine share a position and collapse to one plane). Sorted along the normal."""
    seen = {}
    for ds in datasets:
        plane = ImagePlane.from_dicom(ds, name=name)
        key = tuple(np.round(plane.centre_xyz, 2))
        if key not in seen:
            plane.extra['frames'] = 0
            seen[key] = plane
        seen[key].extra['frames'] += 1
    planes = list(seen.values())
    if not planes:
        return []
    n = planes[0].normal_xyz
    return sorted(planes, key=lambda p: float(np.dot(p.centre_xyz, n)))
