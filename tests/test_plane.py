"""aidmr_utils.plane: DICOM, MRD and exam-memory planes agree when they describe the same slice.

Numbers marked EXAM1 are from AIDMR-PROSPECTIVE 001 (acquired 2026-07-21, probed
2026-09-12): what AMP planned on replay, what the scanner wrote into the DICOMs it
then acquired, and what FIRE's MRD headers said. They are the evidence for the
frame rules in the module docstring, so they are pinned here.
"""
import math
from types import SimpleNamespace

import numpy as np
import pytest

from aidmr_utils.plane import (EXAM_MEMORY_PREFIX, ImagePlane, SliceStack, angle_between_lines_deg,
                               exam_memory_entries, group_dicoms_into_planes, parse_fov_entry,
                               parse_slice_entry, planned_views)

PREFIX = EXAM_MEMORY_PREFIX


def dicom(ipp, iop, spacing, rows, cols, thickness=8.0, desc='cine'):
    return SimpleNamespace(ImagePositionPatient=list(ipp), ImageOrientationPatient=list(iop),
                           PixelSpacing=list(spacing), Rows=rows, Columns=cols,
                           SliceThickness=thickness, SeriesDescription=desc,
                           InPlanePhaseEncodingDirection='ROW')


def slice_entry(name, centre, normal, thick=8.0, df=0.0, pfov=0.75, rfov=340.0, rot=0.0, n=1):
    return ['slicegroup', name, *centre, *normal, thick, df, pfov, rfov, rot, n]


def fov_entry(name, centre, freq_dir, readout_mm, phase_dir, phase_mm):
    f = np.asarray(freq_dir, float) / np.linalg.norm(freq_dir) * readout_mm
    p = np.asarray(phase_dir, float) / np.linalg.norm(phase_dir) * phase_mm
    return ['slicefov', 1, *centre, readout_mm, *f, phase_mm, *p, 0.0, 0.0, 0.0, name]


# ---------------------------------------------------------------------------
# DICOM: the corner pixel is not the centre
# ---------------------------------------------------------------------------

def test_dicom_centre_is_half_a_matrix_in_from_the_first_pixel():
    # axial, identity orientation, 1 mm pixels, 100 x 200: the first pixel centre is at
    # (-99.5, -49.5) if the image centre is the origin
    ds = dicom(ipp=(-99.5, -49.5, 10.0), iop=(1, 0, 0, 0, 1, 0), spacing=(1.0, 1.0), rows=100, cols=200)
    p = ImagePlane.from_dicom(ds)
    assert np.allclose(p.centre_xyz, (0.0, 0.0, 10.0))
    assert p.extent_u_mm == 200.0 and p.extent_v_mm == 100.0        # u = right = columns
    assert np.allclose(p.normal_xyz, (0, 0, 1))
    assert p.pixel_mm == 1.0 and p.thickness_mm == 8.0


def test_dicom_pixel_spacing_is_rows_then_columns():
    # anisotropic spacing to prove which number goes with which axis: 2 mm between rows
    # (along down), 0.5 mm between columns (along right)
    ds = dicom(ipp=(0, 0, 0), iop=(1, 0, 0, 0, 1, 0), spacing=(2.0, 0.5), rows=10, cols=40)
    p = ImagePlane.from_dicom(ds)
    assert p.extent_u_mm == 20.0      # 40 columns x 0.5
    assert p.extent_v_mm == 20.0      # 10 rows x 2.0
    assert np.allclose(p.centre_xyz, (39 * 0.5 / 2, 9 * 2.0 / 2, 0))


def test_exam1_aov_3ch_dicom_geometry():
    # EXAM1: the acquired 3ch (series ' AoV_3ch'), 224 x 168 at 1.51786 mm
    ds = dicom(ipp=(-133.45, -117.30, 48.33),
               iop=(0.2955, 0.9526, -0.0719, 0.812, -0.2901, -0.5065),
               spacing=(1.51786, 1.51786), rows=224, cols=168)
    p = ImagePlane.from_dicom(ds)
    assert np.allclose(p.centre_xyz, (41.4, -45.7, -46.5), atol=0.15)
    assert np.allclose(np.abs(p.normal_xyz), (0.503, 0.091, 0.859), atol=2e-3)
    assert p.extent_u_mm == pytest.approx(255.0, abs=0.1)     # 168 columns: the phase axis here
    assert p.extent_v_mm == pytest.approx(340.0, abs=0.1)     # 224 rows: the readout ran DOWN the image


# ---------------------------------------------------------------------------
# Exam memory
# ---------------------------------------------------------------------------

def test_slice_and_fov_entries_decode_by_position_not_by_guess():
    s = slice_entry('ThreeChamber', (42.3, -45.2, -46.9), (0.503, -0.091, 0.859),
                    thick=8, df=0, pfov=0.71875, rfov=340, rot=math.radians(20), n=1)
    d = parse_slice_entry([str(x) for x in s])          # values arrive as strings in the meta
    assert d['name'] == 'ThreeChamber' and d['n_slices'] == 1
    assert d['phase_fov_prop'] == 0.71875 and d['readout_fov_mm'] == 340.0
    assert d['in_plane_rot_rad'] == pytest.approx(math.radians(20))
    f = fov_entry('ThreeChamber', (42.3, -45.2, -46.9), (0.812, -0.290, -0.506), 340.0,
                  (-0.296, -0.953, 0.072), 244.4)
    e = parse_fov_entry([str(x) for x in f])
    assert e['readout_mm'] == 340.0 and e['phase_mm'] == 244.4
    assert np.linalg.norm(e['freq_vec']) == pytest.approx(340.0)
    assert e['fov_name'] == 'ThreeChamber'


def test_planned_views_finds_every_slice_group_and_pairs_its_fov():
    meta = {
        f'{PREFIX}TwoChamber_Target': ['string', 'x'],
        f'{PREFIX}TwoChamber_Slice': slice_entry('TwoChamber', (41.6, -42.5, -44.9), (0.757, 0.635, 0.154)),
        f'{PREFIX}TwoChamber_FOV': fov_entry('TwoChamber', (41.6, -42.5, -44.9), (-0.612, 0.772, -0.172), 340,
                                             (-0.228, 0.035, 0.973), 308.1),
        f'{PREFIX}ShimMemory_Slice': slice_entry('ShimMemory', (18.3, -46.6, -36.4), (0.12, 0.505, -0.855),
                                                 thick=93.2, df=0.2, pfov=0.65, rfov=286.9),
        'SiemensExamMemory_WIP_AIAAH_Slice_List': ['string array append', 'a', 'b'],
        'ImageComments': 'AID-MR test',
    }
    entries = exam_memory_entries(meta)
    assert set(entries) == {'TwoChamber', 'ShimMemory'}
    assert entries['ShimMemory'][1] is None
    views = planned_views(meta)
    two = views['TwoChamber']
    assert two.axes == 'readout/phase' and two.extent_u_mm == 340 and two.extent_v_mm == 308.1
    assert np.allclose(np.abs(two.normal_xyz), (0.757, 0.635, 0.154), atol=2e-3)
    assert views['ShimMemory'].axes.startswith('unknown')


# ---------------------------------------------------------------------------
# The three sources agree on the same slice
# ---------------------------------------------------------------------------

def test_plan_and_dicom_of_the_same_slice_match_regardless_of_which_axis_is_readout():
    centre = (42.3, -45.2, -46.9)
    n = np.array((0.503, -0.091, 0.859)); n /= np.linalg.norm(n)
    freq = np.cross(n, (0, 1, 0)); freq /= np.linalg.norm(freq)        # some in-plane direction
    phase = np.cross(n, freq)
    plan = ImagePlane.from_exam_memory(
        slice_entry('ThreeChamber', centre, n, pfov=0.71875, rfov=340),
        fov_entry('ThreeChamber', centre, freq, 340.0, phase, 244.4))
    # The scanner stores the readout DOWN the image: right = phase (flipped), down = freq.
    right, down = -phase, freq
    spacing = 340.0 / 224
    rows, cols = 224, int(round(244.4 / spacing))
    ipp = np.array(centre) - right * spacing * (cols - 1) / 2 - down * spacing * (rows - 1) / 2
    ds = dicom(ipp=ipp, iop=(*right, *down), spacing=(spacing, spacing), rows=rows, cols=cols)
    acquired = ImagePlane.from_dicom(ds)

    assert plan.centre_distance_mm(acquired) < 1e-6
    assert plan.normal_angle_deg(acquired) < 1e-6              # sign-insensitive
    assert plan.in_plane_rotation_deg(acquired) < 1e-6
    ext = plan.extents_along(acquired)
    assert ext['u'][0] == pytest.approx(244.4) and ext['u'][1] == pytest.approx(cols * spacing, abs=spacing)
    assert ext['v'] == pytest.approx((340.0, 340.0, 'u')) or ext['v'][:2] == pytest.approx((340.0, 340.0))
    assert ext['v'][2] == 'u'                                 # the plan's readout lies along the DICOM's down


def test_mrd_exam_frame_comes_from_the_light_marker_not_the_header_position():
    # EXAM1: the acquired 2ch cine. Header position and SlicePosLightMarker differ by 24 mm in z;
    # the light marker equals the planned TwoChamber centre (41.6, -42.5, -44.9).
    head = SimpleNamespace(position=(41.6, -42.5, -20.93), read_dir=(-0.612, 0.772, -0.172),
                           phase_dir=(-0.228, 0.035, 0.973), matrix_size=(224, 204, 1),
                           field_of_view=(340.0, 309.6, 8.0))
    meta = {'SlicePosLightMarker': ['41.6007041931152344', '-42.5', '-44.9340343475341797'],
            'ImageRowDir': ['0.2279496', '-0.0354646', '-0.9730268'],
            'ImageColumnDir': ['-0.6122927', '0.7717908', '-0.1715709'],
            'SequenceDescription': '2ch 4Ch (LAX2SAX & CMRQ)'}
    exam = ImagePlane.from_mrd(head=head, meta=meta)
    assert np.allclose(exam.centre_xyz, (41.6, -42.5, -44.93), atol=0.01)
    assert exam.axes == 'right/down' and exam.pixel_mm == pytest.approx(1.518, abs=1e-3)
    assert exam.extent_u_mm == 340.0 and exam.extent_v_mm == 309.6 and exam.thickness_mm == 8.0
    ice = ImagePlane.from_mrd(head=head, meta=meta, frame='header')
    assert np.allclose(ice.centre_xyz, (41.6, -42.5, -20.93))
    assert exam.centre_distance_mm(ice) == pytest.approx(24.0, abs=0.01)
    with pytest.raises(KeyError):
        ImagePlane.from_mrd(head=head, meta={'ImageRowDir': meta['ImageRowDir'],
                                             'ImageColumnDir': meta['ImageColumnDir']})


def test_exam1_two_chamber_plan_matches_the_acquired_mrd_cine():
    plan = ImagePlane.from_exam_memory(slice_entry('TwoChamber', (41.6007, -42.5, -44.934), (0.757, 0.635, 0.154),
                                                   pfov=0.90625, rfov=340, rot=math.radians(10)))
    head = SimpleNamespace(position=(41.6, -42.5, -20.93), read_dir=(-0.612, 0.772, -0.172),
                           phase_dir=(-0.228, 0.035, 0.973), matrix_size=(224, 204, 1),
                           field_of_view=(340.0, 309.6, 8.0))
    meta = {'SlicePosLightMarker': ['41.6007041931152344', '-42.5', '-44.9340343475341797']}
    acquired = ImagePlane.from_mrd(head=head, meta=meta)      # header read/phase fallback for axes
    assert plan.centre_distance_mm(acquired) < 0.01
    assert plan.normal_angle_deg(acquired) < 0.2


# ---------------------------------------------------------------------------
# Stacks
# ---------------------------------------------------------------------------

def test_stack_from_exam_memory_spaces_slices_by_thickness_times_one_plus_distance_factor():
    s = slice_entry('SAXStack', (53.2, -42.9, -38.2), (0.766, -0.544, -0.343), thick=8, df=0.25,
                    pfov=0.8945, rfov=340, rot=math.radians(10), n=14)
    st = SliceStack.from_exam_memory(s)
    assert st.n_slices == 14 and st.spacing_mm == pytest.approx(10.0)
    assert np.allclose(st.centre_xyz, (53.2, -42.9, -38.2))
    assert st.extent_along_normal_mm == pytest.approx(130.0)
    first, last = st.slices[0], st.slices[-1]
    assert first.centre_distance_mm(last) == pytest.approx(130.0)


def test_stack_from_dicoms_collapses_frames_and_measures_spacing():
    # EXAM1 lvsa: 14 positions x 25 phases, 10 mm apart, normal +-(0.773, -0.539, -0.335)
    right, down = np.array((0.495, 0.842, -0.214)), np.array((-0.397, 0.0, -0.918))
    down = down - np.dot(down, right) / np.dot(right, right) * right; down /= np.linalg.norm(down); right /= np.linalg.norm(right)
    n = np.cross(right, down)
    base = np.array((3.3, -0.5, -16.1))
    datasets = []
    for i in range(14):
        centre = base + i * 10.0 * (-n)          # the scanner stepped against the DICOM normal here
        ipp = centre - right * 1.8333 * (180 - 1) / 2 - down * 1.8333 * (240 - 1) / 2
        for _ in range(25):
            datasets.append(dicom(ipp=ipp, iop=(*right, *down), spacing=(1.8333, 1.8333), rows=240, cols=180,
                                  desc='cine_LVSA'))
    planes = group_dicoms_into_planes(datasets)
    assert len(planes) == 14 and all(p.extra['frames'] == 25 for p in planes)
    st = SliceStack.from_planes(planes)
    assert st.n_slices == 14 and st.spacing_mm == pytest.approx(10.0, abs=1e-6)
    assert st.extent_along_normal_mm == pytest.approx(130.0, abs=1e-6)
    assert np.allclose(st.centre_xyz, base - 65.0 * n, atol=1e-6)
    plan = SliceStack.from_exam_memory(slice_entry('SAXStack', st.centre_xyz, -n, thick=8, df=0.25, n=14))
    assert angle_between_lines_deg(plan.normal_xyz, st.normal_xyz) < 1e-6
    assert np.linalg.norm(plan.centre_xyz - st.centre_xyz) < 1e-6


def test_non_parallel_slices_are_refused():
    a = ImagePlane.from_dicom(dicom((0, 0, 0), (1, 0, 0, 0, 1, 0), (1, 1), 10, 10))
    b = ImagePlane.from_dicom(dicom((0, 0, 5), (1, 0, 0, 0, 0.94, 0.34), (1, 1), 10, 10))
    with pytest.raises(ValueError):
        SliceStack.from_planes([a, b])
