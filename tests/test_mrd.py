"""Building and reading MRD images.

The headline regression here is the outgoing layout declaration: BPF fixed it on
2026-08-29, AMP never got it, and CMRQ has it commented out writing the wrong key.
"""

import ctypes

import numpy as np
import pytest

ismrmrd = pytest.importorskip('ismrmrd', reason="needs the [mrd] extra")

from aidmr_utils.geometry import Plane, canonical_right_down
from aidmr_utils.mrd import (INT16_MAX, REPORT_COL_DIR, REPORT_ROW_DIR,
                             parse_h5_to_images,
                             direction_meta, ecg_from_waveforms, image_geometry,
                             meta_string_list_to_array, native_pixel_spacing_mm,
                             np_float_to_mrd, padded_square_geometry,
                             report_image_to_mrd, slice_normal,
                             sort_mrd_images_by_phase, standard_image_meta,
                             validate_existing_mrd_image_dims,
                             validate_row_col_parity)
from aidmr_utils.orientation import choose_transform


def a_head(rows=64, cols=64, fov=(240.0, 240.0, 8.0),
           read_dir=(1.0, 0.0, 0.0), phase_dir=(0.0, 1.0, 0.0),
           position=(0.0, 0.0, 0.0), table=(0.0, 0.0, 0.0), phase=0):
    """An ImageHeader standing in for one the scanner sent."""
    head = ismrmrd.ImageHeader()
    head.matrix_size = tuple(ctypes.c_ushort(v) for v in (cols, rows, 1))
    head.field_of_view = tuple(ctypes.c_float(v) for v in fov)
    head.read_dir = tuple(ctypes.c_float(v) for v in read_dir)
    head.phase_dir = tuple(ctypes.c_float(v) for v in phase_dir)
    head.position = tuple(ctypes.c_float(v) for v in position)
    head.patient_table_position = tuple(ctypes.c_float(v) for v in table)
    head.phase = phase
    return head


def an_image(**kw):
    head = a_head(**kw)
    rows = int(head.matrix_size[1])
    cols = int(head.matrix_size[0])
    img = ismrmrd.Image.from_array(np.zeros((rows, cols), np.int16), transpose=False)
    head.data_type = img.data_type
    img.setHead(head)
    return img


def build(img, **kw):
    """np_float_to_mrd with sensible defaults for the boilerplate arguments."""
    defaults = dict(
        position_xyz=(0.0, 0.0, 0.0),
        fov_freq_phase_slice=(240.0, 240.0, 8.0),
        phase_encoding_dir_xyz=(0.0, 1.0, 0.0),
        freq_encoding_dir_xyz=(1.0, 0.0, 0.0),
        template_head=None,
        image_index=1,
        series_index=1,
        attribute_string='',
    )
    defaults.update(kw)
    return np_float_to_mrd(img, **defaults)


# ---------------------------------------------------------------------------
# THE REGRESSION: outgoing images must declare their layout
# ---------------------------------------------------------------------------

def test_meta_declares_row_and_column_dirs():
    """Without these the scanner lays the matrix out by read_dir/phase_dir - a
    different pair of vectors - and rotates the image. AMP omits them entirely."""
    meta = standard_image_meta('cine', row_dir=(1, 0, 0), col_dir=(0, 1, 0))
    assert meta['ImageRowDir'] == ['1.000000000000000000', '0.000000000000000000',
                                   '0.000000000000000000']
    assert meta['ImageColumnDir'] == ['0.000000000000000000', '1.000000000000000000',
                                      '0.000000000000000000']


def test_the_key_is_ImageColumnDir_not_ImageColDir():
    """CMRQ's commented-out fix writes `ImageColDir`, which nothing reads."""
    meta = standard_image_meta('cine', row_dir=(1, 0, 0), col_dir=(0, 1, 0))
    assert 'ImageColumnDir' in meta
    assert meta.get('ImageColDir') is None


def test_an_existing_pair_on_the_incoming_meta_wins():
    """Matches the `if tmpMeta.get('ImageRowDir') is None` guard upstream."""
    base = ismrmrd.Meta()
    base['ImageRowDir'] = ['0.0', '1.0', '0.0']
    meta = standard_image_meta('cine', base_meta=base, row_dir=(1, 0, 0),
                               col_dir=(0, 1, 0))
    assert meta['ImageRowDir'] == ['0.0', '1.0', '0.0']


def test_direction_meta_normalises():
    assert meta_string_list_to_array(direction_meta((0, 5, 0)))[1] == pytest.approx(1.0)


def test_direction_meta_rejects_a_zero_vector():
    with pytest.raises(AssertionError, match='zero-length'):
        direction_meta((0, 0, 0))


# ---------------------------------------------------------------------------
# Report images
# ---------------------------------------------------------------------------

def test_report_dirs_are_a_fixed_point_of_the_reorientation():
    """The reason a report page comes back the way it was drawn.

    REPORT_ROW_DIR/COL_DIR are exactly canonical axial, so reorienting into
    DICOM norm is the identity rather than something being suppressed.
    """
    right, down = canonical_right_down(Plane.AXIAL)
    assert np.allclose(REPORT_ROW_DIR, right)
    assert np.allclose(REPORT_COL_DIR, down)
    name, score, plane = choose_transform(np.array(REPORT_ROW_DIR),
                                          np.array(REPORT_COL_DIR))
    assert (name, plane) == ('identity', Plane.AXIAL)
    assert score == pytest.approx(2.0)


def test_report_image_declares_the_identity_pair():
    page = np.linspace(0, 1, 64 * 64).reshape(64, 64)
    out = report_image_to_mrd(page, None, image_index=1, series_index=9,
                              series_desc='measurements')
    meta = ismrmrd.Meta.deserialize(out.attribute_string)
    assert meta_string_list_to_array(meta['ImageRowDir']).tolist() == list(REPORT_ROW_DIR)
    assert meta_string_list_to_array(meta['ImageColumnDir']).tolist() == list(REPORT_COL_DIR)


def test_report_image_has_square_one_mm_pixels():
    page = np.zeros((48, 32))
    out = report_image_to_mrd(page, None, 1, 1, 'measurements')
    head = out.getHead()
    assert float(head.field_of_view[0]) == float(head.matrix_size[0])
    assert float(head.field_of_view[1]) == float(head.matrix_size[1])


def test_report_image_inherits_only_table_position_and_timestamp():
    source = an_image(table=(0.0, 12.0, -3.0))
    source.acquisition_time_stamp = 4242
    out = report_image_to_mrd(np.zeros((16, 16)), source, 1, 1, 'measurements')
    assert tuple(out.getHead().patient_table_position) == pytest.approx((0.0, 12.0, -3.0))
    assert out.getHead().acquisition_time_stamp == 4242
    # but NOT the acquired direction cosines
    assert tuple(out.getHead().read_dir) == pytest.approx(REPORT_ROW_DIR)


def test_report_values_are_carried_in_the_meta():
    out = report_image_to_mrd(np.zeros((16, 16)), None, 1, 1, 'measurements',
                              report_values={'LVEF': '55', 'LVEDV': '142'})
    meta = ismrmrd.Meta.deserialize(out.attribute_string)
    assert meta['LVEF'] == '55' and meta['LVEDV'] == '142'


def test_report_image_type_is_not_the_invalid_default():
    out = report_image_to_mrd(np.zeros((16, 16)), None, 1, 1, 'r')
    assert out.getHead().image_type == ismrmrd.IMTYPE_MAGNITUDE


# ---------------------------------------------------------------------------
# np_float_to_mrd
# ---------------------------------------------------------------------------

def test_direction_vectors_are_unit_normalised():
    """CMRQ's copy stamped whatever the caller passed straight onto the header."""
    out = build(np.zeros((64, 64)), freq_encoding_dir_xyz=(5.0, 0.0, 0.0),
                phase_encoding_dir_xyz=(0.0, 3.0, 0.0))
    assert np.linalg.norm(out.getHead().read_dir) == pytest.approx(1.0)
    assert np.linalg.norm(out.getHead().phase_dir) == pytest.approx(1.0)


def test_greyscale_scales_to_int16():
    out = build(np.array([[0.0, 1.0]] * 2))
    assert out.data.dtype == np.int16
    assert out.data.max() == INT16_MAX


def test_rgb_is_uint16_holding_0_255():
    rgb = np.zeros((32, 32, 3))
    rgb[..., 0] = 1.0
    out = build(rgb, use_rgb=True)
    assert out.data.dtype == np.uint16
    assert out.data.max() == 255, "RGB must be 0-255 in a uint16, not 0-65535"
    assert out.getHead().image_type == 6
    assert out.getHead().channels == 3


def test_greyscale_array_can_be_sent_as_rgb():
    out = build(np.full((16, 16), 0.5), use_rgb=True)
    assert out.getHead().channels == 3
    assert out.data.shape[0] == 3


def test_colour_needs_use_rgb():
    with pytest.raises(AssertionError, match='use_rgb=True'):
        build(np.zeros((8, 8, 3)))


def test_out_of_range_input_is_rejected():
    with pytest.raises(AssertionError, match=r'\[0, 1\]'):
        build(np.array([[0.0, 2.0]] * 2))


def test_table_position_is_added_when_asked():
    head = a_head(table=(1.0, 2.0, 3.0))
    out = build(np.zeros((64, 64)), template_head=head, position_xyz=(10.0, 0.0, 0.0))
    assert tuple(out.getHead().position) == pytest.approx((11.0, 2.0, 3.0))


def test_table_position_is_skipped_when_told():
    head = a_head(table=(1.0, 2.0, 3.0))
    out = build(np.zeros((64, 64)), template_head=head, position_xyz=(10.0, 0.0, 0.0),
                use_table_position=False)
    assert tuple(out.getHead().position) == pytest.approx((10.0, 0.0, 0.0))


def test_matrix_size_is_cols_rows_not_rows_cols():
    out = build(np.zeros((32, 32)))
    head = out.getHead()
    assert (int(head.matrix_size[0]), int(head.matrix_size[1])) == (32, 32)
    validate_existing_mrd_image_dims(out)


def test_anisotropic_output_is_refused():
    """Siemens cannot display non-square in-plane pixels; it refuses the recon."""
    with pytest.raises(AssertionError, match='non-square pixels'):
        build(np.zeros((64, 64)), fov_freq_phase_slice=(240.0, 180.0, 8.0))


def test_padded_square_geometry_satisfies_the_square_pixel_rule():
    """The two halves of the fix have to agree, or every output would assert."""
    src = an_image(rows=180, cols=224, fov=(340.0, 273.2, 8.0))
    geom = padded_square_geometry(src)
    out = build(np.zeros((192, 192)), **geom)
    assert out is not None


# ---------------------------------------------------------------------------
# Row/column parity
# ---------------------------------------------------------------------------

def test_mismatched_parity_is_refused_by_default():
    with pytest.raises(ValueError, match='parity'):
        build(np.zeros((64, 65)), fov_freq_phase_slice=(240.0, 236.31, 8.0))


def test_mismatched_parity_can_be_coerced():
    out = build(np.zeros((64, 65)), fov_freq_phase_slice=(243.75, 240.0, 8.0),
                coerce_odd_even_dims=True)
    head = out.getHead()
    assert (int(head.matrix_size[1]), int(head.matrix_size[0])) == (64, 66)


def test_validate_row_col_parity():
    validate_row_col_parity(rows=4, cols=6)
    validate_row_col_parity(rows=5, cols=7)
    with pytest.raises(ValueError):
        validate_row_col_parity(rows=4, cols=5)


# ---------------------------------------------------------------------------
# Incoming geometry
# ---------------------------------------------------------------------------

def test_native_pixel_spacing_pairs_fov_with_the_right_matrix_axis():
    src = an_image(rows=64, cols=64, fov=(256.0, 256.0, 8.0))
    assert native_pixel_spacing_mm(src) == pytest.approx((4.0, 4.0))


def test_native_pixel_spacing_catches_a_mis_paired_axis():
    """A non-square acquisition makes a wrong pairing obvious immediately."""
    src = an_image(rows=208, cols=256, fov=(340.0, 340.0, 8.0))
    with pytest.raises(AssertionError, match='non-square pixels'):
        native_pixel_spacing_mm(src)


def test_image_geometry_is_verbatim():
    src = an_image(fov=(300.0, 200.0, 6.0), position=(1.0, 2.0, 3.0))
    geom = image_geometry(src)
    assert geom['fov_freq_phase_slice'] == pytest.approx((300.0, 200.0, 6.0))
    assert geom['position_xyz'] == pytest.approx((1.0, 2.0, 3.0))


def test_padded_square_geometry_makes_the_fov_square():
    src = an_image(rows=180, cols=224, fov=(340.0, 273.2, 8.0))
    geom = padded_square_geometry(src)
    fov = geom['fov_freq_phase_slice']
    assert fov[0] == fov[1] == pytest.approx(340.0)
    assert geom['use_table_position'] is False


def test_padded_square_geometry_does_not_shift_an_even_shortfall():
    src = an_image(rows=180, cols=224, fov=(340.0, 273.2, 8.0))
    assert padded_square_geometry(src)['position_xyz'] == pytest.approx((0.0, 0.0, 0.0))


def test_padded_square_geometry_shifts_an_odd_shortfall_by_half_a_pixel():
    src = an_image(rows=63, cols=64, fov=(256.0, 252.0, 8.0))
    shifted = padded_square_geometry(src)['position_xyz']
    assert not np.allclose(shifted, (0.0, 0.0, 0.0))


def test_slice_normal_is_a_unit_vector():
    src = an_image(read_dir=(1.0, 0.0, 0.0), phase_dir=(0.0, 1.0, 0.0))
    assert np.allclose(slice_normal(src), (0.0, 0.0, 1.0))


def test_sort_mrd_images_by_phase():
    imgs = [an_image(phase=p) for p in (3, 0, 2, 1)]
    assert [i.getHead().phase for i in sort_mrd_images_by_phase(imgs)] == [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# ECG
# ---------------------------------------------------------------------------

class FakeWaveform:
    def __init__(self, channels, data):
        self.channels = channels
        self.data = np.asarray(data)


def test_ecg_from_waveforms_reshapes_to_five_channels():
    wfs = [FakeWaveform(5, np.arange(5 * 10).reshape(5, 10)) for _ in range(3)]
    ecg = ecg_from_waveforms(wfs)
    assert ecg.shape == (5, 30)


def test_ecg_is_none_when_absent():
    """An anonymised or retrospectively saved study often has no waveforms."""
    assert ecg_from_waveforms([]) is None
    assert ecg_from_waveforms([FakeWaveform(1, np.zeros((1, 4)))]) is None


# ---------------------------------------------------------------------------
# Meta defaults
# ---------------------------------------------------------------------------

def test_program_appears_in_history_and_description():
    meta = standard_image_meta('cine', program='bpf')
    assert meta['ImageProcessingHistory'] == ['FIRE', 'PYTHON', 'BPF']
    assert meta['SequenceDescriptionAdditional'] == 'AID-MR BPF'


def test_no_program_keeps_the_plain_labels():
    meta = standard_image_meta('cine')
    assert meta['ImageProcessingHistory'] == ['FIRE', 'PYTHON']
    assert meta['SequenceDescriptionAdditional'] == 'AID-MR'


def test_window_is_not_overwritten_when_a_base_meta_is_given():
    """CMRQ set WindowCenter unconditionally, clobbering the incoming choice."""
    base = ismrmrd.Meta()
    base['WindowCenter'] = '100'
    assert standard_image_meta('cine', base_meta=base)['WindowCenter'] == '100'


def test_window_defaults_cover_the_int16_range_written():
    meta = standard_image_meta('cine')
    assert float(meta['WindowWidth']) == 2 ** 15


# ---------------------------------------------------------------------------
# .h5 capture reading
# ---------------------------------------------------------------------------

def _minimal_header_xml():
    """The smallest ismrmrdHeader the schema will accept."""
    head = ismrmrd.xsd.ismrmrdHeader(
        experimentalConditions=ismrmrd.xsd.experimentalConditionsType(H1resonanceFrequency_Hz=63600000))
    return ismrmrd.xsd.ToXML(head)


def _write_capture(path, groups: dict):
    """An ISMRMRD capture with the given {group_name: n_images}."""
    ds = ismrmrd.Dataset(str(path), '/dataset', True)
    ds.write_xml_header(_minimal_header_xml())
    for name, n in groups.items():
        for i in range(n):
            img = ismrmrd.Image.from_array(
                np.full((4, 4), i, dtype=np.int16), transpose=False)
            ds.append_image(name, img)
    ds.close()


def test_parse_h5_reads_every_image_group(tmp_path):
    """A capture can hold several image groups - one per series sent.

    AMP's copy tried 'images_0' then 'image_0' and returned the first it found,
    silently dropping every later group. AIFS's took all of them, and AIFS's
    test asserts a total frame count, so this is load-bearing.
    """
    p = tmp_path / 'multi.h5'
    _write_capture(p, {'image_0': 3, 'image_1': 4, 'image_2': 2})
    assert len(parse_h5_to_images(str(p))) == 9


def test_parse_h5_handles_the_newer_group_name(tmp_path):
    p = tmp_path / 'new.h5'
    _write_capture(p, {'images_0': 5})
    assert len(parse_h5_to_images(str(p))) == 5


def test_parse_h5_group_order_is_deterministic(tmp_path):
    p = tmp_path / 'order.h5'
    _write_capture(p, {'image_2': 1, 'image_0': 1, 'image_1': 1})
    first = [im.data.sum() for im in parse_h5_to_images(str(p))]
    second = [im.data.sum() for im in parse_h5_to_images(str(p))]
    assert first == second


def test_parse_h5_says_so_when_there_are_no_images(tmp_path):
    p = tmp_path / 'empty.h5'
    ds = ismrmrd.Dataset(str(p), '/dataset', True)
    ds.write_xml_header(_minimal_header_xml())
    ds.close()
    with pytest.raises(LookupError, match='no image group'):
        parse_h5_to_images(str(p))


def test_square_pixel_check_can_be_switched_off():
    """Not every program echoes acquired geometry onto a padded square.

    BPF does, and there anisotropic pixels mean it got the geometry wrong. AMP
    PRESCRIBES slices with a deliberately rectangular field of view. Making the
    check unconditional imposed BPF's invariant on AMP and broke 36 of its tests.
    """
    img = np.zeros((64, 64))
    with pytest.raises(AssertionError, match='non-square pixels'):
        build(img, fov_freq_phase_slice=(240.0, 180.0, 8.0))
    # ...and the same call is fine when the caller says so
    out = build(img, fov_freq_phase_slice=(240.0, 180.0, 8.0),
                require_square_pixels=False)
    assert out is not None


def test_the_check_still_defaults_to_on():
    """BPF relies on the default: the assert is why its geometry bug stayed fixed."""
    with pytest.raises(AssertionError, match='non-square pixels'):
        build(np.zeros((64, 64)), fov_freq_phase_slice=(240.0, 100.0, 8.0))
