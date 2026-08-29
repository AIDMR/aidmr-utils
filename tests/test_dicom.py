"""The DICOM read path, and the regressions that motivated this package."""

import numpy as np
import pytest

from aidmr_utils import dicom as dcm_mod
from aidmr_utils.dicom import (MixedFrameSizesError, load_pixels, read_image,
                               read_series, sorted_by_instance_number,
                               window_spec_for)
from aidmr_utils.windowing import DicomWindow, PercentileWindow


# ---------------------------------------------------------------------------
# load_pixels: rescale is not optional
# ---------------------------------------------------------------------------

def test_load_pixels_applies_slope_and_intercept(make_dicom):
    raw = np.array([[0, 100], [200, 300]], dtype=np.uint16)
    d = make_dicom(raw, slope=2.5, intercept=-100)
    assert np.allclose(load_pixels(d), raw * 2.5 - 100)


def test_load_pixels_defaults_when_tags_absent(make_dicom):
    raw = np.array([[1, 2], [3, 4]], dtype=np.uint16)
    assert np.allclose(load_pixels(make_dicom(raw)), raw)


def test_load_pixels_returns_float_so_negatives_survive(make_dicom):
    """An intercept can take values below zero; an unsigned array would wrap."""
    out = load_pixels(make_dicom(np.array([[0, 10]], dtype=np.uint16),
                                 slope=1, intercept=-50))
    assert out.dtype == np.float64
    assert out.min() == -50


def test_load_pixels_rescales_in_float64(make_dicom):
    """float32 here shifts ~1% of pixels by one uint8 level. See the docstring."""
    raw = np.arange(4096, dtype=np.uint16).reshape(64, 64)
    slope = 0.0030518
    out = load_pixels(make_dicom(raw, slope=slope, intercept=0))
    assert out.dtype == np.float64
    assert np.array_equal(out, raw.astype(np.float64) * slope)
    # the float32 route really would have differed
    f32 = raw.astype(np.float32) * np.float32(slope)
    assert not np.array_equal(f32.astype(np.float64), out)


# ---------------------------------------------------------------------------
# THE PHILIPS REGRESSION
# ---------------------------------------------------------------------------

def test_dicom_window_on_philips_data_is_not_blank(philips_like):
    """The bug this package exists for.

    Applying WC/WW to un-rescaled Philips values clips everything to one value —
    a uniform frame. `read_image` must rescale first and produce a real image.
    """
    d, raw, slope, intercept = philips_like
    out = read_image(d, 'dicom')

    assert len(np.unique(out)) > 20, (
        "windowed Philips image collapsed to near-uniform - RescaleSlope/"
        "RescaleIntercept were probably not applied before windowing")
    assert out.min() < 40 and out.max() > 215, "window is not spanning the data"


def test_the_old_buggy_path_really_would_have_been_blank(philips_like):
    """Guards the test above: prove the failure mode is real, not hypothetical."""
    d, raw, slope, intercept = philips_like
    wc, ww = float(d.WindowCenter), float(d.WindowWidth)
    lo, hi = wc - ww / 2, wc + ww / 2
    # what the pre-package code did: window the RAW array against a rescaled window
    assert ((raw >= lo) & (raw <= hi)).mean() == 0.0
    # and what this package does instead
    rescaled = load_pixels(d)
    assert ((rescaled >= lo) & (rescaled <= hi)).mean() > 0.9


# ---------------------------------------------------------------------------
# The string-truthiness bug
# ---------------------------------------------------------------------------

def test_read_image_rejects_a_bare_bool(make_dicom):
    """`read_image_from_dicom(dcm, use_dicom_level)` is not this function.

    Two repos called the old signature with `cfg.data.windowing`, a non-empty
    string and therefore always truthy, so a config asking for '0-100' silently
    got DICOM WC/WW. Taking a spec instead makes that impossible; refusing a bool
    makes an unmigrated call site fail loudly rather than being reinterpreted.
    """
    d = make_dicom(np.ones((4, 4), dtype=np.uint16))
    with pytest.raises(TypeError, match="not a boolean"):
        read_image(d, True)
    with pytest.raises(TypeError, match="not a boolean"):
        read_image(d, False)


@pytest.mark.parametrize('spec_str', ['0_100', '0-100', '5_95', '5-95'])
def test_percentile_strings_do_not_become_dicom_windowing(make_dicom, spec_str):
    """A percentile config must take the percentile path even with no WC/WW tags."""
    d = make_dicom(np.arange(256, dtype=np.uint16).reshape(16, 16))
    assert not hasattr(d, 'WindowCenter')
    out = read_image(d, spec_str)          # would raise if it went down WC/WW
    assert out.min() == 0 and out.max() == 255


# ---------------------------------------------------------------------------
# window_spec_for
# ---------------------------------------------------------------------------

def test_window_spec_for_resolves_dicom_from_tags(make_dicom):
    d = make_dicom(np.ones((4, 4), dtype=np.uint16),
                   window_center=100, window_width=50)
    resolved = window_spec_for(d, 'dicom')
    assert isinstance(resolved, DicomWindow)
    assert resolved.bounds() == (75.0, 125.0)


def test_window_spec_for_explains_itself_when_tags_are_missing(make_dicom):
    d = make_dicom(np.ones((4, 4), dtype=np.uint16))
    with pytest.raises(ValueError, match="percentile window"):
        window_spec_for(d, 'dicom')


def test_window_spec_for_takes_first_of_a_multivalue(make_dicom):
    d = make_dicom(np.ones((4, 4), dtype=np.uint16),
                   window_center=[100, 200], window_width=[50, 400])
    assert window_spec_for(d, 'dicom').bounds() == (75.0, 125.0)


def test_window_spec_for_passes_percentile_through(make_dicom):
    d = make_dicom(np.ones((4, 4), dtype=np.uint16))
    assert window_spec_for(d, '5_95') == PercentileWindow(5, 95)


# ---------------------------------------------------------------------------
# MONOCHROME1
# ---------------------------------------------------------------------------

def test_monochrome1_is_inverted(make_dicom):
    pixels = np.arange(256, dtype=np.uint16).reshape(16, 16)
    normal = read_image(make_dicom(pixels), '0_100')
    inverted = read_image(make_dicom(pixels, photometric='MONOCHROME1'), '0_100')
    assert np.array_equal(inverted, 255 - normal)


def test_monochrome1_inversion_happens_in_display_space(make_dicom):
    """Float output inverts about 1.0, not about the stored maximum."""
    pixels = np.arange(256, dtype=np.uint16).reshape(16, 16)
    out = read_image(make_dicom(pixels, photometric='MONOCHROME1'),
                     '0_100', as_uint8=False)
    assert out.max() == pytest.approx(1.0)
    assert out.min() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Series handling
# ---------------------------------------------------------------------------

def test_series_is_windowed_together_not_per_frame(make_dicom):
    """A bright frame must not rescale the dim frames' contrast.

    Per-frame windowing stretches each frame to its own extremes, so the cine
    flickers and the deployed side (which percentiles the whole MRD stack) sees
    something different from training.
    """
    dim = make_dicom(np.full((8, 8), 100, dtype=np.uint16))
    bright = make_dicom(np.full((8, 8), 1000, dtype=np.uint16))
    frames = read_series([dim, bright], '0_100')

    assert frames[0].max() < frames[1].min(), (
        "frames were windowed independently - the dim frame was stretched to "
        "the same range as the bright one")


def test_series_matches_per_frame_for_dicom_windowing(make_dicom):
    """WC/WW are per-image tags, so a 'dicom' series is per-frame by definition."""
    a = make_dicom(np.full((4, 4), 100, np.uint16), window_center=100, window_width=200)
    b = make_dicom(np.full((4, 4), 900, np.uint16), window_center=900, window_width=200)
    series = read_series([a, b], 'dicom')
    assert np.array_equal(series[0], read_image(a, 'dicom'))
    assert np.array_equal(series[1], read_image(b, 'dicom'))


def test_mixed_frame_sizes_raises_before_decoding_pixels(make_dicom):
    a = make_dicom(np.ones((8, 8), np.uint16), series_uid='1.2.3')
    b = make_dicom(np.ones((16, 16), np.uint16), series_uid='1.2.3')
    with pytest.raises(MixedFrameSizesError) as exc:
        read_series([a, b], '0_100')
    assert exc.value.shapes == [(8, 8), (16, 16)]
    assert '1.2.3' in str(exc.value)


def test_empty_series_is_empty_not_an_error():
    assert read_series([], '5_95') == []


def test_sorted_by_instance_number(make_dicom):
    d = [make_dicom(np.ones((2, 2), np.uint16), instance_number=n) for n in (3, 1, 2)]
    assert [x.InstanceNumber for x in sorted_by_instance_number(d)] == [1, 2, 3]


def test_sorted_by_instance_number_keeps_untagged_frames(make_dicom):
    tagged = [make_dicom(np.ones((2, 2), np.uint16), instance_number=n) for n in (2, 1)]
    untagged = make_dicom(np.ones((2, 2), np.uint16))
    out = sorted_by_instance_number(tagged + [untagged])
    assert len(out) == 3
    assert getattr(out[-1], 'InstanceNumber', None) is None


# ---------------------------------------------------------------------------
# Missing extra
# ---------------------------------------------------------------------------

def test_helpful_error_when_pydicom_absent(monkeypatch):
    monkeypatch.setattr(dcm_mod, 'pydicom', None)
    with pytest.raises(ImportError, match=r"aidmr-utils\[dicom\]"):
        dcm_mod._as_dataset('/some/path.dcm')
