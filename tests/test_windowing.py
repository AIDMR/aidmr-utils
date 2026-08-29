"""Windowing: the format, the invariance, and the traps."""

import numpy as np
import pytest

from aidmr_utils.windowing import (DEPLOYABLE_WINDOWINGS, DicomWindow,
                                   PercentileWindow, apply_window,
                                   check_window_method, parse_window_spec,
                                   window, window_series)


# ---------------------------------------------------------------------------
# Parsing: one spec object, several config spellings
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('text,expected', [
    ('5_95', PercentileWindow(5, 95)),
    ('5-95', PercentileWindow(5, 95)),      # CMRQ / AIFS spelling
    ('0_100', PercentileWindow(0, 100)),
    ('0-100', PercentileWindow(0, 100)),
    (' 5_95 ', PercentileWindow(5, 95)),
    ('2.5_97.5', PercentileWindow(2.5, 97.5)),
])
def test_parse_percentile_forms(text, expected):
    assert parse_window_spec(text) == expected


def test_hyphen_and_underscore_normalise_to_one_string():
    """So the string stamped into model metadata does not depend on the config."""
    assert str(parse_window_spec('5-95')) == '5_95'
    assert str(parse_window_spec('0-100')) == '0_100'


def test_parse_dicom():
    assert str(parse_window_spec('dicom')) == 'dicom'


@pytest.mark.parametrize('bad', ['', 'percentile', '5_95_99', 'a_b', '5'])
def test_parse_rejects_nonsense(bad):
    with pytest.raises(ValueError):
        parse_window_spec(bad)


def test_percentile_bounds_must_be_ordered():
    with pytest.raises(ValueError):
        PercentileWindow(95, 5)
    with pytest.raises(ValueError):
        PercentileWindow(-1, 50)


def test_yaml_octal_trap():
    """Unquoted 0_100 in YAML 1.1 is octal 64, an int, not the string we want."""
    with pytest.raises(TypeError, match='octal'):
        check_window_method(0o100)
    with pytest.raises(TypeError, match='octal'):
        parse_window_spec(64)


def test_deployable_flags():
    assert parse_window_spec('5_95').deployable
    assert parse_window_spec('0_100').deployable
    assert not parse_window_spec('2_98').deployable
    assert not parse_window_spec('dicom').deployable
    assert set(DEPLOYABLE_WINDOWINGS) == {'5_95', '0_100'}


def test_dicom_placeholder_refuses_to_invent_bounds():
    """'dicom' has no bounds until an image supplies WC/WW."""
    with pytest.raises(ValueError, match='WindowCenter'):
        parse_window_spec('dicom').bounds(np.ones((4, 4)))


# ---------------------------------------------------------------------------
# The invariance that hid the Philips bug
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('slope,intercept', [
    (1, 0), (2.72869, -5587), (4.3, 0), (0.5, 1000),
])
def test_percentile_windowing_is_invariant_to_affine_rescale(slope, intercept):
    """percentile(a*x+b) == a*percentile(x)+b, and the normalisation divides it out.

    This is why forgetting RescaleSlope/Intercept was harmless in the repos that
    used a percentile window, and catastrophic in the ones that used WC/WW.
    """
    rng = np.random.default_rng(1)
    raw = rng.integers(0, 4096, size=(64, 64)).astype(np.float64)
    spec = PercentileWindow(5, 95)
    baseline = window(raw, spec, as_uint8=False)
    rescaled = window(raw * slope + intercept, spec, as_uint8=False)
    assert np.allclose(baseline, rescaled, atol=1e-6)


def test_dicom_windowing_is_NOT_invariant_to_rescale():
    """The other half of the same fact - stated as a test so it cannot be forgotten."""
    raw = np.linspace(1960, 2475, 256).reshape(16, 16)
    slope, intercept = 2.72869, -5587.0
    spec = DicomWindow(center=460, width=1400)   # quoted in rescaled units
    on_raw = window(raw, spec, as_uint8=False)
    on_rescaled = window(raw * slope + intercept, spec, as_uint8=False)
    assert not np.allclose(on_raw, on_rescaled)
    assert len(np.unique(on_raw)) == 1           # everything clipped: uniform
    assert len(np.unique(on_rescaled)) > 20


# ---------------------------------------------------------------------------
# apply_window arithmetic
# ---------------------------------------------------------------------------

def test_apply_window_clips_and_scales():
    img = np.array([[-10.0, 0.0, 50.0, 100.0, 200.0]])
    out = apply_window(img, 0, 100, as_uint8=False)
    assert np.allclose(out, [[0.0, 0.0, 0.5, 1.0, 1.0]])


def test_apply_window_on_unsigned_input_does_not_wrap():
    """Clamping a uint array against a negative bound in-place would wrap."""
    img = np.array([[0, 5, 10]], dtype=np.uint16)
    out = apply_window(img, -10, 10, as_uint8=False)
    assert out.min() >= 0.0 and np.allclose(out, [[0.5, 0.75, 1.0]])


def test_uint8_output_spans_the_full_range():
    out = apply_window(np.array([[0.0, 0.5, 1.0]]), 0, 1, as_uint8=True)
    assert out.dtype == np.uint8
    assert list(out[0]) == [0, 127, 255]


def test_uniform_input_does_not_divide_by_zero():
    flat = np.full((8, 8), 42.0)
    out = window(flat, PercentileWindow(5, 95), as_uint8=False)
    assert np.all(np.isfinite(out))
    assert np.all(out == 0.0)


# ---------------------------------------------------------------------------
# Half-width arithmetic: the `// 2` bug
# ---------------------------------------------------------------------------

def test_odd_window_width_uses_true_half():
    """Three of the five pre-package copies used `window_width // 2`."""
    assert DicomWindow(center=100, width=51).bounds() == (74.5, 125.5)


def test_zero_width_is_rejected():
    with pytest.raises(ValueError):
        DicomWindow(center=100, width=0)


# ---------------------------------------------------------------------------
# window_series
# ---------------------------------------------------------------------------

def test_window_series_uses_one_set_of_bounds():
    frames = [np.full((4, 4), 10.0), np.full((4, 4), 90.0)]
    out = window_series(frames, PercentileWindow(0, 100), as_uint8=False)
    assert np.allclose(out[0], 0.0)
    assert np.allclose(out[1], 1.0)


def test_window_series_differs_from_per_frame():
    """Per-frame stretches each frame to its own extremes; the cine then flickers.

    Both frames span 20 counts here, but around very different levels. Windowed
    apart they come out identical — the level information is destroyed. Windowed
    together the dim frame stays dim, which is what the network was trained on
    and what the scanner side reproduces.
    """
    dim = np.linspace(10, 30, 16).reshape(4, 4)
    bright = np.linspace(900, 920, 16).reshape(4, 4)
    spec = PercentileWindow(0, 100)

    together = window_series([dim, bright], spec, as_uint8=False)
    apart = [window(f, spec, as_uint8=False) for f in (dim, bright)]

    # apart: both frames stretched to full range, so indistinguishable
    # (atol, not the default rtol: these are float32 ramps over different levels)
    assert np.allclose(apart[0], apart[1], atol=1e-4)
    # together: the dim frame stays at the bottom of the range
    assert together[0].max() < 0.05
    assert together[1].min() > 0.95
    assert not np.allclose(together[0], apart[0])
