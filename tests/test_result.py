"""The result schema, and the version check that replaced byte-identity."""

import json

import pytest

from aidmr_utils import result as result_mod
from aidmr_utils.result import (SCHEMA_VERSION, SEVERITY_HIGH, AIDMRResult,
                                _check_utils_version, _version_tuple,
                                resolved_by, verdict)

META = {
    'studyId': '1.2.3',
    'studyDate': '20260829',
    'studyTime': '120000',
    'measurementID': 'meas-1',
    'seriesTime': '120100',
    'seriesNumber': '4',
    'protocolName': 'cine 2ch 4ch',
}


def a_result(tab='function', program='bpf'):
    return AIDMRResult(dict(META), tab=tab, program=program)


# ---------------------------------------------------------------------------
# Round trip
# ---------------------------------------------------------------------------

def test_round_trip_preserves_the_payload():
    r = a_result()
    r.add_section('biplane', 'biplane', image_dicts=[],
                  interpretation=[verdict('ef', 'EF 55%', good=True)])
    dd = r.to_dict()
    back = AIDMRResult.from_dict(dd)
    assert back.to_dict() == dd


def test_rehydration_keeps_the_programs_inference_time():
    """FIRE rehydrating a payload is not a new inference."""
    r = a_result()
    dd = r.to_dict()
    dd['createdAt'] = '2020-01-01T00:00:00+00:00'
    assert AIDMRResult.from_dict(dd).created_at == '2020-01-01T00:00:00+00:00'


def test_version_1_payload_fails_loudly():
    """A program still on the old schema must stop, not write an undrawable tab."""
    dd = a_result().to_dict()
    del dd['tab']
    with pytest.raises(KeyError):
        AIDMRResult.from_dict(dd)


def test_schema_version_is_stamped():
    assert a_result().to_dict()['schemaVersion'] == SCHEMA_VERSION


# ---------------------------------------------------------------------------
# The utilsVersion check
# ---------------------------------------------------------------------------

def test_utils_version_is_stamped():
    assert 'utilsVersion' in a_result().to_dict()


@pytest.mark.parametrize('text,expected', [
    ('0.1.0', (0, 1)),
    ('1.2.3', (1, 2)),
    ('2.0', (2, 0)),
    ('1.4.0+local', (1, 4)),
    ('0+unknown', None),
    ('', None),
    ('not-a-version', None),
])
def test_version_tuple_parsing(text, expected):
    assert _version_tuple(text) == expected


def test_a_newer_writer_is_refused(monkeypatch):
    """Fields added since this host was built would be dropped on write.

    That is precisely the silent data loss the old "copies must be
    byte-identical" rule existed to prevent, and could not enforce.
    """
    monkeypatch.setattr(result_mod, 'UTILS_VERSION', '0.1.0')
    with pytest.raises(ValueError, match='written by aidmr-utils 0.2.0'):
        _check_utils_version({'utilsVersion': '0.2.0'})


def test_an_older_writer_is_accepted(monkeypatch):
    """Every field it knows, this host knows. Defaults fill the rest."""
    monkeypatch.setattr(result_mod, 'UTILS_VERSION', '0.2.0')
    _check_utils_version({'utilsVersion': '0.1.0'})


def test_a_patch_release_is_never_incompatible(monkeypatch):
    monkeypatch.setattr(result_mod, 'UTILS_VERSION', '0.2.0')
    _check_utils_version({'utilsVersion': '0.2.9'})


def test_missing_version_is_not_an_error(monkeypatch):
    """Results written before this field existed must stay readable."""
    monkeypatch.setattr(result_mod, 'UTILS_VERSION', '0.1.0')
    _check_utils_version({})


def test_uninstalled_checkout_does_not_judge(monkeypatch):
    monkeypatch.setattr(result_mod, 'UTILS_VERSION', '0+unknown')
    _check_utils_version({'utilsVersion': '99.0.0'})


def test_from_dict_runs_the_check(monkeypatch):
    monkeypatch.setattr(result_mod, 'UTILS_VERSION', '0.1.0')
    dd = a_result().to_dict()
    dd['utilsVersion'] = '5.0.0'
    with pytest.raises(ValueError, match='Upgrade the host'):
        AIDMRResult.from_dict(dd)


# ---------------------------------------------------------------------------
# Structural guards
# ---------------------------------------------------------------------------

def test_tab_and_program_are_required():
    with pytest.raises(AssertionError, match='tab'):
        AIDMRResult(dict(META), tab='', program='bpf')
    with pytest.raises(AssertionError, match='program'):
        AIDMRResult(dict(META), tab='function', program='')


def test_a_header_without_a_study_id_is_refused():
    """Otherwise it stringifies to 'None' and creates a phantom db/None/ study."""
    meta = dict(META, studyId=None)
    with pytest.raises(AssertionError, match='studyInstanceUID'):
        AIDMRResult(meta, tab='function', program='bpf')


def test_a_header_without_a_measurement_id_is_refused():
    meta = dict(META, measurementID=None)
    with pytest.raises(AssertionError, match='measurementID'):
        AIDMRResult(meta, tab='function', program='bpf')


# ---------------------------------------------------------------------------
# Issues are interpretation entries carrying a severity
# ---------------------------------------------------------------------------

def test_a_severity_makes_a_verdict_an_issue():
    v = verdict('rwma', 'RWMA present', good=False, severity=SEVERITY_HIGH)
    assert v['severity'] == SEVERITY_HIGH


def test_no_severity_means_a_plain_note():
    assert 'severity' not in verdict('bsa', 'BSA 1.9', good=True)


def test_resolution_match_needs_all_three_keys():
    with pytest.raises(AssertionError, match='needs all of'):
        resolved_by('Repeat the 4ch', tab='function')


def test_resolution_without_a_match_is_just_text():
    assert resolved_by('Record height and weight') == {
        'text': 'Record height and weight'}


def test_resolution_with_a_full_match():
    r = resolved_by('Repeat the 4ch', tab='qc', section_id='4c', verdict_id='misplan')
    assert r['match'] == {'tab': 'qc', 'sectionId': '4c', 'verdictId': 'misplan'}


# ---------------------------------------------------------------------------
# write_to_disk
# ---------------------------------------------------------------------------

def test_write_to_disk_lands_in_tab_and_measurement(tmp_path):
    r = a_result(tab='function', program='bpf')
    r.write_to_disk(tmp_path)
    written = tmp_path / '1.2.3' / 'function' / 'meas-1.json'
    assert written.exists()
    payload = json.loads(written.read_text())
    assert payload['program'] == 'bpf'
    assert payload['tab'] == 'function'
