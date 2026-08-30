"""The config loader and the scanner-instruction differ."""

import json

import pytest

from aidmr_utils import testing as t

yaml = pytest.importorskip('yaml', reason="needs the [config] extra")
pytest.importorskip('easydict', reason="needs the [config] extra")

from aidmr_utils.config import (dot_check, load_config, unflatten_dot, update,
                                update_config)


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

def write_cfg(tmp_path, name, body):
    p = tmp_path / name
    p.write_text(yaml.safe_dump(body), encoding='utf-8')
    return p


def test_load_stamps_the_filename(tmp_path):
    cfg = load_config(write_cfg(tmp_path, '033.yaml', {'desc': '033 - a thing'}))
    assert cfg.filename == '033.yaml'


def test_desc_must_match_the_filename_stem(tmp_path):
    """Catches a config copied from another with its desc left behind."""
    with pytest.raises(AssertionError, match='desc starts with'):
        load_config(write_cfg(tmp_path, '034.yaml', {'desc': '033 - copied'}))


def test_desc_check_uses_the_whole_first_token_not_three_characters(tmp_path):
    """CMRQ/BPF/LGEP compared desc[:3], which breaks past 999 configs."""
    load_config(write_cfg(tmp_path, '1234.yaml', {'desc': '1234 - fine'}))
    with pytest.raises(AssertionError):
        load_config(write_cfg(tmp_path, '1234.yaml', {'desc': '123 - wrong'}))


def test_desc_check_can_be_turned_off(tmp_path):
    cfg = load_config(write_cfg(tmp_path, 'x.yaml', {'desc': 'anything'}),
                      check_desc=False)
    assert cfg.desc == 'anything'


# --- dot_check: the guard that never ran -----------------------------------

def test_dot_check_rejects_a_dotted_key_at_the_top():
    with pytest.raises(ValueError, match="contains a '\\.'"):
        dot_check({'a.b': 1})


def test_dot_check_rejects_a_dotted_key_when_NESTED():
    """The copies recursed into `d` rather than `v`, so this was unreachable -
    any nested dict was infinite recursion, which is why it was never called."""
    with pytest.raises(ValueError, match="contains a '\\.'"):
        dot_check({'data': {'inner': {'bad.key': 1}}})


def test_dot_check_names_the_full_path():
    with pytest.raises(ValueError, match='data.inner.bad'):
        dot_check({'data': {'inner': {'bad.key': 1}}})


def test_dot_check_terminates_on_a_nested_dict():
    """Recursing on `d` instead of `v` would never return."""
    dot_check({'a': {'b': {'c': {'d': 1}}}})


def test_load_config_actually_runs_dot_check(tmp_path):
    with pytest.raises(ValueError, match="contains a '\\.'"):
        load_config(write_cfg(tmp_path, '001.yaml',
                              {'desc': '001 - x', 'data': {'a.b': 1}}))


# --- sweep flattening -------------------------------------------------------

def test_unflatten_dot():
    assert unflatten_dot({'a.b.c': 1, 'a.b.d': 2, 'e': 3}) == {
        'a': {'b': {'c': 1, 'd': 2}}, 'e': 3}


def test_update_keeps_nested_siblings():
    """dict.update would replace `data` wholesale and lose `windowing`."""
    d = {'data': {'windowing': '0_100', 'n_frames': 24}}
    update(d, {'data': {'n_frames': 32}})
    assert d == {'data': {'windowing': '0_100', 'n_frames': 32}}


def test_update_config_folds_dotted_overrides_in():
    cfg = update_config({'data': {'windowing': '0_100'}, 'data.windowing': '5_95'})
    assert cfg.data.windowing == '5_95'


# ---------------------------------------------------------------------------
# testing: the differ
# ---------------------------------------------------------------------------

def instr(test, fn, meta=None, header=None):
    return {'test': test, 'fn': fn, 'meta': meta or {}, 'header': header or {}}


SLICE = ['slicegroup', 'AorticValve', '-8.7380', '-1.0254', '76.0265',
         '-0.5339', '-0.0', '0.8454', '6.0', '0', '0.59375', '340',
         '0.17453292519943298', '3']
KEY = 'SiemensExamMemory_wip_070_fire_ICEOut_AorticValve_Slice'


def dump(tmp_path, name, recs):
    p = tmp_path / name
    p.write_text(json.dumps(recs), encoding='utf-8')
    return str(p)


def test_identical_instructions_report_no_differences(tmp_path, capsys):
    recs = [instr('t::a', 'process:amp3d', {KEY: SLICE})]
    n = t.diff(dump(tmp_path, 'a.json', recs), dump(tmp_path, 'b.json', recs))
    assert n == 0
    assert 'byte-for-byte the same' in capsys.readouterr().out


def test_a_changed_in_plane_rotation_is_caught(tmp_path, capsys):
    changed = list(SLICE); changed[12] = '0.20000000000000001'
    n = t.diff(dump(tmp_path, 'a.json', [instr('t::a', 'process:amp3d', {KEY: SLICE})]),
               dump(tmp_path, 'b.json', [instr('t::a', 'process:amp3d', {KEY: changed})]))
    assert n == 1
    assert '0.17453292519943298' in capsys.readouterr().out


def test_a_changed_phase_fov_is_caught(tmp_path):
    changed = list(SLICE); changed[10] = '0.75'
    assert t.diff(
        dump(tmp_path, 'a.json', [instr('t::a', 'process:amp3d', {KEY: SLICE})]),
        dump(tmp_path, 'b.json', [instr('t::a', 'process:amp3d', {KEY: changed})])) == 1


def test_expected_keys_are_not_counted_as_differences(tmp_path, capsys):
    a = [instr('t::a', 'process:amp3d', {KEY: SLICE})]
    b = [instr('t::a', 'process:amp3d', {KEY: SLICE, 'ImageRowDir': ['1', '0', '0']})]
    assert t.diff(dump(tmp_path, 'a.json', a), dump(tmp_path, 'b.json', b)) == 0
    assert 'expected differences' in capsys.readouterr().out


def test_an_unexpected_meta_change_is_surfaced_but_not_fatal(tmp_path, capsys):
    a = [instr('t::a', 'process:amp3d', {KEY: SLICE, 'SequenceDescription': 'x'})]
    b = [instr('t::a', 'process:amp3d', {KEY: SLICE, 'SequenceDescription': 'y'})]
    assert t.diff(dump(tmp_path, 'a.json', a), dump(tmp_path, 'b.json', b)) == 0
    assert 'OTHER differences' in capsys.readouterr().out


def test_recording_nothing_is_reported_as_a_probe_failure(tmp_path, capsys):
    """The trap this tool exists to avoid: an empty comparison reads as a pass."""
    empty = [instr('t::a', 'process:amp3d', {})]
    t.diff(dump(tmp_path, 'a.json', empty), dump(tmp_path, 'b.json', empty))
    out = capsys.readouterr().out
    assert 'NOTHING COMPARED' in out and 'probe failure' in out


def test_keying_survives_a_changed_count_of_another_builder(tmp_path):
    """An extra preview must not shift the process() records out of alignment."""
    a = [instr('t::a', 'np_float_to_mrd'),
         instr('t::a', 'process:amp3d', {KEY: SLICE})]
    b = [instr('t::a', 'np_float_to_mrd'), instr('t::a', 'np_float_to_mrd'),
         instr('t::a', 'process:amp3d', {KEY: SLICE})]
    assert t.diff(dump(tmp_path, 'a.json', a), dump(tmp_path, 'b.json', b)) == 0
