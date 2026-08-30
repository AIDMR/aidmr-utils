"""Prove a change does not alter what the scanner is told.

Refactoring the FIRE-side code raises one question that no unit test answers:
are the SliceGroup, SliceFOV, SliceList and ShimMemory entries - the position,
normal, thickness, phase FOV, readout FOV, in-plane rotation and slice count -
still bit-for-bit what they were? This records them from a real test run and
diffs two branches.

    # on each branch, in the repo root
    pytest src/tests -p aidmr_utils.testing -q
    # then
    python -m aidmr_utils.testing main.json branch.json

`--instr-out PATH` sets the output file (default `_instr_probe.json`).

TWO THINGS THAT ARE EASY TO GET WRONG, AND WHY THIS EXISTS
-----------------------------------------------------------
**Record what `process()` RETURNS, not what the builders produce.** AMP attaches
every slice-memory and shim-memory entry to the keypoint mosaic at the *end* of
`process()`, long after `np_float_to_mrd` handed that image back. A probe hooked
only to the builders sees the mosaic mid-construction and reports no scanner
instructions at all - which reads as "nothing to compare" rather than as a bug
in the probe.

**Rebind at the import sites, not just the defining module.** Test modules do
`from ...fire_inference import process as process_amp`, which binds before a
plugin loads, so patching the defining module alone misses every call the tests
actually make. Rebinding walks `sys.modules` comparing by IDENTITY - `val in
mapping` raises `TypeError: unhashable type: 'ModuleSpec'` on the first module
attribute it meets.

Pixels are not recorded. Previews are allowed to differ in appearance; the
instructions are not.
"""

import json
import sys
from collections import Counter

#: Everything the scanner acts on lives under this prefix.
INSTRUCTION_PREFIX = 'SiemensExamMemory_'

#: Reported separately rather than as failures - changing these is how the
#: outgoing layout and the padded-square geometry were fixed in the first place.
EXPECTED_META_KEYS = {'ImageRowDir', 'ImageColumnDir'}
EXPECTED_HEADER_KEYS = {'field_of_view'}

_calls = []
_rebind = {}
_current = {'test': '?'}
_out_path = ['_instr_probe.json']

#: Modules searched for `process` and for the image builders.
FIRE_MODULES = (
    'src.lib.fire_inference',
    'src.lib.inference.mrd',
    'src.lib.inference.amp3d.fire_inference',
    'src.lib.inference.amplax2sax.fire_inference',
    'src.lib.inference.amploc.fire_inference',
)
BUILDERS = ('np_float_to_mrd', 'report_image_to_mrd')


def describe(img) -> dict:
    """An outgoing image's header geometry and its full MetaAttributes."""
    h = img.getHead()
    header = {
        'matrix_size': [int(v) for v in h.matrix_size],
        'field_of_view': [round(float(v), 5) for v in h.field_of_view],
        'position': [round(float(v), 5) for v in h.position],
        'read_dir': [round(float(v), 6) for v in h.read_dir],
        'phase_dir': [round(float(v), 6) for v in h.phase_dir],
        'slice_dir': [round(float(v), 6) for v in h.slice_dir],
        'patient_table_position': [round(float(v), 5) for v in h.patient_table_position],
        'image_type': int(h.image_type),
        'image_index': int(h.image_index),
        'image_series_index': int(h.image_series_index),
    }
    meta = {}
    s = getattr(img, 'attribute_string', '') or ''
    if s.strip():
        import ismrmrd
        try:
            m = ismrmrd.Meta.deserialize(s)
            for k in m.keys():
                v = m[k]
                meta[k] = ([str(x) for x in v]
                           if isinstance(v, (list, tuple)) else str(v))
        except Exception as e:            # a malformed meta is itself a finding
            meta = {'__unparseable__': str(e)}
    return {'header': header, 'meta': meta}


# ---------------------------------------------------------------------------
# pytest plugin
# ---------------------------------------------------------------------------

def pytest_addoption(parser):
    parser.addoption('--instr-out', default='_instr_probe.json',
                     help='where to write the recorded scanner instructions')


def pytest_configure(config):
    import importlib
    _out_path[0] = config.getoption('--instr-out')

    def record(**kw):
        _calls.append({'test': _current['test'], **kw})

    def wrap_builder(fn, label):
        def wrapped(*a, **kw):
            out = fn(*a, **kw)
            try:
                record(fn=label, **describe(out))
            except Exception as e:
                record(fn=label, error=f"{type(e).__name__}: {e}")
            return out
        return wrapped

    def wrap_process(proc, prog):
        def wrapped(*a, **kw):
            ret = proc(*a, **kw)
            imgs = ret[0] if isinstance(ret, tuple) else ret
            for i, im in enumerate(imgs or []):
                try:
                    record(fn=f'process:{prog}', idx=i, **describe(im))
                except Exception as e:
                    record(fn=f'process:{prog}', idx=i,
                           error=f"{type(e).__name__}: {e}")
            return ret
        return wrapped

    loaded = {}
    for name in FIRE_MODULES:
        try:
            loaded[name] = importlib.import_module(name)
        except Exception:
            continue

    # process() first - it is the authoritative record
    for name, mod in loaded.items():
        proc = getattr(mod, 'process', None)
        if proc is None or proc in _rebind.values():
            continue
        w = wrap_process(proc, name.split('.')[-2])
        _rebind[proc] = w
        setattr(mod, 'process', w)

    # then the builders, wherever they are bound
    for builder in BUILDERS:
        base = None
        for mod in loaded.values():
            base = getattr(mod, builder, None)
            if base is not None:
                break
        if base is None:
            continue
        w = wrap_builder(base, builder)
        for mod in loaded.values():
            if getattr(mod, builder, None) is not None:
                setattr(mod, builder, w)


def pytest_collection_modifyitems(session, config, items):
    """Rebind `process` wherever a test module imported it directly."""
    if not _rebind:
        return
    n = 0
    for mod in list(sys.modules.values()):
        if mod is None or not hasattr(mod, '__dict__'):
            continue
        for attr in list(vars(mod)):
            try:
                val = getattr(mod, attr)
            except Exception:
                continue
            # identity, never `in` - most module attributes are unhashable
            for orig, wrapper in _rebind.items():
                if val is orig:
                    try:
                        setattr(mod, attr, wrapper)
                        n += 1
                    except Exception:
                        pass
                    break
    print(f"[aidmr-utils] rebound process() at {n} import site(s)")


def pytest_runtest_setup(item):
    _current['test'] = item.nodeid


def pytest_sessionfinish(session, exitstatus):
    with open(_out_path[0], 'w') as f:
        json.dump(_calls, f, indent=1, sort_keys=True)
    print(f"\n[aidmr-utils] {len(_calls)} outgoing images -> {_out_path[0]}")


# ---------------------------------------------------------------------------
# the differ
# ---------------------------------------------------------------------------

def _load(path):
    """Key by (test, builder, nth-of-that-builder) so a change in the count of
    one kind of image cannot shift the indices of another."""
    with open(path) as f:
        recs = json.load(f)
    keyed, seen = {}, Counter()
    for r in recs:
        k = (r.get('test', '?'), r.get('fn', '?'))
        keyed[k + (seen[k],)] = r
        seen[k] += 1
    return keyed


def diff(before_path, after_path, stream=None) -> int:
    """Compare two recordings. Returns the number of instruction differences.

    `stream` resolves at call time, not import time: a default of `sys.stdout`
    binds the real stdout when this module loads, which then bypasses anything
    that replaces it later - pytest's capsys, a redirect, a log capture.
    """
    a, b = _load(before_path), _load(after_path)
    out = sys.stdout if stream is None else stream
    p = lambda s='': print(s, file=out)
    p(f"  before : {len(a)} outgoing images")
    p(f"  after  : {len(b)} outgoing images")

    only_a, only_b = sorted(set(a) - set(b)), sorted(set(b) - set(a))
    if only_a:
        p(f"\n  {len(only_a)} image(s) only in the first run, e.g. {only_a[:2]}")
    if only_b:
        p(f"\n  {len(only_b)} image(s) only in the second run, e.g. {only_b[:2]}")

    shared = sorted(set(a) & set(b))
    crit, expected, other, compared = [], Counter(), Counter(), 0
    for k in shared:
        ma, mb = a[k].get('meta', {}), b[k].get('meta', {})
        ha, hb = a[k].get('header', {}), b[k].get('header', {})
        for mk in sorted(set(ma) | set(mb)):
            va, vb = ma.get(mk), mb.get(mk)
            if mk.startswith(INSTRUCTION_PREFIX):
                compared += 1
                if va != vb:
                    crit.append((k, mk, va, vb))
            elif va != vb:
                (expected if mk in EXPECTED_META_KEYS else other)[mk] += 1
        for hk in sorted(set(ha) | set(hb)):
            if ha.get(hk) != hb.get(hk):
                (expected if hk in EXPECTED_HEADER_KEYS
                 else other)[f"header.{hk}"] += 1

    p(f"\n{'=' * 66}")
    p(f"SCANNER INSTRUCTIONS ({INSTRUCTION_PREFIX}*)")
    p('=' * 66)
    p(f"  {compared} instruction entries compared across {len(shared)} images")
    if crit:
        p(f"  {len(crit)} DIFFER:\n")
        for k, mk, va, vb in crit[:10]:
            p(f"    {k[0].split('::')[-1][:52]} [{k[1]} #{k[2]}]")
            p(f"      key   : {mk}")
            p(f"      before: {va}")
            p(f"      after : {vb}")
    elif compared:
        p("  IDENTICAL - every scanner instruction is byte-for-byte the same")
    else:
        p("  NOTHING COMPARED - no instruction entries were recorded at all.")
        p("  That is a probe failure, not a pass: check process() was wrapped.")

    if expected:
        p("\n  expected differences (deliberate):")
        for kk, n in expected.most_common():
            p(f"    {kk:24s} on {n} image(s)")
    if other:
        p("\n  OTHER differences (review these):")
        for kk, n in other.most_common():
            p(f"    {kk:24s} on {n} image(s)")
    return len(crit)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 2:
        print(__doc__)
        return 2
    return 1 if diff(argv[0], argv[1]) else 0


if __name__ == '__main__':
    sys.exit(main())
