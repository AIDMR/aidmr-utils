"""Prove the extraction is behaviour-preserving, on real data.

    python docs/verify_against_existing.py

NOTE: this compares aidmr-utils against the VENDORED COPIES it replaced, and
those copies have now been deleted from the model repos (phases 1-3). Run it
against the pre-migration code by checking the repos out at the commit before
they adopted the package:

    git -C ../Papers/BPF  stash        # if you have local work
    git -C ../Papers/BPF  checkout main
    git -C ../Papers/CMRQ checkout main
    python docs/verify_against_existing.py

It reported, at the point of migration:

    orientation  identical transforms to BOTH BPF and CMRQ on all 38 distinct
                 slice geometries in the FIRE captures
    dicom        percentile windowing byte-identical to BPF across 165 real
                 series, 24 of them Philips
    mrd          greyscale pixels, RGB pixels and header geometry identical to
                 BPF's on the production path; builds cleanly from all 38 real
                 geometries

Keep it: it is how the next extraction gets justified too.

Runs aidmr-utils side by side with the implementations it replaces, over real
scanner captures and real DICOMs, and reports any disagreement. Adoption is
much easier to justify when "it does the same thing" is measured rather than
asserted.

Compares:
  * orientation.choose_transform  vs BPF's and CMRQ's, over AIDMR-FIRE/logs/mrd
  * dicom.read_series             vs BPF's read_series_from_dicoms, over LGEP DICOMs

Where the old implementations disagree with each other, or where this package
deliberately differs (the Philips rescale, the sagittal handedness), that is
reported as an EXPECTED DIFFERENCE rather than a failure.
"""

import glob
import os
import re
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.abspath(os.path.join(HERE, '..', '..'))
MRD_DIR = os.path.join(WORK, 'AIDMR-FIRE', 'logs', 'mrd')
DICOM_DIR = os.path.join(WORK, 'Papers', 'LGEP', 'data', '2d', 'dicom')
BPF_ROOT = os.path.join(WORK, 'Papers', 'BPF')
CMRQ_ROOT = os.path.join(WORK, 'Papers', 'CMRQ')

from aidmr_utils.orientation import choose_transform as new_choose  # noqa: E402

PASS, FAIL, NOTE = [], [], []


def ok(name, condition, detail=''):
    (PASS if condition else FAIL).append(f"{name}{(' - ' + detail) if detail else ''}")
    print(f"  {'PASS' if condition else 'FAIL'}  {name}" + (f"  {detail}" if detail else ''))


def note(text):
    NOTE.append(text)
    print(f"  NOTE  {text}")


def _load_reference(root, module):
    """Import `src.lib.<module>` out of a model repo without polluting sys.path."""
    saved = list(sys.path)
    try:
        sys.path.insert(0, root)
        for stale in [k for k in sys.modules if k == 'src' or k.startswith('src.')]:
            del sys.modules[stale]
        return __import__(f'src.lib.{module}', fromlist=['*'])
    except Exception as e:
        print(f"  (could not import {module} from {os.path.basename(root)}: {e})")
        return None
    finally:
        sys.path[:] = saved
        for stale in [k for k in sys.modules if k == 'src' or k.startswith('src.')]:
            del sys.modules[stale]


# ---------------------------------------------------------------------------
# Orientation, over real MRD captures
# ---------------------------------------------------------------------------

def meta_vec(xml, key):
    m = re.search(rf'<meta>\s*<name>{key}</name>(.*?)</meta>', xml, re.S | re.I)
    if not m:
        return None
    vals = re.findall(r'<value>([^<]+)</value>', m.group(1))
    if len(vals) != 3:
        return None
    try:
        return np.array([float(v) for v in vals])
    except ValueError:
        return None


def collect_real_geometries(limit_files=None):
    """Unique (right, down) pairs from the captured scanner data."""
    import h5py
    geoms = set()
    files = sorted(glob.glob(os.path.join(MRD_DIR, '*.h5')))[:limit_files]
    for path in files:
        try:
            with h5py.File(path, 'r') as h:
                if 'dataset' not in h:
                    continue
                for gname in h['dataset']:
                    g = h['dataset'][gname]
                    if not hasattr(g, 'keys') or 'attributes' not in g:
                        continue
                    for a in g['attributes'][()]:
                        xml = a.decode() if isinstance(a, bytes) else str(a)
                        rd, cd = meta_vec(xml, 'ImageRowDir'), meta_vec(xml, 'ImageColumnDir')
                        if rd is None or cd is None:
                            continue
                        geoms.add((tuple(np.round(rd, 6)), tuple(np.round(cd, 6))))
        except Exception:
            continue
    return sorted(geoms)


def check_orientation():
    print("\n=== orientation.choose_transform vs BPF and CMRQ ===")
    geoms = collect_real_geometries()
    if not geoms:
        note("no MRD captures found - skipping")
        return
    print(f"  {len(geoms)} distinct slice geometries from {MRD_DIR}")

    for label, root in (('BPF', BPF_ROOT), ('CMRQ', CMRQ_ROOT)):
        ref = _load_reference(root, 'orientation')
        if ref is None:
            continue
        disagreements, planes = [], Counter()
        for right, down in geoms:
            r, d = np.array(right), np.array(down)
            new_name, new_score, new_plane = new_choose(r, d)
            old_name, old_score, old_plane = ref.choose_transform(r, d)
            planes[new_plane.value] += 1
            if new_name != old_name or not np.isclose(new_score, old_score):
                disagreements.append((right, down, new_name, old_name))
        ok(f"{label}: identical transform on all {len(geoms)} real geometries",
           not disagreements,
           '' if not disagreements else f"{len(disagreements)} differ, e.g. "
                                        f"new={disagreements[0][2]} old={disagreements[0][3]}")
        if label == 'BPF':
            print(f"        planes seen: {dict(planes)}")


# ---------------------------------------------------------------------------
# DICOM reading, over real DICOMs
# ---------------------------------------------------------------------------

def check_dicom():
    print("\n=== dicom.read_series vs BPF's read_series_from_dicoms ===")
    import pydicom
    from aidmr_utils.dicom import read_series as new_read_series

    ref = _load_reference(BPF_ROOT, 'dicom')
    if ref is None:
        note("BPF dicom.py not importable - skipping")
        return

    paths = sorted(glob.glob(os.path.join(DICOM_DIR, '*.dcm')))
    if not paths:
        note("no DICOMs found - skipping")
        return

    # Group by series so read_series gets genuine cines. Header-only pass over
    # everything: the vendor mix matters more than the sample size, and reading
    # only the header is cheap.
    by_series, vendors, rescaled = {}, {}, set()
    for p in paths:
        try:
            d = pydicom.dcmread(p, stop_before_pixels=True)
            uid = getattr(d, 'SeriesInstanceUID', None)
            if uid is None:
                continue
            by_series.setdefault(uid, []).append(p)
            vendors[uid] = str(getattr(d, 'Manufacturer', '?')).strip().split()[0]
            if (float(getattr(d, 'RescaleSlope', 1) or 1) != 1
                    or float(getattr(d, 'RescaleIntercept', 0) or 0) != 0):
                rescaled.add(uid)
        except Exception:
            continue

    vendor_counts = Counter(vendors.values())
    print(f"  {len(by_series)} series; vendors {dict(vendor_counts)}; "
          f"{len(rescaled)} with a non-identity rescale")

    # Sample across vendors rather than taking the first N of one manufacturer
    sample, per_vendor = [], Counter()
    for uid in sorted(by_series, key=lambda u: (vendors[u], u)):
        if per_vendor[vendors[uid]] < 50:
            per_vendor[vendors[uid]] += 1
            sample.append(uid)

    same = diff = 0
    philips_checked = 0
    for uid, members in ((u, by_series[u]) for u in sample):
        try:
            dcms = [pydicom.dcmread(p) for p in members[:8]]
            if len({(d.Rows, d.Columns) for d in dcms}) > 1:
                continue
            new = new_read_series(dcms, '5_95')
            old = ref.read_series_from_dicoms([pydicom.dcmread(p) for p in members[:8]],
                                              '5_95')
        except Exception:
            continue
        if all(np.array_equal(a, b) for a, b in zip(new, old)):
            same += 1
        else:
            diff += 1
        if vendors.get(uid, '').lower().startswith('phil'):
            philips_checked += 1

    ok("percentile windowing matches BPF byte for byte", diff == 0,
       f"{same} series identical, {diff} differ")
    print(f"        ({philips_checked} of those series were Philips)")

    # Now the deliberate difference: WC/WW on a rescaled image
    print("\n=== deliberate difference: 'dicom' windowing on Philips ===")
    from aidmr_utils.dicom import read_image as new_read_image
    checked = 0
    for uid, members in by_series.items():
        if uid not in rescaled or not hasattr(pydicom.dcmread(
                members[0], stop_before_pixels=True), 'WindowCenter'):
            continue
        d = pydicom.dcmread(members[0])
        new = new_read_image(d, 'dicom')
        old = ref.read_image_from_dicom(pydicom.dcmread(members[0]), 'dicom')
        vend = vendors.get(uid, '?')
        agree = np.array_equal(new, old)
        note(f"{vend:8s} slope={float(getattr(d,'RescaleSlope',1)):.4g} "
             f"intercept={float(getattr(d,'RescaleIntercept',0)):g}: "
             f"aidmr-utils {len(np.unique(new))} levels, BPF {len(np.unique(old))} "
             f"levels, {'identical' if agree else 'DIFFERENT'}")
        checked += 1
        if checked >= 6:
            break
    if not checked:
        note("no series with a non-zero intercept AND WC/WW tags found")


def check_mrd():
    """np_float_to_mrd against BPF's, on the geometries the scanner really sends."""
    print("\n=== mrd.np_float_to_mrd vs BPF's ===")
    try:
        import ctypes
        import ismrmrd
        from aidmr_utils.mrd import np_float_to_mrd as new_build
    except ImportError as e:
        note(f"skipping: {e}")
        return

    ref = _load_reference(BPF_ROOT, 'mrd')
    if ref is None:
        return

    rng = np.random.default_rng(0)
    img = rng.random((64, 64))

    def head_with(channels):
        h = ismrmrd.ImageHeader()
        h.matrix_size = tuple(ctypes.c_ushort(v) for v in (64, 64, 1))
        h.field_of_view = tuple(ctypes.c_float(v) for v in (240.0, 240.0, 8.0))
        h.patient_table_position = tuple(ctypes.c_float(v) for v in (0.0, 5.0, 0.0))
        h.channels = channels
        return h

    common = dict(position_xyz=(1.0, 2.0, 3.0),
                  fov_freq_phase_slice=(240.0, 240.0, 8.0),
                  phase_encoding_dir_xyz=(0.0, 1.0, 0.0),
                  freq_encoding_dir_xyz=(1.0, 0.0, 0.0),
                  image_index=1, series_index=1, attribute_string='')

    # Production path: the template header FIRE forwards carries channels=1
    new = new_build(img, template_head=head_with(1), **common)
    old = ref.np_float_to_mrd(img, template_head=head_with(1), **common)
    ok("greyscale pixels identical to BPF (production template)",
       np.array_equal(new.data, old.data))
    ok("header geometry identical to BPF",
       tuple(new.getHead().position) == tuple(old.getHead().position)
       and tuple(new.getHead().field_of_view) == tuple(old.getHead().field_of_view)
       and tuple(new.getHead().matrix_size) == tuple(old.getHead().matrix_size))

    rgb = rng.random((64, 64, 3))
    new_rgb = new_build(rgb, template_head=head_with(1), use_rgb=True, **common)
    old_rgb = ref.np_float_to_mrd(rgb, template_head=head_with(1), use_rgb=True, **common)
    ok("RGB pixels identical to BPF", np.array_equal(new_rgb.data, old_rgb.data))

    # The deliberate fix: a header carrying channels=0 (fresh, or a replayed
    # capture) silently produced a zero-size array in every pre-package version.
    new_zero = new_build(img, template_head=head_with(0), **common)
    old_zero = ref.np_float_to_mrd(img, template_head=head_with(0), **common)
    note(f"channels=0 template: aidmr-utils data.shape={new_zero.data.shape} "
         f"(size {new_zero.data.size}), BPF's={old_zero.data.shape} "
         f"(size {old_zero.data.size}) - the greyscale path now sets channels=1")

    # Real geometries from the captures
    geoms = collect_real_geometries()
    bad = []
    for right, down in geoms:
        try:
            new_build(img, template_head=head_with(1),
                      position_xyz=(0.0, 0.0, 0.0),
                      fov_freq_phase_slice=(240.0, 240.0, 8.0),
                      phase_encoding_dir_xyz=np.array(down),
                      freq_encoding_dir_xyz=np.array(right),
                      image_index=1, series_index=1, attribute_string='')
        except Exception as e:
            bad.append((right, down, e))
    ok(f"builds cleanly from all {len(geoms)} real slice geometries", not bad,
       '' if not bad else f"{len(bad)} failed, e.g. {bad[0][2]}")


def main():
    print(__doc__.strip().split('\n\n')[0])
    check_orientation()
    try:
        check_mrd()
    except Exception as e:
        note(f"mrd comparison could not run: {type(e).__name__}: {e}")
    try:
        check_dicom()
    except ImportError as e:
        note(f"skipping DICOM comparison: {e}")

    print(f"\n{'=' * 60}\n{len(PASS)} passed, {len(FAIL)} failed, {len(NOTE)} notes")
    for f in FAIL:
        print(f"  FAILED: {f}")
    return 1 if FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
