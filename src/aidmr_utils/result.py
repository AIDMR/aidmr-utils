"""AIDMR DB result — the `tab` / `section` schema (schemaVersion 2).

**There is one copy of this file and it is this one.** It used to be vendored by
hand into six repos with the instruction that the copies be byte-identical,
because AIDMR-FIRE rehydrates a program's dict with *its own* copy and calls
`write_to_disk` — so a program writing a field its host cannot read is silent
data loss. That invariant was already broken when this moved here: FIRE's copy
had drifted (an extra import and CRLF endings) and `verify_aidmr_result.py`
existed in three mutually different versions.

The byte-identity rule is now a version check instead. `to_dict` stamps
`utilsVersion`, and `from_dict` refuses a payload written by a *newer*
aidmr-utils than the host is running — see `SCHEMA_VERSION` and
`_check_utils_version`. A mismatch is now a loud failure at the boundary rather
than a field that quietly vanishes.

Deploying still means deploying together: FIRE and every program must move to a
compatible aidmr-utils in one go. What has changed is that "compatible" is now
asserted rather than assumed.

What changed from version 1
---------------------------
Version 1 had a single `series_type` doing three jobs at once: it named the
directory on disk, it named the dashboard tab, and it implied how to render the
payload. That worked while one program owned one tab. It stopped working the
moment two programs shared a tab — CMRQ and BPF would both have written
`db/<study>/cine/<measurementID>.json`, the same path, and the second would have
silently clobbered the first.

Three jobs, three fields:

* **tab** — which dashboard tab this measurement belongs in, and the directory
  it is written to: `qc`, `function`, `flow`, `planning`.
* **program** — which model produced it: `cmrq`, `bpf`, `aifs`, `amp3d`,
  `amplax2sax`. Shown in the UI and used to tell two programs apart inside one
  tab.
* **section** — one renderable block within that tab, with a `kind` saying how
  to draw it. A tab holds any number of sections, of any mix of kinds, from any
  number of measurements.

`kind` is the renderer, `sectionId` is the label. CMRQ's sections are named
after views (`2c`, `4c`, ...) but all draw the same way, so they are all
`kind='quality'`. BPF's two sections have distinct names *and* distinct
renderers (`biplane`, `rwma`).

No version-1 compatibility
--------------------------
The version-1 mirror lists (`data.vessels`, `data.cines`, `data.planning`) and
the `seriesType` key are **gone**, not deprecated. `data` is `sections` plus
`ecg`. The consequence is that this file and the dashboard must be deployed
together, and a program still on version 1 fails loudly at `from_dict` (missing
`tab`) rather than writing a payload nothing can draw. That is deliberate: a
staged rollout that half-works is harder to diagnose than one that stops.

`data.ecg` is not a section. It is a per-measurement sidecar — the ECG belongs
to the acquisition, not to any one program's interpretation of it — and the
dashboard's ECG tab collects it across every tab.

Issues
------
There is no `data.issues`. An issue is an **interpretation entry that carries a
`severity`**, so one verdict is drawn both in its section's box and in the
study-wide Issues panel, and the two can never disagree. Build them with
`verdict()` and `resolved_by()`; see the Issues block below.
"""

import io
import json
import os
import urllib.request
import uuid
from datetime import datetime, timezone

import imageio
import imageio.v3 as iio
import numpy as np
from loguru import logger

#: Bumped when the shape of `data` changes. The dashboard may branch on it.
SCHEMA_VERSION = 2

#: This package's version, stamped into every payload as `utilsVersion`.
#: Read from the installed distribution so there is exactly one place to bump it
#: (`pyproject.toml`); a source checkout that was never installed reports
#: '0+unknown', which `_check_utils_version` treats as "do not judge".
try:  # pragma: no cover - trivial, and the fallback is the interesting branch
    from importlib.metadata import PackageNotFoundError, version as _pkg_version
    try:
        UTILS_VERSION = _pkg_version('aidmr-utils')
    except PackageNotFoundError:
        UTILS_VERSION = '0+unknown'
except ImportError:
    UTILS_VERSION = '0+unknown'


def _version_tuple(v: str):
    """(major, minor) of a version string, or None if it is not parseable.

    Only the first two components matter: a patch release must never change the
    payload shape, so it cannot make a host incompatible.
    """
    if not v or v == '0+unknown':
        return None
    parts = v.split('+')[0].split('.')
    try:
        return int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
    except (ValueError, IndexError):
        return None


def _check_utils_version(dd: dict) -> None:
    """Refuse a payload written by a newer aidmr-utils than this one.

    This replaces the old "the copies must be byte-identical" rule, which was
    unenforceable and was in fact already violated. The asymmetry is deliberate:

    * A payload from a NEWER writer may contain fields this host has never heard
      of. `to_dict` would drop them on the way back out, so AIDMR-FIRE would
      write a truncated result and nothing would say so. That is the silent data
      loss the byte-identity rule existed to prevent, and it is now an exception.
    * A payload from an OLDER writer is fine. Every field it knows about, this
      host also knows about, and the defaults in `from_dict` fill the rest.

    An unparseable or absent `utilsVersion` is not an error: results written
    before this field existed must still be readable, and a source checkout that
    was never pip-installed has no version to report.
    """
    theirs = _version_tuple(dd.get('utilsVersion', ''))
    ours = _version_tuple(UTILS_VERSION)
    if theirs is None or ours is None:
        return
    if theirs > ours:
        raise ValueError(
            f"result was written by aidmr-utils {dd['utilsVersion']} but this "
            f"host is running {UTILS_VERSION}. Fields added since "
            f"{UTILS_VERSION} would be silently dropped on write. Upgrade the "
            f"host (AIDMR-FIRE and the dashboard move together with the "
            f"programs) rather than downgrading the program."
        )

#: Recognised section renderers. Adding one means adding a component to
#: `src/components/sections/` in the dashboard and registering it in
#: `SectionHost.vue` — nothing else here needs to change.
#:
#:   quality  — 1 image + metrics with mean/std/threshold, drawn as normal
#:              distributions, plus an interpretation box        (CMRQ)
#:   biplane  — exactly 2 ordered cines + a calculated-values box (BPF)
#:   rwma     — 1..N cines in a grid + an interpretation banner   (BPF)
#:   flow     — images + graphs + metrics + aliasing verdict      (AIFS)
#:   planning — images + found/missing views + slice-memory XML   (AMP)
SECTION_KINDS = ('quality', 'biplane', 'rwma', 'flow', 'planning')

# ---------------------------------------------------------------------------
# Issues
# ---------------------------------------------------------------------------
# An issue is not a separate object: it is an *interpretation entry carrying a
# severity*. The same verdict is then drawn twice — in its own section's box, and
# in the study-wide Issues panel — without any program sending it twice, and
# without the two copies ever being able to disagree.
#
# An interpretation entry with no severity is a plain note (the BSA line), shown
# in its section and nowhere else.
#
# `dismissed` and `resolved` are deliberately NOT fields here. Dismissal is a
# user action, and resolution is derived by looking for a later measurement that
# satisfies `resolution['match']`. Neither is something the model knows.

SEVERITY_INFO = 1        #: worth recording, needs nobody's attention
SEVERITY_LOW = 2         #: borderline; report it, do not act on it
SEVERITY_MODERATE = 3    #: a real finding — e.g. RWMAs present
SEVERITY_HIGH = 4        #: acquisition should be repeated
SEVERITY_CRITICAL = 5    #: the measurement cannot be reported as it stands

SEVERITIES = (SEVERITY_INFO, SEVERITY_LOW, SEVERITY_MODERATE,
              SEVERITY_HIGH, SEVERITY_CRITICAL)


def utc_now_iso() -> str:
    """When this result was produced, as a sortable UTC ISO-8601 string.

    This — not anything in the MRD header — is what the dashboard orders
    measurements by, and therefore what "a *later* measurement" means when
    resolving or superseding an issue.

    Header timestamps cannot do the job. A fully anonymised study has its dates
    and times anonymised too, and several scanners leave `seriesTime` and
    `initialSeriesNumber` unset regardless. Ordering on inference time instead
    means the newest thing the dashboard has seen is always the newest thing
    shown, which is also the behaviour a radiographer expects after a repeat.

    Timezone-aware on purpose: a bare local timestamp is ambiguous across a DST
    change and `Date.parse` in the browser would guess.
    """
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def resolved_by(text: str, tab: str = None, section_id: str = None,
                verdict_id: str = None) -> dict:
    """How a radiographer can clear an issue.

    `text` is the instruction, and is always shown — "Repeat the 4-chamber cine".

    The optional match names the verdict that would clear it: the issue is
    resolved once a **later** measurement in `tab` has a section `section_id`
    whose interpretation entry `verdict_id` is `good: true`. The comparison that
    decides `good` stays here in Python, next to the threshold — the dashboard
    only ever checks a boolean and an acquisition time.

    Pass `text` alone for an issue a repeat cannot clear ("Record height and
    weight in the patient registration").
    """
    match_keys = (tab, section_id, verdict_id)
    if any(k is not None for k in match_keys):
        assert all(k is not None for k in match_keys), (
            "a resolution match needs all of tab, section_id and verdict_id "
            f"(got {match_keys})")
        return {'text': text,
                'match': {'tab': tab, 'sectionId': section_id,
                          'verdictId': verdict_id}}
    return {'text': text}


def verdict(id: str, text: str, good: bool, severity: int = None,
            needs_resolution: bool = False, resolution: dict = None,
            label: str = None, value=None, threshold=None,
            detail: str = None) -> dict:
    """Build one interpretation entry, and an issue if `severity` is given.

    :param good: true -> green tick, false -> red cross. Sent rather than
        computed, because for RWMA it is the *negative* result that is good and
        letting the renderer decide which way round each verdict runs is exactly
        the kind of thing that ends up inverted on a Friday.
    :param severity: 1..5. Its presence is what makes this an issue; the panel
        sorts on it and styles by it.
    :param needs_resolution: does a radiographer have to *do* something? False
        for RWMAs present (a finding to report), true for a cine that must be
        re-acquired. This is the axis the interface may later split on.
    :param resolution: from `resolved_by()`.
    """
    entry = {'id': id, 'text': text, 'good': bool(good)}

    if label is not None:
        entry['label'] = label
    if value is not None:
        entry['value'] = value
    if threshold is not None:
        entry['threshold'] = threshold
    if detail is not None:
        entry['detail'] = detail

    if severity is not None:
        assert severity in SEVERITIES, \
            f"severity must be one of {SEVERITIES}, got {severity!r}"
        entry['severity'] = int(severity)
        entry['needs_resolution'] = bool(needs_resolution)
        if resolution is not None:
            entry['resolution'] = resolution
    else:
        assert not needs_resolution, \
            "needs_resolution on a note with no severity — give it a severity " \
            "so it reaches the Issues panel, or drop the flag"
        assert resolution is None, \
            "resolution on a note with no severity — only issues can be resolved"

    return entry


def _check_interpretation(interpretation: list[dict]) -> None:
    """Fail at build time rather than rendering something inconsistent."""
    for entry in interpretation:
        missing = {'id', 'text', 'good'} - set(entry)
        assert not missing, f"interpretation entry missing {sorted(missing)}: {entry}"
        assert isinstance(entry['good'], bool), \
            f"interpretation 'good' must be a bool, got {entry['good']!r}"

        if 'severity' in entry:
            assert entry['severity'] in SEVERITIES, \
                f"severity must be one of {SEVERITIES}, got {entry['severity']!r}"
        else:
            assert 'resolution' not in entry, \
                f"only an issue (one with a severity) can have a resolution: {entry}"

        res = entry.get('resolution')
        if res is not None:
            assert res.get('text'), f"a resolution needs instruction text: {res}"
            match = res.get('match')
            if match is not None:
                missing = {'tab', 'sectionId', 'verdictId'} - set(match)
                assert not missing, \
                    f"resolution match missing {sorted(missing)}: {match}"


class AIDMRResult:
    """One measurement's worth of results, destined for one tab.

    :param metadata: an ismrmrd header, or a metadata dict from `to_dict()`
    :param tab:      dashboard tab / directory name, e.g. 'function'
    :param program:  which model produced this, e.g. 'bpf'

    Both `tab` and `program` are required. `program` in particular is what lets
    two models share a tab without the renderer having to sniff at the data, and
    what `write_to_disk` uses to refuse to overwrite another model's result.
    """

    def __init__(self, metadata, tab: str, program: str):
        assert tab, "AIDMRResult needs a `tab` (the dashboard tab and directory)"
        assert program, "AIDMRResult needs a `program` (which model produced this)"

        self.metadata = self._extract_metadata(metadata)
        self.tab = tab
        self.program = program
        self.created_at = utc_now_iso()
        self.study_id = self.metadata['studyId']
        self.measurement_id = self.metadata['measurementID']

        # The whole DB layout is db/<studyId>/<tab>/<measurementID>.json, so these
        # two are structural. A header without them used to stringify to 'None' and
        # quietly produce a phantom `db/None/` study; fail here instead.
        assert self.study_id, \
            "header has no studyInstanceUID — cannot file this result under a study"
        assert self.measurement_id, \
            "header has no measurementID — cannot name this result"

        self._data = {'sections': [], 'ecg': []}
        self._images = {}
        self._waveforms = {}

    # ------------------------------------------------------------------
    # construction / serialisation
    # ------------------------------------------------------------------
    @classmethod
    def from_dict(cls, dd: dict) -> "AIDMRResult":
        """Rehydrate a payload returned by a program's `process()`.

        A `KeyError` on 'tab' or 'program' means the calling program is still on
        the version-1 schema and has not been migrated. Fail here rather than
        writing to `db/<study>/None/`.
        """
        _check_utils_version(dd)
        inst = cls(dd['metadata'], tab=dd['tab'], program=dd['program'])
        # Keep the *program's* inference time. The constructor just stamped a new
        # one, but AIDMR-FIRE rehydrating a payload is not a new inference, and
        # overwriting it would date every result to when the server happened to
        # write it.
        inst.created_at = dd.get('createdAt') or inst.created_at
        inst._images = dd.get('images', {}).copy()
        inst._waveforms = dd.get('waveforms', {}).copy()
        data = dd.get('data', {}).copy()
        data.setdefault('sections', [])
        data.setdefault('ecg', [])
        inst._data = data
        return inst

    @staticmethod
    def _extract_metadata(header) -> dict:
        if isinstance(header, dict):
            return header.copy()

        def field(value):
            """Stringify, but keep a missing field missing.

            `str(None)` is `'None'`, and a downstream reader cannot tell that from
            a real value — it parses as neither a time nor a number, so it poisons
            anything that sorts on it. Several scanners leave `seriesTime` and
            `initialSeriesNumber` unset, which is how every measurement in a study
            ended up with an identical sort key and the dashboard listed them in
            directory order. Emit JSON null instead.
            """
            return None if value is None else str(value)

        si = header.studyInformation
        mi = header.measurementInformation
        return {
            'studyId': field(si.studyInstanceUID),
            'studyDate': field(si.studyDate),
            'studyTime': field(si.studyTime),
            'measurementID': field(mi.measurementID),
            'seriesTime': field(mi.seriesTime),
            'seriesNumber': field(mi.initialSeriesNumber),
            'protocolName': field(mi.protocolName),
        }

    def to_dict(self) -> dict:
        return {
            'metadata': self.metadata,
            'schemaVersion': SCHEMA_VERSION,
            # Which aidmr-utils wrote this. `from_dict` checks it; the dashboard
            # ignores it. See `_check_utils_version`.
            'utilsVersion': UTILS_VERSION,
            'tab': self.tab,
            'program': self.program,
            # What the dashboard sorts on. See utc_now_iso().
            'createdAt': self.created_at,
            'images': self._images,
            'waveforms': self._waveforms,
            'data': self._data,
        }

    # ------------------------------------------------------------------
    # images
    # ------------------------------------------------------------------
    def _asset_path(self, uid: str, file_type: str) -> str:
        return os.path.join(self.tab, self.measurement_id,
                            f"{uid}.{file_type}").replace('\\', '/')

    def add_image(self, image_data, file_type: str, **kwargs) -> str:
        """Store one image (or one GIF stack) and return its relative path.

        `image_data` may be raw encoded bytes, a uint8 array, or a float array.
        A float array whose maximum exceeds 1.0 is min-max normalised across the
        *whole* array — for a (T, H, W) stack that means one windowing for every
        frame, so the cine does not flicker.
        """
        ext = file_type.lower()
        assert ext in ['png', 'jpg', 'jpeg', 'gif']

        if isinstance(image_data, (bytes, bytearray)):
            image_bytes = bytes(image_data)
        else:
            if image_data.dtype != np.uint8:
                image_data = image_data.astype(np.float32)  # also handles big uints
                if image_data.max() > 1.0:
                    image_data = image_data - image_data.min()
                    image_data = image_data / image_data.max()
                image_data = (image_data * 255).astype(np.uint8)

            buf = io.BytesIO()
            if ext == 'gif':
                frames = (list(image_data)
                          if hasattr(image_data, 'ndim') and image_data.ndim == 4
                          else image_data)
                imageio.mimsave(buf, frames, format='GIF', **kwargs)
            else:
                iio.imwrite(buf, image_data, format=ext, extension=f'.{ext}', **kwargs)
            image_bytes = buf.getvalue()

        path = self._asset_path(uuid.uuid4().hex, ext)
        self._images[path] = image_bytes
        return path

    def _check_images(self, image_dicts: list[dict]) -> None:
        missing = [d['image_path'] for d in image_dicts
                   if d['image_path'] not in self._images]
        if missing:
            raise KeyError(f"Unknown image paths: {missing}")

    # ------------------------------------------------------------------
    # waveforms / ECG
    # ------------------------------------------------------------------
    def add_waveform(self, waveform_data, file_type: str = 'json', **kwargs) -> str:
        """Save a 1D waveform. Default = JSON holding `{"waveform": [...]}`."""
        ext = file_type.lower()
        assert ext in ['json', 'npz'], f"Unsupported waveform file type: {file_type}"

        waveform_array = np.asarray(waveform_data).squeeze()
        assert waveform_array.ndim == 1, "Waveform data must be 1D"

        if waveform_array.dtype != np.float32:
            waveform_array = waveform_array.astype(np.float32)

        if ext == 'json':
            payload = {"waveform": waveform_array.tolist()}
            if kwargs:
                payload.update(kwargs)
            waveform_bytes = json.dumps(payload, separators=(',', ':')).encode('utf-8')
        else:
            buf = io.BytesIO()
            np.savez_compressed(buf, waveform=waveform_array, **kwargs)
            waveform_bytes = buf.getvalue()

        path = self._asset_path(uuid.uuid4().hex, ext)
        self._waveforms[path] = waveform_bytes
        return path

    def add_ecg(self, ecg, waveform_ids=("I", "II", "III", "VECG", "R-waves"),
                file_type: str = 'json', **waveform_kwargs) -> list[dict]:
        """Add a multichannel ECG as a per-measurement sidecar.

        Not a section: the ECG belongs to the acquisition rather than to any one
        program's interpretation of it, and the dashboard's ECG tab gathers it
        across every tab.
        """
        ecg_array = np.asarray(ecg)

        if ecg_array.ndim == 1:
            channels = [ecg_array]
        elif ecg_array.ndim == 2:
            channels = [ecg_array[i] for i in range(ecg_array.shape[0])]
        else:
            raise AssertionError("ECG data must be 1D or 2D")

        if len(channels) != len(waveform_ids):
            raise ValueError(
                f"ECG has {len(channels)} channels but {len(waveform_ids)} ids")

        entries = [{"id": wid,
                    "waveform_path": self.add_waveform(data, file_type=file_type,
                                                       **waveform_kwargs)}
                   for data, wid in zip(channels, waveform_ids)]
        self._data['ecg'].extend(entries)
        return entries

    # ------------------------------------------------------------------
    # sections
    # ------------------------------------------------------------------
    def add_section(self, section_id: str, kind: str, image_dicts: list[dict],
                    metrics: list[dict] = None, interpretation: list[dict] = None,
                    label: str = None, **extra) -> dict:
        """Append one renderable block. Every other adder is a thin wrapper.

        `extra` carries kind-specific fields (`graphs`, `aliasing`,
        `found_views`, `xml`, ...) so a new kind needs no change here.
        """
        assert kind in SECTION_KINDS, \
            f"Unknown section kind {kind!r}, want one of {SECTION_KINDS}"
        self._check_images(image_dicts)
        _check_interpretation(interpretation or [])

        section = {
            'sectionId': section_id,
            'kind': kind,
            'label': label if label is not None else section_id,
            'program': self.program,
            'images': image_dicts,
            'metrics': list(metrics or []),
            'interpretation': list(interpretation or []),
        }
        section.update(extra)
        self._data['sections'].append(section)
        return section

    def add_quality_data(self, section_id: str, image_dicts: list[dict],
                         metrics: list[dict], interpretation: list[dict] = None,
                         label: str = None) -> dict:
        """CMRQ: one cine, metrics drawn as normal distributions, advice box.

        `metrics` here are scores positioned against a reference distribution —
        `value` plus `mean`, `std` and `threshold` — not physical measurements.
        `interpretation` is sent rather than recomputed in the dashboard, so the
        thresholds live in exactly one place.
        """
        return self.add_section(section_id, 'quality', image_dicts, metrics,
                                interpretation, label)

    def add_biplanar_data(self, section_id: str, image_dicts: list[dict],
                          metrics: list[dict], interpretation: list[dict] = None,
                          label: str = None) -> dict:
        """BPF: a fixed pair of cines plus a calculated-values box.

        `image_dicts` is ordered — the dashboard draws them left to right — and
        `metrics` are physical values (`value` + `unit`, optionally `indexed` +
        `indexed_unit`). `indexed` must be **absent**, not None, when the header
        carries no height and weight.
        """
        return self.add_section(section_id, 'biplane', image_dicts, metrics,
                                interpretation, label)

    def add_rwma_data(self, section_id: str, image_dicts: list[dict],
                      interpretation: list[dict], metrics: list[dict] = None,
                      label: str = None) -> dict:
        """BPF: 1..N heatmap cines in a grid, under an interpretation banner.

        Deliberately not limited to two: the same section holds a short-axis
        stack, or a single view, without a schema change.
        """
        assert image_dicts, "an rwma section needs at least one cine"
        return self.add_section(section_id, 'rwma', image_dicts, metrics,
                                interpretation, label)

    def add_flow_data(self, section_id: str, image_dicts: list[dict],
                      graphs: list[dict], aliasing=None,
                      metrics: list[dict] = None,
                      interpretation: list[dict] = None,
                      label: str = None) -> dict:
        """AIFS: one vessel — masked cines, flow/velocity graphs, values box.

        `metrics` carry forward/backward/net flow and peak velocity as numbers.
        They used to be scraped back out of `graphs[].subtitle` with a regex in
        the dashboard; send them properly instead.
        """
        return self.add_section(section_id, 'flow', image_dicts, metrics,
                                interpretation, label,
                                graphs=graphs, aliasing=aliasing)

    def add_planning_data(self, image_dicts: list[dict], found_views: list[str],
                          missing_views: list[str], xml: list[str] = None,
                          interpretation: list[dict] = None,
                          section_id: str = 'planning',
                          label: str = None) -> dict:
        """AMP: planned slices, which views were found, and the slice-memory XML.

        The old `program=` argument is gone — `program` is set once on the
        result and copied onto every section, so `amp3d` and `amplax2sax` are
        told apart by the constructor rather than per call.

        `interpretation` is where a missing view becomes an issue: a bare
        `missing_views` list renders as red crosses in the section but reaches
        nobody who is not already looking at the planning tab.
        """
        return self.add_section(section_id, 'planning', image_dicts,
                                interpretation=interpretation,
                                label=label,
                                found_views=found_views,
                                missing_views=missing_views,
                                xml=list(xml or []))

    # ------------------------------------------------------------------
    # writing
    # ------------------------------------------------------------------
    def write_to_disk(self, db_dir) -> None:
        study_path = os.path.join(db_dir, self.metadata['studyId'])
        os.makedirs(study_path, exist_ok=True)

        meta = {k: self.metadata[k] for k in ('studyId', 'studyDate', 'studyTime')}
        with open(os.path.join(study_path, 'metadata.json'), 'w') as mf:
            json.dump(meta, mf, indent=2)

        for rel_path, blob in (*self._images.items(), *self._waveforms.items()):
            out_path = os.path.join(study_path, rel_path)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, 'wb') as f:
                f.write(blob)

        payload = self.to_dict()
        payload.pop('images')       # written above, as files
        payload.pop('waveforms')    # ditto

        class _NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, (np.integer, np.floating)):
                    return obj.item()
                return super().default(obj)

        series_json = os.path.join(study_path, self.tab,
                                   f"{self.measurement_id}.json")
        os.makedirs(os.path.dirname(series_json), exist_ok=True)
        self._refuse_to_clobber(series_json)
        with open(series_json, 'w') as jf:
            json.dump(payload, jf, indent=2, cls=_NumpyEncoder)

        self._notify()

    def _refuse_to_clobber(self, series_json: str) -> None:
        """Raise rather than overwrite another program's result at this path.

        The filename is `<tab>/<measurementID>.json`, and the measurement ID
        comes from the MRD header — so two programs in the same tab, dispatched
        as a chain against the same header, resolve to the same path. Splitting
        `series_type` into tab + program stopped that happening for the pairs we
        run today (CMRQ is `qc`, BPF is `function`), but the schema still allows
        several programs per tab, and a future pair would clobber in silence.

        Overwriting our *own* previous result is a re-run and is allowed.
        """
        if not os.path.exists(series_json):
            return
        try:
            with open(series_json) as f:
                existing = json.load(f).get('program')
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Could not read {series_json} to check ownership: {e}")
            return
        if existing and existing != self.program:
            raise FileExistsError(
                f"{series_json} was written by program {existing!r}; "
                f"{self.program!r} would overwrite it. Two programs are sharing "
                f"tab {self.tab!r} for one measurement — give one of them its "
                f"own tab, or make the filename program-specific."
            )

    def _notify(self) -> None:
        msg_dict = {
            'event': 'new_measurement',
            'study_id': self.metadata['studyId'],
            'tab': self.tab,
            # Old key name, same value: the dashboard routes on the tab name and
            # the tab name is the route path. Kept so a websocket client that
            # has not been redeployed still refreshes the right tab.
            'series_type': self.tab,
            'program': self.program,
            'measurement_id': self.measurement_id,
        }
        notify_url = os.environ.get('AIDMR_NOTIFY_URL',
                                    'http://localhost:8000/api/notify')
        try:
            req = urllib.request.Request(
                notify_url, data=json.dumps(msg_dict).encode('utf-8'),
                headers={'Content-Type': 'application/json'}, method='POST')
            urllib.request.urlopen(req, timeout=2)
            logger.info("Websocket update successful.")
        except Exception as e:
            logger.warning(
                f"Failed to broadcast measurement {self.measurement_id}: {e}")
