"""Loading a YAML experiment config, and the hyperparameter-sweep flattening.

One implementation of what was five near-identical copies of `lib/config.py`.
`dot_check` and `update_config` were byte-identical in all five; `load_config`
and `unflatten_dot` differed only in local variable names and one assertion.

Needs the `config` extra (pyyaml, easydict).

TWO THINGS ARE DELIBERATELY DIFFERENT FROM THE COPIES
-----------------------------------------------------
**`dot_check` now runs, and works.** In all five repos it was defined inside
`load_config` and never called - and it would not have survived being called,
because it recursed on `d` rather than on `v`, so any nested dict was infinite
recursion. It is now a module-level function, recurses correctly, and
`load_config` actually invokes it. The check matters: a key containing a dot
collides with the `a.b.c` flattening `update_config` uses, so a sweep would
silently write to the wrong place.

**The `desc` check takes the first whitespace-separated token.** CMRQ, BPF and
LGEP compared `desc[:3]` to the filename stem, which only works while every
config is three characters (`033.yaml`); AIFS used `desc.split()[0]`, which is
the same answer for those and keeps working at `1234.yaml`. AIFS's is used.
"""

import collections.abc
import os
from pathlib import Path

try:
    import yaml
    from easydict import EasyDict as edict
except ImportError:  # pragma: no cover - exercised by the extras, not the suite
    yaml = None
    edict = None


def _require_deps():
    if yaml is None or edict is None:
        raise ImportError(
            "config loading needs the 'config' extra: "
            "pip install 'aidmr-utils[config]'")


def dot_check(d, _path='') -> None:
    """Refuse a key containing a dot, anywhere in the tree.

    `update_config` treats `a.b.c` as a path into the nested config, so a key
    that already contains a dot is ambiguous: a sweep setting it would write
    somewhere else entirely and nothing would say so.

    Recurses into `v`, not into `d`. The copies this replaces recursed into `d`,
    which is why none of them could ever have been called.
    """
    for k, v in d.items():
        here = f"{_path}.{k}" if _path else str(k)
        if '.' in str(k):
            raise ValueError(
                f"config key {here!r} contains a '.', which collides with the "
                f"dotted paths used for hyperparameter sweeps - rename it")
        if isinstance(v, collections.abc.Mapping):
            dot_check(v, here)


def load_config(filepath, check_desc: bool = True):
    """Read a YAML config into an EasyDict, stamping `filename` onto it.

    :param check_desc: assert the first token of `desc` matches the filename
        stem, so a config copied from another cannot keep the wrong description.
        The copies this replaces compared only the first three characters.
    """
    _require_deps()
    filepath = Path(filepath)
    with open(filepath, 'r') as f:
        cfg = edict(yaml.safe_load(f))
    cfg.filename = os.path.basename(filepath)

    if check_desc and (desc := cfg.get('desc')):
        stem = os.path.splitext(cfg.filename)[0]
        first = str(desc).split()[0] if str(desc).split() else ''
        assert first == stem, (
            f"desc starts with {first!r} but the file is {cfg.filename!r} - "
            f"this config was probably copied and its desc not updated")

    dot_check(cfg)
    return cfg


def unflatten_dot(dictionary: dict) -> dict:
    """`{'a.b': 1}` -> `{'a': {'b': 1}}`, the shape a sweep sends."""
    result = dict()
    for key, value in dictionary.items():
        parts = str(key).split('.')
        d = result
        for part in parts[:-1]:
            d = d.setdefault(part, dict())
        d[parts[-1]] = value
    return result


def update(d: dict, u: collections.abc.Mapping) -> dict:
    """Recursive dict update.

    `dict.update` replaces a nested dict wholesale, dropping the siblings that
    were not being set - which for a sweep overriding one leaf would silently
    discard the rest of that branch.
    """
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def update_config(cfg: dict):
    """Fold any dotted top-level keys back into the nested config.

    A sweep passes overrides flat - `{'data.windowing': '5_95'}` - so they have
    to be unflattened and merged before anything reads `cfg.data.windowing`.
    """
    _require_deps()
    hyper_params = unflatten_dot({k: v for k, v in cfg.items() if '.' in str(k)})
    update(cfg, hyper_params)
    return edict(cfg)
