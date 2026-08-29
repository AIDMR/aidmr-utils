"""Shared imaging, geometry and result-schema code for the AID-MR programs.

One implementation of the things AMP, CMRQ, AIFS, BPF and LGEP all need, so that
a fix made once is a fix everywhere. Submodules are imported lazily-ish: the
heavy ones (`dicom` needs pydicom, `mrd` needs ismrmrd) are optional extras, so
importing this package must never require them.

    from aidmr_utils.result import AIDMRResult, verdict, resolved_by
    from aidmr_utils.windowing import parse_window_spec, window_series
    from aidmr_utils.dicom import load_pixels, read_series      # [dicom]
    from aidmr_utils.geometry import Plane, closest_plane
    from aidmr_utils.orientation import canonicalise_cine
"""

from .geometry import (CANONICAL_RIGHT_DOWN, Plane, canonical_right_down,
                       closest_plane, normal_for_named_plane, plane_of, unit)
from .windowing import (DEPLOYABLE_WINDOWINGS, DicomWindow, PercentileWindow,
                        WindowSpec, apply_window, parse_window_spec, window,
                        window_series)

__all__ = [
    # geometry
    'CANONICAL_RIGHT_DOWN', 'Plane', 'canonical_right_down', 'closest_plane',
    'normal_for_named_plane', 'plane_of', 'unit',
    # windowing
    'DEPLOYABLE_WINDOWINGS', 'DicomWindow', 'PercentileWindow', 'WindowSpec',
    'apply_window', 'parse_window_spec', 'window', 'window_series',
]

try:
    from importlib.metadata import PackageNotFoundError, version as _v
    try:
        __version__ = _v('aidmr-utils')
    except PackageNotFoundError:
        __version__ = '0+unknown'
except ImportError:  # pragma: no cover
    __version__ = '0+unknown'
