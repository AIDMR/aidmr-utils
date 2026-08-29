"""Shared imaging, geometry and result-schema code for the AID-MR programs.

One implementation of the things AMP, CMRQ, AIFS, BPF and LGEP all need, so that
a fix made once is a fix everywhere. Submodules are imported lazily-ish: the
heavy ones (`dicom` needs pydicom, `mrd` needs ismrmrd) are optional extras, so
importing this package must never require them.

    from aidmr_utils.result import AIDMRResult, verdict, resolved_by
    from aidmr_utils.windowing import parse_window_spec, window_series
    from aidmr_utils.dicom import load_pixels, read_series      # [dicom]
    from aidmr_utils.mrd import np_float_to_mrd, standard_image_meta   # [mrd]
    from aidmr_utils.geometry import Plane, closest_plane
    from aidmr_utils.orientation import canonicalise_cine
    from aidmr_utils.pixel import mm_per_pixel_for_side, assert_square_pixels
    from aidmr_utils.onnx import load_model_onnx, warm_up            # [onnx]
    from aidmr_utils.imaging import put_text_on_img                  # [imaging]
    from aidmr_utils.fire import ReplayableConnection, save_and_return  # [mrd]

`dicom`, `mrd`, `onnx`, `imaging` and `fire` are NOT re-exported here.
Importing this package has to work in an environment that has none of pydicom,
ismrmrd, onnxruntime or cv2, which is the whole reason those are extras - so
import those five by module.
"""

from .geometry import (CANONICAL_RIGHT_DOWN, Plane, canonical_right_down,
                       closest_plane, normal_for_named_plane, plane_of, unit)
from .pixel import (SQUARE_PIXEL_TOLERANCE, anisotropy, assert_square_pixels,
                    mm_per_pixel_for_side)
from .windowing import (DEPLOYABLE_WINDOWINGS, DicomWindow, PercentileWindow,
                        WindowSpec, apply_window, parse_window_spec, window,
                        window_series)

__all__ = [
    # geometry
    'CANONICAL_RIGHT_DOWN', 'Plane', 'canonical_right_down', 'closest_plane',
    'normal_for_named_plane', 'plane_of', 'unit',
    # pixel
    'SQUARE_PIXEL_TOLERANCE', 'anisotropy', 'assert_square_pixels',
    'mm_per_pixel_for_side',
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
