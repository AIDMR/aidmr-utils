"""Shared fixtures. Synthetic DICOMs so the suite needs no patient data."""

import numpy as np
import pytest

pydicom = pytest.importorskip('pydicom', reason="needs the [dicom] extra")

from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian


def make_dcm(pixels,
             slope=None,
             intercept=None,
             window_center=None,
             window_width=None,
             photometric='MONOCHROME2',
             instance_number=None,
             series_uid=None,
             image_orientation=None):
    """A minimal but genuinely readable DICOM dataset backed by real PixelData."""
    pixels = np.asarray(pixels, dtype=np.uint16)
    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.Rows, ds.Columns = int(pixels.shape[0]), int(pixels.shape[1])
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0          # unsigned
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = photometric
    ds.PixelData = pixels.tobytes()
    if slope is not None:
        ds.RescaleSlope = slope
    if intercept is not None:
        ds.RescaleIntercept = intercept
    if window_center is not None:
        ds.WindowCenter = window_center
    if window_width is not None:
        ds.WindowWidth = window_width
    if instance_number is not None:
        ds.InstanceNumber = instance_number
    if series_uid is not None:
        ds.SeriesInstanceUID = series_uid
    if image_orientation is not None:
        ds.ImageOrientationPatient = [float(v) for v in image_orientation]
    return ds


@pytest.fixture
def make_dicom():
    return make_dcm


@pytest.fixture
def philips_like():
    """A Philips-style image: real rescale, and WC/WW quoted in rescaled units.

    Modelled on a measured LGE series — RescaleSlope 2.72869, RescaleIntercept
    -5587, raw values sitting in a narrow band around 2000. Applying the WC/WW to
    un-rescaled values puts nothing inside the window.
    """
    rng = np.random.default_rng(0)
    raw = rng.integers(1960, 2476, size=(32, 32)).astype(np.uint16)
    slope, intercept = 2.72869352869352, -5587.0
    rescaled = raw * slope + intercept
    # A window that genuinely covers the rescaled data
    center = float((rescaled.min() + rescaled.max()) / 2)
    width = float(rescaled.max() - rescaled.min())
    return make_dcm(raw, slope=slope, intercept=intercept,
                    window_center=center, window_width=width), raw, slope, intercept
