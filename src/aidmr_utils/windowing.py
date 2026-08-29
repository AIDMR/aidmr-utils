"""Intensity windowing: mapping stored pixel values onto a display range.

Deliberately free of pydicom, ismrmrd and torch. Both sides of every deployment
import this module and the two MUST apply the identical mapping: training reads
DICOMs (`aidmr_utils.dicom`), FIRE receives MRD images, and if these diverge the
model sees intensities it was never trained on and fails silently. That is the
single most expensive class of bug in this programme, which is why the mapping
lives in one dependency-free place rather than being re-implemented per repo.

WINDOWING IS A POLICY; RESCALING IS NOT
---------------------------------------
Nothing here applies RescaleSlope/RescaleIntercept. That is `dicom.load_pixels`'
job and it is not optional — see the note there. This module assumes it is being
handed values already in the rescaled (output) domain, which is the domain
`WindowCenter` / `WindowWidth` are defined in.

The distinction matters more than it looks:

* A **percentile** window is exactly invariant to a positive affine rescale.
  `percentile(a*x+b) == a*percentile(x)+b`, and the normalisation divides the
  offset back out, so forgetting to rescale changes nothing.
* A **DICOM WC/WW** window is not invariant at all. The tags are quoted in
  rescaled units, so applying them to raw stored values silently clips. Measured
  on real Philips LGE data with `RescaleIntercept=-5587`: 0.0% of pixels land
  inside the window, i.e. a uniform image.

FORMAT
------
A percentile window is written `'<low>_<high>'`. Hyphenated forms (`'5-95'`) are
also accepted because CMRQ and AIFS spell it that way in their configs; they are
normalised on parse so that one string format reaches the model metadata.
"""

import numpy as np

#: Windowings the scanner can reproduce. MRD carries no DICOM WindowCenter /
#: WindowWidth, so a model deployed to FIRE must be trained on one of these —
#: there is no way to recover a per-image DICOM window on the scanner side.
DEPLOYABLE_WINDOWINGS = ('5_95', '0_100')

#: The one spelling that means "use this image's own WC/WW tags".
DICOM_WINDOW = 'dicom'


class WindowSpec:
    """Base class. Subclasses know how to produce (low, high) for some pixels."""

    def bounds(self, pixels) -> tuple[float, float]:  # pragma: no cover
        raise NotImplementedError

    @property
    def deployable(self) -> bool:
        """Whether FIRE can reproduce this window from MRD images alone."""
        return False


class PercentileWindow(WindowSpec):
    """Map the [low, high] percentiles of the data onto [0, 1].

    Pass one frame for a per-frame window, or a whole stacked series for a
    per-series one. Per-series is what both sides of a cine deployment use — see
    `window_series` and `dicom.read_series`.
    """

    def __init__(self, low_pct: float, high_pct: float):
        if not 0 <= low_pct < high_pct <= 100:
            raise ValueError(
                f"need 0 <= low < high <= 100, got {low_pct} and {high_pct}")
        self.low_pct = float(low_pct)
        self.high_pct = float(high_pct)

    def bounds(self, pixels) -> tuple[float, float]:
        low, high = np.percentile(np.asarray(pixels, dtype=np.float64),
                                  [self.low_pct, self.high_pct])
        if high <= low:
            # A uniform region (air, a saturated frame). Any non-zero span keeps
            # the division defined and maps the whole thing to 0.
            high = low + 1.0
        return float(low), float(high)

    @property
    def deployable(self) -> bool:
        return str(self) in DEPLOYABLE_WINDOWINGS

    def __str__(self):
        def fmt(p):
            return str(int(p)) if float(p).is_integer() else str(p)
        return f"{fmt(self.low_pct)}_{fmt(self.high_pct)}"

    def __repr__(self):
        return f"PercentileWindow({self.low_pct!r}, {self.high_pct!r})"

    def __eq__(self, other):
        return (isinstance(other, PercentileWindow)
                and (self.low_pct, self.high_pct) == (other.low_pct, other.high_pct))

    def __hash__(self):
        return hash((self.low_pct, self.high_pct))


class DicomWindow(WindowSpec):
    """The window this image's own WindowCenter / WindowWidth tags ask for.

    Not reproducible on the scanner: MRD images carry no such tags. Training a
    deployed model on this is a train/deploy mismatch — `deployable` is False and
    export-time checks should refuse it.
    """

    def __init__(self, center: float, width: float):
        self.center = float(center)
        self.width = float(width)
        if self.width <= 0:
            raise ValueError(f"WindowWidth must be positive, got {width}")

    def bounds(self, pixels=None) -> tuple[float, float]:
        # Half-width in floating point. Integer `// 2` (as three of the five
        # pre-package copies used) is off by half a level on odd widths and,
        # applied to an unsigned array, can wrap on the clamp.
        return self.center - self.width / 2.0, self.center + self.width / 2.0

    def __str__(self):
        return DICOM_WINDOW

    def __repr__(self):
        return f"DicomWindow({self.center!r}, {self.width!r})"

    def __eq__(self, other):
        return (isinstance(other, DicomWindow)
                and (self.center, self.width) == (other.center, other.width))

    def __hash__(self):
        return hash((self.center, self.width))


def check_window_method(window_method):
    """Guard the YAML octal trap.

    Unquoted `0_100` is read by YAML 1.1 as octal 0100, i.e. the integer 64, and
    every downstream comparison then silently compares numbers. Quote it in the
    config.
    """
    if not isinstance(window_method, str):
        raise TypeError(
            f"windowing must be a string, got {window_method!r} "
            f"({type(window_method).__name__}). YAML reads an unquoted 0_100 as "
            f"octal 64 - write it as \"0_100\"."
        )
    return window_method


def parse_window_spec(window_method) -> WindowSpec:
    """`'5_95'` / `'5-95'` / `'dicom'` -> a WindowSpec.

    `'dicom'` yields a *placeholder* DicomWindow: the real centre and width are
    per-image tags, so `dicom.read_image` substitutes them. Anything asking for
    bounds from this placeholder is asking the wrong question and gets told so.
    """
    method = check_window_method(window_method).strip()
    if method.lower() == DICOM_WINDOW:
        return _DicomWindowPlaceholder()
    parts = method.replace('-', '_').split('_')
    if len(parts) != 2:
        raise ValueError(
            f"unrecognised windowing {window_method!r}: expected '<low>_<high>' "
            f"(e.g. '5_95'), a hyphenated '<low>-<high>', or '{DICOM_WINDOW}'")
    try:
        low, high = float(parts[0]), float(parts[1])
    except ValueError:
        raise ValueError(
            f"unrecognised windowing {window_method!r}: "
            f"{parts[0]!r} and {parts[1]!r} are not numbers") from None
    return PercentileWindow(low, high)


class _DicomWindowPlaceholder(DicomWindow):
    """'dicom' as a config value, before any image has supplied the tags."""

    def __init__(self):
        self.center = None
        self.width = None

    def bounds(self, pixels=None):
        raise ValueError(
            "'dicom' windowing has no bounds of its own - the centre and width "
            "come from each image's WindowCenter/WindowWidth tags. Use "
            "aidmr_utils.dicom.read_image, which substitutes them per image.")

    def __repr__(self):
        return "parse_window_spec('dicom')"

    def __eq__(self, other):
        return isinstance(other, _DicomWindowPlaceholder)

    def __hash__(self):
        return hash(DICOM_WINDOW)


def apply_window(img, low: float, high: float, as_uint8: bool = True):
    """Map [low, high] onto [0, 1] (or 0-255), clipping outside.

    Always computes in float: clamping an unsigned integer array against a
    negative bound wraps instead of clipping.
    """
    img = (np.asarray(img, dtype=np.float32) - low) / (high - low)
    img = np.clip(img, 0.0, 1.0)
    return (img * 255).astype(np.uint8) if as_uint8 else img


def window(pixels, spec: WindowSpec, as_uint8: bool = True):
    """Window one array with `spec`, using that array's own extremes."""
    low, high = spec.bounds(pixels)
    return apply_window(pixels, low, high, as_uint8=as_uint8)


def window_series(frames, spec: WindowSpec, as_uint8: bool = True) -> list:
    """Window a stack of frames TOGETHER, under one set of bounds.

    Two reasons this is the default for a cine rather than per-frame windowing:

    * It is what the scanner can do. FIRE receives the whole cine as MRD images
      with no window tags, so the deployed side percentiles the stack. Windowing
      per frame in training and per cine at deployment is a silent mismatch.
    * Per-frame min-max stretches every frame to its own extremes, so a bright
      artefact in one frame changes that frame's contrast alone and the cine
      flickers through the cardiac cycle - noise the network must absorb.

    A DicomWindow is per-image by construction (the tags are per-image), so it
    is applied frame by frame and this collapses to the obvious thing.
    """
    stack = np.stack([np.asarray(f) for f in frames])
    low, high = spec.bounds(stack)
    return [apply_window(f, low, high, as_uint8=as_uint8) for f in stack]
