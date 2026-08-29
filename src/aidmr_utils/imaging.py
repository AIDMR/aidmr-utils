"""Drawing on an image: text, crosses, lines, discs.

Presentation primitives, shared because every program draws the same annotations
onto the previews it sends back. Deliberately NOT part of `mrd` - dragging cv2
into a module the FIRE container imports on every request is what kept
`np_float_to_mrd` from being shareable in the first place.

Needs the `imaging` extra (opencv-python-headless); the FIRE container already
has opencv, so this costs it nothing.

DELIBERATELY NOT HERE: pad_to_square. There are three variants across the repos
and they do not agree - AMP pads with HALF THE IMAGE MAX, BPF and CMRQ pad with
zeros, and AMP's is 2-D while the other two take a cine. That is not an
accident to be tidied away: padding is part of each model's preprocessing
contract, tied to what it was trained on and (for BPF) to
`mrd.padded_square_geometry`. A single shared version would have to pick a
default, and picking wrong changes what a network is fed. Each repo keeps its
own.
"""

import math

import numpy as np
import skimage.draw

try:
    import cv2
except ImportError:  # pragma: no cover - exercised by the extras, not the suite
    cv2 = None


def _require_cv2():
    if cv2 is None:
        raise ImportError(
            "drawing needs the 'imaging' extra: pip install 'aidmr-utils[imaging]'")


def put_text_on_img(img, text):
    _require_cv2()
    if img.max() <= 1.0:
        was_float = True
        img = (img * 255).astype(np.uint8)
    else:
        was_float = False

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    # white colour
    color = 255
    thickness = 1  # Thickness of the text

    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    # Calculate the (x, y) coordinate of the text position
    # Here we are putting the text at the center of the image
    x = (img.shape[1] - text_width) // 2
    y = (img.shape[0] + text_height) // 2

    # Put the text on the image
    img_with_text = cv2.putText(img, text, (x, y), font, font_scale, color, thickness)

    if was_float:
        img_with_text = img_with_text / 255.0

    return img_with_text


def add_cross_to_array(array_2d, coord_y_x, length=3, thickness=1, val=None):
    if val is None:
        val = array_2d.max()

    y, x = [round(v) for v in coord_y_x]

    # Ensure the thickness is at least 1
    thickness = max(1, thickness)

    # Vertical line of the cross
    y_from = max(y - length, 0)
    y_to = min(y + length + 1, array_2d.shape[0])
    x_from = max(x - thickness // 2, 0)
    x_to = min(x + (thickness + 1) // 2, array_2d.shape[1])
    array_2d[y_from: y_to, x_from: x_to] = val

    # Horizontal line of the cross
    y_from = max(y - thickness // 2, 0)
    y_to = min(y + (thickness + 1) // 2, array_2d.shape[0])
    x_from = max(x - length, 0)
    x_to = min(x + length + 1, array_2d.shape[1])
    array_2d[y_from: y_to, x_from: x_to] = val

    return array_2d


def add_dashed_cross_to_array(array_2d, coord_y_x, length=3, val=None, step=2):
    """A "dotted" cross - centre pixel plus single-pixel dots at ``step`` intervals along each arm.

    Visually distinct from ``add_cross_to_array`` so two overlapping crosses can
    be read as "chosen one" (solid, ``add_cross_to_array``) vs "candidate that
    wasn't chosen" (dotted, this function).

    Parameters
    ----------
    array_2d : np.ndarray
        Array to draw on. Modified in place.
    coord_y_x : tuple[float, float]
        Centre of the cross in ``(y, x)`` pixel coordinates.
    length : int
        Half-length of each cross arm in pixels (longest distance from centre).
    val : optional
        Value to draw with. Defaults to ``array_2d.max()``.
    step : int
        Spacing between dots along each arm (in pixels). With step=2 and length=3
        each arm has dots at distances 0, 2 from centre - so 5 pixels per arm.
    """
    if val is None:
        val = array_2d.max()
    step = max(1, int(step))
    length = max(0, int(length))
    h, w = array_2d.shape[:2]
    y0, x0 = int(round(coord_y_x[0])), int(round(coord_y_x[1]))

    # Centre pixel
    if 0 <= y0 < h and 0 <= x0 < w:
        array_2d[y0, x0] = val

    # Vertical and horizontal arm dots at every ``step`` pixels
    for d in range(step, length + 1, step):
        for yy in (y0 - d, y0 + d):
            if 0 <= yy < h:
                array_2d[yy, x0] = val
        for xx in (x0 - d, x0 + d):
            if 0 <= xx < w:
                array_2d[y0, xx] = val

    return array_2d


def _clip_segment_to_bounds(y1, x1, y2, x2, height, width):
    """Clip a segment to array bounds using Liang–Barsky clipping.

    Returns ``(y1, x1, y2, x2)`` ints inside ``[0, height-1] x [0, width-1]`` or
    ``None`` if the segment lies entirely outside. Clipping is essential before
    walking/drawing a segment: intersection endpoints (e.g. from
    ``DICOMPlane.get_intersection_and_points_with_abcd``) can be enormous, and an
    unclipped per-pixel walk over such a segment never terminates in practice.
    """
    x_min, x_max = 0.0, float(width - 1)
    y_min, y_max = 0.0, float(height - 1)

    x0, y0 = float(x1), float(y1)
    x1_, y1_ = float(x2), float(y2)

    dx = x1_ - x0
    dy = y1_ - y0

    p = (-dx, dx, -dy, dy)
    q = (x0 - x_min, x_max - x0, y0 - y_min, y_max - y0)

    u1, u2 = 0.0, 1.0

    for pi, qi in zip(p, q):
        if pi == 0.0:
            if qi < 0.0:
                return None
            continue

        t = qi / pi
        if pi < 0.0:
            u1 = max(u1, t)
        else:
            u2 = min(u2, t)

        if u1 > u2:
            return None

    clipped_x0 = x0 + u1 * dx
    clipped_y0 = y0 + u1 * dy
    clipped_x1 = x0 + u2 * dx
    clipped_y1 = y0 + u2 * dy

    return (
        int(round(clipped_y0)),
        int(round(clipped_x0)),
        int(round(clipped_y1)),
        int(round(clipped_x1)),
    )


def add_line_to_array(array_2d, coord1_y_x, coord2_y_x, thickness=1, val=None):
    """Draw a line on ``array_2d`` between ``coord1_y_x`` and ``coord2_y_x``.

    Parameters
    ----------
    array_2d : np.ndarray
        Array to draw on. Modified in place.
    coord1_y_x, coord2_y_x : tuple[float, float]
        End points of the line in ``(y, x)`` pixel coordinates.  Floats are
        rounded to the nearest integer.
    thickness : int, optional
        Thickness of the line in pixels.
    val : optional
        Value to draw with.  Defaults to the maximum of ``array_2d``.

    Returns
    -------
    np.ndarray
        The array with the line drawn on it.
    """

    if val is None:
        val = array_2d.max()

    y1, x1 = [round(v) for v in coord1_y_x]
    y2, x2 = [round(v) for v in coord2_y_x]

    clipped = _clip_segment_to_bounds(y1, x1, y2, x2, array_2d.shape[0], array_2d.shape[1])
    if clipped is None:
        return array_2d

    y1, x1, y2, x2 = clipped

    rr, cc = skimage.draw.line(y1, x1, y2, x2)

    thickness = max(1, thickness)
    for t in range(-(thickness // 2), (thickness + 1) // 2):
        rr_t = np.clip(rr + t, 0, array_2d.shape[0] - 1)
        cc_t = np.clip(cc + t, 0, array_2d.shape[1] - 1)
        array_2d[rr_t, cc_t] = val

    return array_2d


def add_dotted_line_to_array(array_2d,
                             coord1_y_x,
                             coord2_y_x,
                             val=None,
                             spacing=8,
                             radius=2):
    """Draw a "dotted" line - filled discs of ``radius`` pixels every ``spacing`` pixels along the line.

    Useful for drawing a reference line (e.g. the mitral plane) in a way that is
    clearly distinct from a solid thin line (e.g. SAX slice intersections) even
    when the two overlap. A radius of 0 degenerates to a single-pixel dot.

    Parameters
    ----------
    array_2d : np.ndarray
        Array to draw on. Modified in place.
    coord1_y_x, coord2_y_x : tuple[float, float]
        End points of the line in ``(y, x)`` pixel coordinates.
    val : optional
        Value to draw with. Defaults to the maximum of ``array_2d``.
    spacing : int
        Centre-to-centre spacing between dots in pixels (must be > 0).
    radius : int
        Disc radius in pixels (gives a (2*radius+1)-wide dot). 0 -> single pixel.
    """
    if val is None:
        val = array_2d.max()

    # Clip to the array first: intersection-line endpoints can be enormous
    # (e.g. from DICOMPlane.get_intersection_and_points_with_abcd, which uses
    # FOV-scaled, non-unit plane normals), and an unclipped walk would try to
    # drop millions of dots and never finish.
    clipped = _clip_segment_to_bounds(coord1_y_x[0], coord1_y_x[1],
                                      coord2_y_x[0], coord2_y_x[1],
                                      array_2d.shape[0], array_2d.shape[1])
    if clipped is None:
        return array_2d
    y1, x1, y2, x2 = (float(v) for v in clipped)

    dy = y2 - y1
    dx = x2 - x1
    length = math.sqrt(dy * dy + dx * dx)
    if length < 1e-6:
        # Degenerate "line" - just draw a single dot at the start
        _paint_disc(array_2d, y1, x1, radius, val)
        return array_2d

    spacing = max(1, int(spacing))
    radius = max(0, int(radius))

    # Unit step along line
    uy = dy / length
    ux = dx / length

    # Walk from 0 to length inclusive, dropping a disc every `spacing` pixels
    n_dots = int(math.floor(length / spacing)) + 1
    for k in range(n_dots):
        s = k * spacing
        cy = y1 + uy * s
        cx = x1 + ux * s
        _paint_disc(array_2d, cy, cx, radius, val)

    return array_2d


def _paint_disc(array_2d, cy, cx, radius, val):
    """Paint a filled disc of ``radius`` pixels at (cy, cx). Clipped to array bounds."""
    h, w = array_2d.shape[:2]
    cy_r, cx_r = int(round(cy)), int(round(cx))
    if radius <= 0:
        if 0 <= cy_r < h and 0 <= cx_r < w:
            array_2d[cy_r, cx_r] = val
        return
    r = int(radius)
    y_lo = max(0, cy_r - r)
    y_hi = min(h, cy_r + r + 1)
    x_lo = max(0, cx_r - r)
    x_hi = min(w, cx_r + r + 1)
    if y_lo >= y_hi or x_lo >= x_hi:
        return
    ys = np.arange(y_lo, y_hi)[:, None]
    xs = np.arange(x_lo, x_hi)[None, :]
    mask = (ys - cy_r) ** 2 + (xs - cx_r) ** 2 <= r * r
    array_2d[y_lo:y_hi, x_lo:x_hi][mask] = val