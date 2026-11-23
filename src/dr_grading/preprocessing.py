"""Fundus preprocessing and the two normalization arms.

The ONLY difference between the arms is geometry:

- ``geometric`` (baseline): crop the fundus tightly, then resize the whole
  fundus to fill a square frame (aspect-preserving + pad). This is the common,
  scale-*naive* approach -- a small-FOV and a wide-FOV image are both stretched
  to fill the frame, so physical scale (mm/pixel) is inconsistent across images.

- ``physical_scale``: rescale the image so the optic-disc diameter maps to a
  fixed number of pixels, then crop a fixed physical field-of-view centred on
  the fundus. This yields an (approximately) constant mm/pixel across images.

Everything else (fundus crop, optional CLAHE / colour normalization, final
resize, tensor normalization) is shared and applied identically to both arms,
so any performance difference is attributable to scale handling alone.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Shared low-level operations
# ---------------------------------------------------------------------------
def crop_fundus_bbox(image: np.ndarray, thresh: int = 10) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Crop the black border around the circular fundus.

    Returns the cropped image and the (x, y, w, h) bounding box. If no fundus
    is found (degenerate image), the original image and full bbox are returned.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    _, mask = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        h, w = image.shape[:2]
        return image, (0, 0, w, h)
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    return image[y : y + h, x : x + w], (x, y, w, h)


def fundus_center_radius(image: np.ndarray, thresh: int = 10) -> Tuple[Tuple[int, int], int]:
    """Estimate the fundus centre and radius from the circular foreground."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    _, mask = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        h, w = image.shape[:2]
        return (w // 2, h // 2), min(h, w) // 2
    (cx, cy), r = cv2.minEnclosingCircle(max(contours, key=cv2.contourArea))
    return (int(cx), int(cy)), int(r)


def resize_pad_square(image: np.ndarray, size: int, pad_value: int = 0) -> np.ndarray:
    """Aspect-preserving resize so the longer side == ``size``, then pad."""
    h, w = image.shape[:2]
    scale = size / max(h, w)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    if image.ndim == 3:
        canvas = np.full((size, size, image.shape[2]), pad_value, dtype=image.dtype)
    else:
        canvas = np.full((size, size), pad_value, dtype=image.dtype)
    y0, x0 = (size - nh) // 2, (size - nw) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return canvas


def crop_pad_centered(image: np.ndarray, center: Tuple[int, int], box: int, pad_value: int = 0) -> np.ndarray:
    """Extract a ``box``x``box`` window centred on ``center`` (zero-padded)."""
    cx, cy = center
    half = box // 2
    if image.ndim == 3:
        canvas = np.full((box, box, image.shape[2]), pad_value, dtype=image.dtype)
    else:
        canvas = np.full((box, box), pad_value, dtype=image.dtype)
    x1, y1 = cx - half, cy - half
    sx1, sy1 = max(0, x1), max(0, y1)
    sx2, sy2 = min(image.shape[1], x1 + box), min(image.shape[0], y1 + box)
    dx1, dy1 = sx1 - x1, sy1 - y1
    canvas[dy1 : dy1 + (sy2 - sy1), dx1 : dx1 + (sx2 - sx1)] = image[sy1:sy2, sx1:sx2]
    return canvas


def apply_clahe(image: np.ndarray, clip: float = 2.0, grid: int = 8) -> np.ndarray:
    """CLAHE on the L channel of LAB (local contrast enhancement)."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def color_normalize(image: np.ndarray, method: str) -> np.ndarray:
    """Shared colour normalization applied identically to both arms.

    - ``none``: no change
    - ``clahe``: CLAHE contrast enhancement
    - ``graham``: Ben Graham background subtraction (common for DR grading)
    """
    if method == "none":
        return image
    if method == "clahe":
        return apply_clahe(image)
    if method == "graham":
        h, w = image.shape[:2]
        blur = cv2.GaussianBlur(image, (0, 0), sigmaX=max(h, w) / 30.0)
        return cv2.addWeighted(image, 4, blur, -4, 128)
    raise ValueError(f"Unknown color_normalize method: {method}")


# ---------------------------------------------------------------------------
# Preprocessing configuration + the two arms
# ---------------------------------------------------------------------------
@dataclass
class PreprocessConfig:
    size: int = 224                 # final square output size
    color_norm: str = "clahe"       # shared across arms: none|clahe|graham
    # physical-scale arm only:
    target_od_diameter: int = 90    # OD diameter mapped to this many pixels
    # Canonical field of view = N x OD diameter. The window fixes two things
    # at once. It sets how much of the frame the retina fills, which is
    # effective resolution, and it sets how much of the retina falls inside the
    # frame at all. Measured on 298 APTOS training images, coverage first
    # reaches 1.0 and the linear size first matches the geometric arm at the
    # same window, N = 1/rho = 7.04. Below it the window crops the periphery,
    # above it the retina is rendered smaller. N is fixed from the training
    # distribution alone.
    fov_in_od_units: float = 7.0
    fundus_pad_thresh: int = 10


def normalize_geometric(image_bgr: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """Scale-naive baseline: crop fundus, colour-norm, resize-pad to square."""
    cropped, _ = crop_fundus_bbox(image_bgr, cfg.fundus_pad_thresh)
    cropped = color_normalize(cropped, cfg.color_norm)
    return resize_pad_square(cropped, cfg.size, pad_value=0)


def normalize_physical_scale(
    image_bgr: np.ndarray,
    cfg: PreprocessConfig,
    od_diameter_px: Optional[float],
) -> np.ndarray:
    """OD-referenced normalization to a consistent physical scale.

    ``od_diameter_px`` is the detected optic-disc diameter (in pixels of the
    *original* image). When it is ``None`` (no detection), we gracefully fall
    back to the geometric arm so the pipeline never crashes -- such fallbacks
    are tracked by the caller for reporting.
    """
    if od_diameter_px is None or od_diameter_px <= 1:
        return normalize_geometric(image_bgr, cfg)

    # 1) Rescale so OD diameter -> target pixels.
    scale = cfg.target_od_diameter / float(od_diameter_px)
    scaled = cv2.resize(
        image_bgr, (0, 0), fx=scale, fy=scale,
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC,
    )
    # 2) Crop a canonical physical field-of-view centred on the fundus.
    center, _ = fundus_center_radius(scaled, cfg.fundus_pad_thresh)
    box = int(round(cfg.fov_in_od_units * cfg.target_od_diameter))
    window = crop_pad_centered(scaled, center, box, pad_value=0)
    # 3) Shared colour norm + final resize.
    window = color_normalize(window, cfg.color_norm)
    return resize_pad_square(window, cfg.size, pad_value=0)


def preprocess(
    image_bgr: np.ndarray,
    arm: str,
    cfg: PreprocessConfig,
    od_diameter_px: Optional[float] = None,
) -> np.ndarray:
    """Dispatch to the requested normalization arm. Returns a BGR uint8 image."""
    if arm == "geometric":
        return normalize_geometric(image_bgr, cfg)
    if arm == "physical_scale":
        return normalize_physical_scale(image_bgr, cfg, od_diameter_px)
    raise ValueError(f"Unknown normalization arm: {arm}")
