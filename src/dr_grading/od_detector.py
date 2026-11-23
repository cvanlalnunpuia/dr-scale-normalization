"""Stage 2: optic-disc (OD) detector.

A compact U-Net segments the optic disc. Trained on REFUGE2 and validated
externally on IDRiD, it is then run over APTOS + Aizawl to estimate each image's
OD diameter (in original-image pixels). Those diameters drive the
``physical_scale`` normalization arm.

Design notes:
- Images/masks are **letterboxed** (aspect-preserving resize + pad) to a square
  so a single uniform scale factor maps detector pixels back to the original
  image -- essential for a correct OD-diameter in original coordinates.
- REFUGE2 mask convention: 0 = cup, 128 = disc rim, 255 = background, so the
  full optic disc is ``mask <= 128``. IDRiD OD masks are binary (>0 = disc).
- OD diameter is the area-equivalent diameter of the largest predicted
  component (robust to small spurious blobs).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from .config import IMAGENET_MEAN, IMAGENET_STD
from .preprocessing import fundus_center_radius, resize_pad_square


# ---------------------------------------------------------------------------
# Letterbox helpers (return the uniform scale so we can map back to original)
# ---------------------------------------------------------------------------
def letterbox(image: np.ndarray, size: int, interp: int = cv2.INTER_AREA) -> Tuple[np.ndarray, float]:
    """Aspect-preserving resize+pad to ``size`` square. Returns (canvas, scale)."""
    h, w = image.shape[:2]
    scale = size / max(h, w)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(image, (nw, nh), interpolation=interp)
    if image.ndim == 3:
        canvas = np.zeros((size, size, image.shape[2]), dtype=image.dtype)
    else:
        canvas = np.zeros((size, size), dtype=image.dtype)
    y0, x0 = (size - nh) // 2, (size - nw) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return canvas, scale


def _normalize_img(rgb: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float().div_(255.0)
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (t - mean) / std


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
class RefugeODDataset(Dataset):
    """REFUGE2 optic-disc segmentation split (images/ + mask/)."""

    def __init__(self, split_dir: Path, size: int = 512, train: bool = False) -> None:
        self.size = size
        self.train = train
        img_dir = Path(split_dir) / "images"
        msk_dir = Path(split_dir) / "mask"
        masks = {p.stem: p for p in msk_dir.iterdir() if p.is_file()}
        self.pairs: List[Tuple[Path, Path]] = []
        for ip in sorted(img_dir.iterdir()):
            if ip.stem in masks:
                self.pairs.append((ip, masks[ip.stem]))

    def __len__(self) -> int:
        return len(self.pairs)

    def _augment(self, img: np.ndarray, msk: np.ndarray, rng) -> Tuple[np.ndarray, np.ndarray]:
        if rng.random() < 0.5:
            img, msk = img[:, ::-1], msk[:, ::-1]
        if rng.random() < 0.5:
            img, msk = img[::-1], msk[::-1]
        if rng.random() < 0.5:
            angle = float(rng.uniform(-20, 20))
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR)
            msk = cv2.warpAffine(msk, M, (w, h), flags=cv2.INTER_NEAREST)
        return np.ascontiguousarray(img), np.ascontiguousarray(msk)

    def __getitem__(self, idx: int):
        ip, mp = self.pairs[idx]
        bgr = cv2.imread(str(ip))
        raw_mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        od = (raw_mask <= 128).astype(np.uint8)  # disc = cup + rim
        rgb, _ = letterbox(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), self.size)
        msk, _ = letterbox(od, self.size, interp=cv2.INTER_NEAREST)
        if self.train:
            rgb, msk = self._augment(rgb, msk, np.random.default_rng())
        return _normalize_img(rgb), torch.from_numpy(msk[None].astype(np.float32))


class IdridODDataset(Dataset):
    """IDRiD optic-disc masks for external detector validation."""

    def __init__(self, images_dir: Path, masks_dir: Path, size: int = 512) -> None:
        self.size = size
        masks = {p.stem.replace("_OD", ""): p for p in Path(masks_dir).iterdir() if p.is_file()}
        self.pairs: List[Tuple[Path, Path]] = []
        for ip in sorted(Path(images_dir).iterdir()):
            if ip.stem in masks:
                self.pairs.append((ip, masks[ip.stem]))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        ip, mp = self.pairs[idx]
        bgr = cv2.imread(str(ip))
        raw_mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        od = (raw_mask > 0).astype(np.uint8)
        rgb, _ = letterbox(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), self.size)
        msk, _ = letterbox(od, self.size, interp=cv2.INTER_NEAREST)
        return _normalize_img(rgb), torch.from_numpy(msk[None].astype(np.float32))


# ---------------------------------------------------------------------------
# Compact U-Net
# ---------------------------------------------------------------------------
class _DoubleConv(nn.Module):
    def __init__(self, ci: int, co: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ci, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
            nn.Conv2d(co, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_ch: int = 3, base: int = 32) -> None:
        super().__init__()
        c = [base, base * 2, base * 4, base * 8]
        self.inc = _DoubleConv(in_ch, c[0])
        self.d1, self.e1 = nn.MaxPool2d(2), _DoubleConv(c[0], c[1])
        self.d2, self.e2 = nn.MaxPool2d(2), _DoubleConv(c[1], c[2])
        self.d3, self.e3 = nn.MaxPool2d(2), _DoubleConv(c[2], c[3])
        self.u3 = _DoubleConv(c[3] + c[2], c[2])
        self.u2 = _DoubleConv(c[2] + c[1], c[1])
        self.u1 = _DoubleConv(c[1] + c[0], c[0])
        self.outc = nn.Conv2d(c[0], 1, 1)

    @staticmethod
    def _up(x, ref):
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x):
        x0 = self.inc(x)
        x1 = self.e1(self.d1(x0))
        x2 = self.e2(self.d2(x1))
        x3 = self.e3(self.d3(x2))
        y = self.u3(torch.cat([self._up(x3, x2), x2], 1))
        y = self.u2(torch.cat([self._up(y, x1), x1], 1))
        y = self.u1(torch.cat([self._up(y, x0), x0], 1))
        return self.outc(y)


# ---------------------------------------------------------------------------
# Losses + metrics + diameter estimation
# ---------------------------------------------------------------------------
def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    num = 2 * (prob * target).sum(dim=(1, 2, 3)) + eps
    den = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps
    return (1 - num / den).mean()


def bce_dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, target) + dice_loss(logits, target)


@torch.no_grad()
def dice_score(logits: torch.Tensor, target: torch.Tensor, thresh: float = 0.5) -> float:
    prob = (torch.sigmoid(logits) > thresh).float()
    num = 2 * (prob * target).sum(dim=(1, 2, 3))
    den = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return ((num + 1e-6) / (den + 1e-6)).mean().item()


def largest_component_diameter(prob_mask: np.ndarray, thresh: float = 0.5) -> Optional[float]:
    """Area-equivalent diameter (px) of the largest blob in a probability mask."""
    binary = (prob_mask > thresh).astype(np.uint8)
    if binary.sum() == 0:
        return None
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    area = float(areas.max())
    return float(2.0 * np.sqrt(area / np.pi))


@torch.no_grad()
def predict_od_diameter(
    model: nn.Module, image_bgr: np.ndarray, size: int, device: str, thresh: float = 0.5,
    min_od_frac: float = 0.04, max_od_frac: float = 0.30,
) -> Optional[float]:
    """Return the OD diameter in ORIGINAL-image pixels, or None if not found.

    A plausibility check rejects detections whose diameter is implausible
    relative to *this image's own* fundus diameter (the optic disc is ~6-22% of
    the fundus width). This is scale-invariant, so it filters detector failures
    (tiny blobs / whole-fundus blobs) without discarding genuine resolution
    differences between cohorts -- returning None triggers a geometric fallback.
    """
    rgb, scale = letterbox(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), size)
    x = _normalize_img(rgb).unsqueeze(0).to(device)
    prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()
    d_letterbox = largest_component_diameter(prob, thresh)
    if d_letterbox is None:
        return None
    d_orig = d_letterbox / scale  # map back to original-image pixels
    _, fundus_r = fundus_center_radius(image_bgr)
    if fundus_r > 0:
        frac = d_orig / (2.0 * fundus_r)
        if frac < min_od_frac or frac > max_od_frac:
            return None  # implausible -> treat as no detection
    return d_orig
