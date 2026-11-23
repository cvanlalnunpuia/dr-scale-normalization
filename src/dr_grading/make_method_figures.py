"""Method and agreement figures for the scale-normalization paper.

Produces three figures that no training is needed for:

- ``fig_pipeline_schematic.png``  schematic of the two scaling arms, showing
  which stages are shared and where the arms diverge. The paper uses the
  image-bearing version from ``make_pipeline_figure`` instead.
- ``fig_example_arms.png``  one real fundus carried through both arms, with the
  optic disc marked, so the near-equivalence of the outputs is visible.
- ``fig_bland_altman.png``  Bland-Altman agreement between the detector and
  ground-truth optic-disc diameter on the annotated IDRiD images.

Run:
    python -m src.dr_grading.make_method_figures
"""
from __future__ import annotations

import glob
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

from .config import PATHS
from .preprocessing import (PreprocessConfig, crop_fundus_bbox,
                            fundus_center_radius, normalize_geometric,
                            normalize_physical_scale)

SHARED = "#4C72B0"
GEO = "#55A868"
PHY = "#C44E52"


# --------------------------------------------------------------------------
# Figure 1: pipeline schematic
# --------------------------------------------------------------------------
def _box(ax, x, y, w, h, text, color, fontsize=8.5, alpha=0.16):
    # Small pad: the pad extends the patch beyond (x, y, w, h), so a large pad
    # silently makes neighbouring boxes overlap.
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.005",
                                linewidth=1.1, edgecolor=color,
                                facecolor=color, alpha=alpha))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color="#1a1a1a", linespacing=1.35)


def _arrow(ax, x1, y1, x2, y2, color="#666666"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=11, linewidth=1.0, color=color))


def pipeline_figure(path):
    fig, ax = plt.subplots(figsize=(11.4, 4.5))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    h = 0.20
    # Shared input stage
    _box(ax, 0.020, 0.40, 0.130, h, "Fundus\nphotograph", SHARED)
    _box(ax, 0.175, 0.40, 0.130, h, "Crop fundus\nfrom border", SHARED)
    _arrow(ax, 0.153, 0.50, 0.172, 0.50)

    # Split to the two arms
    _arrow(ax, 0.310, 0.57, 0.356, 0.73)
    _arrow(ax, 0.310, 0.43, 0.356, 0.27)

    # Geometric branch
    _box(ax, 0.360, 0.66, 0.200, h,
         "Resize field of view\nto $S \\times S$, pad", GEO)
    ax.text(0.460, 0.905, "Geometric arm", ha="center", va="center",
            fontsize=9.5, color=GEO, weight="bold")
    ax.text(0.460, 0.617, "$s_{\\mathrm{geo}} = S/D$", ha="center",
            va="center", fontsize=9)

    # Physical branch, with a real gap between the two boxes
    _box(ax, 0.360, 0.14, 0.155, h,
         "Detect optic disc,\nrescale $d \\rightarrow T$ px", PHY)
    _box(ax, 0.545, 0.14, 0.135, h, "Crop canonical\nwindow", PHY)
    _arrow(ax, 0.522, 0.24, 0.539, 0.24)
    ax.text(0.460, 0.385, "Physical-scale arm", ha="center", va="center",
            fontsize=9.5, color=PHY, weight="bold")
    ax.text(0.460, 0.097, "$s_{\\mathrm{phy}} = T/d$", ha="center",
            va="center", fontsize=9)

    # Rejoin
    _arrow(ax, 0.565, 0.73, 0.742, 0.58)
    _arrow(ax, 0.685, 0.25, 0.742, 0.42)

    _box(ax, 0.745, 0.40, 0.145, h, "CLAHE, resize\nto $224^2$", SHARED)
    _box(ax, 0.915, 0.40, 0.065, h, "Grade", SHARED, fontsize=8)
    _arrow(ax, 0.893, 0.50, 0.912, 0.50)

    ax.text(0.50, 0.985, "Both arms share every stage except the rescaling step",
            ha="center", va="center", fontsize=10)
    ax.text(0.818, 0.345, "shared", ha="center", fontsize=8.5, color=SHARED)
    ax.text(0.085, 0.345, "shared", ha="center", fontsize=8.5, color=SHARED)

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# Figure 2: one fundus through both arms
# --------------------------------------------------------------------------
def _idrid_annotated():
    """Return (image_bgr, gt_diameter_px) for one annotated IDRiD image."""
    mask_dirs = glob.glob("data/raw/IDRID/A. Segmentation/A. Segmentation/"
                          "2. All Segmentation Groundtruths/*/5. Optic Disc")
    img_dirs = glob.glob("data/raw/IDRID/A. Segmentation/A. Segmentation/"
                         "1. Original Images/*")
    orig = {}
    for d in img_dirs:
        for p in glob.glob(os.path.join(d, "*.jpg")):
            orig[os.path.splitext(os.path.basename(p))[0]] = p
    for d in sorted(mask_dirs):
        for m in sorted(glob.glob(os.path.join(d, "*_OD.tif"))):
            stem = os.path.basename(m).replace("_OD.tif", "")
            if stem not in orig:
                continue
            mask = cv2.imread(m, cv2.IMREAD_GRAYSCALE)
            area = int((mask > 0).sum())
            if area < 50:
                continue
            ys, xs = np.nonzero(mask > 0)
            cx, cy = float(xs.mean()), float(ys.mean())
            return cv2.imread(orig[stem]), 2.0 * np.sqrt(area / np.pi), (cx, cy), stem
    raise RuntimeError("no annotated IDRiD image found")


def example_figure(path):
    img, gt_d, (cx, cy), stem = _idrid_annotated()
    cfg = PreprocessConfig(size=224)
    cropped, (bx, by, bw, bh) = crop_fundus_bbox(img, cfg.fundus_pad_thresh)
    (_, _), r = fundus_center_radius(img)
    D = 2.0 * r
    rho = gt_d / D

    geo = normalize_geometric(img, cfg)
    phy = normalize_physical_scale(img, cfg, gt_d)

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 4.1))
    # Panel A: cropped fundus with the optic disc marked
    axA = axes[0]
    axA.imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
    axA.add_patch(Circle((cx - bx, cy - by), gt_d / 2.0, fill=False,
                         edgecolor="#FFD24A", linewidth=2.0))
    axA.set_title(f"Cropped fundus\n$D$ = {D:.0f} px, $d$ = {gt_d:.0f} px, "
                  f"$\\rho$ = {rho:.3f}", fontsize=9.5)
    axA.axis("off")

    for ax, im, ttl, col in ((axes[1], geo, "Geometric arm", GEO),
                             (axes[2], phy, "Physical-scale arm", PHY)):
        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        ax.set_title(ttl, fontsize=10.5, color=col)
        ax.axis("off")

    # No in-figure title: the caption carries the description.
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return stem


# --------------------------------------------------------------------------
# Figure 3: Bland-Altman agreement
# --------------------------------------------------------------------------
def bland_altman_figure(path):
    d = pd.read_csv(PATHS.reports / "od_fov_paired.csv")
    d = d.dropna(subset=["gt_od_fov", "det_od_fov"])
    gt = d["gt_od_fov"].to_numpy() * d["fov"].to_numpy()
    det = d["det_od_fov"].to_numpy() * d["fov"].to_numpy()
    mean = (gt + det) / 2.0
    diff_pct = 100.0 * (det - gt) / gt
    bias = diff_pct.mean()
    sd = diff_pct.std(ddof=1)
    lo, hi = bias - 1.96 * sd, bias + 1.96 * sd

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.scatter(mean, diff_pct, s=26, alpha=0.75, color=SHARED,
               edgecolor="black", linewidth=0.3)
    ax.axhline(0, color="#888888", lw=1.0, ls=":")
    ax.axhline(bias, color=PHY, lw=1.4,
               label=f"bias {bias:+.1f}%")
    ax.axhline(hi, color=PHY, lw=1.0, ls="--",
               label=f"95% limits {lo:+.0f} to {hi:+.0f}%")
    ax.axhline(lo, color=PHY, lw=1.0, ls="--")
    ax.set_xlabel("Mean of ground-truth and detector optic-disc diameter (px)")
    ax.set_ylabel("Detector minus ground truth (% of ground truth)")
    ax.set_title("Detector agreement with ground-truth optic-disc diameter")
    # Headroom so no point collides with the legend.
    ax.set_ylim(min(lo, diff_pct.min()) - 12, max(hi, diff_pct.max()) + 26)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return bias, lo, hi


def main() -> int:
    PATHS.ensure_outputs()
    p1 = PATHS.reports / "fig_pipeline_schematic.png"
    p2 = PATHS.reports / "fig_example_arms.png"
    p3 = PATHS.reports / "fig_bland_altman.png"
    pipeline_figure(p1)
    print(f"[fig] {p1}")
    stem = example_figure(p2)
    print(f"[fig] {p2}  (image {stem})")
    bias, lo, hi = bland_altman_figure(p3)
    print(f"[fig] {p3}  bias {bias:+.1f}%, limits {lo:+.1f} to {hi:+.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
