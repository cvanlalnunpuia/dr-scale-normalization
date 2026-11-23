"""Method figure: the two scaling arms as a flow, with image evidence.

Each arm is a box with a header band and numbered steps, so the heading is
clearly a label and the steps below it are clearly the operations. Connectors are
stepped rather than straight, and the branch is symmetric about the shared row.
Photographs appear only where a label cannot substitute: the optic disc being
measured, and the network inputs, where the window choice is plainly visible.

Run:
    python -m src.dr_grading.make_pipeline_figure
"""
from __future__ import annotations

import glob
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

from .config import PATHS
from .preprocessing import (PreprocessConfig, color_normalize, crop_fundus_bbox,
                            crop_pad_centered, fundus_center_radius,
                            resize_pad_square)
from .od_fov_analysis import _fov_width

SHARED, GEO, PHY, BAD, GREY = "#2F5C8F", "#2E7D46", "#B03A3E", "#8C2F32", "#6E6E6E"
MID = 0.500                      # vertical centre of the shared row
IW, IH = 0.126, 0.252            # square panels on a 2:1 canvas


def _annotated_image(target_rho=1.0 / 7.0):
    """The annotated IDRiD image whose rho is closest to ``target_rho``."""
    md = glob.glob("data/raw/IDRID/A. Segmentation/A. Segmentation/"
                   "2. All Segmentation Groundtruths/*/5. Optic Disc")
    idirs = glob.glob("data/raw/IDRID/A. Segmentation/A. Segmentation/"
                      "1. Original Images/*")
    orig = {}
    for d in idirs:
        for p in glob.glob(os.path.join(d, "*.jpg")):
            orig[os.path.splitext(os.path.basename(p))[0]] = p
    best = None
    for d in sorted(md):
        for m in sorted(glob.glob(os.path.join(d, "*_OD.tif"))):
            stem = os.path.basename(m).replace("_OD.tif", "")
            if stem not in orig:
                continue
            mask = cv2.imread(m, cv2.IMREAD_GRAYSCALE)
            area = int((mask > 0).sum())
            if area < 50:
                continue
            img = cv2.imread(orig[stem])
            fov = _fov_width(img)
            if fov <= 0:
                continue
            diam = 2.0 * np.sqrt(area / np.pi)
            ys, xs = np.nonzero(mask > 0)
            cand = (abs(diam / fov - target_rho), img, diam,
                    (float(xs.mean()), float(ys.mean())), stem)
            if best is None or cand[0] < best[0]:
                best = cand
    if best is None:
        raise RuntimeError("no annotated IDRiD image found")
    return best[1], best[2], best[3], best[4]


def _square(bgr):
    h, w = bgr.shape[:2]
    side = max(h, w)
    c = np.zeros((side, side, 3), dtype=bgr.dtype)
    y0, x0 = (side - h) // 2, (side - w) // 2
    c[y0:y0 + h, x0:x0 + w] = bgr
    return c, x0, y0


def _physical(img, cfg, d):
    sc = cfg.target_od_diameter / float(d)
    scaled = cv2.resize(img, (0, 0), fx=sc, fy=sc,
                        interpolation=cv2.INTER_AREA if sc < 1 else cv2.INTER_CUBIC)
    centre, _ = fundus_center_radius(scaled, cfg.fundus_pad_thresh)
    box = int(round(cfg.fov_in_od_units * cfg.target_od_diameter))
    win = crop_pad_centered(scaled, centre, box, pad_value=0)
    return resize_pad_square(color_normalize(win, cfg.color_norm), cfg.size, 0)


def _fill(bgr) -> float:
    return float((cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) > 10).mean())


def build(path):
    img, d, (cx, cy), stem = _annotated_image()
    c7 = PreprocessConfig(size=224, fov_in_od_units=7.0)
    c9 = PreprocessConfig(size=224, fov_in_od_units=9.0)
    cropped, (bx, by, _, _) = crop_fundus_bbox(img, c7.fundus_pad_thresh)
    D = _fov_width(img)
    rho = d / D
    geo = resize_pad_square(color_normalize(cropped, c7.color_norm), c7.size, 0)
    phy7, phy9 = _physical(img, c7, d), _physical(img, c9, d)

    fig = plt.figure(figsize=(13.4, 7.0))

    def panel(x, y, im, title, sub, colour, circle=None):
        ax = fig.add_axes([x, y, IW, IH]); ax.set_zorder(3)
        sq, ox, oy = _square(im)
        ax.imshow(cv2.cvtColor(sq, cv2.COLOR_BGR2RGB))
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(colour); sp.set_linewidth(1.2)
        ax.set_title(title, fontsize=9.2, pad=4, color=colour)
        if sub:
            ax.set_xlabel(sub, fontsize=9, labelpad=3, color=colour)
        if circle:
            ccx, ccy, cr = circle
            ax.add_patch(Circle((ccx + ox, ccy + oy), cr, fill=False,
                                edgecolor="#FFC61E", linewidth=1.8))

    def armbox(x, y, w, h, header, steps, colour):
        """Header band on top, numbered steps beneath, so the two read apart."""
        fig.add_artist(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004",
                                      transform=fig.transFigure, zorder=1,
                                      facecolor="white", edgecolor=colour,
                                      linewidth=1.5))
        hb = 0.052
        fig.add_artist(Rectangle((x, y + h - hb), w, hb, transform=fig.transFigure,
                                 zorder=2, facecolor=colour, edgecolor="none"))
        fig.text(x + w / 2, y + h - hb / 2, header, ha="center", va="center",
                 fontsize=10.2, color="white", weight="bold", zorder=4)
        for k, (num, txt) in enumerate(steps):
            yy = y + h - hb - 0.048 - k * 0.052
            fig.text(x + 0.016, yy, num, ha="left", va="center", fontsize=8.6,
                     color="white", zorder=5,
                     bbox=dict(boxstyle="circle,pad=0.22", fc=colour, ec="none"))
            fig.text(x + 0.046, yy, txt, ha="left", va="center", fontsize=9.2,
                     color="#222222", zorder=4)

    def step_arrow(x1, y1, x2, y2, colour=GREY, rad=0.0):
        """Stepped connector: leaves horizontally, turns once, arrives horizontally."""
        # angleA=90 leaves the shared row vertically, angleB=0 arrives horizontally
        # into the left edge of the target box.
        style = "arc3,rad=0" if y1 == y2 else f"angle,angleA=90,angleB=0,rad={rad}"
        fig.add_artist(FancyArrowPatch(
            (x1, y1), (x2, y2), transform=fig.transFigure, arrowstyle="-|>",
            mutation_scale=12, linewidth=1.5, color=colour, zorder=2,
            connectionstyle=style))

    # Shared stages, centred on MID.
    py = MID - IH / 2
    panel(0.026, py, img, "Fundus photograph", None, SHARED)
    step_arrow(0.156, MID, 0.180, MID)
    panel(0.184, py, cropped, "Crop fundus from border",
          f"$D$ = {D:.0f} px,  $d$ = {d:.0f} px", SHARED,
          circle=(cx - bx, cy - by, d / 2.0))
    fig.text(0.247, MID - IH / 2 - 0.062, f"$\\rho = d/D = {rho:.3f}$",
             ha="center", fontsize=9.8, color=SHARED, weight="bold")

    # Symmetric branch out of the shared row.
    GY, PY_ = MID + 0.205, MID - 0.205        # arm-box centres, equal offsets
    fig.add_artist(plt.Line2D([0.316, 0.338], [MID, MID], transform=fig.transFigure,
                              color=GREY, linewidth=1.5, zorder=2))
    step_arrow(0.338, MID, 0.362, GY, GEO, rad=8)
    step_arrow(0.338, MID, 0.362, PY_, PHY, rad=8)

    BW, BH = 0.238, 0.186
    armbox(0.366, GY - BH / 2, BW, BH, "Geometric arm",
           [("1", "resize field of view to $S \\times S$"),
            ("2", "scale factor $s_{\\mathrm{geo}} = S/D$")], GEO)
    armbox(0.366, PY_ - BH / 2, BW, BH, "Physical-scale arm",
           [("1", "rescale so $d \\rightarrow T$,  $s_{\\mathrm{phy}} = T/d$"),
            ("2", "crop window of $N \\times T$")], PHY)

    # Into the network inputs, symmetric.
    step_arrow(0.608, GY, 0.646, GY, GEO)
    step_arrow(0.608, PY_, 0.646, PY_, PHY)
    panel(0.650, GY - IH / 2, geo, "Network input",
          f"fills {100*_fill(geo):.0f}%", GEO)
    panel(0.650, PY_ - IH / 2, phy7, "Network input, $N = 7$",
          f"matched, fills {100*_fill(phy7):.0f}%", PHY)

    # The wider window, joined to the same arm rather than left floating.
    step_arrow(0.780, PY_, 0.854, PY_, BAD)
    panel(0.858, PY_ - IH / 2, phy9, "Network input, $N = 9$",
          f"fills {100*_fill(phy9):.0f}%", BAD)
    fig.text(0.817, PY_ + 0.026, "larger\nwindow", ha="center", va="bottom",
             fontsize=8.4, color=BAD, linespacing=1.25)

    fig.savefig(path, dpi=240, facecolor="white", bbox_inches="tight",
                pad_inches=0.06)
    plt.close(fig)
    return stem, _fill(geo), _fill(phy7), _fill(phy9)


def main() -> int:
    PATHS.ensure_outputs()
    p = PATHS.reports / "fig_pipeline.png"
    stem, g, p6, p9 = build(p)
    print(f"[fig] {p} ({stem})  fill: geo {100*g:.0f}%, N=7 {100*p6:.0f}%, "
          f"N=9 {100*p9:.0f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
