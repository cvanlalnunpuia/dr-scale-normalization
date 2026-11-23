"""Optic-disc-to-field-of-view (OD/FOV) analysis: the mechanism behind the
physical-scale result.

Physical-scale normalization rescales each image so the optic-disc (OD) diameter
maps to a fixed pixel size. It can only differ from plain geometric FOV resizing
to the extent that true OD size varies relative to the fundus field of view (FOV)
across images. This script quantifies that variation and the noise the OD
detector adds on top of it.

Two measurements:
1. Cohort OD/FOV distributions (APTOS, Aizawl, IDRiD) from the trained U-Net OD
   detector (cached diameters) + measured fundus radius.
2. Paired ground-truth-vs-detector OD/FOV on the 81 IDRiD segmentation images
   that ship GT optic-disc masks (the only images with a true OD size). This
   separates real OD-size variation from detector noise on identical images.

Outputs (no training):
- artifacts/reports/od_fov_ratios.csv      raw per-image ratios (cohort scan)
- artifacts/reports/od_fov_paired.csv      per-image GT vs detector (81 IDRiD)
- artifacts/reports/od_fov_table.{md,json}  summary stats + noise decomposition
- artifacts/reports/fig_od_fov_analysis.png the mechanism figure

Run:
    python -m src.dr_grading.od_fov_analysis
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .config import PATHS
from .data import od_key, load_idrid_external
from .od_detector import UNet, predict_od_diameter
from .preprocessing import crop_fundus_bbox, fundus_center_radius

# Predicted ratio from anatomy and camera geometry. The area-equivalent optic
# disc diameter is sqrt(1.76 * 1.92) = 1.84 mm (Jonas et al., 1988) and the
# retinal extent of a field of theta degrees is 17 mm * theta in radians, using
# the nodal-point-to-retina distance adopted in retinal morphometry.
DISC_MM = (1.76 * 1.92) ** 0.5
NODAL_MM = 17.0
def _predicted(deg: float) -> float:
    import math
    return DISC_MM / (NODAL_MM * math.radians(deg))
ANATOMICAL = _predicted(45.0)
N_PER_COHORT = 300  # deterministic sample per cohort for the distribution panel


def _fov_width(img) -> float:
    """Field-of-view width as the bounding-box extent of the fundus.

    The geometric arm crops to this bounding box and resizes it to the output
    frame, so this extent is the denominator that governs its scale. An
    enclosing circle would give a larger denominator on images whose fundus is
    clipped at the top and bottom, which most fundus cameras produce.
    """
    _, (_, _, w, h) = crop_fundus_bbox(img)
    return float(max(w, h))


def _cohort_paths():
    ap = pd.read_csv(PATHS.aptos_csv)
    ap_paths = [PATHS.aptos_images / f"{str(i).strip()}.png" for i in ap[ap.columns[0]]]
    ap_paths = [p for p in ap_paths if p.exists()]
    az = pd.read_csv(PATHS.aizawl_csv)
    az_paths = [PATHS.aizawl_images / str(i).strip() for i in az[az.columns[0]]]
    az_paths = [p for p in az_paths if p.exists()]
    idr = load_idrid_external()
    idr_paths = [Path(p) for p in idr["path"]]
    return {"APTOS": ap_paths, "Aizawl": az_paths, "IDRiD": idr_paths}


def scan_cohorts(od_cache) -> pd.DataFrame:
    rows = []
    for name, paths in _cohort_paths().items():
        paths = list(paths)
        rng = np.random.default_rng(0)
        rng.shuffle(paths)
        n = 0
        for p in paths:
            d = od_cache.get(od_key(p))
            if not d or d <= 1:
                continue
            img = cv2.imread(str(p))
            if img is None:
                continue
            fov = _fov_width(img)
            if fov > 0:
                rows.append({"cohort": name, "od_fov": d / fov})
                n += 1
            if n >= N_PER_COHORT:
                break
    return pd.DataFrame(rows)


def paired_idrid_gt(device) -> pd.DataFrame:
    """GT vs detector OD/FOV on the 81 IDRiD segmentation images."""
    mask_dirs = glob.glob("data/raw/IDRID/A. Segmentation/A. Segmentation/"
                          "2. All Segmentation Groundtruths/*/5. Optic Disc")
    img_dirs = glob.glob("data/raw/IDRID/A. Segmentation/A. Segmentation/"
                         "1. Original Images/*")
    orig = {}
    for d in img_dirs:
        for p in glob.glob(os.path.join(d, "*.jpg")):
            orig[os.path.splitext(os.path.basename(p))[0]] = p

    ck = torch.load(PATHS.checkpoints / "od_unet_best.pt", map_location=device,
                    weights_only=False)
    size = ck.get("size", 512)
    model = UNet(base=ck.get("base_channels", 32)).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    rows = []
    for d in mask_dirs:
        for m in glob.glob(os.path.join(d, "*_OD.tif")):
            stem = os.path.basename(m).replace("_OD.tif", "")
            if stem not in orig:
                continue
            mask = cv2.imread(m, cv2.IMREAD_GRAYSCALE)
            area = int((mask > 0).sum())
            if area < 50:
                continue
            gt_d = 2.0 * np.sqrt(area / np.pi)
            img = cv2.imread(orig[stem])
            fov = _fov_width(img)
            if fov <= 0:
                continue
            det_d = predict_od_diameter(model, img, size, device)
            rows.append({
                "stem": stem, "fov": fov,
                "gt_od_fov": gt_d / fov,
                "det_od_fov": (det_d / fov) if det_d else np.nan,
            })
    return pd.DataFrame(rows)


def _cv(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(x.std() / x.mean()) if len(x) and x.mean() else float("nan")


def build_table(scan: pd.DataFrame, paired: pd.DataFrame) -> dict:
    out = {"anatomical_od_fov": ANATOMICAL, "cohorts": {}}
    for name, g in scan.groupby("cohort"):
        r = g["od_fov"].to_numpy()
        out["cohorts"][name] = {
            "n": int(len(r)), "median": float(np.median(r)), "mean": float(r.mean()),
            "cv": _cv(r), "iqr_lo": float(np.percentile(r, 25)),
            "iqr_hi": float(np.percentile(r, 75)),
        }
    gt = paired["gt_od_fov"].to_numpy()
    det = paired["det_od_fov"].to_numpy()
    mask = np.isfinite(gt) & np.isfinite(det)
    gt, det = gt[mask], det[mask]
    cv_gt, cv_det = _cv(gt), _cv(det)
    # Detector-induced scale error (per image), relative to GT OD size.
    rel_err = np.abs(det - gt) / gt
    out["idrid_paired"] = {
        "n": int(len(gt)),
        "gt_median": float(np.median(gt)), "gt_cv": cv_gt,
        "det_median": float(np.median(det)), "det_cv": cv_det,
        "cv_from_detector_noise": float(np.sqrt(max(cv_det**2 - cv_gt**2, 0.0))),
        "median_abs_rel_scale_error": float(np.median(rel_err)),
        "frac_within_10pct": float((rel_err < 0.10).mean()),
        "frac_gross_error_gt30pct": float((rel_err > 0.30).mean()),
    }
    return out


def write_table(tab: dict) -> Path:
    md = []
    md.append("# Optic-disc-to-field-of-view (OD/FOV) analysis\n")
    md.append(f"Anatomical prior: OD diameter is about {tab['anatomical_od_fov']:.3f} "
              "of the fundus field-of-view width for 45-degree fundus cameras.\n")
    md.append("## OD/FOV ratio by cohort (U-Net detector)\n")
    md.append("| cohort | n | median | IQR | CV |")
    md.append("|---|---|---|---|---|")
    for name, c in tab["cohorts"].items():
        md.append(f"| {name} | {c['n']} | {c['median']:.3f} | "
                  f"[{c['iqr_lo']:.3f}, {c['iqr_hi']:.3f}] | {c['cv']:.2f} |")
    p = tab["idrid_paired"]
    md.append("\n## IDRiD paired ground-truth vs detector (81 segmentation images)\n")
    md.append("| source | median OD/FOV | CV |")
    md.append("|---|---|---|")
    md.append(f"| ground truth | {p['gt_median']:.3f} | {p['gt_cv']:.2f} |")
    md.append(f"| U-Net detector | {p['det_median']:.3f} | {p['det_cv']:.2f} |")
    md.append(f"\nTrue OD size varies little relative to FOV (ground-truth CV "
              f"{p['gt_cv']:.2f}). The detector inflates the apparent variation to "
              f"CV {p['det_cv']:.2f}, so about CV {p['cv_from_detector_noise']:.2f} "
              f"is detector noise. The detector is accurate for most images "
              f"({100*p['frac_within_10pct']:.0f}% within 10% of the true OD size, "
              f"median error {100*p['median_abs_rel_scale_error']:.0f}%), and grossly "
              f"wrong for a minority ({100*p['frac_gross_error_gt30pct']:.0f}% with a "
              f"scale error above 30%). That heavy tail of bad detections is what "
              f"injects scale noise into the physical-scale arm.\n")
    md.append("Because OD size is a near-constant fraction of the FOV, rescaling to "
              "a fixed OD size is close to geometric FOV resizing. The physical-scale "
              "arm mainly adds detector noise, which is consistent with its small "
              "cross-dataset degradation.\n")
    path = PATHS.reports / "od_fov_table.md"
    path.write_text("\n".join(md), encoding="utf-8")
    json.dump(tab, open(PATHS.reports / "od_fov_table.json", "w"), indent=2)
    return path


def build_figure(scan: pd.DataFrame, paired: pd.DataFrame, tab: dict) -> Path:
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.6))

    # Panel A: OD/FOV distribution per cohort (detector) + IDRiD GT.
    order = ["APTOS", "Aizawl", "IDRiD"]
    data = [scan.loc[scan.cohort == c, "od_fov"].to_numpy() for c in order]
    gt = paired["gt_od_fov"].to_numpy()
    gt = gt[np.isfinite(gt)]
    data.append(gt)
    labels = ["APTOS\n(det)", "Aizawl\n(det)", "IDRiD\n(det)", "IDRiD\n(GT)"]
    colors = ["#4C72B0", "#4C72B0", "#4C72B0", "#55A868"]
    parts = axA.boxplot(data, labels=labels, patch_artist=True, showfliers=False,
                        widths=0.6, medianprops=dict(color="black"))
    for patch, col in zip(parts["boxes"], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.75)
    axA.axhspan(_predicted(50.0), _predicted(40.0), color="#C44E52", alpha=0.12,
                label="predicted, 40 to 50 degree field")
    axA.axhline(ANATOMICAL, color="#C44E52", ls="--", lw=1.3,
                label=f"predicted, 45 degrees ({ANATOMICAL:.3f})")
    axA.set_ylabel("OD diameter / FOV width")
    axA.set_title("Optic-disc size is a near-constant fraction of the field of view")
    axA.set_ylim(0, max(0.32, float(np.nanpercentile(np.concatenate(data), 99))))
    axA.legend(loc="upper right", fontsize=9, framealpha=0.9)
    axA.grid(axis="y", alpha=0.3)

    # Panel B: paired GT vs detector on the same 81 IDRiD images.
    d = paired.dropna(subset=["gt_od_fov", "det_od_fov"])
    axB.scatter(d["gt_od_fov"], d["det_od_fov"], s=22, alpha=0.7,
                color="#4C72B0", edgecolor="black", linewidth=0.3)
    lo, hi = 0.08, 0.30
    axB.plot([lo, hi], [lo, hi], color="#555555", ls="--", lw=1.2, label="y = x")
    axB.axvline(np.median(d["gt_od_fov"]), color="#55A868", lw=1.0, alpha=0.7)
    axB.set_xlim(lo, hi)
    axB.set_ylim(lo, hi)
    axB.set_xlabel("Ground-truth OD / FOV")
    axB.set_ylabel("Detector OD / FOV")
    p = tab["idrid_paired"]
    axB.set_title(f"Detector adds scale noise (GT CV {p['gt_cv']:.2f} to "
                  f"detector CV {p['det_cv']:.2f})")
    axB.legend(loc="upper left", fontsize=9, framealpha=0.9)
    axB.grid(alpha=0.3)

    fig.tight_layout()
    path = PATHS.reports / "fig_od_fov_analysis.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main() -> int:
    PATHS.ensure_outputs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    od_cache = json.load(open(PATHS.cache / "od_diameters.json"))

    scan = scan_cohorts(od_cache)
    scan.to_csv(PATHS.reports / "od_fov_ratios.csv", index=False)
    print(f"[scan] {len(scan)} images across cohorts")

    paired = paired_idrid_gt(device)
    paired.to_csv(PATHS.reports / "od_fov_paired.csv", index=False)
    print(f"[paired] {len(paired)} IDRiD GT images")

    tab = build_table(scan, paired)
    tpath = write_table(tab)
    fpath = build_figure(scan, paired, tab)
    print(f"[table] {tpath}")
    print(f"[figure] {fpath}")
    print("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
