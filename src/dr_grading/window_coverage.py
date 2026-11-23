"""Retinal coverage and frame fill against the canonical window.

The canonical window sets two things at once. It fixes how much of the output
frame the retina fills, which is effective resolution, and it fixes how much of
the retina is inside the window at all, which is anatomical coverage. Reporting
fill alone leaves the two entangled, so both are measured here.

Everything is computed in original-image pixels. Rescaling the image so the
optic disc spans T pixels and then cropping a window of N x T pixels is
equivalent to cropping a window of N x d pixels in the original image, where d
is the detected disc diameter, so no resampling is needed.

Run:
    python -m src.dr_grading.window_coverage
"""
from __future__ import annotations

import json
from typing import Dict, List

import cv2
import numpy as np
import pandas as pd

from .config import PATHS
from .data import load_idrid_external, load_split_frames, od_key
from .preprocessing import PreprocessConfig
from .utils import save_json

WINDOWS: List[float] = [5.0, 5.5, 6.0, 6.5, 6.7, 7.0, 7.1, 7.5, 8.0, 9.0]
N_SAMPLE = 300
SEED = 42


def _mask(path: str, thresh: int):
    img = cv2.imread(str(path))
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, m = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    return m > 0


def _mask_summary(mask: np.ndarray):
    """Row and column profiles give extent, centroid and area in one pass."""
    rows = mask.sum(axis=1).astype(np.int64)
    cols = mask.sum(axis=0).astype(np.int64)
    total = float(rows.sum())
    if total == 0:
        return None
    ry, rx = np.nonzero(rows)[0], np.nonzero(cols)[0]
    idx_y, idx_x = np.arange(mask.shape[0]), np.arange(mask.shape[1])
    cy = float((rows * idx_y).sum()) / total
    cx = float((cols * idx_x).sum()) / total
    bh = int(ry[-1] - ry[0] + 1)
    bw = int(rx[-1] - rx[0] + 1)
    return {"total": total, "cx": cx, "cy": cy, "bw": bw, "bh": bh,
            "geo_fill": total / float(max(bw, bh) ** 2)}


def _window_stats(mask: np.ndarray, cx: float, cy: float, total: float,
                  d: float, n: float) -> Dict[str, float]:
    """Coverage and fill for a window of ``n`` disc diameters."""
    half = 0.5 * n * d
    x0, x1 = int(round(cx - half)), int(round(cx + half))
    y0, y1 = int(round(cy - half)), int(round(cy + half))
    side = max(x1 - x0, 1)
    inside = mask[max(y0, 0):max(y1, 0), max(x0, 0):max(x1, 0)]
    kept = float(inside.sum()) if inside.size else 0.0
    return {"coverage": kept / total if total else float("nan"),
            "fill": kept / float(side * side)}


def run() -> Dict:
    cfg = PreprocessConfig()
    thresh = cfg.fundus_pad_thresh
    diam = json.loads((PATHS.artifacts / "cache" / "od_diameters.json").read_text())

    train, _, _, aizawl = load_split_frames(0.15, seed=SEED, test_fraction=0.15)
    cohorts = {"APTOS (train)": train, "Aizawl": aizawl, "IDRiD": load_idrid_external()}

    rows = []
    for name, frame in cohorts.items():
        f = frame.sample(min(N_SAMPLE, len(frame)), random_state=SEED)
        for path in f["path"]:
            d = diam.get(od_key(path))
            if not d or d <= 1:
                continue
            m = _mask(path, thresh)
            if m is None:
                continue
            info = _mask_summary(m)
            if info is None:
                continue
            fov = float(max(info["bw"], info["bh"]))
            rec = {"cohort": name, "rho": float(d) / fov, "geo_fill": info["geo_fill"]}
            for n in WINDOWS:
                s = _window_stats(m, info["cx"], info["cy"], info["total"],
                                  float(d), n)
                rec[f"cov_{n}"] = s["coverage"]
                rec[f"fill_{n}"] = s["fill"]
            rows.append(rec)

    df = pd.DataFrame(rows)
    df.to_csv(PATHS.reports / "window_coverage.csv", index=False)

    out: Dict = {"n_windows": WINDOWS, "cohorts": {}}
    for name, g in df.groupby("cohort"):
        geo = float(g["geo_fill"].median())
        per_n = {}
        for n in WINDOWS:
            fill = float(g[f"fill_{n}"].median())
            per_n[str(n)] = {
                "coverage": float(g[f"cov_{n}"].median()),
                "coverage_p05": float(g[f"cov_{n}"].quantile(0.05)),
                "fill": fill,
                "linear_ratio": float(np.sqrt(fill / geo)),
                "frac_cropped": float((g[f"cov_{n}"] < 0.999).mean()),
            }
        out["cohorts"][name] = {
            "n": int(len(g)),
            "rho_median": float(g["rho"].median()),
            "rho_cv": float(g["rho"].std() / g["rho"].mean()),
            "n_matched": float(1.0 / g["rho"].median()),
            "geometric_fill": geo,
            "windows": per_n,
        }
    save_json(out, PATHS.reports / "window_coverage.json")
    return out


def main() -> int:
    PATHS.ensure_outputs()
    out = run()
    for name, c in out["cohorts"].items():
        print(f"\n{name}  n={c['n']}  rho={c['rho_median']:.4f}  "
              f"1/rho={c['n_matched']:.2f}  geometric fill={c['geometric_fill']:.3f}")
        print(f"  {'N':>5} {'coverage':>9} {'cov p05':>8} {'fill':>7} "
              f"{'lin.ratio':>10} {'cropped':>8}")
        for n, w in c["windows"].items():
            print(f"  {n:>5} {w['coverage']:>9.3f} {w['coverage_p05']:>8.3f} "
                  f"{w['fill']:>7.3f} {w['linear_ratio']:>10.3f} "
                  f"{w['frac_cropped']:>7.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
