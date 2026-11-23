"""Stage 2: run the OD detector over APTOS + Aizawl -> od_diameters.json.

Produces:
- ``artifacts/cache/od_diameters.json``: {image_stem: od_diameter_px} in
  ORIGINAL-image pixels, consumed by the physical_scale arm.
- ``artifacts/reports/stage2_od_diameter_distributions.png``: per-dataset OD
  diameter histograms (evidence that physical scale differs across cohorts).
- ``artifacts/reports/stage2_od_inference_summary.json``: coverage + stats.

Run:
    python -m src.dr_grading.infer_od --size 512
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .config import PATHS, SEED
from .data import od_key
from .od_detector import UNet, predict_od_diameter
from .utils import save_json, set_seed


def _list_images(images_dir: Path, csv_path: Path, id_col: str, ext: str) -> List[Path]:
    df = pd.read_csv(csv_path)
    df[id_col] = df[id_col].astype(str).str.strip()
    paths: List[Path] = []
    for rid in df[id_col]:
        for key in (rid, f"{rid}{ext}", str(Path(rid).stem)):
            cand = images_dir / key
            if cand.exists():
                paths.append(cand)
                break
    return paths


def _run(model, paths: List[Path], size: int, device: str) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {}
    for p in tqdm(paths, desc=f"OD infer ({len(paths)})", leave=False, mininterval=30.0):
        img = cv2.imread(str(p))
        if img is None:
            out[od_key(p)] = None
            continue
        out[od_key(p)] = predict_od_diameter(model, img, size, device)
    return out


def _stats(diam: Dict[str, Optional[float]]) -> Dict:
    vals = np.array([v for v in diam.values() if v is not None], dtype=float)
    n_total = len(diam)
    n_found = len(vals)
    s = {"n_total": n_total, "n_detected": n_found,
         "coverage": n_found / n_total if n_total else 0.0}
    if n_found:
        s.update({"median_px": float(np.median(vals)), "mean_px": float(vals.mean()),
                  "std_px": float(vals.std()), "min_px": float(vals.min()),
                  "max_px": float(vals.max())})
    return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=str(PATHS.checkpoints / "od_unet_best.pt"))
    ap.add_argument("--size", type=int, default=None, help="Overrides size stored in ckpt.")
    a = ap.parse_args()

    set_seed(SEED)
    PATHS.ensure_outputs()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    state = torch.load(a.ckpt, map_location=device, weights_only=False)
    size = a.size or state.get("size", 512)
    model = UNet(base=state.get("base_channels", 32)).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    aptos_paths = _list_images(PATHS.aptos_images, PATHS.aptos_csv, "id_code", ".png")
    aizawl_paths = _list_images(PATHS.aizawl_images, PATHS.aizawl_csv, "id_code", "")
    # IDRiD grading images (second external cohort): pooled train+test.
    idrid_paths: List[Path] = []
    for img_dir, csv in ((PATHS.idrid_grade_train_images, PATHS.idrid_grade_train_csv),
                         (PATHS.idrid_grade_test_images, PATHS.idrid_grade_test_csv)):
        if Path(img_dir).exists() and Path(csv).exists():
            idrid_paths += _list_images(img_dir, csv, "Image name", ".jpg")
    print(f"APTOS images={len(aptos_paths)}  Aizawl images={len(aizawl_paths)}  "
          f"IDRiD images={len(idrid_paths)}  size={size}")

    aptos_d = _run(model, aptos_paths, size, device)
    aizawl_d = _run(model, aizawl_paths, size, device)
    idrid_d = _run(model, idrid_paths, size, device) if idrid_paths else {}

    # Single shared cache keyed by stem (APTOS hex ids, Aizawl numeric ids, and
    # IDRiD_xxx ids do not collide). Missing detections -> physical_scale falls
    # back to geometric.
    cache: Dict[str, float] = {}
    for d in (aptos_d, aizawl_d, idrid_d):
        for k, v in d.items():
            if v is not None:
                cache[k] = round(v, 2)
    cache_path = PATHS.cache / "od_diameters.json"
    save_json(cache, cache_path)

    summary = {"aptos": _stats(aptos_d), "aizawl": _stats(aizawl_d),
               "idrid": _stats(idrid_d),
               "size": size, "ckpt": a.ckpt, "cache": str(cache_path)}
    save_json(summary, PATHS.reports / "stage2_od_inference_summary.json")

    # Distribution figure (motivates physical-scale normalization).
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, d, color in [("APTOS", aptos_d, "#3b6ea5"), ("Aizawl", aizawl_d, "#b5562f")]:
        vals = [v for v in d.values() if v is not None]
        if vals:
            ax.hist(vals, bins=40, alpha=0.55, label=f"{name} (n={len(vals)})", color=color)
    ax.set_xlabel("Estimated OD diameter (original-image pixels)")
    ax.set_ylabel("count")
    ax.set_title("Optic-disc diameter distribution: APTOS vs Aizawl")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PATHS.reports / "stage2_od_diameter_distributions.png", dpi=130)
    plt.close(fig)

    print("\n" + "=" * 60)
    print(f"APTOS  : {summary['aptos']}")
    print(f"Aizawl : {summary['aizawl']}")
    print(f"Cache  -> {cache_path}  ({len(cache)} entries)")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
