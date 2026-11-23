"""Test-time scale-perturbation sensitivity of a trained grader.

The physical-scale arm rescales each image by a factor derived from an estimated
optic-disc diameter. When that estimate is wrong by a relative amount, the image
reaches the network at the wrong scale. This script measures how much external
agreement a trained geometric-arm model loses as a function of a known,
deliberately injected relative scale error, so the mechanism proposed for the
physical arm can be checked against a dose-response curve.

The perturbation is applied at evaluation time only. The model is not retrained,
so the curve isolates the sensitivity of a fixed grader to input scale.

Outputs:
- artifacts/reports/scale_jitter.csv   QWK per cohort per perturbation
- artifacts/reports/fig_scale_jitter.png  the dose-response curve

Run:
    python -m src.dr_grading.scale_jitter --ckpt <path to a geometric checkpoint>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .config import IMAGENET_MEAN, IMAGENET_STD, PATHS, TrainConfig
from .data import load_idrid_external, load_split_frames
from .metrics import quadratic_weighted_kappa as qwk
from .model import create_backbone
from .preprocessing import (PreprocessConfig, color_normalize,
                            crop_fundus_bbox, resize_pad_square)

DELTAS = [-0.30, -0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20, 0.30]


def _center_fit(img: np.ndarray, size: int) -> np.ndarray:
    """Center-crop or center-pad a square image to ``size``."""
    h = img.shape[0]
    if h == size:
        return img
    if h > size:
        o = (h - size) // 2
        return img[o:o + size, o:o + size]
    canvas = np.zeros((size, size, img.shape[2]), dtype=img.dtype)
    o = (size - h) // 2
    canvas[o:o + h, o:o + h] = img
    return canvas


def geometric_jitter(image_bgr: np.ndarray, cfg: PreprocessConfig,
                     delta: float) -> np.ndarray:
    """Geometric arm with a known relative scale error ``delta``.

    The fundus is rendered at ``size * (1 + delta)`` and then fitted to ``size``
    about the centre, which changes millimetres per pixel by ``1 + delta`` while
    leaving every other stage of the pipeline untouched.
    """
    cropped, _ = crop_fundus_bbox(image_bgr, cfg.fundus_pad_thresh)
    cropped = color_normalize(cropped, cfg.color_norm)
    target = max(8, int(round(cfg.size * (1.0 + delta))))
    scaled = resize_pad_square(cropped, target, pad_value=0)
    return _center_fit(scaled, cfg.size)


class JitterDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, cfg: PreprocessConfig, delta: float):
        self.frame = frame.reset_index(drop=True)
        self.cfg = cfg
        self.delta = delta
        self.mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, i):
        row = self.frame.iloc[i]
        img = cv2.imread(row["path"])
        if img is None:
            raise FileNotFoundError(row["path"])
        proc = geometric_jitter(img, self.cfg, self.delta)
        rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0)
        return (t - self.mean) / self.std, int(row["label"])


@torch.no_grad()
def _evaluate(model, frame, cfg, delta, device, batch_size, workers):
    loader = DataLoader(JitterDataset(frame, cfg, delta), batch_size=batch_size,
                        shuffle=False, num_workers=workers, pin_memory=True)
    gts, preds = [], []
    for xb, yb in loader:
        logits = model(xb.to(device, non_blocking=True))
        preds.extend(logits.argmax(1).cpu().tolist())
        gts.extend(yb.tolist())
    return qwk(np.array(gts), np.array(preds))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="trained geometric-arm checkpoint")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=0)
    a = ap.parse_args()

    PATHS.ensure_outputs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(a.ckpt, map_location=device, weights_only=False)
    saved = ck.get("cfg", {})
    cfg_t = TrainConfig(backbone=saved.get("backbone", "vit_l_imagenet"),
                        retfound_ckpt="")
    model, _ = create_backbone(cfg_t)
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    print(f"loaded {Path(a.ckpt).name} backbone={cfg_t.backbone} device={device}",
          flush=True)

    pre = PreprocessConfig(size=cfg_t.img_size)
    _, _, _, aizawl = load_split_frames(cfg_t.val_fraction, seed=saved.get("seed", 42),
                                        test_fraction=cfg_t.test_fraction)
    idrid = load_idrid_external()
    cohorts = {"aizawl": aizawl, "idrid": idrid}

    rows = []
    for name, frame in cohorts.items():
        for d in DELTAS:
            k = _evaluate(model, frame, pre, d, device, a.batch_size, a.num_workers)
            rows.append({"cohort": name, "delta": d, "qwk": k, "n": len(frame)})
            print(f"  {name:7s} delta={d:+.2f}  QWK={k:.4f}", flush=True)
    df = pd.DataFrame(rows)
    out = PATHS.reports / "scale_jitter.csv"
    df.to_csv(out, index=False)
    print(f"[csv] {out}")

    # dose-response figure
    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    styles = {"aizawl": ("#C44E52", "o", "Aizawl"), "idrid": ("#4C72B0", "s", "IDRiD")}
    for name, g in df.groupby("cohort"):
        col, mk, lbl = styles[name]
        g = g.sort_values("delta")
        base = float(g.loc[g.delta == 0.0, "qwk"].iloc[0])
        ax.plot(100 * g["delta"], g["qwk"], marker=mk, color=col, lw=1.6,
                label=f"{lbl} (unperturbed {base:.3f})")
    ax.axvline(0, color="#888888", ls=":", lw=1.0)
    ax.set_xlabel("Injected relative scale error (%)")
    ax.set_ylabel("External quadratic weighted kappa")
    ax.set_title("Sensitivity of a trained grader to input scale error")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, framealpha=0.9)
    fig.tight_layout()
    figp = PATHS.reports / "fig_scale_jitter.png"
    fig.savefig(figp, dpi=200)
    plt.close(fig)
    print(f"[fig] {figp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
