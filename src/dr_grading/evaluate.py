"""Standalone evaluation of a trained checkpoint on the external Aizawl set.

Produces metrics (QWK + CIs), a JSON report and confusion-matrix plots for the
in-domain (APTOS-val) and external (Aizawl) sets.

Run:
    python -m src.dr_grading.evaluate --ckpt artifacts/checkpoints/retfound_geometric_best.pt \
        --arm geometric --tag retfound_geometric_eval
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .config import ICDR_CLASSES, NUM_CLASSES, PATHS, SEED, TrainConfig
from .data import build_dataloaders
from .metrics import bootstrap_ci, compute_metrics
from .model import build_model
from .preprocessing import PreprocessConfig
from .train import evaluate as run_inference
from .utils import save_json, set_seed


def plot_confusion(cm: np.ndarray, title: str, out_png: Path) -> None:
    classes = [ICDR_CLASSES[c] for c in range(NUM_CLASSES)]
    cm = np.asarray(cm, dtype=float)
    norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(NUM_CLASSES), classes, rotation=30, ha="right")
    ax.set_yticks(range(NUM_CLASSES), classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, f"{int(cm[i, j])}", ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--arm", choices=["geometric", "physical_scale"], default="geometric")
    ap.add_argument("--backbone", choices=["retfound", "vit_l_imagenet"], default="retfound")
    ap.add_argument("--od-cache", type=str, default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--tag", type=str, default="eval")
    a = ap.parse_args()

    set_seed(SEED)
    PATHS.ensure_outputs()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    od_diameters = None
    if a.od_cache:
        import json
        with open(a.od_cache, "r", encoding="utf-8") as f:
            od_diameters = {str(k): float(v) for k, v in json.load(f).items()}

    cfg = TrainConfig()
    loaders = build_dataloaders(
        arm=a.arm, pre_cfg=PreprocessConfig(size=cfg.img_size), batch_size=a.batch_size,
        num_workers=a.num_workers, val_fraction=cfg.val_fraction, od_diameters=od_diameters,
    )

    model = build_model(a.backbone, NUM_CLASSES)
    state = torch.load(a.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.to(device)

    results: Dict = {"ckpt": a.ckpt, "arm": a.arm, "backbone": a.backbone}
    for split in ("val", "test"):
        gt, pred = run_inference(model, loaders[split], device)
        m = compute_metrics(gt, pred)
        m["qwk_ci"] = bootstrap_ci(gt, pred, seed=SEED)
        name = "in_domain_val" if split == "val" else "external_aizawl"
        results[name] = m
        plot_confusion(
            np.array(m["confusion_matrix"]),
            f"{name} QWK={m['qwk']:.3f}",
            PATHS.reports / f"{a.tag}_{name}_cm.png",
        )
    results["generalization_gap_qwk"] = (
        results["in_domain_val"]["qwk"] - results["external_aizawl"]["qwk"]
    )
    out = PATHS.reports / f"{a.tag}_eval.json"
    save_json(results, out)
    print(f"in-domain QWK={results['in_domain_val']['qwk']:.4f} "
          f"| external QWK={results['external_aizawl']['qwk']:.4f} "
          f"| gap={results['generalization_gap_qwk']:.4f}")
    print(f"Report -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
