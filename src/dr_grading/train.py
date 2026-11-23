"""Stage 1/3 trainer: fine-tune RETFound on APTOS and evaluate the external gap.

Trains on APTOS (in-domain), selects the best epoch by validation QWK, then
reports in-domain (APTOS-val) vs external (Aizawl) metrics -- the generalization
gap that the physical-scale arm aims to shrink.

Examples:
    # Geometric baseline (Stage 1)
    python -m src.dr_grading.train --arm geometric --epochs 50

    # Physical-scale arm (Stage 3) with an OD-diameter cache
    python -m src.dr_grading.train --arm physical_scale --od-cache artifacts/cache/od_diameters.json

Set the RETFound checkpoint via the RETFOUND_CKPT env var (or --retfound-ckpt).
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import NUM_CLASSES, PATHS, SEED, TrainConfig
from .data import (
    build_dataloaders,
    class_weights_from_frame,
    fallback_stats,
    load_idrid_external,
    load_split_frames,
)
from .metrics import bootstrap_ci, compute_metrics
from .model import create_backbone, param_groups_lrd
from .preprocessing import PreprocessConfig
from .utils import save_json, set_seed


def _lr_factor(epoch_f: float, warmup: int, total: int, min_ratio: float) -> float:
    """Linear warmup then cosine decay; ``epoch_f`` may be fractional."""
    if epoch_f < warmup:
        return epoch_f / max(1, warmup)
    progress = (epoch_f - warmup) / max(1, total - warmup)
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


def _load_od_cache(path: Optional[str]) -> Optional[Dict[str, float]]:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return {str(k): float(v) for k, v in json.load(f).items()}


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str,
             max_batches: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds: List[int] = []
    gts: List[int] = []
    for bi, (xb, yb) in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        xb2 = xb.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda" if device == "cuda" else "cpu",
                            enabled=(device == "cuda")):
            logits = model(xb2)
        preds.extend(logits.argmax(1).cpu().tolist())
        gts.extend(yb.tolist())
    return np.array(gts), np.array(preds)


def train(cfg: TrainConfig, od_cache_path: Optional[str], tag: str,
          max_train_batches: Optional[int] = None,
          max_eval_batches: Optional[int] = None) -> Dict:
    set_seed(cfg.seed)
    PATHS.ensure_outputs()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pre_cfg = PreprocessConfig(size=cfg.img_size,
                               fov_in_od_units=cfg.fov_in_od_units)
    od_diameters = _load_od_cache(od_cache_path)
    loaders = build_dataloaders(
        arm=cfg.norm_arm, pre_cfg=pre_cfg, batch_size=cfg.batch_size,
        num_workers=cfg.num_workers, val_fraction=cfg.val_fraction, od_diameters=od_diameters,
        seed=cfg.seed, test_fraction=cfg.test_fraction,
    )
    train_df, val_df, intest_df, ext_df = load_split_frames(
        cfg.val_fraction, seed=cfg.seed, test_fraction=cfg.test_fraction
    )

    # Report cohort sizes before training starts, so a dataset that failed to
    # mount is visible in seconds instead of after hours of training.
    n_idrid = len(load_idrid_external())
    print(f"[cohorts] train={len(train_df)} val={len(val_df)} "
          f"intest={len(intest_df)} aizawl={len(ext_df)} idrid={n_idrid}")
    if n_idrid == 0:
        print("[warn] IDRiD grading set not found. Training continues and the "
              "run will report Aizawl only. Attach the IDRiD dataset and set "
              "PATHS.idrid_flat_images / PATHS.idrid_flat_csv to include it.")

    model, load_report = create_backbone(cfg)
    model.to(device)

    # Weighted, label-smoothed cross-entropy for the imbalanced training set.
    weights = class_weights_from_frame(train_df, NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=cfg.label_smoothing)

    groups = param_groups_lrd(model, cfg.lr, cfg.weight_decay, cfg.layer_decay)
    optimizer = torch.optim.AdamW(groups, lr=cfg.lr, betas=(0.9, 0.999))
    base_lrs = [g["lr"] for g in optimizer.param_groups]
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.use_amp and device == "cuda")

    steps_per_epoch = max(1, len(loaders["train"]))
    best_qwk = -1.0
    best_epoch = -1
    history: List[Dict] = []
    ckpt_path = PATHS.checkpoints / f"{tag}_best.pt"
    patience = 0

    for epoch in range(cfg.epochs):
        model.train()
        # Vary deterministic augmentation per epoch (reproducible across reruns).
        loaders["train"].dataset.set_epoch(epoch)
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        pbar = tqdm(loaders["train"], desc=f"epoch {epoch+1}/{cfg.epochs}", leave=False,
                    mininterval=30.0)
        for step, (xb, yb) in enumerate(pbar):
            if max_train_batches is not None and step >= max_train_batches:
                break
            epoch_f = epoch + step / steps_per_epoch
            factor = _lr_factor(epoch_f, cfg.warmup_epochs, cfg.epochs, cfg.min_lr / cfg.lr)
            for g, base in zip(optimizer.param_groups, base_lrs):
                g["lr"] = base * factor

            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda" if device == "cuda" else "cpu",
                                enabled=cfg.use_amp and device == "cuda"):
                loss = criterion(model(xb), yb) / cfg.grad_accum_steps
            scaler.scale(loss).backward()
            if (step + 1) % cfg.grad_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            running += loss.item() * cfg.grad_accum_steps
            pbar.set_postfix(loss=f"{running/(step+1):.3f}")

        # Validation (in-domain).
        val_gt, val_pred = evaluate(model, loaders["val"], device, max_eval_batches)
        val_m = compute_metrics(val_gt, val_pred)
        history.append({"epoch": epoch + 1, "train_loss": running / steps_per_epoch,
                        "val_qwk": val_m["qwk"], "val_acc": val_m["accuracy"]})
        print(f"epoch {epoch+1}: train_loss={running/steps_per_epoch:.3f} "
              f"val_qwk={val_m['qwk']:.4f} val_acc={val_m['accuracy']:.4f}")

        if val_m["qwk"] > best_qwk:
            best_qwk, best_epoch, patience = val_m["qwk"], epoch + 1, 0
            torch.save({"model": model.state_dict(), "cfg": vars(cfg), "epoch": epoch + 1}, ckpt_path)
        else:
            patience += 1
            if patience >= cfg.early_stop_patience:
                print(f"Early stopping at epoch {epoch+1} (best epoch {best_epoch}).")
                break

    # Final evaluation with the best checkpoint.
    if ckpt_path.exists():
        model.load_state_dict(
            torch.load(ckpt_path, map_location=device, weights_only=False)["model"]
        )

    val_gt, val_pred = evaluate(model, loaders["val"], device, max_eval_batches)
    test_gt, test_pred = evaluate(model, loaders["test"], device, max_eval_batches)
    val_metrics = compute_metrics(val_gt, val_pred)
    test_metrics = compute_metrics(test_gt, test_pred)
    val_metrics["qwk_ci"] = bootstrap_ci(val_gt, val_pred, seed=cfg.seed)
    test_metrics["qwk_ci"] = bootstrap_ci(test_gt, test_pred, seed=cfg.seed)

    # Held-out in-domain test (NEVER used for selection -> unbiased estimate).
    intest_metrics = None
    intest_gt = intest_pred = np.array([], dtype=int)
    if "intest" in loaders:
        intest_gt, intest_pred = evaluate(model, loaders["intest"], device, max_eval_batches)
        intest_metrics = compute_metrics(intest_gt, intest_pred)
        intest_metrics["qwk_ci"] = bootstrap_ci(intest_gt, intest_pred, seed=cfg.seed)

    # Second external cohort: IDRiD grading set (evaluated when present).
    idrid_metrics = None
    idrid_gt = idrid_pred = np.array([], dtype=int)
    if "idrid" in loaders:
        idrid_gt, idrid_pred = evaluate(model, loaders["idrid"], device, max_eval_batches)
        idrid_metrics = compute_metrics(idrid_gt, idrid_pred)
        idrid_metrics["qwk_ci"] = bootstrap_ci(idrid_gt, idrid_pred, seed=cfg.seed)
    idrid_df = load_idrid_external()

    # Per-split fallback rate for the physical_scale arm (geometric contrast dilution).
    fallbacks = {
        "train": fallback_stats(train_df, cfg.norm_arm, od_diameters),
        "val": fallback_stats(val_df, cfg.norm_arm, od_diameters),
        "intest": fallback_stats(intest_df, cfg.norm_arm, od_diameters),
        "external_aizawl": fallback_stats(ext_df, cfg.norm_arm, od_diameters),
        "external_idrid": fallback_stats(idrid_df, cfg.norm_arm, od_diameters),
    }

    # Unbiased gap uses the held-out in-domain test when available; else val.
    in_domain_ref = intest_metrics if intest_metrics is not None else val_metrics
    report = {
        "tag": tag,
        "config": vars(cfg),
        "backbone_load": load_report,
        "best_epoch": best_epoch,
        "best_val_qwk": best_qwk,
        "in_domain_val": val_metrics,
        "in_domain_test": intest_metrics,
        "external_aizawl": test_metrics,
        "external_idrid": idrid_metrics,
        # Legacy (selection-biased) gap kept for backward compatibility.
        "generalization_gap_qwk": val_metrics["qwk"] - test_metrics["qwk"],
        # Unbiased gap from the held-out in-domain test (preferred when present).
        "generalization_gap_qwk_heldout": in_domain_ref["qwk"] - test_metrics["qwk"],
        "fallback_stats": fallbacks,
        "history": history,
        "device": device,
    }
    if idrid_metrics is not None:
        report["generalization_gap_qwk_heldout_idrid"] = in_domain_ref["qwk"] - idrid_metrics["qwk"]
    out = PATHS.reports / f"{tag}_results.json"
    save_json(report, out)
    # Save raw predictions for later paired statistics across arms/seeds.
    np.savez(PATHS.reports / f"{tag}_preds.npz",
             val_gt=val_gt, val_pred=val_pred, test_gt=test_gt, test_pred=test_pred,
             intest_gt=intest_gt, intest_pred=intest_pred,
             idrid_gt=idrid_gt, idrid_pred=idrid_pred)

    print("\n" + "=" * 60)
    print(f"[{tag}] in-domain QWK = {val_metrics['qwk']:.4f} "
          f"(95% CI {val_metrics['qwk_ci']['lo']:.3f}-{val_metrics['qwk_ci']['hi']:.3f})")
    if intest_metrics is not None:
        print(f"[{tag}] in-domain TEST QWK = {intest_metrics['qwk']:.4f} "
              f"(held-out, unbiased; 95% CI "
              f"{intest_metrics['qwk_ci']['lo']:.3f}-{intest_metrics['qwk_ci']['hi']:.3f})")
    print(f"[{tag}] external (Aizawl) QWK = {test_metrics['qwk']:.4f} "
          f"(95% CI {test_metrics['qwk_ci']['lo']:.3f}-{test_metrics['qwk_ci']['hi']:.3f})")
    if idrid_metrics is not None:
        print(f"[{tag}] external (IDRiD)  QWK = {idrid_metrics['qwk']:.4f} "
              f"(95% CI {idrid_metrics['qwk_ci']['lo']:.3f}-{idrid_metrics['qwk_ci']['hi']:.3f})")
    print(f"[{tag}] generalization gap (val-ext) = {report['generalization_gap_qwk']:.4f}")
    print(f"[{tag}] generalization gap (heldout-ext) = "
          f"{report['generalization_gap_qwk_heldout']:.4f}")
    print(f"Report -> {out}")
    print("=" * 60)
    return report


def parse_args() -> argparse.Namespace:
    cfg = TrainConfig()
    ap = argparse.ArgumentParser(description="Fine-tune RETFound for DR grading.")
    ap.add_argument("--arm", choices=["geometric", "physical_scale"], default=cfg.norm_arm)
    ap.add_argument("--backbone", choices=["retfound", "vit_l_imagenet"], default=cfg.backbone)
    ap.add_argument("--epochs", type=int, default=cfg.epochs)
    ap.add_argument("--batch-size", type=int, default=cfg.batch_size)
    ap.add_argument("--grad-accum", type=int, default=cfg.grad_accum_steps)
    ap.add_argument("--lr", type=float, default=cfg.lr)
    ap.add_argument("--num-workers", type=int, default=cfg.num_workers)
    ap.add_argument("--seed", type=int, default=cfg.seed)
    ap.add_argument("--fov-in-od-units", type=float, default=cfg.fov_in_od_units,
                    help="Canonical window for the physical arm, in optic-disc diameters.")
    ap.add_argument("--retfound-ckpt", type=str, default=cfg.retfound_ckpt)
    ap.add_argument("--od-cache", type=str, default=None,
                    help="JSON {image_stem: od_diameter_px} for the physical_scale arm.")
    ap.add_argument("--tag", type=str, default=None)
    ap.add_argument("--max-train-batches", type=int, default=None,
                    help="Debug/dry-run: cap training batches per epoch.")
    ap.add_argument("--max-eval-batches", type=int, default=None,
                    help="Debug/dry-run: cap eval batches.")
    return ap.parse_args()


def main() -> int:
    a = parse_args()
    cfg = TrainConfig(
        seed=a.seed, epochs=a.epochs, batch_size=a.batch_size, grad_accum_steps=a.grad_accum,
        lr=a.lr, num_workers=a.num_workers, norm_arm=a.arm, backbone=a.backbone,
        retfound_ckpt=a.retfound_ckpt, fov_in_od_units=a.fov_in_od_units,
    )
    # The window is part of the identity of a physical-arm run, so it is in
    # the tag. Geometric runs keep the bare tag, which the existing artifacts use.
    _w = "" if a.arm == "geometric" else f"_n{a.fov_in_od_units:g}".replace(".", "p")
    tag = a.tag or f"{a.backbone}_{a.arm}_s{a.seed}{_w}"
    t0 = time.time()
    train(cfg, a.od_cache, tag, a.max_train_batches, a.max_eval_batches)
    print(f"Elapsed: {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
