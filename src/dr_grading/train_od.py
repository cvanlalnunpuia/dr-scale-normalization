"""Stage 2: train the optic-disc U-Net on REFUGE2, validate externally on IDRiD.

Run:
    python -m src.dr_grading.train_od --epochs 40 --size 512 --batch-size 8

Saves the best checkpoint (by REFUGE2-val Dice) to
``artifacts/checkpoints/od_unet_best.pt`` and a JSON report with REFUGE2
train/val/test and IDRiD (external) Dice.
"""
from __future__ import annotations

import argparse
import time
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import PATHS, SEED
from .od_detector import (
    IdridODDataset,
    RefugeODDataset,
    UNet,
    bce_dice_loss,
    dice_score,
)
from .utils import save_json, set_seed


@torch.no_grad()
def eval_dice(model, loader, device) -> float:
    model.eval()
    scores: List[float] = []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        with torch.autocast(device_type="cuda" if device == "cuda" else "cpu",
                            enabled=device == "cuda"):
            logits = model(xb)
        scores.append(dice_score(logits.float(), yb))
    return float(np.mean(scores)) if scores else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--base-channels", type=int, default=32)
    ap.add_argument("--patience", type=int, default=12,
                    help="Stop if REFUGE2-val Dice does not improve for this many epochs.")
    ap.add_argument("--max-minutes", type=float, default=180.0,
                    help="Wall-clock budget for the training loop; saves best + proceeds when hit.")
    a = ap.parse_args()

    set_seed(SEED)
    PATHS.ensure_outputs()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    r = PATHS.refuge2_root
    train_ds = RefugeODDataset(r / "train", size=a.size, train=True)
    val_ds = RefugeODDataset(r / "val", size=a.size)
    test_ds = RefugeODDataset(r / "test", size=a.size)
    print(f"REFUGE2 train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    train_dl = DataLoader(train_ds, batch_size=a.batch_size, shuffle=True,
                          num_workers=a.num_workers, pin_memory=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=a.batch_size, shuffle=False, num_workers=a.num_workers)
    test_dl = DataLoader(test_ds, batch_size=a.batch_size, shuffle=False, num_workers=a.num_workers)

    model = UNet(base=a.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=a.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    ckpt = PATHS.checkpoints / "od_unet_best.pt"
    best_dice = -1.0
    no_improve = 0
    history: List[Dict] = []
    t_start = time.time()
    stop_reason = "completed"
    for epoch in range(a.epochs):
        model.train()
        running = 0.0
        # mininterval keeps the log readable (avoids thousands of bar lines).
        for xb, yb in tqdm(train_dl, desc=f"OD epoch {epoch+1}/{a.epochs}",
                           leave=False, mininterval=30.0):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda" if device == "cuda" else "cpu",
                                enabled=device == "cuda"):
                loss = bce_dice_loss(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()
        scheduler.step()
        vdice = eval_dice(model, val_dl, device)
        elapsed_min = (time.time() - t_start) / 60.0
        history.append({"epoch": epoch + 1, "train_loss": running / max(1, len(train_dl)),
                        "val_dice": vdice, "elapsed_min": round(elapsed_min, 1)})
        print(f"epoch {epoch+1}: loss={running/max(1,len(train_dl)):.4f} "
              f"val_dice={vdice:.4f} ({elapsed_min:.1f} min)")
        if vdice > best_dice:
            best_dice = vdice
            no_improve = 0
            torch.save({"model": model.state_dict(), "size": a.size,
                        "base_channels": a.base_channels}, ckpt)
        else:
            no_improve += 1
        if no_improve >= a.patience:
            stop_reason = f"early-stop (no val improvement for {a.patience} epochs)"
            print(f"[stop] {stop_reason}")
            break
        if elapsed_min >= a.max_minutes:
            stop_reason = f"time budget reached ({a.max_minutes:.0f} min)"
            print(f"[stop] {stop_reason}")
            break

    # Load best and report on REFUGE2-test + external IDRiD.
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False)["model"])
    refuge_test_dice = eval_dice(model, test_dl, device)

    idrid_dice = float("nan")
    if PATHS.idrid_images.exists() and PATHS.idrid_od_masks.exists():
        idrid_ds = IdridODDataset(PATHS.idrid_images, PATHS.idrid_od_masks, size=a.size)
        if len(idrid_ds):
            idrid_dl = DataLoader(idrid_ds, batch_size=a.batch_size, shuffle=False,
                                  num_workers=a.num_workers)
            idrid_dice = eval_dice(model, idrid_dl, device)
            print(f"IDRiD external n={len(idrid_ds)} dice={idrid_dice:.4f}")
    else:
        print("IDRiD not found - skipping external detector validation.")

    report = {
        "best_refuge_val_dice": best_dice,
        "refuge_test_dice": refuge_test_dice,
        "idrid_external_dice": idrid_dice,
        "size": a.size,
        "base_channels": a.base_channels,
        "epochs_requested": a.epochs,
        "epochs_run": len(history),
        "stop_reason": stop_reason,
        "history": history,
        "checkpoint": str(ckpt),
    }
    out = PATHS.reports / "stage2_od_detector_results.json"
    save_json(report, out)
    print("\n" + "=" * 60)
    print(f"OD detector: REFUGE val={best_dice:.4f} test={refuge_test_dice:.4f} "
          f"IDRiD(ext)={idrid_dice:.4f}")
    print(f"Report -> {out}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
