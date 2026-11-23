"""Re-evaluate the trained optic-disc detector on every split.

The training run recorded the best validation Dice reached over epochs, which is
a maximum over 54 evaluations and therefore optimistic, and it left the external
Dice unfilled. This script loads the saved checkpoint once and scores REFUGE
train, validation and test, plus the annotated IDRiD images, under one
protocol, so the numbers are comparable.

Run:
    python -m src.dr_grading.eval_od_detector
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import ConcatDataset, DataLoader

from .config import PATHS
from .od_detector import IdridODDataset, RefugeODDataset, UNet, dice_score
from .utils import save_json



@torch.no_grad()
def _scores(model, loader, device) -> Dict[str, float]:
    per_image = []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        for i in range(xb.shape[0]):
            per_image.append(float(dice_score(logits[i:i + 1].float(), yb[i:i + 1])))
    t = torch.tensor(per_image)
    return {"n": len(per_image), "dice_mean": float(t.mean()),
            "dice_median": float(t.median()), "dice_std": float(t.std()),
            "frac_below_0.5": float((t < 0.5).float().mean())}


def main() -> int:
    PATHS.ensure_outputs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(PATHS.checkpoints / "od_unet_best.pt", map_location=device,
                    weights_only=False)
    model = UNet(base=32)
    model.load_state_dict(ck["model"] if "model" in ck else ck)
    model.to(device).eval()

    out: Dict = {"device": device, "splits": {}}
    for split in ("train", "val", "test"):
        ds = RefugeODDataset(PATHS.refuge2_root / split, size=384)
        dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
        out["splits"][f"refuge_{split}"] = _scores(model, dl, device)
        print(f"refuge/{split}: {out['splits'][f'refuge_{split}']}", flush=True)

    # IDRiD ships optic-disc masks for both segmentation subsets, 54 in the
    # training set and 27 in the testing set. Both are scored so the external
    # count matches the 81 used elsewhere in the paper.
    seg = PATHS.idrid_images.parents[1]
    gt = PATHS.idrid_od_masks.parents[1]
    subsets = [("a. Training Set", "a. Training Set"), ("b. Testing Set", "b. Testing Set")]
    parts = []
    for img_sub, gt_sub in subsets:
        idir, mdir = seg / "1. Original Images" / img_sub, gt / gt_sub / "5. Optic Disc"
        if idir.is_dir() and mdir.is_dir():
            parts.append(IdridODDataset(idir, mdir, size=384))
    if parts:
        ds = ConcatDataset(parts) if len(parts) > 1 else parts[0]
        dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
        out["splits"]["idrid_external"] = _scores(model, dl, device)
        out["idrid_subsets"] = [len(p) for p in parts]
        print(f"idrid: {out['splits']['idrid_external']}", flush=True)

    save_json(out, PATHS.reports / "od_detector_eval.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
