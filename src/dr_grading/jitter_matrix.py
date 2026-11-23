"""Scale-perturbation sensitivity across every trained geometric-arm model.

The single-checkpoint curve leaves the reader unable to tell a property of the
method from a property of one run. This module repeats the perturbation for
every geometric-arm checkpoint present, so the curve carries a mean and a spread
over seeds and both backbones.

Checkpoints are large, so this is meant to run in the same session that trained
them. Only the CSV needs to be carried out.

Run:
    python -m src.dr_grading.jitter_matrix
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch

from .config import PATHS, TrainConfig
from .data import load_idrid_external, load_split_frames
from .model import create_backbone
from .preprocessing import PreprocessConfig
from .scale_jitter import DELTAS, _evaluate

CKPT_RE = re.compile(r"^(?P<backbone>.+)_geometric_s(?P<seed>\d+)_best\.pt$")


def _checkpoints(extra_roots: Optional[List[str]] = None) -> List[Dict]:
    """Every geometric-arm checkpoint found under the checkpoint directories.

    Extra directories can be given to search checkpoints stored elsewhere
    without copying them.
    """
    found: Dict[tuple, Dict] = {}
    roots = [PATHS.checkpoints, PATHS.artifacts / "ckpt_jitter"]
    for r in (extra_roots or []):
        roots.append(Path(r))
    for d in roots:
        if not d.is_dir():
            continue
        paths = sorted(d.glob("*_geometric_s*_best.pt"))
        if not paths:
            paths = sorted(d.rglob("*_geometric_s*_best.pt"))
        for p in paths:
            m = CKPT_RE.match(p.name)
            if m:
                key = (m.group("backbone"), int(m.group("seed")))
                found.setdefault(key, {"path": p, "backbone": key[0], "seed": key[1]})
    return [found[k] for k in sorted(found)]


def run(batch_size: int = 16, num_workers: int = 0,
        extra_roots: Optional[List[str]] = None) -> pd.DataFrame:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpts = _checkpoints(extra_roots)
    if not ckpts:
        # A session that trained only physical-arm runs holds no geometric
        # checkpoint. Report and return empty so the session still saves its
        # training output.
        print("No geometric-arm checkpoints reachable. Skipping the sweep. "
              "Attach the version outputs that hold them and run this stage alone.")
        return pd.DataFrame(columns=["backbone", "seed", "cohort", "delta", "qwk", "n"])
    print(f"{len(ckpts)} checkpoint(s) on {device}", flush=True)

    out = PATHS.reports / "scale_jitter_matrix.csv"
    # Resume support. The sweep is long, so completed checkpoints are kept and
    # the file is rewritten after each one.
    rows: List[Dict] = []
    done = set()
    if out.exists():
        prev = pd.read_csv(out)
        rows = prev.to_dict("records")
        for (b, sd), g in prev.groupby(["backbone", "seed"]):
            if len(g) == len(DELTAS) * 2:
                done.add((b, int(sd)))
        print(f"resuming, {len(done)} checkpoint(s) already complete", flush=True)
        rows = [r for r in rows if (r["backbone"], int(r["seed"])) in done]

    for c in ckpts:
        if (c["backbone"], c["seed"]) in done:
            print(f"[skip] {c['backbone']} s{c['seed']} already complete", flush=True)
            continue
        ck = torch.load(c["path"], map_location=device, weights_only=False)
        saved = ck.get("cfg", {})
        cfg_t = TrainConfig(backbone=saved.get("backbone", c["backbone"]),
                            retfound_ckpt="")
        model, _ = create_backbone(cfg_t)
        model.load_state_dict(ck["model"])
        model.to(device).eval()

        # The split depends on the seed, so each checkpoint is scored against
        # the cohorts its own run was evaluated on.
        pre = PreprocessConfig(size=cfg_t.img_size)
        _, _, _, aizawl = load_split_frames(cfg_t.val_fraction, seed=c["seed"],
                                            test_fraction=cfg_t.test_fraction)
        cohorts = {"aizawl": aizawl, "idrid": load_idrid_external()}
        for name, frame in cohorts.items():
            for d in DELTAS:
                k = _evaluate(model, frame, pre, d, device, batch_size, num_workers)
                rows.append({"backbone": c["backbone"], "seed": c["seed"],
                             "cohort": name, "delta": d, "qwk": k, "n": len(frame)})
                print(f"  {c['backbone']} s{c['seed']} {name:7s} "
                      f"delta={d:+.2f} QWK={k:.4f}", flush=True)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"[saved] {len(rows)} rows so far", flush=True)

    df = pd.DataFrame(rows)
    print(f"[csv] {out}  ({len(df)} rows)")
    return df


def main() -> int:
    PATHS.ensure_outputs()
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
