"""Recompute every value reported in the Results section.

Two sources of variation are reported separately. Seed-level statistics treat
each training run as the unit and capture variability from initialisation, the
data split and augmentation order. Image-level paired bootstrap treats the test
images as the unit, resampling indices once and applying them to both arms so
the variance shared by the arms cancels.

Run:
    python -m src.dr_grading.final_analysis
"""
from __future__ import annotations

import json
import os
from typing import Dict, List

import numpy as np
from scipy import stats

from .config import PATHS
from .metrics import quadratic_weighted_kappa as qwk
from .utils import save_json

SEEDS = [42, 1, 2, 3, 4]
BACKBONES = [("retfound", "RETFound"), ("vit_l_imagenet", "ViT-L (ImageNet)")]
ARMS = [("geo", "geometric"), ("7", "physical, N = 7"), ("9", "physical, N = 9")]
COHORTS = [("external_aizawl", "test", "Aizawl"), ("external_idrid", "idrid", "IDRiD")]
N_BOOT = 5000


def _tag(backbone: str, arm: str, seed: int) -> str:
    if arm == "geo":
        return f"{backbone}_geometric_s{seed}"
    return f"{backbone}_physical_scale_s{seed}_n{arm}"


def _path(tag: str, suffix: str) -> str:
    for d in (PATHS.reports, PATHS.artifacts / "frozen_fov9"):
        p = d / f"{tag}{suffix}"
        if p.exists():
            return str(p)
    raise FileNotFoundError(tag + suffix)


def _report(tag: str) -> Dict:
    return json.load(open(_path(tag, "_results.json")))


def _preds(tag: str, key: str):
    return np.load(_path(tag, "_preds.npz"))[key]


def _cell(backbone: str, arm: str, metric: str) -> np.ndarray:
    return np.array([_report(_tag(backbone, arm, s))[metric]["qwk"] for s in SEEDS])


def _seed_contrast(a: np.ndarray, b: np.ndarray) -> Dict:
    """Paired contrast over seeds, with a t interval on 5 runs."""
    d = a - b
    m = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(len(d)))
    lo, hi = stats.t.interval(0.95, len(d) - 1, loc=m, scale=se)
    t, p = stats.ttest_1samp(d, 0.0)
    return {"per_seed": [float(x) for x in d], "mean": m, "sd": float(d.std(ddof=1)),
            "ci_lo": float(lo), "ci_hi": float(hi), "p": float(p)}


def _image_bootstrap(backbone: str, arm: str, gt_key: str, pred_key: str,
                     rng: np.random.Generator) -> Dict:
    """Paired bootstrap over images, seed-averaged inside each resample."""
    gts, pa, pb = [], [], []
    for s in SEEDS:
        t_phy, t_geo = _tag(backbone, arm, s), _tag(backbone, "geo", s)
        gts.append(_preds(t_geo, gt_key))
        pa.append(_preds(t_phy, pred_key))
        pb.append(_preds(t_geo, pred_key))
    n = len(gts[0])
    obs = float(np.mean([qwk(g, x) - qwk(g, y) for g, x, y in zip(gts, pa, pb)]))
    draws = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = rng.integers(0, n, n)
        draws[i] = np.mean([qwk(g[idx], x[idx]) - qwk(g[idx], y[idx])
                            for g, x, y in zip(gts, pa, pb)])
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"observed": obs, "ci_lo": float(lo), "ci_hi": float(hi),
            "p_two_sided": float(2 * min((draws <= 0).mean(), (draws >= 0).mean())),
            "n_boot": N_BOOT, "n_images": int(n)}


def run() -> Dict:
    rng = np.random.default_rng(0)
    out: Dict = {"seeds": SEEDS, "matrix": {}, "contrasts": {}}

    for b, blab in BACKBONES:
        for a, alab in ARMS:
            v = {}
            for metric in ("in_domain_test", "external_aizawl", "external_idrid"):
                x = _cell(b, a, metric)
                v[metric] = {"mean": float(x.mean()), "sd": float(x.std(ddof=1)),
                             "per_seed": [float(y) for y in x]}
            out["matrix"][f"{b}/{alab}"] = v

    for b, blab in BACKBONES:
        for coh, pred_stub, clab in COHORTS:
            geo = _cell(b, "geo", coh)
            for a in ("7", "9"):
                key = f"{b}/{a}/{clab}"
                out["contrasts"][key] = {
                    "seed_level": _seed_contrast(_cell(b, a, coh), geo),
                    "image_level": _image_bootstrap(
                        b, a, f"{pred_stub}_gt", f"{pred_stub}_pred", rng),
                }
            out["contrasts"][f"{b}/7v9/{clab}"] = {
                "seed_level": _seed_contrast(_cell(b, "7", coh), _cell(b, "9", coh))}

    # Fallback rate, identical across physical runs by construction.
    r = _report(_tag("retfound", "7", 42))
    out["fallback"] = {k: r["fallback_stats"][k]["fallback_rate"]
                       for k in r["fallback_stats"]}
    save_json(out, PATHS.reports / "final_results.json")
    return out


def main() -> int:
    PATHS.ensure_outputs()
    out = run()
    print(f"{'backbone':18s} {'arm':18s} {'in-domain':>16s} {'Aizawl':>16s} {'IDRiD':>16s}")
    for k, v in out["matrix"].items():
        b, a = k.split("/")
        f = lambda m: f"{v[m]['mean']:6.4f} +/-{v[m]['sd']:5.3f}"
        print(f"{b:18s} {a:18s} {f('in_domain_test'):>16s} "
              f"{f('external_aizawl'):>16s} {f('external_idrid'):>16s}")
    print()
    print(f"{'contrast':34s} {'seed mean':>10s} {'seed 95% CI':>20s} {'boot 95% CI':>20s}")
    for k, v in out["contrasts"].items():
        s = v["seed_level"]
        img = v.get("image_level")
        b = (f"[{img['ci_lo']:+6.3f},{img['ci_hi']:+6.3f}]" if img else "")
        print(f"{k:34s} {s['mean']:+10.4f} "
              f"[{s['ci_lo']:+6.3f},{s['ci_hi']:+6.3f}]{'':>2s} {b:>20s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
