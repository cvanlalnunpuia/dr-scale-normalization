"""Multi-seed, multi-backbone experiment runner and aggregation.

Runs the {backbone} x {arm} x {seed} matrix one combination at a time and skips
any combination whose results JSON already exists, so an interrupted run can be
resumed. After running, it aggregates:

- per (backbone, arm): mean and standard deviation of in-domain and external
  QWK over seeds
- per backbone and seed: paired-bootstrap QWK difference (physical_scale minus
  geometric) on the shared external test set, with 95% CI and p-value

Run:
    python -m src.dr_grading.run_experiments \
        --backbones retfound vit_l_imagenet --arms geometric physical_scale \
        --seeds 42 1 2 3 4 \
        --od-cache artifacts/cache/od_diameters.json --retfound-ckpt /path/to.pth

Use ``--aggregate-only`` to rebuild the summary from existing results.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import PATHS, TrainConfig
from .metrics import paired_bootstrap_qwk_diff
from .train import train
from .utils import save_json


def _window_suffix(arm: str, window: Optional[float]) -> str:
    """Physical-arm runs carry their canonical window in the tag.

    A physical-arm result is only interpretable alongside the window it used, so
    the window belongs in the artifact name. Geometric runs keep the bare tag,
    which is what the existing artifacts are named.
    """
    if arm == "geometric" or window is None:
        return ""
    return f"_n{window:g}".replace(".", "p")


def _tag(backbone: str, arm: str, seed: int, window: Optional[float] = None) -> str:
    return f"{backbone}_{arm}_s{seed}{_window_suffix(arm, window)}"


def _result_paths(backbone: str, arm: str, seed: int,
                  window: Optional[float] = None) -> Tuple[Path, Path]:
    """Return (results_json, preds_npz), honouring the legacy seed-42 names."""
    tag = _tag(backbone, arm, seed, window)
    rj = PATHS.reports / f"{tag}_results.json"
    rn = PATHS.reports / f"{tag}_preds.npz"
    if not rj.exists() and seed == 42:
        legacy = PATHS.reports / f"{backbone}_{arm}_results.json"
        legacy_n = PATHS.reports / f"{backbone}_{arm}_preds.npz"
        if legacy.exists():
            return legacy, legacy_n
    return rj, rn


def run_matrix(
    backbones: List[str],
    arms: List[str],
    seeds: List[int],
    epochs: int,
    batch_size: int,
    grad_accum: int,
    num_workers: int,
    retfound_ckpt: str,
    od_cache: Optional[str],
    max_new_runs: Optional[int] = None,
    max_minutes: Optional[float] = None,
    fov_in_od_units: Optional[float] = None,
) -> int:
    """Train missing combos. Stops after ``max_new_runs`` new trainings or once
    ``max_minutes`` of wall-clock has elapsed, so a time-limited session does a
    bounded chunk and the rest resumes on the next call. Returns the number of
    combos trained this call."""
    trained = 0
    t_start = time.time()
    # Order: backbone (headline first) -> seed -> arm, so each seed yields a
    # complete arm pair for early paired comparison.
    for backbone in backbones:
        for seed in seeds:
            for arm in arms:
                tag = _tag(backbone, arm, seed, fov_in_od_units)
                rj, _ = _result_paths(backbone, arm, seed, fov_in_od_units)
                if rj.exists():
                    print(f"[skip] {tag} (found {rj.name})")
                    continue
                if arm == "physical_scale" and not (od_cache and Path(od_cache).exists()):
                    print(f"[skip] {tag}: physical_scale needs --od-cache (missing). "
                          f"Run Stage 2 first.")
                    continue
                if max_new_runs is not None and trained >= max_new_runs:
                    print(f"[budget] reached max_new_runs={max_new_runs}; "
                          f"stopping (remaining combos resume next commit).")
                    return trained
                if max_minutes is not None and (time.time() - t_start) / 60.0 >= max_minutes:
                    print(f"[budget] reached max_minutes={max_minutes:.0f}; "
                          f"stopping (remaining combos resume next commit).")
                    return trained
                cfg = TrainConfig(
                    seed=seed, backbone=backbone, norm_arm=arm,
                    retfound_ckpt=retfound_ckpt, epochs=epochs, batch_size=batch_size,
                    grad_accum_steps=grad_accum, num_workers=num_workers,
                )
                if fov_in_od_units is not None:
                    cfg.fov_in_od_units = fov_in_od_units
                print(f"\n{'#'*70}\n# RUN {tag}  (new run {trained+1}"
                      f"{'/'+str(max_new_runs) if max_new_runs else ''})\n{'#'*70}")
                t0 = time.time()
                train(cfg, od_cache_path=(od_cache if arm == "physical_scale" else None), tag=tag)
                trained += 1
                print(f"[done] {tag} in {(time.time()-t0)/60:.1f} min")
    return trained


# External test cohorts: (name, results-JSON key, npz gt key, npz pred key).
# Aizawl is the primary cohort (kept in legacy top-level summary keys); IDRiD is
# a second external cohort added later and is skipped for runs that predate it.
EXTERNAL_COHORTS = [
    ("aizawl", "external_aizawl", "test_gt", "test_pred"),
    ("idrid", "external_idrid", "idrid_gt", "idrid_pred"),
]


def aggregate(backbones: List[str], arms: List[str], seeds: List[int],
              fov_in_od_units: Optional[float] = None) -> Dict:
    per_cell: Dict[str, Dict] = {}
    for backbone in backbones:
        for arm in arms:
            in_q, seeds_done = [], []
            # Per-cohort external QWK and gap (val - external).
            coh_q: Dict[str, List[float]] = {c[0]: [] for c in EXTERNAL_COHORTS}
            coh_gap: Dict[str, List[float]] = {c[0]: [] for c in EXTERNAL_COHORTS}
            for seed in seeds:
                rj, _ = _result_paths(backbone, arm, seed, fov_in_od_units)
                if not rj.exists():
                    continue
                r = json.load(open(rj))
                vq = r["in_domain_val"]["qwk"]
                in_q.append(vq)
                seeds_done.append(seed)
                for name, key, _, _ in EXTERNAL_COHORTS:
                    m = r.get(key)
                    if m:
                        coh_q[name].append(m["qwk"])
                        coh_gap[name].append(vq - m["qwk"])
            if seeds_done:
                cell = {
                    "seeds": seeds_done,
                    "in_domain_qwk_mean": float(np.mean(in_q)),
                    "in_domain_qwk_std": float(np.std(in_q)),
                    "external": {},
                }
                for name in coh_q:
                    if coh_q[name]:
                        cell["external"][name] = {
                            "n_seeds": len(coh_q[name]),
                            "qwk_mean": float(np.mean(coh_q[name])),
                            "qwk_std": float(np.std(coh_q[name])),
                            "gap_mean": float(np.mean(coh_gap[name])),
                            "gap_std": float(np.std(coh_gap[name])),
                        }
                # Legacy top-level keys mirror the primary (Aizawl) cohort.
                aiz = cell["external"].get("aizawl")
                if aiz:
                    cell["external_qwk_mean"] = aiz["qwk_mean"]
                    cell["external_qwk_std"] = aiz["qwk_std"]
                    cell["gap_mean"] = aiz["gap_mean"]
                    cell["gap_std"] = aiz["gap_std"]
                per_cell[f"{backbone}/{arm}"] = cell

    # Paired arm comparison (physical_scale vs geometric) per backbone, seed, cohort.
    paired: Dict[str, List[Dict]] = {c[0]: [] for c in EXTERNAL_COHORTS}
    if "geometric" in arms and "physical_scale" in arms:
        for backbone in backbones:
            for seed in seeds:
                _, gn = _result_paths(backbone, "geometric", seed)
                _, pn = _result_paths(backbone, "physical_scale", seed)
                if not (gn.exists() and pn.exists()):
                    continue
                g = np.load(gn)
                p = np.load(pn)
                for name, _, gt_key, pred_key in EXTERNAL_COHORTS:
                    if gt_key not in g.files or gt_key not in p.files:
                        continue
                    if len(g[gt_key]) == 0 or len(p[gt_key]) == 0:
                        continue
                    if not np.array_equal(g[gt_key], p[gt_key]):
                        print(f"[warn] {name} gt order differs for {backbone} seed {seed}; "
                              f"skipping paired test.")
                        continue
                    stat = paired_bootstrap_qwk_diff(g[gt_key], g[pred_key], p[pred_key], seed=seed)
                    paired[name].append({"backbone": backbone, "seed": seed, **stat})

    summary = {
        "cells": per_cell,
        "paired_external_diff": paired,
        # Legacy key: the primary (Aizawl) paired comparison.
        "paired_external_diff_phys_minus_geo": paired["aizawl"],
    }
    out = PATHS.reports / "experiment_matrix_summary.json"
    save_json(summary, out)
    comparisons = paired["aizawl"]

    # Console tables.
    print("\n" + "=" * 78)
    print("EXPERIMENT MATRIX  (mean +/- std over seeds)")
    print("=" * 78)
    print(f"{'backbone/arm':<30}{'in-domain QWK':>16}"
          f"{'Aizawl QWK':>16}{'IDRiD QWK':>16}")
    for key, c in per_cell.items():
        ext = c.get("external", {})
        aiz = ext.get("aizawl", {})
        idr = ext.get("idrid", {})
        aiz_s = f"{aiz['qwk_mean']:.4f}+/-{aiz['qwk_std']:.3f}" if aiz else "n/a"
        idr_s = f"{idr['qwk_mean']:.4f}+/-{idr['qwk_std']:.3f}" if idr else "n/a"
        print(f"{key:<30}{c['in_domain_qwk_mean']:.4f}+/-{c['in_domain_qwk_std']:.3f}"
              f"   {aiz_s:>13}{idr_s:>16}")
    for cohort, comps in paired.items():
        if not comps:
            continue
        print(f"\nPaired external-QWK difference ({cohort}, physical_scale - geometric):")
        for cmp in comps:
            sig = "*" if (cmp["p_value"] == cmp["p_value"] and cmp["p_value"] < 0.05) else " "
            print(f"  {cmp['backbone']} seed {cmp['seed']}: "
                  f"diff={cmp['diff']:+.4f}  95%CI[{cmp['lo']:+.4f},{cmp['hi']:+.4f}]  "
                  f"p={cmp['p_value']:.4f}{sig}")
        diffs = [c["diff"] for c in comps]
        print(f"  mean diff over {len(diffs)} pair(s): {np.mean(diffs):+.4f}")
    print(f"\nSummary -> {out}")
    print("=" * 78)
    return summary


def export_paper_artifacts(
    backbones: List[str], arms: List[str], seeds: List[int],
    fov_in_od_units: Optional[float] = None,
) -> Dict[str, str]:
    """Write paper-ready artifacts from the (already trained) matrix:

    - ``results_per_seed.csv``  : one row per (backbone, arm, seed) with
      in-domain QWK, external QWK, and the generalization gap.
    - ``results_table.md``      : mean +/- std table + paired physical-vs-geometric
      bootstrap diffs (markdown, drop-in for the paper).
    - ``results_external_qwk.png`` : grouped bar chart of external QWK by
      backbone x arm (mean +/- std) with per-seed points overlaid.

    Reads only saved ``*_results.json`` / ``*_preds.npz`` files, so it is
    deterministic and requires no training. Returns the written paths.
    """
    import csv

    PATHS.ensure_outputs()

    # --- gather per-seed rows -------------------------------------------------
    rows: List[Dict] = []
    for backbone in backbones:
        for arm in arms:
            for seed in seeds:
                rj, _ = _result_paths(backbone, arm, seed, fov_in_od_units)
                if not rj.exists():
                    continue
                r = json.load(open(rj))
                # Newer runs add a held-out in-domain test + fallback stats;
                # fall back gracefully for legacy result JSONs.
                intest = r.get("in_domain_test") or {}
                fb = r.get("fallback_stats") or {}
                fb_ext = fb.get("external_aizawl") or {}
                fb_idr = fb.get("external_idrid") or {}
                idr = r.get("external_idrid") or {}
                rows.append({
                    "backbone": backbone,
                    "arm": arm,
                    "seed": seed,
                    "in_domain_val_qwk": r["in_domain_val"]["qwk"],
                    "in_domain_test_qwk": intest.get("qwk", float("nan")),
                    "external_aizawl_qwk": r["external_aizawl"]["qwk"],
                    "external_idrid_qwk": idr.get("qwk", float("nan")),
                    "gen_gap_qwk": r["generalization_gap_qwk"],
                    "gen_gap_qwk_heldout": r.get("generalization_gap_qwk_heldout",
                                                 float("nan")),
                    "gen_gap_qwk_heldout_idrid": r.get("generalization_gap_qwk_heldout_idrid",
                                                       float("nan")),
                    "aizawl_fallback_rate": fb_ext.get("fallback_rate", float("nan")),
                    "idrid_fallback_rate": fb_idr.get("fallback_rate", float("nan")),
                })

    csv_path = PATHS.reports / "results_per_seed.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "backbone", "arm", "seed", "in_domain_val_qwk", "in_domain_test_qwk",
            "external_aizawl_qwk", "external_idrid_qwk", "gen_gap_qwk",
            "gen_gap_qwk_heldout", "gen_gap_qwk_heldout_idrid",
            "aizawl_fallback_rate", "idrid_fallback_rate"])
        w.writeheader()
        for row in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v)
                        for k, v in row.items()})

    # --- aggregate cells + paired diffs (reuse the canonical computation) -----
    summary = aggregate(backbones, arms, seeds)
    cells = summary["cells"]
    comparisons = summary["paired_external_diff_phys_minus_geo"]

    # --- markdown table -------------------------------------------------------
    md: List[str] = []
    def _ext(c, cohort, field):
        v = (c.get("external", {}).get(cohort) or {}).get(field)
        return v

    md.append("# Experiment matrix (mean +/- std over seeds)\n")
    md.append("| backbone | arm | in-domain QWK | external QWK (Aizawl) | "
              "external QWK (IDRiD) |")
    md.append("|---|---|---|---|---|")
    for backbone in backbones:
        for arm in arms:
            c = cells.get(f"{backbone}/{arm}")
            if not c:
                continue
            def cell_str(cohort):
                m, s = _ext(c, cohort, "qwk_mean"), _ext(c, cohort, "qwk_std")
                return f"{m:.4f} +/- {s:.3f}" if m is not None else "n/a"
            md.append(
                f"| {backbone} | {arm} | "
                f"{c['in_domain_qwk_mean']:.4f} +/- {c['in_domain_qwk_std']:.3f} | "
                f"{cell_str('aizawl')} | {cell_str('idrid')} |")

    for cohort, comps in summary["paired_external_diff"].items():
        if not comps:
            continue
        md.append(f"\n## Paired external-QWK difference, {cohort} "
                  f"(physical_scale - geometric)\n")
        md.append("| backbone | seed | diff | 95% CI | p-value | sig |")
        md.append("|---|---|---|---|---|---|")
        for cmp in comps:
            sig = "*" if (cmp["p_value"] == cmp["p_value"] and cmp["p_value"] < 0.05) else ""
            md.append(
                f"| {cmp['backbone']} | {cmp['seed']} | {cmp['diff']:+.4f} | "
                f"[{cmp['lo']:+.4f}, {cmp['hi']:+.4f}] | {cmp['p_value']:.4f} | {sig} |")
        md.append(f"\nMean paired diff per backbone ({cohort}):\n")
        md.append("| backbone | mean diff | n seeds |")
        md.append("|---|---|---|")
        for backbone in backbones:
            ds = [c["diff"] for c in comps if c["backbone"] == backbone]
            if ds:
                md.append(f"| {backbone} | {np.mean(ds):+.4f} | {len(ds)} |")
    md_path = PATHS.reports / "results_table.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")

    # --- bar chart ------------------------------------------------------------
    png_path = PATHS.reports / "results_external_qwk.png"
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        labels, means, stds = [], [], []
        for backbone in backbones:
            for arm in arms:
                c = cells.get(f"{backbone}/{arm}")
                if not c:
                    continue
                labels.append(f"{backbone}\n{arm}")
                means.append(c["external_qwk_mean"])
                stds.append(c["external_qwk_std"])
        x = np.arange(len(labels))
        colors = ["#4C72B0" if "geometric" in l else "#C44E52" for l in labels]
        fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(labels)), 4.5))
        ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.85,
               edgecolor="black", linewidth=0.6)
        # overlay per-seed points
        for i, (backbone, arm) in enumerate(
                [(b, a) for b in backbones for a in arms
                 if cells.get(f"{b}/{a}")]):
            pts = [row["external_aizawl_qwk"] for row in rows
                   if row["backbone"] == backbone and row["arm"] == arm]
            ax.scatter([x[i]] * len(pts), pts, color="black", zorder=3, s=18)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("External (Aizawl) QWK")
        ax.set_title("Cross-dataset generalization: external QWK by backbone x arm")
        ax.set_ylim(0, max(means) + max(stds) + 0.1)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
    except Exception as e:  # pragma: no cover - plotting is best-effort
        print(f"[warn] could not render bar chart: {e}")
        png_path = None

    print(f"[export] per-seed CSV -> {csv_path}")
    print(f"[export] markdown table -> {md_path}")
    if png_path:
        print(f"[export] bar chart -> {png_path}")
    return {
        "csv": str(csv_path),
        "markdown": str(md_path),
        "figure": str(png_path) if png_path else "",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", nargs="+", default=["retfound"],
                    choices=["retfound", "vit_l_imagenet"])
    ap.add_argument("--arms", nargs="+", default=["geometric", "physical_scale"],
                    choices=["geometric", "physical_scale"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 1, 2])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--retfound-ckpt", type=str, default=TrainConfig().retfound_ckpt)
    ap.add_argument("--od-cache", type=str, default=None)
    ap.add_argument("--max-new-runs", type=int, default=None,
                    help="Train at most this many new combos this call, then stop (resumable).")
    ap.add_argument("--max-minutes", type=float, default=None,
                    help="Stop launching new combos after this much wall-clock (resumable).")
    ap.add_argument("--aggregate-only", action="store_true")
    a = ap.parse_args()

    PATHS.ensure_outputs()
    if not a.aggregate_only:
        run_matrix(a.backbones, a.arms, a.seeds, a.epochs, a.batch_size, a.grad_accum,
                   a.num_workers, a.retfound_ckpt, a.od_cache,
                   max_new_runs=a.max_new_runs, max_minutes=a.max_minutes)
    aggregate(a.backbones, a.arms, a.seeds)
    return 0


if __name__ == "__main__":
    sys.exit(main())
