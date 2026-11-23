"""Emit the Results tables as LaTeX fragments, computed from the result files.

Every value printed in the paper's Results section comes from here, so a rerun
regenerates the tables rather than requiring them to be edited by hand.

Run:
    python -m src.dr_grading.make_result_tables
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats

from .config import PATHS
from .od_fov_analysis import _predicted

OUT = PATHS.reports / "tables"
BACKBONES = [("retfound", "RETFound"), ("vit_l_imagenet", "ViT-L")]
ARMS = [("geometric", "geometric"), ("physical, N = 7", "physical, $N=7$"),
        ("physical, N = 9", "physical, $N=9$")]


def _w(name: str, body: str) -> None:
    (OUT / name).write_text(body, encoding="utf-8")
    print(f"[tex] {OUT / name}")


def ratio_table(od: dict) -> None:
    rows = []
    for k in ("APTOS", "Aizawl", "IDRiD"):
        c = od["cohorts"][k]
        rows.append(f"{k} & {c['n']} & {c['median']:.4f} & {c['cv']:.2f} & "
                    f"{1.0 / c['median']:.2f} \\\\")
    p = od["idrid_paired"]
    rows.append("\hline")
    rows.append(f"IDRiD, ground truth & {p['n']} & {p['gt_median']:.4f} & "
                f"{p['gt_cv']:.2f} & {1.0 / p['gt_median']:.2f} \\\\")
    _w("tab_ratio.tex", "\n".join(rows))


def signal_table(od: dict, paired: pd.DataFrame) -> None:
    p = od["idrid_paired"]
    r, _ = stats.pearsonr(paired.gt_od_fov, paired.det_od_fov)
    rel = (paired.det_od_fov - paired.gt_od_fov) / paired.gt_od_fov
    rows = [
        f"True ratio, CV & {p['gt_cv']:.3f} \\\\",
        f"Detector ratio, CV & {p['det_cv']:.3f} \\\\",
        f"Correlation with truth, $r$ & {r:.3f} \\\\",
        f"Variance recovered, $r^{{2}}$ & {r ** 2:.3f} \\\\",
        f"Mean relative detector error & ${rel.mean() * 100:+.1f}$\% \\\\",
        f"Relative detector error, SD & {rel.std(ddof=1):.3f} \\\\",
        f"Within 10\% of truth & {p['frac_within_10pct'] * 100:.0f}\% \\\\",
        f"Error above 30\% & {p['frac_gross_error_gt30pct'] * 100:.0f}\% \\\\",
    ]
    _w("tab_signal.tex", "\n".join(rows))


def detector_table(ev: dict) -> None:
    names = [("refuge_train", "REFUGE, train"), ("refuge_val", "REFUGE, validation"),
             ("refuge_test", "REFUGE, test"), ("idrid_external", "IDRiD, external")]
    rows = [f"{lab} & {ev['splits'][k]['n']} & {ev['splits'][k]['dice_mean']:.3f} & "
            f"{ev['splits'][k]['dice_median']:.3f} & "
            f"{ev['splits'][k]['frac_below_0.5'] * 100:.0f}\% \\\\"
            for k, lab in names]
    _w("tab_detector.tex", "\n".join(rows))


def window_table(cov: dict) -> None:
    rows = []
    for name in ("APTOS (train)", "Aizawl", "IDRiD"):
        c = cov["cohorts"][name]
        for n in ("6.0", "7.0", "9.0"):
            w = c["windows"][n]
            lab = name.replace(" (train)", "") if n == "6.0" else ""
            rows.append(f"{lab} & {float(n):.0f} & {w['coverage']:.3f} & "
                        f"{w['coverage_p05']:.3f} & {w['frac_cropped'] * 100:.0f}\% & "
                        f"{w['linear_ratio']:.3f} \\\\")
        rows.append("\hline" if name != "IDRiD" else "")
    _w("tab_window.tex", "\n".join(r for r in rows if r))


def matrix_table(res: dict) -> None:
    # Short arm labels, because this table is set in one column.
    short = {"geometric": "geometric", "physical, N = 7": "$N=7$",
             "physical, N = 9": "$N=9$"}
    rows = []
    for b, blab in BACKBONES:
        for a, _ in ARMS:
            alab = short[a]
            v = res["matrix"][f"{b}/{a}"]
            f = lambda m: f"{v[m]['mean']:.3f} ({v[m]['sd']:.3f})"
            rows.append(f"{blab} & {alab} & {f('in_domain_test')} & "
                        f"{f('external_aizawl')} & {f('external_idrid')} \\\\")
        rows.append("\hline")
    _w("tab_matrix.tex", "\n".join(rows[:-1]))


def contrast_table(res: dict) -> None:
    rows = []
    for b, blab in BACKBONES:
        for coh in ("Aizawl", "IDRiD"):
            for a, alab in (("7", "$N=7$"), ("9", "$N=9$"), ("7v9", "$N=7$ vs $N=9$")):
                c = res["contrasts"][f"{b}/{a}/{coh}"]
                s = c["seed_level"]
                img = c.get("image_level")
                ib = (f"$[{img['ci_lo']:+.3f}, {img['ci_hi']:+.3f}]$" if img else "--")
                rows.append(f"{blab} & {coh} & {alab} & ${s['mean']:+.4f}$ & "
                            f"$[{s['ci_lo']:+.3f}, {s['ci_hi']:+.3f}]$ & "
                            f"{s['p']:.3f} & {ib} \\\\")
            rows.append("\hline")
    _w("tab_contrast.tex", "\n".join(rows[:-1]))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    od = json.loads((PATHS.reports / "od_fov_table.json").read_text())
    ev = json.loads((PATHS.reports / "od_detector_eval.json").read_text())
    cov = json.loads((PATHS.reports / "window_coverage.json").read_text())
    res = json.loads((PATHS.reports / "final_results.json").read_text())
    paired = pd.read_csv(PATHS.reports / "od_fov_paired.csv").dropna(
        subset=["gt_od_fov", "det_od_fov"])
    ratio_table(od)
    signal_table(od, paired)
    detector_table(ev)
    window_table(cov)
    matrix_table(res)
    contrast_table(res)
    print(f"\npredicted ratio: 40 deg {_predicted(40):.4f}, 45 deg {_predicted(45):.4f}, "
          f"50 deg {_predicted(50):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
