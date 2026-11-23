"""The two resolution measurements in one full-width figure.

Panel (a) is the test-time scale perturbation, averaged over every trained
geometric-arm checkpoint. Panel (b) is the grading outcome at each canonical
window. Both address the same question, so they belong in one float.

Run:
    python -m src.dr_grading.make_resolution_figure
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import PATHS

LABEL = {"retfound": "RETFound", "vit_l_imagenet": "ViT-L (ImageNet)"}
COHORT = {"aizawl": "Aizawl", "idrid": "IDRiD"}
ARMS = [("geometric", "geometric", "#4C72B0"),
        ("physical, N = 7", "physical, $N = 7$", "#DD8452"),
        ("physical, N = 9", "physical, $N = 9$", "#C44E52")]


def build(path):
    jit = pd.read_csv(PATHS.reports / "scale_jitter_matrix.csv")
    res = json.loads((PATHS.reports / "final_results.json").read_text())

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(13.0, 4.6))

    for name, colour, marker in (("aizawl", "#C44E52", "o"), ("idrid", "#4C72B0", "s")):
        g = jit[jit.cohort == name].groupby("delta")["qwk"]
        m, sd = g.mean(), g.std()
        axa.errorbar(100 * m.index, m.values, yerr=sd.values, marker=marker,
                     color=colour, lw=1.6, capsize=3, label=COHORT[name])
    axa.axvline(0, color="#888888", ls=":", lw=1.0)
    axa.set_xlabel("Injected relative scale error (%)")
    axa.set_ylabel("External quadratic weighted kappa")
    axa.set_title("(a) Sensitivity of a trained grader to input scale error",
                  fontsize=10.5)
    axa.grid(alpha=0.3)
    axa.legend(fontsize=9, framealpha=0.9)

    keys = [(b, c) for b in ("retfound", "vit_l_imagenet")
            for c in ("external_aizawl", "external_idrid")]
    x, w = np.arange(len(keys)), 0.26
    for i, (arm, alab, colour) in enumerate(ARMS):
        mean = [res["matrix"][f"{b}/{arm}"][c]["mean"] for b, c in keys]
        sd = [res["matrix"][f"{b}/{arm}"][c]["sd"] for b, c in keys]
        axb.bar(x + (i - 1) * w, mean, w, yerr=sd, capsize=4, label=alab,
                color=colour, edgecolor="black", linewidth=0.5)
    axb.set_xticks(x)
    axb.set_xticklabels([f"{LABEL[b]}\n{'Aizawl' if 'aizawl' in c else 'IDRiD'}"
                         for b, c in keys], fontsize=9)
    axb.set_ylabel("External quadratic weighted kappa")
    axb.set_title("(b) External agreement by canonical window", fontsize=10.5)
    axb.grid(axis="y", alpha=0.3)
    axb.legend(fontsize=8.8, framealpha=0.9)
    axb.set_ylim(0, 0.85)

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> int:
    PATHS.ensure_outputs()
    p = PATHS.reports / "fig_resolution_effect.png"
    build(p)
    print(f"[fig] {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
