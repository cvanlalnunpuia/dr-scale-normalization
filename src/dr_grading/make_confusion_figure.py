"""External confusion matrices for the paper, one row per resizing arm.

The artifact tags for the physical arm carry the canonical window, so the
naming follows ``final_analysis`` rather than the exploratory figures in
``make_paper_figures``. Columns are backbone crossed with cohort, rows are
the three arms, and each panel is row-normalized to recall and pooled over
the five seeds.

Run:
    python -m src.dr_grading.make_confusion_figure
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

from .config import PATHS
from .final_analysis import SEEDS, _preds, _tag
from .metrics import quadratic_weighted_kappa as qwk

CLASSES = [0, 1, 2, 3, 4]
ARMS = [("geo", "geometric"), ("7", "physical, $N = 7$"), ("9", "physical, $N = 9$")]
COLUMNS = [("retfound", "RETFound", "Aizawl", "test_gt", "test_pred"),
           ("retfound", "RETFound", "IDRiD", "idrid_gt", "idrid_pred"),
           ("vit_l_imagenet", "ViT-L (ImageNet)", "Aizawl", "test_gt", "test_pred"),
           ("vit_l_imagenet", "ViT-L (ImageNet)", "IDRiD", "idrid_gt", "idrid_pred")]


def _pooled(backbone, arm, gt_key, pred_key):
    cm = np.zeros((5, 5), dtype=int)
    ks = []
    for s in SEEDS:
        gt, pred = _preds(_tag(backbone, arm, s), gt_key), _preds(_tag(backbone, arm, s), pred_key)
        cm += confusion_matrix(gt, pred, labels=CLASSES)
        ks.append(qwk(gt, pred))
    return cm, float(np.mean(ks))


def build(path):
    fig, axes = plt.subplots(len(ARMS), len(COLUMNS), figsize=(13.4, 10.2))
    im = None
    for r, (arm, alab) in enumerate(ARMS):
        for c, (b, blab, cohort, gk, pk) in enumerate(COLUMNS):
            ax = axes[r, c]
            cm, k = _pooled(b, arm, gk, pk)
            rows = cm.sum(axis=1, keepdims=True)
            norm = np.divide(cm, rows, out=np.zeros_like(cm, dtype=float), where=rows > 0)
            im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
            for y in range(5):
                for x in range(5):
                    frac = norm[y, x]
                    ax.text(x, y, f"{frac:.2f}\n{cm[y, x]}", ha="center", va="center",
                            fontsize=7.5, color="white" if frac > 0.5 else "#222222")
            ax.set_xticks(CLASSES)
            ax.set_yticks(CLASSES)
            ax.set_xticklabels(CLASSES, fontsize=8)
            ax.set_yticklabels(CLASSES, fontsize=8)
            title = f"QWK {k:.3f}"
            if r == 0:
                title = f"{blab}, {cohort}\n{title}"
            ax.set_title(title, fontsize=9.5)
            if r == len(ARMS) - 1:
                ax.set_xlabel("Predicted grade", fontsize=8.5)
            if c == 0:
                ax.set_ylabel(f"{alab}\nTrue grade", fontsize=9)

    fig.tight_layout(rect=(0, 0, 0.93, 1))
    cax = fig.add_axes([0.945, 0.16, 0.012, 0.68])
    fig.colorbar(im, cax=cax, label="Recall (row fraction)")
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> int:
    PATHS.ensure_outputs()
    p = PATHS.reports / "fig_confusion_external_both.png"
    build(p)
    print(f"[fig] {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
