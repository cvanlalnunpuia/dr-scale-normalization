"""Central configuration: paths, seeds, label scheme, and hyperparameters.

All paths are resolved relative to the repository root.
Override any path via environment variables of the same name.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _repo_root() -> Path:
    # src/dr_grading/config.py -> repo root is two levels up.
    return Path(__file__).resolve().parents[2]


REPO_ROOT = Path(os.environ.get("DR_REPO_ROOT", _repo_root()))

# Reproducibility
SEED = int(os.environ.get("DR_SEED", 42))

# ICDR 5-class diabetic retinopathy grading scheme (shared by APTOS + Aizawl).
ICDR_CLASSES = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative",
}
NUM_CLASSES = len(ICDR_CLASSES)


def _p(*parts: str) -> Path:
    return REPO_ROOT.joinpath(*parts)


@dataclass
class DataPaths:
    """Resolved locations for every dataset used in the study."""

    # APTOS 2019 (training source). Images are <id_code>.png; labels in CSV.
    aptos_images: Path = field(default_factory=lambda: _p("Kdata", "Kdata"))
    aptos_csv: Path = field(default_factory=lambda: _p("Kdata", "Kdatacsv.csv"))
    aptos_id_col: str = "id_code"
    aptos_label_col: str = "diagnosis"
    aptos_ext: str = ".png"

    # Aizawl clinical set (external test). Images are <id>.jpg; labels in CSV.
    aizawl_images: Path = field(default_factory=lambda: _p("Eye Care Hospital", "DR_images"))
    aizawl_csv: Path = field(default_factory=lambda: _p("Eye Care Hospital", "dr_labels.csv"))
    aizawl_id_col: str = "id_code"   # values already include ".jpg"
    aizawl_label_col: str = "Label"

    # REFUGE2 (optic-disc detector training) and IDRiD (OD-detector validation).
    refuge2_root: Path = field(default_factory=lambda: _p("REFUGE2"))
    idrid_seg_root: Path = field(
        default_factory=lambda: _p("data", "raw", "IDRID", "A. Segmentation", "A. Segmentation")
    )
    # IDRiD original images (train set) used to validate the OD detector.
    idrid_images: Path = field(
        default_factory=lambda: _p("data", "raw", "IDRID", "A. Segmentation", "A. Segmentation",
                                   "1. Original Images", "a. Training Set")
    )
    idrid_od_masks: Path = field(
        default_factory=lambda: _p("data", "raw", "IDRID", "A. Segmentation", "A. Segmentation",
                                   "2. All Segmentation Groundtruths", "a. Training Set",
                                   "5. Optic Disc")
    )

    # IDRiD disease-grading set (second external test cohort). Retinopathy grade
    # is on the ICDR 0-4 scale, so it is directly comparable to APTOS/Aizawl.
    # The folder is oddly double-nested in the official release.
    idrid_grade_train_images: Path = field(
        default_factory=lambda: _p("data", "raw", "IDRID", "B. Disease Grading",
                                   "B. Disease Grading", "1. Original Images", "a. Training Set")
    )
    idrid_grade_test_images: Path = field(
        default_factory=lambda: _p("data", "raw", "IDRID", "B. Disease Grading",
                                   "B. Disease Grading", "1. Original Images", "b. Testing Set")
    )
    idrid_grade_train_csv: Path = field(
        default_factory=lambda: _p("data", "raw", "IDRID", "B. Disease Grading",
                                   "B. Disease Grading", "2. Groundtruths",
                                   "a. IDRiD_Disease Grading_Training Labels.csv")
    )
    idrid_grade_test_csv: Path = field(
        default_factory=lambda: _p("data", "raw", "IDRID", "B. Disease Grading",
                                   "B. Disease Grading", "2. Groundtruths",
                                   "b. IDRiD_Disease Grading_Testing Labels.csv")
    )
    # Flat repackaging of the IDRiD grading set (built by build_idrid_flat.py).
    # Preferred by load_idrid_external when present.
    idrid_flat_images: Path = field(
        default_factory=lambda: _p("data", "idrid_grading_flat", "images"))
    idrid_flat_csv: Path = field(
        default_factory=lambda: _p("data", "idrid_grading_flat", "labels.csv"))

    # Outputs
    artifacts: Path = field(default_factory=lambda: _p("artifacts"))
    reports: Path = field(default_factory=lambda: _p("artifacts", "reports"))
    checkpoints: Path = field(default_factory=lambda: _p("artifacts", "checkpoints"))
    cache: Path = field(default_factory=lambda: _p("artifacts", "cache"))

    def ensure_outputs(self) -> None:
        for d in (self.artifacts, self.reports, self.checkpoints, self.cache):
            d.mkdir(parents=True, exist_ok=True)


@dataclass
class TrainConfig:
    """Hyperparameters for fine-tuning RETFound on APTOS."""

    seed: int = SEED              # controls weights init, split, augmentation
    img_size: int = 224           # RETFound / ViT-L native input
    batch_size: int = 8           # fits a 16 GB GPU with AMP
    grad_accum_steps: int = 4     # effective batch 32
    epochs: int = 50
    warmup_epochs: int = 5
    lr: float = 1e-4              # head/backbone fine-tune LR (layer-wise decay applied)
    layer_decay: float = 0.65
    weight_decay: float = 0.05
    drop_path: float = 0.2
    min_lr: float = 1e-6
    label_smoothing: float = 0.1
    use_amp: bool = True
    val_fraction: float = 0.15    # stratified val split (model selection)
    test_fraction: float = 0.15   # held-out in-domain APTOS test (unbiased; 0=legacy)
    num_workers: int = 4
    early_stop_patience: int = 10
    # Normalization arm: "geometric" (baseline) or "physical_scale".
    norm_arm: str = "geometric"
    # Canonical window for the physical arm, in optic-disc diameters. Fixed from
    # the training distribution as 1/rho, where rho is the median ratio of disc
    # diameter to field-of-view width on APTOS. Recorded per run so the window a
    # result came from is never inferred from the code version.
    fov_in_od_units: float = 7.0
    # Backbone selector: "retfound" | "vit_l_imagenet" (ablation).
    backbone: str = "retfound"
    # Path to the RETFound ViT-L checkpoint (.pth). Set via env.
    retfound_ckpt: str = os.environ.get("RETFOUND_CKPT", "")


# ImageNet normalization stats (RETFound fine-tuning uses these per its repo).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


PATHS = DataPaths()
