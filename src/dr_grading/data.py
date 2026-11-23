"""Datasets and dataloaders for APTOS (train/val) and Aizawl (external test).

Key design choices:
- The normalization *arm* (geometric vs physical_scale) is applied inside the
  Dataset via :func:`preprocessing.preprocess`, so swapping arms is a one-line
  config change and both arms share every other step.
- Augmentation is deliberately **scale-preserving** (flips, small rotation,
  mild colour jitter). We avoid RandomResizedCrop/zoom because that would undo
  the physical-scale normalization and confound the central experiment.
- Optic-disc diameters for the physical-scale arm are supplied via an optional
  ``od_diameters`` mapping {image_stem: diameter_px}; missing entries fall back
  to the geometric arm (tracked for reporting in Stage 3).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from .config import IMAGENET_MEAN, IMAGENET_STD, PATHS, SEED
from .preprocessing import PreprocessConfig, preprocess


# Every split frame carries these columns. Constructing an empty DataFrame
# without them yields a column-less frame, and downstream code that indexes
# frame["path"] then raises KeyError instead of seeing an empty cohort.
FRAME_COLS = ["path", "label"]


def od_key(path) -> str:
    """Globally unique OD-cache key for an image path.

    APTOS (hex ids) and Aizawl (numeric ids) live in flat folders and keep their
    bare stem. IDRiD's grading train and test sets both number from IDRiD_001, so
    their stems collide; we disambiguate by the split folder name so each image
    gets its own OD diameter. The same key is used when writing the cache
    (:mod:`infer_od`) and when reading it (dataset + fallback stats).
    """
    p = Path(path)
    parent = p.parent.name.lower()
    if "testing set" in parent:
        return f"{p.stem}__test"
    if "training set" in parent:
        return f"{p.stem}__train"
    return p.stem


def _resolve_path(images_dir: Path, rid: str, ext: str) -> Optional[Path]:
    for key in (rid, f"{rid}{ext}", str(Path(rid).stem)):
        cand = images_dir / key
        if cand.exists():
            return cand
    return None


def load_split_frames(
    val_fraction: float, seed: int = SEED, test_fraction: float = 0.0
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return stratified APTOS train/val/in-domain-test frames + Aizawl external.

    Each frame has columns: ``path`` (str) and ``label`` (int). The returned
    tuple is ``(train, val, intest, external)``.

    - ``val`` is used for early stopping / best-epoch selection.
    - ``intest`` is a held-out in-domain APTOS test set, NEVER used for model
      selection, so its QWK is an unbiased in-domain estimate. When
      ``test_fraction == 0`` it is returned empty (legacy 3-way behaviour).
    - ``external`` is the Aizawl clinical set (the cross-dataset test).
    """
    p = PATHS
    aptos = pd.read_csv(p.aptos_csv)
    aptos[p.aptos_id_col] = aptos[p.aptos_id_col].astype(str).str.strip()
    aptos_rows = []
    for _, r in aptos.iterrows():
        path = _resolve_path(p.aptos_images, r[p.aptos_id_col], p.aptos_ext)
        if path is not None:
            aptos_rows.append({"path": str(path), "label": int(r[p.aptos_label_col])})
    aptos_df = pd.DataFrame(aptos_rows, columns=FRAME_COLS)

    # Carve out the held-out in-domain test set first (stratified), so the
    # train/val selection process never touches it.
    if test_fraction and test_fraction > 0:
        trainval_df, intest_df = train_test_split(
            aptos_df,
            test_size=test_fraction,
            stratify=aptos_df["label"],
            random_state=seed,
        )
        # Keep ``val`` the same fraction of the FULL set as before.
        rel_val = val_fraction / (1.0 - test_fraction)
        train_df, val_df = train_test_split(
            trainval_df,
            test_size=min(0.95, rel_val),
            stratify=trainval_df["label"],
            random_state=seed,
        )
    else:
        train_df, val_df = train_test_split(
            aptos_df,
            test_size=val_fraction,
            stratify=aptos_df["label"],
            random_state=seed,
        )
        intest_df = aptos_df.iloc[0:0].copy()

    aiz = pd.read_csv(p.aizawl_csv)
    aiz[p.aizawl_id_col] = aiz[p.aizawl_id_col].astype(str).str.strip()
    aiz_rows = []
    for _, r in aiz.iterrows():
        path = _resolve_path(p.aizawl_images, r[p.aizawl_id_col], "")
        if path is not None:
            aiz_rows.append({"path": str(path), "label": int(r[p.aizawl_label_col])})
    test_df = pd.DataFrame(aiz_rows, columns=FRAME_COLS)

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        intest_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def load_idrid_external() -> pd.DataFrame:
    """IDRiD disease-grading set (train+test pooled) as an external cohort.

    Returns a frame with columns ``path`` and ``label`` using the ICDR-compatible
    ``Retinopathy grade`` (0-4). Returns an empty frame when the IDRiD grading
    data is not present on this machine, so callers can treat IDRiD as optional.
    """
    p = PATHS
    rows = []
    # Preferred: flat package (unique names, single labels.csv).
    if Path(p.idrid_flat_csv).exists() and Path(p.idrid_flat_images).exists():
        df = pd.read_csv(p.idrid_flat_csv)
        df.columns = [str(c).strip() for c in df.columns]
        gcol = "grade" if "grade" in df.columns else \
            [c for c in df.columns if "etinopathy" in c.lower() or c.lower() == "grade"][0]
        for _, r in df.iterrows():
            name = str(r["id_code"]).strip()
            if not name or pd.isna(r[gcol]):
                continue
            path = _resolve_path(Path(p.idrid_flat_images), name, ".jpg")
            if path is not None:
                rows.append({"path": str(path), "label": int(r[gcol])})
        return pd.DataFrame(rows, columns=FRAME_COLS)
    # Fallback: official nested release (train + test folders).
    for img_dir, csv in ((p.idrid_grade_train_images, p.idrid_grade_train_csv),
                         (p.idrid_grade_test_images, p.idrid_grade_test_csv)):
        if not (Path(csv).exists() and Path(img_dir).exists()):
            continue
        df = pd.read_csv(csv)
        df.columns = [str(c).strip() for c in df.columns]
        name_col = df.columns[0]
        grade_cols = [c for c in df.columns if "etinopathy" in c.lower()]
        if not grade_cols:
            continue
        gcol = grade_cols[0]
        for _, r in df.iterrows():
            name = str(r[name_col]).strip()
            if not name or name.lower().startswith("unnamed") or pd.isna(r[gcol]):
                continue
            path = _resolve_path(Path(img_dir), name, ".jpg")
            if path is not None:
                rows.append({"path": str(path), "label": int(r[gcol])})
    return pd.DataFrame(rows, columns=FRAME_COLS)


def fallback_stats(
    frame: pd.DataFrame, arm: str, od_diameters: Optional[Dict[str, float]]
) -> Dict:
    """How many rows of ``frame`` would fall back to the geometric arm.

    For the ``physical_scale`` arm, an image with no plausible OD diameter is
    processed geometrically (see :func:`preprocessing.normalize_physical_scale`).
    A high fallback rate dilutes the arm contrast, so we report it per split.
    """
    # An absent cohort (for example IDRiD when the dataset is not attached) is
    # reported as empty rather than raising, so one missing optional cohort
    # cannot discard a completed training run.
    if frame is None or len(frame) == 0 or "path" not in frame.columns:
        return {"arm": arm, "n": 0, "n_fallback": 0, "fallback_rate": 0.0}
    if arm != "physical_scale":
        return {"arm": arm, "n": int(len(frame)), "n_fallback": 0, "fallback_rate": 0.0}
    od = od_diameters or {}
    n_fb = 0
    for path in frame["path"]:
        d = od.get(od_key(path))
        if d is None or d <= 1:
            n_fb += 1
    n = int(len(frame))
    return {"arm": arm, "n": n, "n_fallback": int(n_fb),
            "fallback_rate": (n_fb / n if n else 0.0)}


class GradingDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        arm: str,
        pre_cfg: PreprocessConfig,
        train: bool = False,
        od_diameters: Optional[Dict[str, float]] = None,
        base_seed: int = SEED,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.arm = arm
        self.pre_cfg = pre_cfg
        self.train = train
        self.od_diameters = od_diameters or {}
        self.base_seed = int(base_seed)
        self.epoch = 0
        self.mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    def set_epoch(self, epoch: int) -> None:
        """Set the current epoch so augmentation varies deterministically.

        Workers are re-spawned each epoch (``persistent_workers=False``), so the
        updated ``epoch`` is pickled into every worker copy of the dataset.
        """
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.frame)

    # --- scale-preserving augmentation on a uint8 RGB HxWx3 array ---
    def _augment(self, rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if rng.random() < 0.5:
            rgb = rgb[:, ::-1, :]
        if rng.random() < 0.5:
            rgb = rgb[::-1, :, :]
        k = int(rng.integers(0, 4))
        if k:
            rgb = np.rot90(rgb, k=k, axes=(0, 1))
        # small free-angle rotation (scale-preserving)
        if rng.random() < 0.5:
            angle = float(rng.uniform(-15, 15))
            h, w = rgb.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            rgb = cv2.warpAffine(rgb, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
        # mild brightness/contrast jitter
        if rng.random() < 0.5:
            alpha = float(rng.uniform(0.9, 1.1))  # contrast
            beta = float(rng.uniform(-10, 10))    # brightness
            rgb = np.clip(rgb.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(rgb)

    def __getitem__(self, idx: int):
        row = self.frame.iloc[idx]
        img = cv2.imread(row["path"])
        if img is None:
            raise FileNotFoundError(row["path"])
        od = self.od_diameters.get(od_key(row["path"]))
        proc_bgr = preprocess(img, self.arm, self.pre_cfg, od_diameter_px=od)
        rgb = cv2.cvtColor(proc_bgr, cv2.COLOR_BGR2RGB)
        if self.train:
            # Deterministic per (base_seed, epoch, sample) RNG: augmentation is
            # now fully reproducible across reruns and independent of worker id.
            rng = np.random.default_rng((self.base_seed, self.epoch, int(idx)))
            rgb = self._augment(rgb, rng)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0)
        tensor = (tensor - self.mean) / self.std
        return tensor, int(row["label"])


def build_dataloaders(
    arm: str,
    pre_cfg: PreprocessConfig,
    batch_size: int,
    num_workers: int,
    val_fraction: float,
    od_diameters: Optional[Dict[str, float]] = None,
    seed: int = SEED,
    test_fraction: float = 0.0,
) -> Dict[str, DataLoader]:
    train_df, val_df, intest_df, test_df = load_split_frames(
        val_fraction, seed=seed, test_fraction=test_fraction
    )
    loaders = {
        "train": DataLoader(
            GradingDataset(train_df, arm, pre_cfg, train=True, od_diameters=od_diameters,
                           base_seed=seed),
            batch_size=batch_size, shuffle=True, num_workers=num_workers,
            pin_memory=True, drop_last=True,
        ),
        "val": DataLoader(
            GradingDataset(val_df, arm, pre_cfg, train=False, od_diameters=od_diameters),
            batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True,
        ),
        "test": DataLoader(
            GradingDataset(test_df, arm, pre_cfg, train=False, od_diameters=od_diameters),
            batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True,
        ),
    }
    # Held-out in-domain test loader (only when a test split was requested).
    if len(intest_df) > 0:
        loaders["intest"] = DataLoader(
            GradingDataset(intest_df, arm, pre_cfg, train=False, od_diameters=od_diameters),
            batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True,
        )
    # Second external cohort: IDRiD grading set (optional; skipped if absent).
    idrid_df = load_idrid_external()
    if len(idrid_df) > 0:
        loaders["idrid"] = DataLoader(
            GradingDataset(idrid_df, arm, pre_cfg, train=False, od_diameters=od_diameters),
            batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True,
        )
    loaders["_sizes"] = {"train": len(train_df), "val": len(val_df),  # type: ignore
                         "intest": len(intest_df), "test": len(test_df),
                         "idrid": len(idrid_df)}
    return loaders


def class_weights_from_frame(frame: pd.DataFrame, num_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights for the imbalanced APTOS training set."""
    counts = np.bincount(frame["label"].to_numpy(), minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    w = counts.sum() / (num_classes * counts)
    return torch.tensor(w, dtype=torch.float32)
