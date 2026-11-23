"""Repackage the official IDRiD disease-grading set into a flat folder, and remap
the OD cache keys to match.

The official IDRiD tree is deeply nested, with a doubled
``B. Disease Grading/B. Disease Grading`` folder. This produces:

    data/idrid_grading_flat/
        images/IDRiD_train_001.jpg ... IDRiD_train_413.jpg
        images/IDRiD_test_001.jpg  ... IDRiD_test_103.jpg   (unique names)
        labels.csv   (columns: id_code, grade, split)

Unique names also remove the train/test stem collision at the file level. The OD
cache stores diameters in original-image pixels and the flat files are byte
copies of the same originals, so we remap the existing IDRiD cache keys
(``IDRiD_001__train`` -> ``IDRiD_train_001``) rather than re-run OD inference.

Run:
    python -m src.dr_grading.build_idrid_flat
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pandas as pd

from .config import PATHS

OUT_DIR = PATHS.artifacts.parent / "data" / "idrid_grading_flat"


def _num(name: str) -> str:
    m = re.search(r"(\d+)", str(name))
    return m.group(1) if m else str(name)


def build() -> pd.DataFrame:
    img_out = OUT_DIR / "images"
    img_out.mkdir(parents=True, exist_ok=True)
    sources = [
        ("train", PATHS.idrid_grade_train_images, PATHS.idrid_grade_train_csv),
        ("test", PATHS.idrid_grade_test_images, PATHS.idrid_grade_test_csv),
    ]
    rows = []
    for split, img_dir, csv in sources:
        if not (Path(csv).exists() and Path(img_dir).exists()):
            print(f"[skip] {split}: missing {csv} or {img_dir}")
            continue
        df = pd.read_csv(csv)
        df.columns = [str(c).strip() for c in df.columns]
        name_col = df.columns[0]
        gcol = [c for c in df.columns if "etinopathy" in c.lower()][0]
        for _, r in df.iterrows():
            name = str(r[name_col]).strip()
            if not name.startswith("IDRiD") or pd.isna(r[gcol]):
                continue
            src = Path(img_dir) / f"{name}.jpg"
            if not src.exists():
                continue
            new_stem = f"IDRiD_{split}_{_num(name)}"
            shutil.copyfile(src, img_out / f"{new_stem}.jpg")
            rows.append({"id_code": new_stem, "grade": int(r[gcol]), "split": split})
    labels = pd.DataFrame(rows)
    labels.to_csv(OUT_DIR / "labels.csv", index=False)
    print(f"[flat] wrote {len(labels)} images + labels.csv to {OUT_DIR}")
    return labels


def remap_od_cache(labels: pd.DataFrame) -> None:
    """Rename existing IDRiD cache keys (IDRiD_<num>__<split>) to flat stems."""
    cache_path = PATHS.cache / "od_diameters.json"
    if not cache_path.exists():
        print("[od] no cache to remap")
        return
    cache = json.load(open(cache_path))
    new = {}
    remapped = 0
    for k, v in cache.items():
        m = re.match(r"^IDRiD_(\d+)__(train|test)$", k)
        if m:
            new[f"IDRiD_{m.group(2)}_{m.group(1)}"] = v
            remapped += 1
        else:
            new[k] = v
    json.dump(new, open(cache_path, "w"))
    print(f"[od] remapped {remapped} IDRiD cache keys to flat stems; total {len(new)}")


def main() -> int:
    labels = build()
    remap_od_cache(labels)
    print("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
