"""
Prepare TrashNet dataset for EfficientNet classifier training.

TrashNet: https://github.com/garythung/trashnet
Expected input layout:
    <trashnet_root>/
        dataset-resized/
            cardboard/  glass/  metal/  paper/  plastic/  trash/

Output layout (folder-per-class, MVP: plastic/paper/metal only):
    data/processed/classifier/
        train/  val/  test/
            plastic/  paper/  metal/
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np

# Map TrashNet folders → MVP material classes (None = skip)
CATEGORY_MAP: dict[str, str | None] = {
    "plastic":   "plastic",
    "paper":     "paper",
    "cardboard": "paper",
    "metal":     "metal",
    "glass":     None,   # out of MVP scope
    "trash":     None,   # ambiguous
}


def convert(
    trashnet_root: Path,
    out_root: Path,
    val_frac: float = 0.15,
    test_frac: float = 0.10,
    seed: int = 42,
) -> None:
    src_root = trashnet_root / "dataset-resized"
    if not src_root.exists():
        raise FileNotFoundError(f"dataset-resized not found at {src_root}")

    rng = np.random.default_rng(seed)

    for folder, material in CATEGORY_MAP.items():
        if material is None:
            continue
        src_dir = src_root / folder
        if not src_dir.exists():
            print(f"  [warn] folder not found: {src_dir}")
            continue

        files = sorted(src_dir.glob("*.jpg")) + sorted(src_dir.glob("*.png"))
        files_arr = np.array(files)
        rng.shuffle(files_arr)
        files = list(files_arr)

        n = len(files)
        n_test = int(n * test_frac)
        n_val = int(n * val_frac)
        splits = {
            "test":  files[:n_test],
            "val":   files[n_test: n_test + n_val],
            "train": files[n_test + n_val:],
        }

        for split, paths in splits.items():
            dst = out_root / split / material
            dst.mkdir(parents=True, exist_ok=True)
            for p in paths:
                shutil.copy2(p, dst / p.name)

        print(f"  {folder:12s} → {material:8s}: train={len(splits['train'])}  val={len(splits['val'])}  test={len(splits['test'])}")

    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare TrashNet for classification")
    parser.add_argument("trashnet_root", type=Path)
    parser.add_argument(
        "--out", type=Path, default=Path("data/processed/classifier"),
    )
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    convert(args.trashnet_root, args.out, args.val_frac, args.test_frac, args.seed)


if __name__ == "__main__":
    main()
