"""
Convert TACO COCO-format annotations to YOLO format.

TACO repo: https://github.com/pedropro/TACO
Expected input layout:
    <taco_root>/
        data/
            annotations.json
            batch_1/  batch_2/  ...  (images live here)

Output layout (YOLO):
    data/processed/detector/
        images/train/  images/val/  images/test/
        labels/train/  labels/val/  labels/test/

All categories are collapsed to a single class (trash, id=0).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np


def _coco_bbox_to_yolo(bbox: list[float], img_w: int, img_h: int) -> tuple[float, float, float, float]:
    """COCO bbox [x, y, w, h] → YOLO [cx, cy, w, h] normalised."""
    x, y, w, h = bbox
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h
    return cx, cy, nw, nh


def convert(
    taco_root: Path,
    out_root: Path,
    val_frac: float = 0.15,
    test_frac: float = 0.10,
    seed: int = 42,
) -> None:
    ann_path = taco_root / "data" / "annotations.json"
    if not ann_path.exists():
        raise FileNotFoundError(f"Annotations not found at {ann_path}")

    with ann_path.open() as f:
        coco: dict[str, Any] = json.load(f)

    images_by_id: dict[int, dict] = {img["id"]: img for img in coco["images"]}
    anns_by_image: dict[int, list[dict]] = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    image_ids = list(images_by_id.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(image_ids)

    n = len(image_ids)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    splits = {
        "test":  image_ids[:n_test],
        "val":   image_ids[n_test: n_test + n_val],
        "train": image_ids[n_test + n_val:],
    }

    for split, ids in splits.items():
        img_out = out_root / "images" / split
        lbl_out = out_root / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img_id in ids:
            meta = images_by_id[img_id]
            src = taco_root / "data" / meta["file_name"]
            if not src.exists():
                continue

            # prefix with batch folder to avoid filename collisions across batches
            batch = src.parent.name          # e.g. "batch_1"
            stem  = f"{batch}_{src.stem}"    # e.g. "batch_1_000006"
            dst   = img_out / f"{stem}{src.suffix}"
            shutil.copy2(src, dst)

            label_lines: list[str] = []
            for ann in anns_by_image.get(img_id, []):
                cx, cy, nw, nh = _coco_bbox_to_yolo(
                    ann["bbox"], meta["width"], meta["height"]
                )
                label_lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

            (lbl_out / f"{stem}.txt").write_text("\n".join(label_lines))

    print(f"Done. train={len(splits['train'])}  val={len(splits['val'])}  test={len(splits['test'])}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert TACO → YOLO format")
    parser.add_argument("taco_root", type=Path, help="Root of cloned TACO repo")
    parser.add_argument(
        "--out", type=Path, default=Path("data/processed/detector"),
        help="Output root (default: data/processed/detector)"
    )
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    convert(args.taco_root, args.out, args.val_frac, args.test_frac, args.seed)


if __name__ == "__main__":
    main()
