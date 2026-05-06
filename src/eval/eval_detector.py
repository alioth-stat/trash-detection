"""Evaluate YOLOv8 detector: compute mAP on a YOLO-format validation split."""

from __future__ import annotations

import argparse
from pathlib import Path


def evaluate(weights: Path, data_yaml: Path, imgsz: int = 640, device: str = "cpu") -> dict:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    metrics = model.val(
        data=str(data_yaml),
        imgsz=imgsz,
        device=device,
        verbose=True,
    )
    results = {
        "mAP50":    float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall":    float(metrics.box.mr),
    }
    print("\n=== Detector Results ===")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("configs/detector_data.yaml"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    evaluate(args.weights, args.data, args.imgsz, args.device)


if __name__ == "__main__":
    main()
