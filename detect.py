"""
Real-time trash detection from a camera feed.

Usage:
    python detect.py
    python detect.py --camera 1 --output detections.jsonl
    python detect.py --config configs/inference.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time trash detection")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--output", type=Path, default=Path("detections.jsonl"))
    parser.add_argument("--config", type=Path, default=Path("configs/inference.yaml"))
    parser.add_argument("--device", default=None, help="Override device: cpu | cuda | mps")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.config.exists():
        print(f"[error] Config not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    with args.config.open() as f:
        cfg = yaml.safe_load(f)

    if args.device:
        cfg["device"] = args.device

    from src.models.detector import Detector
    from src.models.classifier import Classifier
    from src.logging.jsonl_logger import JSONLLogger
    from src.pipeline.video import run_camera

    detector = Detector(
        weights=cfg["detector"]["weights"],
        conf_threshold=cfg["detector"]["conf_threshold"],
        iou_threshold=cfg["detector"]["iou_threshold"],
        imgsz=cfg["detector"]["imgsz"],
        device=cfg.get("device", "cpu"),
    )
    classifier = Classifier(
        weights=cfg["classifier"]["weights"],
        conf_threshold=cfg["classifier"]["conf_threshold"],
        imgsz=cfg["classifier"]["imgsz"],
        device=cfg.get("device", "cpu"),
    )

    print(f"Logging detections to {args.output}")
    with JSONLLogger(args.output, append=True) as logger:
        run_camera(detector, classifier, logger, camera_index=args.camera)


if __name__ == "__main__":
    main()
