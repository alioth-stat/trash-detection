"""Train YOLOv8s detector on TACO data."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def train(config_path: Path) -> None:
    from ultralytics import YOLO

    with config_path.open() as f:
        cfg = yaml.safe_load(f)

    model = YOLO(cfg.pop("model", "yolov8s.pt"))
    model.train(**cfg)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/detector.yaml"))
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
