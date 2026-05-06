# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

Always activate the venv before running any Python command:
```bash
source .venv/bin/activate
```

## Common commands

```bash
# Run all tests
pytest tests/

# Run a single test file
pytest tests/test_schema.py -v

# Download datasets
python src/data/download.py taco
python src/data/download.py trashnet

# Convert datasets after downloading
python src/data/convert_taco.py data/raw/taco
python src/data/convert_trashnet.py data/raw/trashnet

# Train
python train/train_detector.py    # YOLOv8s on TACO
python train/train_classifier.py  # EfficientNet-B0 on TrashNet

# Copy trained weights for inference
cp runs/detector/trash_yolov8s/weights/best.pt weights/detector.pt
cp runs/classifier/best.pt weights/classifier.pt

# Run live camera detection
python detect.py
python detect.py --camera 1 --output detections.jsonl
```

## Architecture

Two-stage pipeline: **YOLO detector → EfficientNet-B0 classifier → JSONL logger**

1. `detect.py` — CLI entrypoint. Loads config, builds models, opens camera, runs the loop.
2. `src/pipeline/video.py` — The live camera loop. Reads frames via OpenCV, calls detector on the full frame, crops each bbox, calls classifier on each crop, draws overlay, logs every detection.
3. `src/models/detector.py` — Wraps `ultralytics.YOLO`. Returns `Detection` dataclasses (bbox, confidence, class_id).
4. `src/models/classifier.py` — Wraps `timm` EfficientNet-B0. Takes a cropped RGB numpy array, returns `ClassifierResult` (label, confidence, per-class scores). Class order is alphabetical: `["metal", "paper", "plastic"]` — must match training data folder order.
5. `src/logging/schema.py` — Pydantic `DetectionRecord`. Single source of truth for valid materials and subtypes. All logging goes through this.
6. `src/logging/jsonl_logger.py` — Append-mode writer. Use `make_record()` to build a record, then `logger.log(record)`.

## Config

`configs/inference.yaml` controls runtime behaviour (device, thresholds, output path). Training configs are in `configs/detector.yaml` and `configs/classifier.yaml`. The YOLO dataset descriptor is `configs/detector_data.yaml` — its `path` field is relative to the config file's own directory.

## Data flow

```
data/raw/taco/          → convert_taco.py    → data/processed/detector/   (YOLO format)
data/raw/trashnet/      → convert_trashnet.py → data/processed/classifier/ (folder-per-class)
```

TrashNet categories `cardboard` and `paper` are both mapped to the `paper` class. `glass` and `trash` are dropped (out of MVP scope).

## Label schema

Top-level materials: `plastic`, `paper`, `metal`, `glass`, `organic`, `other`.
MVP trains on plastic / paper / metal only. Valid subtypes per material are defined in `src/logging/schema.py:VALID_SUBTYPES`. Brand detection is Phase 3 (not implemented).

## Weights

Trained weights go in `weights/`. `detect.py` expects `weights/detector.pt` and `weights/classifier.pt`. These are gitignored.
