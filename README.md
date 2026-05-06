# Trash Detection

Two-stage computer vision pipeline for real-time trash detection and material classification.

**Pipeline:** YOLOv8s detector → EfficientNet-B0 classifier → JSONL logger → Streamlit dashboard

---

## Requirements

- Python 3.10+
- CUDA-capable GPU (recommended) or CPU
- Webcam for live detection

---

## Setup

### 1. Clone and create virtual environment

```bash
git clone https://github.com/alioth-stat/trash-detection.git
cd trash-detection

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

---

## Datasets

Model weights are not included in the repo. You need to download the datasets and train from scratch.

### Detector — TACO

```bash
python src/data/download.py taco
python src/data/convert_taco.py data/raw/taco
```

This produces `data/processed/detector/` in YOLO format (1 class: `trash`).

### Classifier — TrashNet

Download manually from [https://github.com/garythung/trashnet](https://github.com/garythung/trashnet) — get the `dataset-resized.zip` and extract it to `data/raw/trashnet/` so the structure looks like:

```
data/raw/trashnet/
    dataset-resized/
        cardboard/  glass/  metal/  paper/  plastic/  trash/
```

Then convert:

```bash
python src/data/convert_trashnet.py data/raw/trashnet
```

This produces `data/processed/classifier/` with `train/val/test` splits across 3 classes: `metal`, `paper`, `plastic`.

> `cardboard` is merged into `paper`. `glass` and `trash` are dropped (out of MVP scope).

---

## Training

### Train the detector (YOLOv8s on TACO)

```bash
PYTHONPATH=. python train/train_detector.py
```

Weights saved to `runs/detect/detector/trash_yolov8s/weights/best.pt`.

### Train the classifier (EfficientNet-B0 on TrashNet)

```bash
PYTHONPATH=. python train/train_classifier.py
```

Weights saved to `runs/classifier/best.pt`. Trains for up to 50 epochs with early stopping (patience 10).

### Copy weights for inference

```bash
mkdir -p weights
cp runs/detect/detector/trash_yolov8s/weights/best.pt weights/detector.pt
cp runs/classifier/best.pt weights/classifier.pt
```

---

## Running

### Live detection (webcam)

```bash
PYTHONPATH=. python detect.py
```

Detections are logged to `detections.jsonl`. Press `q` to quit.

Options:
```bash
python detect.py --camera 1          # use a different camera index
python detect.py --output out.jsonl  # custom output file
python detect.py --device cpu        # force CPU
```

### Live dashboard

In a second terminal:

```bash
streamlit run dashboard.py
```

Open [http://localhost:8501](http://localhost:8501). The dashboard reads `detections.jsonl` and shows a live grid of detection snapshots (auto-refreshes every 2 seconds). Snapshots are saved to `data/snapshots/`.

---

## Project structure

```
trash-detection/
├── detect.py                   # CLI entrypoint
├── dashboard.py                # Streamlit dashboard
├── configs/
│   ├── inference.yaml          # Runtime config (thresholds, device)
│   ├── detector.yaml           # YOLOv8 training config
│   ├── classifier.yaml         # EfficientNet training config
│   └── detector_data.yaml      # YOLO dataset descriptor
├── src/
│   ├── models/
│   │   ├── detector.py         # YOLOv8 wrapper → Detection dataclass
│   │   └── classifier.py      # EfficientNet-B0 wrapper → ClassifierResult
│   ├── pipeline/
│   │   └── video.py            # Camera loop (detect → classify → log → save)
│   ├── logging/
│   │   ├── schema.py           # DetectionRecord Pydantic model
│   │   └── jsonl_logger.py     # Append-mode JSONL writer
│   └── data/
│       ├── augment.py          # Albumentations transforms
│       ├── convert_taco.py     # TACO → YOLO format
│       ├── convert_trashnet.py # TrashNet → folder-per-class
│       └── download.py         # Dataset downloader
├── train/
│   ├── train_detector.py
│   └── train_classifier.py
└── tests/
```

---

## Label schema

**Materials (MVP):** `plastic`, `paper`, `metal`

**Supported but not trained:** `glass`, `organic`, `other`

Valid subtypes per material are defined in `src/logging/schema.py`.

---

## Running tests

```bash
pytest tests/ -v
```
