"""Two-stage inference pipeline: YOLO detector → EfficientNet classifier → JSONL log."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.models.detector import Detector
from src.models.classifier import Classifier
from src.logging.jsonl_logger import JSONLLogger, make_record


def _crop(image: np.ndarray, bbox: list[float]) -> np.ndarray:
    x1, y1, x2, y2 = (int(v) for v in bbox)
    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    return image[y1:y2, x1:x2]


class Pipeline:
    def __init__(
        self,
        detector: Detector,
        classifier: Classifier,
        logger: JSONLLogger,
    ):
        self.detector = detector
        self.classifier = classifier
        self.logger = logger

    def process_image(
        self,
        image_path: Path,
        frame_id: Optional[int] = None,
    ) -> int:
        """Run pipeline on a single image. Returns number of detections logged."""
        bgr = cv2.imread(str(image_path))
        if bgr is None:
            raise ValueError(f"Cannot read image: {image_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        detections = self.detector.predict(rgb)
        image_id = image_path.name

        for obj_id, det in enumerate(detections):
            crop = _crop(rgb, det.bbox)
            cls_result = None
            if crop.size > 0:
                cls_result = self.classifier.predict(crop)

            record = make_record(
                image_id=image_id,
                object_id=obj_id,
                bbox=det.bbox,
                confidence=det.confidence,
                frame_id=frame_id,
                material=cls_result.label if cls_result else None,
                material_confidence=cls_result.confidence if cls_result else None,
            )
            self.logger.log(record)

        return len(detections)


def build_pipeline(cfg: dict, logger: JSONLLogger) -> Pipeline:
    """Build Pipeline from parsed inference.yaml config."""
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
    return Pipeline(detector, classifier, logger)
