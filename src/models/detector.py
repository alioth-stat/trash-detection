"""YOLOv8 detector wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class Detection:
    bbox: list[float]       # [x1, y1, x2, y2] pixel coords
    confidence: float
    class_id: int


class Detector:
    def __init__(
        self,
        weights: str | Path,
        conf_threshold: float = 0.40,
        iou_threshold: float = 0.45,
        imgsz: int = 640,
        device: str = "cpu",
    ):
        from ultralytics import YOLO
        self.model = YOLO(str(weights))
        self.conf = conf_threshold
        self.iou = iou_threshold
        self.imgsz = imgsz
        self.device = device

    def predict(self, image: np.ndarray) -> list[Detection]:
        """
        Args:
            image: HxWxC uint8 BGR or RGB numpy array
        Returns:
            List of Detection objects
        """
        results = self.model.predict(
            source=image,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        detections: list[Detection] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append(
                    Detection(
                        bbox=[x1, y1, x2, y2],
                        confidence=float(box.conf[0]),
                        class_id=int(box.cls[0]),
                    )
                )
        return detections
