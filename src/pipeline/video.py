"""Real-time camera detection loop."""

from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path

from src.models.detector import Detector, Detection
from src.models.classifier import Classifier, ClassifierResult
from src.logging.jsonl_logger import JSONLLogger, make_record

# label colour per material
COLOURS: dict[str, tuple[int, int, int]] = {
    "plastic": (0, 200, 255),
    "paper":   (50, 220, 50),
    "metal":   (180, 100, 255),
    "glass":   (255, 200, 50),
    "organic": (50, 180, 100),
    "other":   (180, 180, 180),
}
DEFAULT_COLOUR = (200, 200, 200)
SNAPSHOT_DIR = Path("data/snapshots")


def _draw(frame: np.ndarray, det: Detection, cls_result: ClassifierResult | None) -> None:
    x1, y1, x2, y2 = (int(v) for v in det.bbox)
    material = cls_result.label if cls_result else "unknown"
    conf = cls_result.confidence if cls_result else det.confidence
    colour = COLOURS.get(material, DEFAULT_COLOUR)

    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

    label = f"{material} {conf:.0%}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), colour, -1)
    cv2.putText(frame, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)


def run_camera(
    detector: Detector,
    classifier: Classifier,
    logger: JSONLLogger,
    camera_index: int = 0,
) -> None:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_index}")

    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    print("Press  q  to quit.")
    frame_id = 0

    while True:
        ret, bgr = cap.read()
        if not ret:
            print("[warn] empty frame, skipping")
            continue

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        detections = detector.predict(rgb)

        for obj_id, det in enumerate(detections):
            x1, y1, x2, y2 = (int(v) for v in det.bbox)
            crop = rgb[max(0, y1):y2, max(0, x1):x2]
            cls_result = classifier.predict(crop) if crop.size > 0 else None

            snap_path = SNAPSHOT_DIR / f"{frame_id}_{obj_id}.jpg"

            logger.log(make_record(
                image_id=f"camera_{camera_index}",
                frame_id=frame_id,
                object_id=obj_id,
                bbox=det.bbox,
                confidence=det.confidence,
                material=cls_result.label if cls_result else None,
                material_confidence=cls_result.confidence if cls_result else None,
                snapshot_path=str(snap_path),
            ))

            _draw(bgr, det, cls_result)

            x1c = max(0, x1 - 4)
            y1c = max(0, y1 - 20)
            x2c = min(bgr.shape[1], x2 + 4)
            y2c = min(bgr.shape[0], y2 + 4)
            snap_crop = bgr[y1c:y2c, x1c:x2c]
            if snap_crop.size > 0:
                cv2.imwrite(str(snap_path), snap_crop, [cv2.IMWRITE_JPEG_QUALITY, 85])

        count_text = f"frame {frame_id}  detections: {len(detections)}"
        cv2.putText(bgr, count_text, (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("Trash Detection", bgr)
        frame_id += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
