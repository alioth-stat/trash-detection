from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.logging.schema import DetectionRecord


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class JSONLLogger:
    def __init__(self, path: str | Path, append: bool = True):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._mode = "a" if append else "w"
        self._fh = self.path.open(self._mode, encoding="utf-8")

    def log(self, record: DetectionRecord) -> None:
        self._fh.write(record.model_dump_json() + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "JSONLLogger":
        return self

    def __exit__(self, *_) -> None:
        self.close()


def make_record(
    *,
    image_id: str,
    object_id: int,
    bbox: list[float],
    confidence: float,
    frame_id: Optional[int] = None,
    material: Optional[str] = None,
    material_confidence: Optional[float] = None,
    subtype: Optional[str] = None,
    subtype_confidence: Optional[float] = None,
    brand: Optional[str] = None,
    brand_confidence: Optional[float] = None,
    tracking_id: Optional[str] = None,
    notes: Optional[str] = None,
    timestamp: Optional[str] = None,
    snapshot_path: Optional[str] = None,
) -> DetectionRecord:
    return DetectionRecord(
        timestamp=timestamp or _now_iso(),
        image_id=image_id,
        frame_id=frame_id,
        object_id=object_id,
        bbox=bbox,
        confidence=confidence,
        material=material,
        material_confidence=material_confidence,
        subtype=subtype,
        subtype_confidence=subtype_confidence,
        brand=brand,
        brand_confidence=brand_confidence,
        tracking_id=tracking_id,
        notes=notes,
        snapshot_path=snapshot_path,
    )
