"""Smoke test for pipeline using mocked detector and classifier."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.logging.jsonl_logger import JSONLLogger
from src.models.detector import Detection
from src.models.classifier import ClassifierResult
from src.pipeline.inference import Pipeline


def _make_fake_image(path: Path) -> None:
    import cv2
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)


def test_pipeline_smoke():
    detector = MagicMock()
    detector.predict.return_value = [
        Detection(bbox=[50.0, 50.0, 200.0, 300.0], confidence=0.88, class_id=0)
    ]

    classifier = MagicMock()
    classifier.predict.return_value = ClassifierResult(
        label="plastic", confidence=0.93, scores={"plastic": 0.93, "paper": 0.05, "metal": 0.02}
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        img_path = tmp / "test.jpg"
        _make_fake_image(img_path)
        log_path = tmp / "out.jsonl"

        with JSONLLogger(log_path, append=False) as logger:
            pipeline = Pipeline(detector, classifier, logger)
            n = pipeline.process_image(img_path)

        assert n == 1
        lines = log_path.read_text().strip().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["material"] == "plastic"
        assert record["confidence"] == pytest.approx(0.88, abs=1e-3)
        assert record["image_id"] == "test.jpg"
