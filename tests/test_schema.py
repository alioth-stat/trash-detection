import pytest
from pydantic import ValidationError
from src.logging.schema import DetectionRecord


VALID = dict(
    timestamp="2026-04-24T14:32:01.123Z",
    image_id="test.jpg",
    object_id=0,
    bbox=[10.0, 20.0, 100.0, 200.0],
    confidence=0.85,
    material="plastic",
    subtype="bottle",
)


def test_valid_record():
    r = DetectionRecord(**VALID)
    assert r.material == "plastic"
    assert r.subtype == "bottle"


def test_invalid_material():
    with pytest.raises(ValidationError):
        DetectionRecord(**{**VALID, "material": "wood"})


def test_invalid_subtype():
    with pytest.raises(ValidationError):
        DetectionRecord(**{**VALID, "subtype": "spaceship"})


def test_invalid_bbox():
    with pytest.raises(ValidationError):
        DetectionRecord(**{**VALID, "bbox": [100.0, 200.0, 10.0, 20.0]})


def test_none_material_allowed():
    r = DetectionRecord(**{**VALID, "material": None, "subtype": None})
    assert r.material is None


def test_confidence_range():
    with pytest.raises(ValidationError):
        DetectionRecord(**{**VALID, "confidence": 1.5})
