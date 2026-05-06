import json
import tempfile
from pathlib import Path

from src.logging.jsonl_logger import JSONLLogger, make_record


def _make_test_record(obj_id: int = 0):
    return make_record(
        image_id="test.jpg",
        object_id=obj_id,
        bbox=[10.0, 20.0, 100.0, 200.0],
        confidence=0.85,
        material="plastic",
        material_confidence=0.91,
    )


def test_logger_writes_valid_jsonl():
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        path = Path(f.name)

    try:
        with JSONLLogger(path, append=False) as logger:
            logger.log(_make_test_record(0))
            logger.log(_make_test_record(1))

        lines = path.read_text().strip().splitlines()
        assert len(lines) == 2
        for line in lines:
            obj = json.loads(line)
            assert obj["material"] == "plastic"
            assert "timestamp" in obj
            assert len(obj["bbox"]) == 4
    finally:
        path.unlink(missing_ok=True)


def test_logger_append_mode():
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        path = Path(f.name)

    try:
        with JSONLLogger(path, append=False) as logger:
            logger.log(_make_test_record(0))

        with JSONLLogger(path, append=True) as logger:
            logger.log(_make_test_record(1))

        lines = path.read_text().strip().splitlines()
        assert len(lines) == 2
    finally:
        path.unlink(missing_ok=True)
