from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field, field_validator


VALID_MATERIALS = {"plastic", "paper", "metal", "glass", "organic", "other"}

VALID_SUBTYPES: dict[str, set[str]] = {
    "plastic": {"bottle", "bag", "cup", "wrapper", "container", "straw", "unknown"},
    "paper":   {"cardboard", "newspaper", "wrapper", "cup", "bag", "unknown"},
    "metal":   {"can", "foil", "lid", "unknown"},
    "glass":   {"bottle", "jar", "unknown"},
    "organic": {"food_waste", "unknown"},
    "other":   {"unknown"},
}


class DetectionRecord(BaseModel):
    timestamp: str = Field(..., description="ISO 8601 UTC")
    image_id: str
    frame_id: Optional[int] = None
    object_id: int
    bbox: list[float] = Field(..., min_length=4, max_length=4)
    confidence: float = Field(..., ge=0.0, le=1.0)
    material: Optional[str] = None
    material_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    subtype: Optional[str] = None
    subtype_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    brand: Optional[str] = None
    brand_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    tracking_id: Optional[str] = None
    notes: Optional[str] = None
    snapshot_path: Optional[str] = None

    @field_validator("material")
    @classmethod
    def validate_material(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in VALID_MATERIALS:
            raise ValueError(f"material '{v}' not in {VALID_MATERIALS}")
        return v

    @field_validator("subtype")
    @classmethod
    def validate_subtype(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in {s for ss in VALID_SUBTYPES.values() for s in ss}:
            raise ValueError(f"subtype '{v}' is not a recognised subtype")
        return v

    @field_validator("bbox")
    @classmethod
    def validate_bbox(cls, v: list[float]) -> list[float]:
        x1, y1, x2, y2 = v
        if x2 <= x1 or y2 <= y1:
            raise ValueError("bbox must have x2 > x1 and y2 > y1")
        return v
