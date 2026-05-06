"""EfficientNet-B0 material classifier wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


MATERIAL_CLASSES = ["metal", "paper", "plastic"]   # alphabetical — must match training order


@dataclass
class ClassifierResult:
    label: str
    confidence: float
    scores: dict[str, float]


class Classifier:
    def __init__(
        self,
        weights: str | Path,
        conf_threshold: float = 0.50,
        imgsz: int = 224,
        device: str = "cpu",
    ):
        import timm

        self.device = torch.device(device)
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz

        self.model = timm.create_model(
            "efficientnet_b0",
            pretrained=False,
            num_classes=len(MATERIAL_CLASSES),
        )
        state = torch.load(str(weights), map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

        data_cfg = timm.data.resolve_data_config({}, model=self.model)
        self.transform = timm.data.create_transform(**data_cfg)

    def predict(self, crop: np.ndarray) -> ClassifierResult | None:
        """
        Args:
            crop: HxWxC uint8 RGB numpy array (already cropped to bbox)
        Returns:
            ClassifierResult or None if max confidence < threshold
        """
        img = Image.fromarray(crop)
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        top_idx = int(np.argmax(probs))
        top_conf = float(probs[top_idx])

        if top_conf < self.conf_threshold:
            return None

        return ClassifierResult(
            label=MATERIAL_CLASSES[top_idx],
            confidence=top_conf,
            scores={cls: float(probs[i]) for i, cls in enumerate(MATERIAL_CLASSES)},
        )
