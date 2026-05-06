"""Print and optionally save a per-class confusion matrix for the classifier."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from src.models.classifier import MATERIAL_CLASSES


def confusion_matrix(weights: Path, data_dir: Path, split: str = "val", device: str = "cpu") -> None:
    import timm

    dev = torch.device(device)
    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=len(MATERIAL_CLASSES))
    model.load_state_dict(torch.load(str(weights), map_location=dev, weights_only=True))
    model.to(dev)
    model.eval()

    data_cfg = timm.data.resolve_data_config({}, model=model)
    transform = timm.data.create_transform(**data_cfg)
    dataset = datasets.ImageFolder(str(data_dir / split), transform=transform)
    loader = DataLoader(dataset, batch_size=64, num_workers=4)

    nc = len(dataset.classes)
    matrix = np.zeros((nc, nc), dtype=int)

    with torch.no_grad():
        for images, labels in loader:
            preds = model(images.to(dev)).argmax(dim=1).cpu().numpy()
            for p, l in zip(preds, labels.numpy()):
                matrix[l, p] += 1

    print(f"\nConfusion matrix ({split})  rows=actual  cols=predicted")
    header = f"{'':10s}" + "".join(f"{c:>10s}" for c in dataset.classes)
    print(header)
    for i, cls in enumerate(dataset.classes):
        row = f"{cls:10s}" + "".join(f"{matrix[i, j]:>10d}" for j in range(nc))
        print(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/classifier"))
    parser.add_argument("--split", default="val")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    confusion_matrix(args.weights, args.data_dir, args.split, args.device)


if __name__ == "__main__":
    main()
