"""Evaluate EfficientNet-B0 classifier: top-1 accuracy and per-class report."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from src.models.classifier import MATERIAL_CLASSES


def evaluate(weights: Path, data_dir: Path, imgsz: int = 224, device: str = "cpu", batch: int = 64) -> dict:
    import timm

    dev = torch.device(device)
    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=len(MATERIAL_CLASSES))
    state = torch.load(str(weights), map_location=dev, weights_only=True)
    model.load_state_dict(state)
    model.to(dev)
    model.eval()

    data_cfg = timm.data.resolve_data_config({}, model=model)
    transform = timm.data.create_transform(**data_cfg)

    val_dir = data_dir / "val"
    dataset = datasets.ImageFolder(str(val_dir), transform=transform)
    loader = DataLoader(dataset, batch_size=batch, num_workers=4, pin_memory=True)

    correct = 0
    total = 0
    class_correct: dict[int, int] = {}
    class_total: dict[int, int] = {}

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(dev), labels.to(dev)
            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)
            for p, l in zip(preds.cpu().tolist(), labels.cpu().tolist()):
                class_total[l] = class_total.get(l, 0) + 1
                if p == l:
                    class_correct[l] = class_correct.get(l, 0) + 1

    top1 = correct / total if total > 0 else 0.0
    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}

    print(f"\n=== Classifier Results (n={total}) ===")
    print(f"  Top-1 accuracy: {top1:.4f}")
    for idx, cls_name in idx_to_class.items():
        n = class_total.get(idx, 0)
        acc = class_correct.get(idx, 0) / n if n > 0 else 0.0
        print(f"  {cls_name:10s}: {acc:.4f}  ({n} samples)")

    return {"top1": top1, "per_class": {idx_to_class[i]: class_correct.get(i, 0) / class_total.get(i, 1) for i in idx_to_class}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/classifier"))
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args()
    evaluate(args.weights, args.data_dir, args.imgsz, args.device, args.batch)


if __name__ == "__main__":
    main()
