"""Train EfficientNet-B0 material classifier on TrashNet data."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from torchvision import datasets

from src.data.augment import classifier_train_transform, classifier_val_transform


def train(config_path: Path) -> None:
    import timm

    with config_path.open() as f:
        cfg = yaml.safe_load(f)

    data_dir = Path(cfg["data_dir"])
    classes: list[str] = cfg["classes"]
    nc = len(classes)
    imgsz: int = cfg.get("imgsz", 224)
    batch: int = cfg.get("batch_size", 64)
    epochs: int = cfg.get("epochs", 50)
    lr: float = cfg.get("lr", 1e-3)
    wd: float = cfg.get("weight_decay", 1e-4)
    patience: int = cfg.get("patience", 10)
    save_dir = Path(cfg.get("save_dir", "runs/classifier"))
    device_str: str = str(cfg.get("device", "cpu"))
    device = torch.device("cuda" if device_str != "cpu" and torch.cuda.is_available() else "cpu")

    train_tf = classifier_train_transform(imgsz)
    val_tf = classifier_val_transform(imgsz)

    def albu_to_torchvision(transform):
        import numpy as np
        class Wrapper:
            def __call__(self, img):
                arr = np.array(img)
                return transform(image=arr)["image"]
        return Wrapper()

    train_ds = datasets.ImageFolder(str(data_dir / "train"), transform=albu_to_torchvision(train_tf))
    val_ds   = datasets.ImageFolder(str(data_dir / "val"),   transform=albu_to_torchvision(val_tf))

    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True,  num_workers=cfg.get("workers", 4))
    val_loader   = DataLoader(val_ds,   batch_size=batch, shuffle=False, num_workers=cfg.get("workers", 4))

    model = timm.create_model(cfg.get("backbone", "efficientnet_b0"), pretrained=cfg.get("pretrained", True), num_classes=nc)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    save_dir.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                preds = model(images).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += len(labels)
        acc = correct / total if total > 0 else 0.0
        print(f"Epoch {epoch:3d}/{epochs}  val_acc={acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            no_improve = 0
            torch.save(model.state_dict(), save_dir / "best.pt")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"Best val accuracy: {best_acc:.4f}  saved to {save_dir}/best.pt")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/classifier.yaml"))
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
