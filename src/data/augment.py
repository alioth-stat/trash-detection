"""Albumentations augmentation pipelines for detector and classifier training."""

from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2


def detector_train_transform(imgsz: int = 640) -> A.Compose:
    return A.Compose(
        [
            A.RandomResizedCrop(size=(imgsz, imgsz), scale=(0.5, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8),
            A.GaussNoise(p=0.2),
            A.MotionBlur(p=0.1),
            A.RandomShadow(p=0.1),
        ],
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
    )


def classifier_train_transform(imgsz: int = 224) -> A.Compose:
    return A.Compose(
        [
            A.RandomResizedCrop(size=(imgsz, imgsz), scale=(0.7, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05, p=0.7),
            A.GaussNoise(p=0.15),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def classifier_val_transform(imgsz: int = 224) -> A.Compose:
    return A.Compose(
        [
            A.Resize(height=imgsz, width=imgsz),  # Resize still uses height/width
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
