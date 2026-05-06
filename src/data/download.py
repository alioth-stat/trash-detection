"""Helper scripts to download required datasets."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def download_taco(dest: Path = Path("data/raw/taco")) -> None:
    dest.mkdir(parents=True, exist_ok=True)

    if not (dest / "download.py").exists():
        print(f"Cloning TACO into {dest} ...")
        subprocess.run(
            ["git", "clone", "https://github.com/pedropro/TACO.git", str(dest)],
            check=True,
        )

    print("Installing TACO dependencies ...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "requests", "Pillow", "--quiet"],
        check=True,
    )

    ann_path = dest / "data" / "annotations.json"
    if not ann_path.exists():
        raise FileNotFoundError(f"Annotations not found at {ann_path}")

    print("Downloading TACO images from Flickr (this may take 20-30 min, some 404s are normal) ...")
    subprocess.run(
        [sys.executable, "download.py", "--dataset_path", str(ann_path)],
        cwd=str(dest),
        check=True,
    )
    print("TACO download complete.")


def download_trashnet(dest: Path = Path("data/raw/trashnet")) -> None:
    dest.mkdir(parents=True, exist_ok=True)

    if not (dest / "README.md").exists():
        print(f"Cloning TrashNet into {dest} ...")
        subprocess.run(
            ["git", "clone", "https://github.com/garythung/trashnet.git", str(dest)],
            check=True,
        )

    extracted = dest / "dataset-resized"
    if extracted.exists():
        print("TrashNet images already extracted.")
        return

    archive = dest / "dataset-resized.7z"
    if not archive.exists():
        print(
            "\nTrashNet images must be downloaded manually:\n"
            "  1. Go to https://github.com/garythung/trashnet/releases\n"
            "  2. Download dataset-resized.7z\n"
            f"  3. Move it to {archive}\n"
            "  4. Re-run this script\n"
        )
        return

    print("Extracting dataset-resized.7z ...")
    result = subprocess.run(["7z", "x", str(archive), f"-o{dest}"], check=False)
    if result.returncode != 0:
        print("7z not found. Install with: sudo apt install p7zip-full")
    else:
        print("TrashNet extraction complete.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["taco", "trashnet", "all"])
    args = parser.parse_args()
    if args.dataset in ("taco", "all"):
        download_taco()
    if args.dataset in ("trashnet", "all"):
        download_trashnet()
