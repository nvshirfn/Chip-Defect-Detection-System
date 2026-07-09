from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np
from skimage.exposure import equalize_adapthist
from skimage.util import img_as_ubyte

ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = ROOT / "ChipDetection_single_class" / "DIE_CRACK"
OUTPUT_ROOT = ROOT / "ChipDetection_single_class_enhanced_v3_python" / "DIE_CRACK"
SPLITS = ("train", "valid", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Same parameters as enhance_die_crack_v3.m: median denoise, then light CLAHE,
# no sharpening. skimage's clip_limit is on the same 0-1 scale as MATLAB's
# adapthisteq ClipLimit, and its default kernel_size (image_shape / 8) matches
# MATLAB's NumTiles [8 8].
CLIP_LIMIT = 0.008


def enhance_crack_image(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.medianBlur(gray, 3)
    enhanced = equalize_adapthist(denoised, clip_limit=CLIP_LIMIT)
    return img_as_ubyte(enhanced)


def write_data_yaml() -> None:
    data_yaml = OUTPUT_ROOT / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {OUTPUT_ROOT.relative_to(ROOT).as_posix()}",
                "train: train/images",
                "val: valid/images",
                "test: test/images",
                "nc: 1",
                "names:",
                "  - DIE_CRACK",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    if not SOURCE_ROOT.exists():
        raise FileNotFoundError(f"Source dataset not found: {SOURCE_ROOT}")

    for split in SPLITS:
        src_images = SOURCE_ROOT / split / "images"
        src_labels = SOURCE_ROOT / split / "labels"
        out_images = OUTPUT_ROOT / split / "images"
        out_labels = OUTPUT_ROOT / split / "labels"

        if not src_images.exists():
            continue

        out_images.mkdir(parents=True, exist_ok=True)
        out_labels.mkdir(parents=True, exist_ok=True)

        image_paths = sorted(
            path for path in src_images.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS
        )
        print(f"Enhancing {split} with V3 (Python): {len(image_paths)} files")

        for image_path in image_paths:
            enhanced = enhance_crack_image(image_path)
            cv2.imwrite(str(out_images / image_path.name), enhanced)

            label_path = src_labels / f"{image_path.stem}.txt"
            if label_path.exists():
                shutil.copy2(label_path, out_labels / label_path.name)

    write_data_yaml()
    print(f"\nDone. Enhanced V3 (Python) dataset saved to: {OUTPUT_ROOT}")
    print(f"YOLO data file: {OUTPUT_ROOT / 'data.yaml'}")


if __name__ == "__main__":
    main()
