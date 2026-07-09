from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np
from skimage.exposure import equalize_adapthist
from skimage.util import img_as_ubyte

ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = ROOT / "single_class_raw" / "DIE_INK"
OUTPUT_ROOT = ROOT / "ink_enhanced_python" / "DIE_INK"
SPLITS = ("train", "valid", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Mirrors enhance_die_ink_v3.m exactly: grayscale -> median denoise ->
# contrast adjustment -> bottom-hat dark-feature isolation -> subtract ->
# light CLAHE. Bottom-hat isolates dark ink stains against their local
# background; CLAHE then sharpens local contrast on the isolated result.
LOW_PERCENTILE = 1.0
HIGH_PERCENTILE = 99.0
BOTHAT_KERNEL_SIZE = 17  # approximates MATLAB's strel("disk", 8)
CLAHE_CLIP_LIMIT = 0.008  # same 0-1 scale as MATLAB's adapthisteq ClipLimit


def imadjust_like(gray: np.ndarray) -> np.ndarray:
    low, high = np.percentile(gray, [LOW_PERCENTILE, HIGH_PERCENTILE])
    if high <= low:
        return gray.copy()

    stretched = (gray.astype(np.float32) - low) * (255.0 / (high - low))
    return np.clip(stretched, 0, 255).astype(np.uint8)


def enhance_ink_image(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.medianBlur(gray, 3)
    adjusted = imadjust_like(denoised)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (BOTHAT_KERNEL_SIZE, BOTHAT_KERNEL_SIZE))
    dark_features = cv2.morphologyEx(adjusted, cv2.MORPH_BLACKHAT, kernel)
    subtracted = cv2.subtract(adjusted, dark_features)

    enhanced = equalize_adapthist(subtracted, clip_limit=CLAHE_CLIP_LIMIT)
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
                "  - DIE_INK",
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
        print(f"Enhancing {split} with Ink V3 (Python): {len(image_paths)} files")

        for image_path in image_paths:
            enhanced = enhance_ink_image(image_path)
            cv2.imwrite(str(out_images / image_path.name), enhanced)

            label_path = src_labels / f"{image_path.stem}.txt"
            if label_path.exists():
                shutil.copy2(label_path, out_labels / label_path.name)

    write_data_yaml()
    print(f"\nDone. Enhanced Ink V3 (Python) dataset saved to: {OUTPUT_ROOT}")
    print(f"YOLO data file: {OUTPUT_ROOT / 'data.yaml'}")


if __name__ == "__main__":
    main()
