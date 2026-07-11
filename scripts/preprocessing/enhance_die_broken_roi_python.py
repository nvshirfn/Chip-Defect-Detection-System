from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = ROOT / "single_class_v2" / "DIE_BROKEN"
OUTPUT_ROOT = ROOT / "broken_v2split_enhanced_roi_python" / "DIE_BROKEN"
SPLITS = ("train", "valid", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Diagnosis from prior failed attempts: on the old split (no background
# images), CLAHE/sharpening/edge-detection barely hurt DIE_BROKEN. On the
# new split (real background images included), every one of those
# techniques caused a much bigger precision drop -- because they also
# enhance the background pad-grid pattern's natural edges, making
# backgrounds look more defect-like and causing false positives. Since raw
# pixels already perform best within the die region, this version changes
# NOTHING about pixel values -- it only masks out the background (via Otsu
# segmentation, same idea as the ink V4 test) so the model never sees that
# confusing context, without introducing any of the side effects that hurt
# every previous attempt.
ROI_DILATE_SIZE = 15


def segment_die_roi(gray: np.ndarray) -> np.ndarray:
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    if num_labels <= 1:
        return np.full(gray.shape, 255, dtype=np.uint8)

    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = np.where(labels == largest_label, 255, 0).astype(np.uint8)

    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ROI_DILATE_SIZE, ROI_DILATE_SIZE))
    return cv2.dilate(mask, dilate_kernel)


def enhance_broken_image(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    roi_mask = segment_die_roi(gray)

    roi_pixels = image[roi_mask > 0]
    if roi_pixels.size:
        fill_value = roi_pixels.reshape(-1, image.shape[2]).mean(axis=0).astype(np.uint8)
    else:
        fill_value = image.reshape(-1, image.shape[2]).mean(axis=0).astype(np.uint8)

    masked = image.copy()
    masked[roi_mask == 0] = fill_value
    return masked


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
                "  - DIE_BROKEN",
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
        print(f"Masking background for {split} DIE_BROKEN (Python ROI-only): {len(image_paths)} files")

        for image_path in image_paths:
            enhanced = enhance_broken_image(image_path)
            cv2.imwrite(str(out_images / image_path.name), enhanced)

            label_path = src_labels / f"{image_path.stem}.txt"
            if label_path.exists():
                shutil.copy2(label_path, out_labels / label_path.name)

    write_data_yaml()
    print(f"\nDone. Enhanced DIE_BROKEN ROI-only (Python) dataset saved to: {OUTPUT_ROOT}")
    print(f"YOLO data file: {OUTPUT_ROOT / 'data.yaml'}")


if __name__ == "__main__":
    main()
