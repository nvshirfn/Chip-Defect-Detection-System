from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np
from skimage.exposure import equalize_adapthist
from skimage.util import img_as_ubyte

ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = ROOT / "single_class_v2" / "DIE_INK"
OUTPUT_ROOT = ROOT / "ink_v2split_enhanced_v4_python" / "DIE_INK"
SPLITS = ("train", "valid", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# V4 hypothesis: DIE_INK had the highest false-positive rate against
# background of any class (60% of all background false positives were
# mislabeled as ink). V1-V3 only changed contrast within the whole image;
# V4 additionally segments out the die region (light square) from the
# background grid (dark pads) via Otsu thresholding, then masks the
# background to a neutral fill so the model never sees the confusing
# texture at all. Keeps the V3 bottom-hat + CLAHE technique on top, since
# that was the best-performing contrast pipeline so far.
LOW_PERCENTILE = 1.0
HIGH_PERCENTILE = 99.0
BOTHAT_KERNEL_SIZE = 17
CLAHE_CLIP_LIMIT = 0.008
ROI_DILATE_SIZE = 15


def imadjust_like(gray: np.ndarray) -> np.ndarray:
    low, high = np.percentile(gray, [LOW_PERCENTILE, HIGH_PERCENTILE])
    if high <= low:
        return gray.copy()
    stretched = (gray.astype(np.float32) - low) * (255.0 / (high - low))
    return np.clip(stretched, 0, 255).astype(np.uint8)


def segment_die_roi(denoised: np.ndarray) -> np.ndarray:
    _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    if num_labels <= 1:
        return np.full(denoised.shape, 255, dtype=np.uint8)

    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = np.where(labels == largest_label, 255, 0).astype(np.uint8)

    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ROI_DILATE_SIZE, ROI_DILATE_SIZE))
    return cv2.dilate(mask, dilate_kernel)


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
    enhanced = img_as_ubyte(equalize_adapthist(subtracted, clip_limit=CLAHE_CLIP_LIMIT))

    roi_mask = segment_die_roi(denoised)
    roi_pixels = enhanced[roi_mask > 0]
    fill_value = int(roi_pixels.mean()) if roi_pixels.size else int(enhanced.mean())
    masked = np.where(roi_mask > 0, enhanced, fill_value).astype(np.uint8)
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
        print(f"Enhancing {split} with Ink V4 ROI-segmented (Python): {len(image_paths)} files")

        for image_path in image_paths:
            enhanced = enhance_ink_image(image_path)
            cv2.imwrite(str(out_images / image_path.name), enhanced)

            label_path = src_labels / f"{image_path.stem}.txt"
            if label_path.exists():
                shutil.copy2(label_path, out_labels / label_path.name)

    write_data_yaml()
    print(f"\nDone. Enhanced Ink V4 (Python) dataset saved to: {OUTPUT_ROOT}")
    print(f"YOLO data file: {OUTPUT_ROOT / 'data.yaml'}")


if __name__ == "__main__":
    main()
