from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = ROOT / "single_class_v2" / "DIE_BROKEN"
OUTPUT_ROOT = ROOT / "broken_v2split_enhanced_edge_python" / "DIE_BROKEN"
SPLITS = ("train", "valid", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# DIE_BROKEN resisted every contrast-based enhancement tried so far (CLAHE,
# unsharp masking, both plain and tuned) -- none beat the raw baseline.
# This is a different lever: explicit edge detection (Canny) to extract
# fracture/boundary lines directly, rather than adjusting contrast. Lightly
# blended with the original (not replacing it), since a hard over-sharpen
# already proved to hurt this class by amplifying noise into false edges.
CANNY_LOWER_RATIO = 0.66
CANNY_UPPER_RATIO = 1.33
EDGE_BLEND_WEIGHT = 0.15  # light touch, same lesson learned from the sharpening attempts


def enhance_broken_image(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.medianBlur(gray, 3)

    median_val = float(np.median(denoised))
    lower = int(max(0, CANNY_LOWER_RATIO * median_val))
    upper = int(min(255, CANNY_UPPER_RATIO * median_val))
    edges = cv2.Canny(denoised, lower, upper)
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8))

    enhanced = cv2.addWeighted(denoised, 1 - EDGE_BLEND_WEIGHT, edges, EDGE_BLEND_WEIGHT, 0)
    return enhanced


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
        print(f"Enhancing {split} DIE_BROKEN with edge detection (Python): {len(image_paths)} files")

        for image_path in image_paths:
            enhanced = enhance_broken_image(image_path)
            cv2.imwrite(str(out_images / image_path.name), enhanced)

            label_path = src_labels / f"{image_path.stem}.txt"
            if label_path.exists():
                shutil.copy2(label_path, out_labels / label_path.name)

    write_data_yaml()
    print(f"\nDone. Enhanced DIE_BROKEN edge-detection (Python) dataset saved to: {OUTPUT_ROOT}")
    print(f"YOLO data file: {OUTPUT_ROOT / 'data.yaml'}")


if __name__ == "__main__":
    main()
