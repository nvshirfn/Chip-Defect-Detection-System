from __future__ import annotations

import shutil
from pathlib import Path

import cv2
from skimage.exposure import equalize_adapthist
from skimage.util import img_as_ubyte

ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = ROOT / "single_class_raw" / "DIE_BROKEN"
OUTPUT_ROOT = ROOT / "broken_enhanced_python" / "DIE_BROKEN"
SPLITS = ("train", "valid", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# DIE_BROKEN defects are structural (chipped/fractured edges), not thin lines
# (crack) or dark blobs (ink), so this pipeline emphasizes edge/boundary
# definition instead of just local contrast:
# grayscale -> median denoise -> mild CLAHE -> light unsharp masking.
CLIP_LIMIT = 0.01
SHARPEN_SIGMA = 1.0
SHARPEN_AMOUNT = 0.4


def enhance_broken_image(image_path: Path):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.medianBlur(gray, 3)
    contrast = img_as_ubyte(equalize_adapthist(denoised, clip_limit=CLIP_LIMIT))

    blurred = cv2.GaussianBlur(contrast, (0, 0), sigmaX=SHARPEN_SIGMA)
    sharpened = cv2.addWeighted(
        contrast, 1 + SHARPEN_AMOUNT, blurred, -SHARPEN_AMOUNT, 0
    )
    return sharpened


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
        print(f"Enhancing {split} DIE_BROKEN (Python): {len(image_paths)} files")

        for image_path in image_paths:
            enhanced = enhance_broken_image(image_path)
            cv2.imwrite(str(out_images / image_path.name), enhanced)

            label_path = src_labels / f"{image_path.stem}.txt"
            if label_path.exists():
                shutil.copy2(label_path, out_labels / label_path.name)

    write_data_yaml()
    print(f"\nDone. Enhanced DIE_BROKEN (Python) dataset saved to: {OUTPUT_ROOT}")
    print(f"YOLO data file: {OUTPUT_ROOT / 'data.yaml'}")


if __name__ == "__main__":
    main()
