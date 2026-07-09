from __future__ import annotations

import random
import shutil
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = ROOT / "chip-surface-defect-dataset"
OUTPUT_ROOT = ROOT / "single_class_v2"

# Real defective images/labels are mixed-class (one XML can contain boxes for
# several defect types), so every class scans the same real label folders and
# keeps only the boxes matching its own class.
REAL_DEFECTIVE_DIRS = [
    SOURCE_ROOT / "DatasetA-Real" / "defective",
    SOURCE_ROOT / "DatasetB-Real" / "defective",
]

# Synthetic folders are already single-class per subfolder.
SYNTHETIC_DIRS = {
    "DIE_CRACK": [
        SOURCE_ROOT / "DatasetA-Handcrafted-generated" / "CRACK",
        SOURCE_ROOT / "DatasetA-Semantic-generated" / "CRACK1",
        SOURCE_ROOT / "DatasetA-Semantic-generated" / "CRACK2",
    ],
    "DIE_BROKEN": [
        SOURCE_ROOT / "DatasetA-Handcrafted-generated" / "BROKEN",
        SOURCE_ROOT / "DatasetA-Semantic-generated" / "BROKEN",
        SOURCE_ROOT / "DatasetB-Handcrafted-generated" / "BROKEN",
        SOURCE_ROOT / "DatasetB-Semantic-generated" / "BROKEN",
    ],
    "DIE_INK": [
        SOURCE_ROOT / "DatasetA-Handcrafted-generated" / "INK",
        SOURCE_ROOT / "DatasetA-Semantic-generated" / "INK",
        SOURCE_ROOT / "DatasetB-Handcrafted-generated" / "INK",
        SOURCE_ROOT / "DatasetB-Semantic-generated" / "INK",
    ],
    "NO_DIE": [],  # No synthetic NO_DIE images exist in the source dataset.
}

BACKGROUND_DIRS = [
    SOURCE_ROOT / "DatasetA-Real" / "nondefective",
    SOURCE_ROOT / "DatasetB-Real" / "nondefective",
]

CLASSES = ["DIE_BROKEN", "DIE_CRACK", "DIE_INK", "NO_DIE"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
BACKGROUND_RATIO = 0.10  # background images added as ~10% of each class's train positives
TRAIN_FRAC, VALID_FRAC = 0.70, 0.20  # remainder (0.10) goes to test
SEED = 42


def find_image(image_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTENSIONS:
        candidate = image_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def parse_voc_objects(xml_path: Path) -> tuple[int, int, list[tuple[str, int, int, int, int]]]:
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    width = int(size.findtext("width", "0"))
    height = int(size.findtext("height", "0"))

    objects = []
    for obj in root.findall("object"):
        name = obj.findtext("name", "")
        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue
        xmin = int(float(bndbox.findtext("xmin", "0")))
        ymin = int(float(bndbox.findtext("ymin", "0")))
        xmax = int(float(bndbox.findtext("xmax", "0")))
        ymax = int(float(bndbox.findtext("ymax", "0")))
        objects.append((name, xmin, ymin, xmax, ymax))
    return width, height, objects


def to_yolo_line(xmin: int, ymin: int, xmax: int, ymax: int, width: int, height: int) -> str:
    x_center = ((xmin + xmax) / 2) / width
    y_center = ((ymin + ymax) / 2) / height
    box_w = (xmax - xmin) / width
    box_h = (ymax - ymin) / height
    return f"0 {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}"


def gather_real(class_name: str) -> list[tuple[Path, list[str]]]:
    items = []
    for defective_dir in REAL_DEFECTIVE_DIRS:
        label_dir = defective_dir / "label"
        image_dir = defective_dir / "image"
        if not label_dir.exists():
            continue
        for xml_path in sorted(label_dir.glob("*.xml")):
            width, height, objects = parse_voc_objects(xml_path)
            lines = [
                to_yolo_line(xmin, ymin, xmax, ymax, width, height)
                for name, xmin, ymin, xmax, ymax in objects
                if name == class_name
            ]
            if not lines:
                continue
            image_path = find_image(image_dir, xml_path.stem)
            if image_path is None:
                continue
            items.append((image_path, lines))
    return items


def gather_synthetic(class_name: str) -> list[tuple[Path, list[str]]]:
    items = []
    for class_dir in SYNTHETIC_DIRS[class_name]:
        label_dir = class_dir / "label"
        image_dir = class_dir / "image"
        if not label_dir.exists():
            continue
        for xml_path in sorted(label_dir.glob("*.xml")):
            width, height, objects = parse_voc_objects(xml_path)
            lines = [
                to_yolo_line(xmin, ymin, xmax, ymax, width, height)
                for name, xmin, ymin, xmax, ymax in objects
                if name == class_name
            ]
            if not lines:
                continue
            image_path = find_image(image_dir, xml_path.stem)
            if image_path is None:
                continue
            items.append((image_path, lines))
    return items


def gather_background_pool() -> list[Path]:
    pool = []
    for bg_dir in BACKGROUND_DIRS:
        if not bg_dir.exists():
            continue
        pool.extend(sorted(p for p in bg_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS))
    return pool


def split_three(items: list, seed: int) -> tuple[list, list, list]:
    shuffled = items[:]
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    n_train = round(n * TRAIN_FRAC)
    n_valid = round(n * VALID_FRAC)
    train = shuffled[:n_train]
    valid = shuffled[n_train:n_train + n_valid]
    test = shuffled[n_train + n_valid:]
    return train, valid, test


def write_split(class_name: str, split_name: str, entries: list[tuple[Path, list[str]]]) -> None:
    image_dir = OUTPUT_ROOT / class_name / split_name / "images"
    label_dir = OUTPUT_ROOT / class_name / split_name / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    for src_image, lines, prefix, index in entries:
        dst_name = f"{prefix}_{index:05d}{src_image.suffix.lower()}"
        shutil.copy2(src_image, image_dir / dst_name)
        label_text = "\n".join(lines) + ("\n" if lines else "")
        (label_dir / f"{Path(dst_name).stem}.txt").write_text(label_text, encoding="utf-8")


def write_data_yaml(class_name: str) -> None:
    dataset_dir = OUTPUT_ROOT / class_name
    yaml_text = (
        f"path: {dataset_dir.relative_to(ROOT).as_posix()}\n"
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n\n"
        "nc: 1\n"
        "names:\n"
        f"  - {class_name}\n"
    )
    (dataset_dir / "data.yaml").write_text(yaml_text, encoding="utf-8")


def build_class(class_name: str, background_pool: list[Path]) -> None:
    real_items = gather_real(class_name)
    synthetic_items = gather_synthetic(class_name)

    real_train, real_valid, real_test = split_three(real_items, seed=SEED)
    # All synthetic images go to train only.
    synth_train = synthetic_items

    train_positive_count = len(real_train) + len(synth_train)
    background_total = min(round(train_positive_count * BACKGROUND_RATIO), len(background_pool))
    class_seed = SEED + hash(class_name) % 1000
    sampled_backgrounds = random.Random(class_seed).sample(background_pool, background_total) if background_total else []
    bg_train, bg_valid, bg_test = split_three(sampled_backgrounds, seed=class_seed)

    def tag(entries: list[tuple[Path, list[str]]], prefix: str) -> list[tuple[Path, list[str], str, int]]:
        return [(path, lines, prefix, i) for i, (path, lines) in enumerate(entries)]

    def tag_bg(entries: list[Path], prefix: str) -> list[tuple[Path, list[str], str, int]]:
        return [(path, [], prefix, i) for i, path in enumerate(entries)]

    train_entries = tag(real_train, "real") + tag(synth_train, "synth") + tag_bg(bg_train, "bg")
    valid_entries = tag(real_valid, "real") + tag_bg(bg_valid, "bg")
    test_entries = tag(real_test, "real") + tag_bg(bg_test, "bg")

    write_split(class_name, "train", train_entries)
    write_split(class_name, "valid", valid_entries)
    write_split(class_name, "test", test_entries)
    write_data_yaml(class_name)

    print(
        f"{class_name}: "
        f"train={len(train_entries)} (real={len(real_train)}, synth={len(synth_train)}, bg={len(bg_train)}), "
        f"valid={len(valid_entries)} (real={len(real_valid)}, bg={len(bg_valid)}), "
        f"test={len(test_entries)} (real={len(real_test)}, bg={len(bg_test)})"
    )


def main() -> None:
    if not SOURCE_ROOT.exists():
        raise FileNotFoundError(f"Source dataset not found: {SOURCE_ROOT}")

    background_pool = gather_background_pool()
    print(f"Background pool: {len(background_pool)} real non-defective images\n")

    for class_name in CLASSES:
        build_class(class_name, background_pool)

    print(f"\nDone. Datasets saved under: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
