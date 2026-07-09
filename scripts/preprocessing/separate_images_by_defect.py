from __future__ import annotations

import argparse
import shutil
from pathlib import Path


CLASS_NAMES = {
    0: "DIE_BROKEN",
    1: "DIE_CRACK",
    2: "DIE_INK",
    3: "NO_DIE",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy YOLO images into folders based on defect class labels."
    )
    parser.add_argument(
        "--dataset",
        default="ChipDetection.v8i.yolov8",
        help="YOLO dataset folder containing train/valid/test.",
    )
    parser.add_argument(
        "--output",
        default="ChipDetection_by_defect",
        help="Output folder for separated images.",
    )
    parser.add_argument(
        "--mode",
        choices=["first", "multi", "mixed"],
        default="first",
        help=(
            "first: use first label only; multi: copy image into every class it contains; "
            "mixed: put multi-class images into MIXED."
        ),
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move files instead of copying. Not recommended unless you have a backup.",
    )
    return parser.parse_args()


def read_class_ids(label_path: Path) -> list[int]:
    if not label_path.exists() or label_path.stat().st_size == 0:
        return []

    class_ids = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        try:
            class_ids.append(int(float(parts[0])))
        except ValueError:
            continue
    return class_ids


def unique_destination(path: Path) -> Path:
    if not path.exists():
        return path

    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    index = 2
    while True:
        candidate = parent / f"{stem}_{index}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def copy_or_move(src: Path, dst: Path, move: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst = unique_destination(dst)
    if move:
        shutil.move(str(src), str(dst))
    else:
        shutil.copy2(src, dst)


def target_classes(class_ids: list[int], mode: str) -> list[str]:
    if not class_ids:
        return ["BACKGROUND_OR_UNLABELED"]

    names = [CLASS_NAMES.get(class_id, f"CLASS_{class_id}") for class_id in sorted(set(class_ids))]

    if mode == "first":
        return [names[0]]
    if mode == "mixed" and len(names) > 1:
        return ["MIXED"]
    return names


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset)
    output_root = Path(args.output)

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_root}")

    total = 0
    counts: dict[str, int] = {}

    for split in ["train", "valid", "test"]:
        image_dir = dataset_root / split / "images"
        label_dir = dataset_root / split / "labels"

        if not image_dir.exists():
            continue

        for image_path in image_dir.iterdir():
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            label_path = label_dir / f"{image_path.stem}.txt"
            class_ids = read_class_ids(label_path)
            classes = target_classes(class_ids, args.mode)

            for class_name in classes:
                dst_image = output_root / split / class_name / "images" / image_path.name
                copy_or_move(image_path, dst_image, args.move)

                if label_path.exists():
                    dst_label = output_root / split / class_name / "labels" / label_path.name
                    copy_or_move(label_path, dst_label, args.move)

                counts[class_name] = counts.get(class_name, 0) + 1
            total += 1

    print(f"Processed {total} images.")
    print(f"Output folder: {output_root.resolve()}")
    for class_name, count in sorted(counts.items()):
        print(f"{class_name}: {count}")


if __name__ == "__main__":
    main()
