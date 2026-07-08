from __future__ import annotations

import argparse
import shutil
from pathlib import Path


CLASSES = {
    0: "DIE_BROKEN",
    1: "DIE_CRACK",
    2: "DIE_INK",
    3: "NO_DIE",
}

SPLITS = ["train", "valid", "test"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create one single-class YOLO dataset per defect type."
    )
    parser.add_argument("--source", default="ChipDetection.v8i.yolov8", help="Original YOLO dataset folder.")
    parser.add_argument("--output", default="ChipDetection_single_class", help="Output folder.")
    parser.add_argument(
        "--include-backgrounds",
        action="store_true",
        help="Also copy images with empty labels into every single-class dataset.",
    )
    return parser.parse_args()


def image_for_label(image_dir: Path, label_path: Path) -> Path | None:
    for extension in IMAGE_EXTENSIONS:
        candidate = image_dir / f"{label_path.stem}{extension}"
        if candidate.exists():
            return candidate
    return None


def filter_label(label_path: Path, target_class_id: int) -> list[str]:
    if not label_path.exists():
        return []

    lines = []
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        parts = raw_line.strip().split()
        if len(parts) < 5:
            continue

        try:
            class_id = int(float(parts[0]))
        except ValueError:
            continue

        if class_id == target_class_id:
            # Convert target class to 0 because this output dataset has only one class.
            lines.append(" ".join(["0", *parts[1:]]))

    return lines


def write_yaml(dataset_dir: Path, class_name: str) -> None:
    yaml_text = f"""path: {dataset_dir.as_posix()}
train: train/images
val: valid/images
test: test/images

nc: 1
names:
  - {class_name}
"""
    (dataset_dir / "data.yaml").write_text(yaml_text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    source_root = Path(args.source)
    output_root = Path(args.output)

    if not source_root.exists():
        raise FileNotFoundError(f"Source dataset not found: {source_root}")

    output_root.mkdir(parents=True, exist_ok=True)
    counts: dict[str, dict[str, int]] = {name: {split: 0 for split in SPLITS} for name in CLASSES.values()}

    for target_id, class_name in CLASSES.items():
        dataset_dir = output_root / class_name

        for split in SPLITS:
            src_image_dir = source_root / split / "images"
            src_label_dir = source_root / split / "labels"
            dst_image_dir = dataset_dir / split / "images"
            dst_label_dir = dataset_dir / split / "labels"
            dst_image_dir.mkdir(parents=True, exist_ok=True)
            dst_label_dir.mkdir(parents=True, exist_ok=True)

            if not src_label_dir.exists() or not src_image_dir.exists():
                continue

            for label_path in src_label_dir.glob("*.txt"):
                filtered_lines = filter_label(label_path, target_id)
                if not filtered_lines and not args.include_backgrounds:
                    continue

                image_path = image_for_label(src_image_dir, label_path)
                if image_path is None:
                    continue

                shutil.copy2(image_path, dst_image_dir / image_path.name)
                (dst_label_dir / label_path.name).write_text("\n".join(filtered_lines) + ("\n" if filtered_lines else ""), encoding="utf-8")
                counts[class_name][split] += 1

        write_yaml(dataset_dir, class_name)

    print(f"Created single-class datasets in: {output_root.resolve()}")
    for class_name, split_counts in counts.items():
        total = sum(split_counts.values())
        print(
            f"{class_name}: train={split_counts['train']}, "
            f"valid={split_counts['valid']}, test={split_counts['test']}, total={total}"
        )


if __name__ == "__main__":
    main()
