from __future__ import annotations

import argparse
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a baseline YOLO defect detector with no image enhancement."
    )
    parser.add_argument("--data", default="baseline_data.yaml", help="Dataset YAML path.")
    parser.add_argument("--model", default="models/yolov8n.pt", help="YOLO model to train.")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=512, help="Image size.")
    parser.add_argument("--batch", type=int, default=8, help="Batch size.")
    parser.add_argument("--device", default="cpu", help="Use 'cpu' or GPU index such as '0'.")
    parser.add_argument("--workers", type=int, default=0, help="Dataloader workers. Use 0 on Windows.")
    parser.add_argument("--amp", action="store_true", help="Enable automatic mixed precision.")
    parser.add_argument("--project", default="runs/baseline_no_enhancement", help="Output folder.")
    parser.add_argument("--name", default="yolov8n_512", help="Run name.")
    return parser.parse_args()


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    os.environ.setdefault("YOLO_CONFIG_DIR", str(root / ".ultralytics"))

    from ultralytics import YOLO

    args = parse_args()
    project = Path(args.project)
    if not project.is_absolute():
        project = root / project

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        amp=args.amp,
        project=str(project),
        name=args.name,
        exist_ok=True,
        pretrained=True,
    )


if __name__ == "__main__":
    main()
