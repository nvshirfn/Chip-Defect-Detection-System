from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = ROOT / "runs" / "baseline_no_enhancement" / "yolov8n_512" / "weights" / "best.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the baseline YOLO model on the full validation or test set."
    )
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Path to trained .pt model.")
    parser.add_argument("--data", default="baseline_data.yaml", help="Dataset YAML path.")
    parser.add_argument("--split", default="test", choices=["val", "test"], help="Dataset split to evaluate.")
    parser.add_argument("--imgsz", type=int, default=512, help="Image size.")
    parser.add_argument("--batch", type=int, default=8, help="Batch size.")
    parser.add_argument("--device", default="0", help="Use GPU index such as '0' or 'cpu'.")
    parser.add_argument("--workers", type=int, default=0, help="DataLoader workers. Use 0 on Windows for stability.")
    parser.add_argument("--project", default="runs/evaluation_no_enhancement", help="Output folder.")
    parser.add_argument("--name", default="test_metrics", help="Run name.")
    return parser.parse_args()


def main() -> None:
    os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))

    from ultralytics import YOLO

    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    project = Path(args.project)
    if not project.is_absolute():
        project = ROOT / project

    model = YOLO(str(model_path))
    metrics = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(project),
        name=args.name,
        exist_ok=True,
        plots=True,
    )

    results = dict(metrics.results_dict)
    save_dir = Path(metrics.save_dir)
    json_path = save_dir / "summary_metrics.json"
    csv_path = save_dir / "summary_metrics.csv"

    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key, value in results.items():
            writer.writerow([key, value])

    print("\nOverall model performance")
    print(f"Split: {args.split}")
    print(f"Saved to: {save_dir}")
    for key, value in results.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
