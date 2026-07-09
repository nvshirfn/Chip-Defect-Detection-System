from __future__ import annotations

import os
import shutil
from pathlib import Path
from uuid import uuid4

import cv2
from flask import Flask, render_template, request, send_from_directory
from ultralytics import YOLO
from werkzeug.utils import secure_filename


ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "runs" / "baseline_no_enhancement" / "yolov8n_512" / "weights" / "best.pt"
CRACK_NO_ENHANCE_MODEL_PATH = ROOT / "runs" / "baseline_no_enhancement" / "crack_no_enhancement" / "weights" / "best.pt"
CRACK_ENHANCED_MODEL_PATH = ROOT / "runs" / "baseline_no_enhancement" / "crack_enhanced_v3" / "weights" / "best.pt"
INK_NO_ENHANCE_MODEL_PATH = ROOT / "runs" / "baseline_no_enhancement" / "ink_no_enhancement" / "weights" / "best.pt"
INK_ENHANCED_MODEL_PATH = ROOT / "runs" / "baseline_no_enhancement" / "ink_enhanced_v2" / "weights" / "best.pt"
UPLOAD_DIR = ROOT / "web_runs" / "uploads"
RESULT_DIR = ROOT / "web_runs" / "results"
EVAL_DIR = ROOT / "web_runs" / "evaluations"
STEP_DIR = ROOT / "web_runs" / "enhancement_steps"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024

model: YOLO | None = None
crack_no_enhance_model: YOLO | None = None
crack_enhanced_model: YOLO | None = None
ink_no_enhance_model: YOLO | None = None
ink_enhanced_model: YOLO | None = None


def get_model() -> YOLO:
    global model
    if model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
        model = YOLO(str(MODEL_PATH))
    return model


def get_crack_no_enhance_model() -> YOLO:
    global crack_no_enhance_model
    if crack_no_enhance_model is None:
        if not CRACK_NO_ENHANCE_MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found: {CRACK_NO_ENHANCE_MODEL_PATH}")
        crack_no_enhance_model = YOLO(str(CRACK_NO_ENHANCE_MODEL_PATH))
    return crack_no_enhance_model


def get_crack_enhanced_model() -> YOLO:
    global crack_enhanced_model
    if crack_enhanced_model is None:
        if not CRACK_ENHANCED_MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found: {CRACK_ENHANCED_MODEL_PATH}")
        crack_enhanced_model = YOLO(str(CRACK_ENHANCED_MODEL_PATH))
    return crack_enhanced_model


def get_ink_no_enhance_model() -> YOLO:
    global ink_no_enhance_model
    if ink_no_enhance_model is None:
        if not INK_NO_ENHANCE_MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found: {INK_NO_ENHANCE_MODEL_PATH}")
        ink_no_enhance_model = YOLO(str(INK_NO_ENHANCE_MODEL_PATH))
    return ink_no_enhance_model


def get_ink_enhanced_model() -> YOLO:
    global ink_enhanced_model
    if ink_enhanced_model is None:
        if not INK_ENHANCED_MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found: {INK_ENHANCED_MODEL_PATH}")
        ink_enhanced_model = YOLO(str(INK_ENHANCED_MODEL_PATH))
    return ink_enhanced_model


def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def relative_web_path(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def extract_detections(results) -> list[dict[str, object]]:
    detections = []
    if not results:
        return detections

    names = results[0].names
    for box in results[0].boxes:
        class_id = int(box.cls[0])
        score = float(box.conf[0])
        xyxy = [round(float(v), 2) for v in box.xyxy[0].tolist()]
        detections.append(
            {
                "label": names.get(class_id, str(class_id)),
                "confidence": f"{score:.2f}",
                "box": xyxy,
            }
        )
    return detections


def save_crack_enhancement_steps(image_path: Path, output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.medianBlur(gray, 3)
    clahe = cv2.createCLAHE(clipLimit=0.6, tileGridSize=(8, 8))
    contrast = clahe.apply(denoised)

    paths = {
        "Grayscale Image": output_dir / "01_grayscale.jpg",
        "Median Noise Reduction": output_dir / "02_median.jpg",
        "Light CLAHE Contrast Enhancement": output_dir / "03_clahe.jpg",
    }

    cv2.imwrite(str(paths["Grayscale Image"]), gray)
    cv2.imwrite(str(paths["Median Noise Reduction"]), denoised)
    cv2.imwrite(str(paths["Light CLAHE Contrast Enhancement"]), contrast)

    return {label: relative_web_path(path) for label, path in paths.items()}


def save_ink_enhancement_steps(image_path: Path, output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.medianBlur(gray, 3)
    adjusted = cv2.normalize(denoised, None, 0, 255, cv2.NORM_MINMAX)

    paths = {
        "Grayscale Image": output_dir / "01_grayscale.jpg",
        "Median Noise Reduction": output_dir / "02_median.jpg",
        "Contrast Adjustment": output_dir / "03_imadjust.jpg",
    }

    cv2.imwrite(str(paths["Grayscale Image"]), gray)
    cv2.imwrite(str(paths["Median Noise Reduction"]), denoised)
    cv2.imwrite(str(paths["Contrast Adjustment"]), adjusted)

    return {label: relative_web_path(path) for label, path in paths.items()}


def predict_to_saved_image(yolo_model: YOLO, source_path: Path, run_name: str, confidence: float) -> dict[str, object]:
    run_dir = RESULT_DIR / run_name
    if run_dir.exists():
        shutil.rmtree(run_dir)

    results = yolo_model.predict(
        source=str(source_path),
        conf=confidence,
        save=True,
        project=str(RESULT_DIR),
        name=run_name,
        exist_ok=True,
        verbose=False,
    )

    result_image_path = run_dir / source_path.name
    return {
        "image": relative_web_path(result_image_path) if result_image_path.exists() else None,
        "detections": extract_detections(results),
    }


def run_evaluation(split: str) -> dict[str, str]:
    run_name = f"{split}_metrics"
    metrics = get_model().val(
        data=str(ROOT / "baseline_data.yaml"),
        split=split,
        imgsz=512,
        batch=8,
        device="0",
        workers=0,
        project=str(EVAL_DIR),
        name=run_name,
        exist_ok=True,
        plots=True,
    )

    raw = dict(metrics.results_dict)
    summary = {
        "Precision": f"{raw.get('metrics/precision(B)', 0):.4f}",
        "Recall": f"{raw.get('metrics/recall(B)', 0):.4f}",
        "mAP50": f"{raw.get('metrics/mAP50(B)', 0):.4f}",
        "mAP50-95": f"{raw.get('metrics/mAP50-95(B)', 0):.4f}",
        "Fitness": f"{raw.get('fitness', 0):.4f}",
    }

    save_dir = Path(metrics.save_dir)
    summary_path = save_dir / "web_summary.txt"
    summary_path.write_text(
        "\n".join([f"{key}: {value}" for key, value in summary.items()]),
        encoding="utf-8",
    )

    return {
        "split": split,
        "save_dir": relative_web_path(save_dir),
        "confusion_matrix": relative_web_path(save_dir / "confusion_matrix.png")
        if (save_dir / "confusion_matrix.png").exists()
        else None,
        "pr_curve": relative_web_path(save_dir / "PR_curve.png") if (save_dir / "PR_curve.png").exists() else None,
        "metrics": summary,
    }


def summarize_metrics(metrics) -> dict[str, float]:
    raw = dict(metrics.results_dict)
    return {
        "Precision": float(raw.get("metrics/precision(B)", 0)),
        "Recall": float(raw.get("metrics/recall(B)", 0)),
        "mAP50": float(raw.get("metrics/mAP50(B)", 0)),
        "mAP50-95": float(raw.get("metrics/mAP50-95(B)", 0)),
        "Fitness": float(raw.get("fitness", 0)),
    }


def run_single_crack_evaluation(yolo_model: YOLO, data_path: Path, run_name: str, split: str) -> dict[str, object]:
    metrics = yolo_model.val(
        data=str(data_path),
        split=split,
        imgsz=512,
        batch=8,
        device="0",
        workers=0,
        project=str(EVAL_DIR),
        name=run_name,
        exist_ok=True,
        plots=True,
    )

    values = summarize_metrics(metrics)
    save_dir = Path(metrics.save_dir)
    return {
        "metrics": {key: f"{value:.4f}" for key, value in values.items()},
        "raw": values,
        "save_dir": relative_web_path(save_dir),
        "confusion_matrix": relative_web_path(save_dir / "confusion_matrix.png")
        if (save_dir / "confusion_matrix.png").exists()
        else None,
    }


def run_crack_comparison_evaluation(split: str) -> dict[str, object]:
    no_enhancement = run_single_crack_evaluation(
        get_crack_no_enhance_model(),
        ROOT / "ChipDetection_single_class" / "DIE_CRACK" / "data.yaml",
        f"crack_no_enhancement_{split}",
        split,
    )
    enhanced = run_single_crack_evaluation(
        get_crack_enhanced_model(),
        ROOT / "ChipDetection_single_class_enhanced_v3" / "DIE_CRACK" / "data.yaml",
        f"crack_enhanced_v3_{split}",
        split,
    )

    differences = {}
    for key in no_enhancement["raw"]:
        differences[key] = f"{enhanced['raw'][key] - no_enhancement['raw'][key]:+.4f}"

    return {
        "split": split,
        "no_enhancement": no_enhancement,
        "enhanced": enhanced,
        "differences": differences,
    }


def run_ink_comparison_evaluation(split: str) -> dict[str, object]:
    no_enhancement = run_single_crack_evaluation(
        get_ink_no_enhance_model(),
        ROOT / "ChipDetection_single_class" / "DIE_INK" / "data.yaml",
        f"ink_no_enhancement_{split}",
        split,
    )
    enhanced = run_single_crack_evaluation(
        get_ink_enhanced_model(),
        ROOT / "ChipDetection_single_class_enhanced_ink_v2" / "DIE_INK" / "data.yaml",
        f"ink_enhanced_v2_{split}",
        split,
    )

    differences = {}
    for key in no_enhancement["raw"]:
        differences[key] = f"{enhanced['raw'][key] - no_enhancement['raw'][key]:+.4f}"

    return {
        "split": split,
        "no_enhancement": no_enhancement,
        "enhanced": enhanced,
        "differences": differences,
    }


@app.route("/", methods=["GET", "POST"])
def index():
    context = {
        "model_path": relative_web_path(MODEL_PATH) if MODEL_PATH.exists() else str(MODEL_PATH),
        "error": None,
        "original_image": None,
        "result_image": None,
        "detections": [],
        "confidence": 0.25,
        "evaluation": None,
        "crack_compare": None,
        "crack_evaluation": None,
        "ink_compare": None,
        "ink_evaluation": None,
        "crack_models": {
            "no_enhancement": relative_web_path(CRACK_NO_ENHANCE_MODEL_PATH)
            if CRACK_NO_ENHANCE_MODEL_PATH.exists()
            else str(CRACK_NO_ENHANCE_MODEL_PATH),
            "enhanced": relative_web_path(CRACK_ENHANCED_MODEL_PATH)
            if CRACK_ENHANCED_MODEL_PATH.exists()
            else str(CRACK_ENHANCED_MODEL_PATH),
        },
        "ink_models": {
            "no_enhancement": relative_web_path(INK_NO_ENHANCE_MODEL_PATH)
            if INK_NO_ENHANCE_MODEL_PATH.exists()
            else str(INK_NO_ENHANCE_MODEL_PATH),
            "enhanced": relative_web_path(INK_ENHANCED_MODEL_PATH)
            if INK_ENHANCED_MODEL_PATH.exists()
            else str(INK_ENHANCED_MODEL_PATH),
        },
    }

    if request.method == "POST":
        action = request.form.get("action", "detect")

        if action == "evaluate":
            split = request.form.get("split", "test")
            if split not in {"val", "test"}:
                context["error"] = "Choose either validation or test split."
                return render_template("index.html", **context)

            context["evaluation"] = run_evaluation(split)
            return render_template("index.html", **context)

        if action == "evaluate_ink":
            split = request.form.get("split", "test")
            if split not in {"val", "test"}:
                context["error"] = "Choose either validation or test split."
                return render_template("index.html", **context)

            try:
                context["ink_evaluation"] = run_ink_comparison_evaluation(split)
            except FileNotFoundError as exc:
                context["error"] = str(exc)
            return render_template("index.html", **context)

        if action == "evaluate_crack":
            split = request.form.get("split", "test")
            if split not in {"val", "test"}:
                context["error"] = "Choose either validation or test split."
                return render_template("index.html", **context)

            try:
                context["crack_evaluation"] = run_crack_comparison_evaluation(split)
            except FileNotFoundError as exc:
                context["error"] = str(exc)
            return render_template("index.html", **context)

        confidence = float(request.form.get("confidence", "0.25"))
        context["confidence"] = confidence

        file = request.files.get("image")
        if file is None or file.filename == "":
            context["error"] = "Please choose an image file."
            return render_template("index.html", **context)

        if not allowed_file(file.filename):
            context["error"] = "Use a JPG, PNG, BMP, JPEG, or WEBP image."
            return render_template("index.html", **context)

        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        RESULT_DIR.mkdir(parents=True, exist_ok=True)

        upload_id = uuid4().hex[:10]
        safe_name = secure_filename(file.filename)
        upload_path = UPLOAD_DIR / f"{upload_id}_{safe_name}"
        file.save(upload_path)

        if action == "compare_crack":
            use_already_enhanced = request.form.get("already_enhanced") == "1"

            if use_already_enhanced:
                steps = {
                    "MATLAB-Enhanced Input": relative_web_path(upload_path),
                }
                enhanced_source = upload_path
            else:
                step_dir = STEP_DIR / upload_id
                steps = save_crack_enhancement_steps(upload_path, step_dir)
                enhanced_source = step_dir / "03_clahe.jpg"

            no_enhance_result = {"image": None, "detections": [], "missing": None}
            enhanced_result = {"image": None, "detections": [], "missing": None}

            try:
                no_enhance_result = predict_to_saved_image(
                    get_crack_no_enhance_model(),
                    upload_path,
                    f"{upload_id}_crack_no_enhancement",
                    confidence,
                )
            except FileNotFoundError as exc:
                no_enhance_result["missing"] = str(exc)

            try:
                enhanced_result = predict_to_saved_image(
                    get_crack_enhanced_model(),
                    enhanced_source,
                    f"{upload_id}_crack_enhanced",
                    confidence,
                )
            except FileNotFoundError as exc:
                enhanced_result["missing"] = str(exc)

            context["crack_compare"] = {
                "original_image": relative_web_path(upload_path),
                "steps": steps,
                "no_enhancement": no_enhance_result,
                "enhanced": enhanced_result,
                "already_enhanced": use_already_enhanced,
            }
            return render_template("index.html", **context)

        if action == "compare_ink":
            use_already_enhanced = request.form.get("already_enhanced") == "1"

            if use_already_enhanced:
                steps = {
                    "MATLAB-Enhanced Ink Input": relative_web_path(upload_path),
                }
                enhanced_source = upload_path
            else:
                step_dir = STEP_DIR / upload_id
                steps = save_ink_enhancement_steps(upload_path, step_dir)
                enhanced_source = step_dir / "03_imadjust.jpg"

            no_enhance_result = {"image": None, "detections": [], "missing": None}
            enhanced_result = {"image": None, "detections": [], "missing": None}

            try:
                no_enhance_result = predict_to_saved_image(
                    get_ink_no_enhance_model(),
                    upload_path,
                    f"{upload_id}_ink_no_enhancement",
                    confidence,
                )
            except FileNotFoundError as exc:
                no_enhance_result["missing"] = str(exc)

            try:
                enhanced_result = predict_to_saved_image(
                    get_ink_enhanced_model(),
                    enhanced_source,
                    f"{upload_id}_ink_enhanced",
                    confidence,
                )
            except FileNotFoundError as exc:
                enhanced_result["missing"] = str(exc)

            context["ink_compare"] = {
                "original_image": relative_web_path(upload_path),
                "steps": steps,
                "no_enhancement": no_enhance_result,
                "enhanced": enhanced_result,
                "already_enhanced": use_already_enhanced,
            }
            return render_template("index.html", **context)

        run_dir = RESULT_DIR / upload_id
        if run_dir.exists():
            shutil.rmtree(run_dir)

        results = get_model().predict(
            source=str(upload_path),
            conf=confidence,
            save=True,
            project=str(RESULT_DIR),
            name=upload_id,
            exist_ok=True,
            verbose=False,
        )

        result_image_path = run_dir / upload_path.name
        detections = extract_detections(results)

        context.update(
            {
                "original_image": relative_web_path(upload_path),
                "result_image": relative_web_path(result_image_path) if result_image_path.exists() else None,
                "detections": detections,
            }
        )

    return render_template("index.html", **context)


@app.route("/web_runs/<path:filename>")
def web_runs(filename: str):
    return send_from_directory(ROOT / "web_runs", filename)


if __name__ == "__main__":
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    STEP_DIR.mkdir(parents=True, exist_ok=True)
    app.run(host="127.0.0.1", port=5000, debug=False)
