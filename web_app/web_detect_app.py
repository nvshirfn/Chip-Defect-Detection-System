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
UPLOAD_DIR = ROOT / "web_runs" / "uploads"
RESULT_DIR = ROOT / "web_runs" / "results"
EVAL_DIR = ROOT / "web_runs" / "evaluations"
STEP_DIR = ROOT / "web_runs" / "enhancement_steps"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024

model: YOLO | None = None
_model_cache: dict[str, YOLO] = {}


def _run_dir(name: str) -> Path:
    return ROOT / "runs" / "baseline_no_enhancement" / name / "weights" / "best.pt"


def enhance_crack_steps(image_path: Path, output_dir: Path) -> dict[str, str]:
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
        "Light CLAHE Contrast Enhancement": output_dir / "03_final.jpg",
    }
    cv2.imwrite(str(paths["Grayscale Image"]), gray)
    cv2.imwrite(str(paths["Median Noise Reduction"]), denoised)
    cv2.imwrite(str(paths["Light CLAHE Contrast Enhancement"]), contrast)
    return {label: relative_web_path(path) for label, path in paths.items()}


def enhance_ink_steps(image_path: Path, output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.medianBlur(gray, 3)
    adjusted = cv2.normalize(denoised, None, 0, 255, cv2.NORM_MINMAX)

    # Bottom-hat isolates dark blob-like features (ink stains), then light
    # CLAHE sharpens local contrast on the isolated result -- mirroring
    # enhance_die_ink_v3.m's imbothat + imsubtract + adapthisteq pipeline.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    dark_features = cv2.morphologyEx(adjusted, cv2.MORPH_BLACKHAT, kernel)
    subtracted = cv2.subtract(adjusted, dark_features)
    clahe = cv2.createCLAHE(clipLimit=0.4, tileGridSize=(8, 8))
    final = clahe.apply(subtracted)

    paths = {
        "Grayscale Image": output_dir / "01_grayscale.jpg",
        "Median Noise Reduction": output_dir / "02_median.jpg",
        "Contrast Adjustment": output_dir / "03_adjusted.jpg",
        "Bottom-Hat Dark Features": output_dir / "04_bothat.jpg",
        "Final Enhanced Image": output_dir / "05_final.jpg",
    }
    cv2.imwrite(str(paths["Grayscale Image"]), gray)
    cv2.imwrite(str(paths["Median Noise Reduction"]), denoised)
    cv2.imwrite(str(paths["Contrast Adjustment"]), adjusted)
    cv2.imwrite(str(paths["Bottom-Hat Dark Features"]), dark_features)
    cv2.imwrite(str(paths["Final Enhanced Image"]), final)
    return {label: relative_web_path(path) for label, path in paths.items()}


def enhance_broken_steps(image_path: Path, output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.medianBlur(gray, 3)
    clahe = cv2.createCLAHE(clipLimit=0.4, tileGridSize=(8, 8))
    contrast = clahe.apply(denoised)
    blurred = cv2.GaussianBlur(contrast, (0, 0), sigmaX=1.5)
    sharpened = cv2.addWeighted(contrast, 1.15, blurred, -0.15, 0)

    paths = {
        "Grayscale Image": output_dir / "01_grayscale.jpg",
        "Median Noise Reduction": output_dir / "02_median.jpg",
        "Mild CLAHE Contrast": output_dir / "03_clahe.jpg",
        "Light Unsharp Masking": output_dir / "04_final.jpg",
    }
    cv2.imwrite(str(paths["Grayscale Image"]), gray)
    cv2.imwrite(str(paths["Median Noise Reduction"]), denoised)
    cv2.imwrite(str(paths["Mild CLAHE Contrast"]), contrast)
    cv2.imwrite(str(paths["Light Unsharp Masking"]), sharpened)
    return {label: relative_web_path(path) for label, path in paths.items()}


# Single source of truth for every defect class the app can detect/compare.
# NO_DIE has no enhanced_model/enhanced_data/steps_fn because it never got an
# enhancement pipeline (already ~perfect on raw images), so the UI and
# backend both fall back to "no enhancement available" for it.
CLASS_CONFIG = {
    "DIE_CRACK": {
        "no_enhance_model": _run_dir("crack_no_enhancement"),
        "enhanced_model": _run_dir("crack_enhanced_v3"),
        "no_enhance_data": ROOT / "single_class_raw" / "DIE_CRACK" / "data.yaml",
        "enhanced_data": ROOT / "crack_enhanced_matlab" / "DIE_CRACK" / "data.yaml",
        "enhancement_name": "MATLAB CLAHE (V3)",
        "steps_fn": enhance_crack_steps,
        "final_step_key": "Light CLAHE Contrast Enhancement",
    },
    "DIE_INK": {
        "no_enhance_model": _run_dir("ink_no_enhancement"),
        "enhanced_model": _run_dir("ink_enhanced_matlab_v3"),
        "no_enhance_data": ROOT / "single_class_raw" / "DIE_INK" / "data.yaml",
        "enhanced_data": ROOT / "ink_enhanced_matlab_v3" / "DIE_INK" / "data.yaml",
        "enhancement_name": "MATLAB Bottom-Hat + CLAHE (V3)",
        "steps_fn": enhance_ink_steps,
        "final_step_key": "Final Enhanced Image",
    },
    "DIE_BROKEN": {
        "no_enhance_model": _run_dir("broken_no_enhancement"),
        "enhanced_model": _run_dir("broken_enhanced_python"),
        "no_enhance_data": ROOT / "single_class_raw" / "DIE_BROKEN" / "data.yaml",
        "enhanced_data": ROOT / "broken_enhanced_python" / "DIE_BROKEN" / "data.yaml",
        "enhancement_name": "Python CLAHE + Unsharp Mask",
        "steps_fn": enhance_broken_steps,
        "final_step_key": "Light Unsharp Masking",
    },
    "NO_DIE": {
        "no_enhance_model": _run_dir("no_die_no_enhancement"),
        "enhanced_model": None,
        "no_enhance_data": ROOT / "single_class_raw" / "NO_DIE" / "data.yaml",
        "enhanced_data": None,
        "enhancement_name": None,
        "steps_fn": None,
        "final_step_key": None,
    },
}


def get_model() -> YOLO:
    global model
    if model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
        model = YOLO(str(MODEL_PATH))
    return model


def get_class_model(defect_class: str, variant: str) -> YOLO:
    config = CLASS_CONFIG[defect_class]
    path = config[f"{variant}_model"]
    if path is None:
        raise FileNotFoundError(f"No {variant.replace('_', ' ')} model is available for {defect_class}.")

    cache_key = f"{defect_class}:{variant}"
    if cache_key not in _model_cache:
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        _model_cache[cache_key] = YOLO(str(path))
    return _model_cache[cache_key]


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


def run_single_evaluation(yolo_model: YOLO, data_path: Path, run_name: str, split: str) -> dict[str, object]:
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


def run_class_comparison_evaluation(defect_class: str, split: str) -> dict[str, object]:
    config = CLASS_CONFIG[defect_class]
    prefix = defect_class.lower()

    no_enhancement = run_single_evaluation(
        get_class_model(defect_class, "no_enhance"),
        config["no_enhance_data"],
        f"{prefix}_no_enhancement_{split}",
        split,
    )

    if config["enhanced_data"] is None:
        return {
            "defect_class": defect_class,
            "split": split,
            "no_enhancement": no_enhancement,
            "enhanced": None,
            "differences": None,
            "enhancement_name": None,
        }

    enhanced = run_single_evaluation(
        get_class_model(defect_class, "enhanced"),
        config["enhanced_data"],
        f"{prefix}_enhanced_{split}",
        split,
    )
    differences = {
        key: f"{enhanced['raw'][key] - no_enhancement['raw'][key]:+.4f}" for key in no_enhancement["raw"]
    }

    return {
        "defect_class": defect_class,
        "split": split,
        "no_enhancement": no_enhancement,
        "enhanced": enhanced,
        "differences": differences,
        "enhancement_name": config["enhancement_name"],
    }


def class_model_paths(defect_class: str) -> dict[str, str]:
    config = CLASS_CONFIG[defect_class]
    no_enhance_path = config["no_enhance_model"]
    enhanced_path = config["enhanced_model"]
    return {
        "no_enhancement": relative_web_path(no_enhance_path) if no_enhance_path.exists() else str(no_enhance_path),
        "enhanced": (relative_web_path(enhanced_path) if enhanced_path.exists() else str(enhanced_path))
        if enhanced_path is not None
        else None,
        "enhancement_name": config["enhancement_name"],
    }


@app.route("/", methods=["GET", "POST"])
def index():
    class_keys = list(CLASS_CONFIG.keys())
    context = {
        "model_path": relative_web_path(MODEL_PATH) if MODEL_PATH.exists() else str(MODEL_PATH),
        "error": None,
        "original_image": None,
        "result_image": None,
        "detections": [],
        "confidence": 0.25,
        "evaluation": None,
        "class_keys": class_keys,
        "selected_class": class_keys[0],
        "class_models": {key: class_model_paths(key) for key in class_keys},
        "class_compare": None,
        "class_evaluation": None,
    }

    if request.method == "POST":
        action = request.form.get("action", "detect")
        defect_class = request.form.get("defect_class", class_keys[0])
        if defect_class in class_keys:
            context["selected_class"] = defect_class

        if action == "evaluate":
            split = request.form.get("split", "test")
            if split not in {"val", "test"}:
                context["error"] = "Choose either validation or test split."
                return render_template("index.html", **context)

            context["evaluation"] = run_evaluation(split)
            return render_template("index.html", **context)

        if action == "evaluate_class":
            split = request.form.get("split", "test")
            if split not in {"val", "test"}:
                context["error"] = "Choose either validation or test split."
                return render_template("index.html", **context)

            try:
                context["class_evaluation"] = run_class_comparison_evaluation(defect_class, split)
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

        if action == "compare_class":
            config = CLASS_CONFIG[defect_class]
            has_enhanced = config["enhanced_model"] is not None
            use_already_enhanced = has_enhanced and request.form.get("already_enhanced") == "1"

            steps: dict[str, str] = {}
            enhanced_source = upload_path

            if has_enhanced:
                if use_already_enhanced:
                    steps = {"Pre-Enhanced Input": relative_web_path(upload_path)}
                else:
                    step_dir = STEP_DIR / upload_id
                    steps = config["steps_fn"](upload_path, step_dir)
                    final_label = config["final_step_key"]
                    enhanced_source = step_dir / Path(steps[final_label]).name

            no_enhance_result = {"image": None, "detections": [], "missing": None}
            try:
                no_enhance_result = predict_to_saved_image(
                    get_class_model(defect_class, "no_enhance"),
                    upload_path,
                    f"{upload_id}_{defect_class.lower()}_no_enhancement",
                    confidence,
                )
            except FileNotFoundError as exc:
                no_enhance_result["missing"] = str(exc)

            enhanced_result = None
            if has_enhanced:
                enhanced_result = {"image": None, "detections": [], "missing": None}
                try:
                    enhanced_result = predict_to_saved_image(
                        get_class_model(defect_class, "enhanced"),
                        enhanced_source,
                        f"{upload_id}_{defect_class.lower()}_enhanced",
                        confidence,
                    )
                except FileNotFoundError as exc:
                    enhanced_result["missing"] = str(exc)

            context["class_compare"] = {
                "defect_class": defect_class,
                "enhancement_name": config["enhancement_name"],
                "final_step_label": config["final_step_key"],
                "original_image": relative_web_path(upload_path),
                "steps": steps,
                "no_enhancement": no_enhance_result,
                "enhanced": enhanced_result,
                "already_enhanced": use_already_enhanced,
                "has_enhanced": has_enhanced,
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
