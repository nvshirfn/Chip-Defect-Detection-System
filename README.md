# Chip Defect Detection System

YOLOv8-based detection of chip surface defects (`DIE_BROKEN`, `DIE_CRACK`, `DIE_INK`, `NO_DIE`), with a
systematic study of which classical image-processing preprocessing techniques (MATLAB and Python)
actually improve detection accuracy for each defect type.

## Project structure

```
matlab/                  MATLAB preprocessing scripts (run in MATLAB, not Python)
scripts/
  preprocessing/          Dataset builders + Python preprocessing scripts
  training/               train_baseline_yolo.py, evaluate_baseline_yolo.py
  tools/                  show_datasetA_bbox.py
web_app/                  Flask demo app (web_detect_app.py + templates/)
models/                   Base pretrained yolov8n.pt
single_class_raw/         Original per-class dataset (synthetic + real mixed across splits)
single_class_v2/          Properly-split dataset (see "Two dataset versions" below)
runs/baseline_no_enhancement/   All trained model weights, one folder per experiment
```

## Setup

```
py -3.11 -m venv yolo_env
yolo_env\Scripts\activate
pip install ultralytics flask opencv-python scikit-image
```

MATLAB scripts require MATLAB installed locally (Image Processing Toolbox). Run them from MATLAB's
Command Window with the project root as the current folder:
```matlab
cd('path/to/chip-image-processing')
addpath('matlab')
enhance_die_crack_v2split   % or whichever script
```

## Two dataset versions -- read this before trusting any number

- **`single_class_raw`** (original): images randomly split ~70/20/10 across train/valid/test, with
  synthetic and real images mixed at nearly identical ratios in every split. This means valid/test
  are NOT real-only, so metrics from this split can be inflated versus true real-world performance.
- **`single_class_v2`** (properly split, built by `scripts/preprocessing/build_single_class_v2_datasets.py`
  from the raw `chip-surface-defect-dataset` source): real images split ~70/20/10, **synthetic images
  placed 100% in train only**, and background (non-defective) images added at ~10% of each class's
  train positive count. valid/test are 100% real images.

**Always prefer `single_class_v2` results for reporting real-world accuracy.** We found that
`single_class_raw` numbers for `DIE_INK` and `DIE_BROKEN` were significantly inflated by synthetic
test-set contamination (e.g. ink's real recall is ~0.46-0.61, not the ~0.69-0.77 the old split showed).
`DIE_CRACK` was the exception -- synthetic crack images transfer well to real ones, so its old-split
numbers were roughly accurate.

## Recommended models (use these for the web app / demo)

| Class | Model (in `runs/baseline_no_enhancement/`) | Technique | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|
| DIE_CRACK | `crack_v2split_enhanced_matlab` | MATLAB CLAHE | 0.907 | 0.940 | 0.937 | 0.640 |
| DIE_INK | `ink_v2split_enhanced_matlab` | MATLAB bottom-hat + CLAHE (v3) | 0.611 | 0.610 | 0.610 | 0.286 |
| DIE_BROKEN | `broken_v2split_no_enhancement` | none (raw images) | 0.873 | 0.634 | 0.813 | 0.373 |
| NO_DIE | `no_die_v2split_no_enhancement` | none (raw images) | 0.992 | 1.000 | 0.995 | 0.794 |
| Combined 4-class baseline | `yolov8n_512` | none (raw images) | 0.826 | 0.862 | 0.893 | 0.624 |

All numbers above are from `single_class_v2` (real-only test set). Every other trained model in
`runs/baseline_no_enhancement/` is a kept experiment (see below) -- not recommended for production use,
but preserved for the report/comparison evidence.

### Why DIE_INK's recommended number looks worse than other classes

`DIE_INK` also has a version trained/evaluated on the old `single_class_raw` split
(`ink_enhanced_matlab_v3`) that scores much higher (mAP50 0.799 vs 0.610). **Do not use that number as
"the" ink accuracy** -- it's measured on a test set that's ~83% synthetic, so it doesn't reflect real
detection performance. The 0.610 mAP50 above is the honest number: `DIE_INK` has a genuine,
still-unsolved synthetic-to-real generalization gap (most synthetic ink stains don't resemble real ink
stains closely enough for the model to transfer what it learned). This is a legitimate finding, not a
bug -- report it as-is rather than substituting the inflated number.

### Why DIE_BROKEN has no enhancement

Four different preprocessing techniques were tried and evaluated (Python CLAHE+sharpen at two
tuning levels, MATLAB CLAHE+unsharp, Canny edge detection) -- none beat the raw/no-enhancement
baseline on this class, across both dataset splits. Diagnosis: contrast/edge enhancement also
enhances the background pad-grid pattern, making backgrounds look more defect-like and hurting
precision specifically once real background images are in the test set. An ROI-only masking
approach (`enhance_die_broken_roi_python.py`, background masked out, die pixels left completely
raw) was the next hypothesis tested to isolate that effect -- check `runs/baseline_no_enhancement/
broken_v2split_enhanced_roi_python/` for its result if you want to continue this thread.

## Running things

**Train a model:**
```
python scripts/training/train_baseline_yolo.py --data <path-to-data.yaml> --name <run-name> --device 0
```

**Evaluate a model:**
```
python scripts/training/evaluate_baseline_yolo.py --model runs/baseline_no_enhancement/<run-name>/weights/best.pt --data <path-to-data.yaml> --split test
```

**Run the web demo app** (lets you compare no-enhancement vs enhanced models per class, and switch
between the old/new dataset splits):
```
python web_app/web_detect_app.py
```
Then open `http://127.0.0.1:5000`.

## Preprocessing techniques tried, by class

| Class | Techniques tried | Winner |
|---|---|---|
| DIE_CRACK | MATLAB CLAHE, Python CLAHE (skimage) | MATLAB CLAHE, by a small margin |
| DIE_INK | Bottom-hat only (v1), imadjust only (v2), bottom-hat+CLAHE (v3), ROI segmentation+v3 (v4) | v3 (MATLAB and Python are near-identical) |
| DIE_BROKEN | CLAHE+sharpen (2 tunings), MATLAB CLAHE+unsharp, Canny edge detection, ROI-only masking | None beat raw yet (ROI-only pending) |
| NO_DIE | None attempted (already ~perfect on raw images; no faint feature for enhancement to reveal) | N/A |

See commit history for the full experimental trail, including which techniques were tried and
rejected, and the reasoning for each.

## Original raw dataset note

The underlying real+synthetic source data (`chip-surface-defect-dataset/`, not committed here -- see
`.gitignore`) has 2,270 real non-defective samples, 1,241 real defective samples, and ~7,300 synthetic
defective samples, combining `DatasetA/B-Real`, `DatasetA/B-Handcrafted-generated`, and
`DatasetA/B-Semantic-generated` sources. Ask the team for the download link if you need to rebuild
`single_class_v2` from scratch via `scripts/preprocessing/build_single_class_v2_datasets.py`.
