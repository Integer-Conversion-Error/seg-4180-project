# ParkOpticon YOLO11s Pipeline Documentation

## Table of Contents
1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [The Complete Pipeline](#the-complete-pipeline)
4. [Runs Directory & Experiment Tracking](#runs-directory--experiment-tracking)
5. [Reverting Changes & Experimentation](#reverting-changes--experimentation)
6. [Configuration & CLI Options](#configuration--cli-options)
7. [Troubleshooting & Best Practices](#troubleshooting--best-practices)

---

## Overview

The ParkOpticon YOLO11s pipeline is a complete object detection training workflow designed for identifying vehicles in street-level imagery, with specialized focus on parking enforcement vehicles and police cruisers.

**Key Features:**
- **100% Synthetic Data**: Uses Google Street View backgrounds + Gemini API for vehicle insertion
- **4-Class Detection**: `vehicle`, `enforcement_vehicle`, `police_old`, `police_new`
- **Group-Aware Splitting**: Prevents data leakage by splitting by panorama (`pano_id`)
- **Automated Pipeline**: End-to-end workflow from data collection to model training

---

## Project Structure

### Root-Level Layout
```
parkopticon-yolo11s-bootstrap/
├── scripts/                    # Pipeline automation scripts (numbered by execution order)
├── web_ui/                     # Web UI for label review & batch run management
│   ├── app.py                  # FastAPI backend (supports --run-dir)
│   ├── static/                 # Static assets (CSS, JS)
│   └── templates/              # Jinja2 HTML templates
├── utils/                      # Shared utility modules
│   ├── area_sampler_utils.py   # Area sampling helpers
│   ├── flux_api.py             # Flux image API integration
│   ├── gemini_image_api.py     # Gemini image API integration
│   ├── grok_image_api.py       # Grok image API integration
│   ├── metadata.py             # Image metadata utilities
│   └── preprocessing.py        # Image preprocessing utilities
├── runs/                       # Isolated pipeline run folders (preferred)
│   ├── Benchmark_Run_002/
│   ├── Combined_Run_003/
│   ├── Combined_Run_004_Folded/
│   └── Legacy_Root_Run_001/
├── enforcement-dataset/        # Reference vehicle images for synthesis (by model)
├── police-dataset-old/         # Reference Ottawa Police old livery images
├── police-dataset-new/         # Reference Ottawa Police new livery images
├── holdout_new/                # Holdout test set (images + labels + holdout.yaml)
├── test-videos/                # Test video files for inference
├── temp_touchups/              # Temporary touchup images
├── deleted_images/             # Deleted images log + synthetic cleanup
├── prompts/                    # Prompt configurations (touchup_prompts.json)
├── docs/                       # Project documentation
├── dataset.yaml                # YOLO dataset configuration
├── Makefile                    # Pipeline command shortcuts
├── .env / .env.example         # API keys (GSV_API_KEY, GEMINI_API_KEY)
├── requirements.txt            # Python dependencies
├── ParkOpticon_Pipeline.ipynb  # Jupyter notebook pipeline
├── start_web_ui.bat            # Windows launcher for web UI
├── yolo11n.pt                  # YOLO11n pretrained weights
├── yolo11s.pt                  # YOLO11s pretrained weights
└── yolo26n.pt                  # YOLO26n pretrained weights
```

### Scripts (Complete)

**Core Pipeline (numbered by execution order):**
| Script | Purpose |
|--------|---------|
| `01_make_points_template.py` | Generate CSV template for locations |
| `01b_expand_points_by_area.py` | Auto-generate points via area sampling |
| `02_fetch_streetview.py` | Download Street View images + metadata |
| `02b_crop_bottom.py` | Bottom-crop downloaded images, update manifest |
| `02c_rebuild_manifest.py` | Rebuild manifest from existing images |
| `03_detect_empty_frames.py` | Find images without vehicles |
| `03a_yolo_pass.py` | YOLO detection pass (sub-step of empty detection) |
| `03b_gemini_gate.py` | Gemini JSON scene gating (sub-step of empty detection) |
| `04_synthesize_vehicle_edits.py` | Generate synthetic vehicle edits via Gemini |
| `04b_generate_cardinal_references.py` | Generate cardinal direction reference images |
| `04c_augment_synth_images.py` | Augment synthetic images |
| `05_prelabel_boxes.py` | Auto-generate YOLO format bounding box labels |
| `05b_sweep_prelabel_confidence.py` | Sweep prelabel confidence thresholds |
| `06_split_dataset.py` | Group-aware train/val/test split |
| `07_train_yolo.py` | Train YOLO model |
| `07b_batch_run_manager.py` | Run multiple sequential split+train jobs |
| `08_eval_report.py` | Evaluate model and generate report |
| `09_analyze_dataset.py` | Dataset analysis utilities |
| `09_count_boxes.py` | Count bounding boxes in dataset |
| `09_detect_duplicates.py` | Detect duplicate images |
| `09_visualize_boxes.py` | Visualize bounding boxes on images |
| `10_collect_hard_negatives.py` | Collect hard negative examples |
| `11_live_inference.py` | Live inference script |
| `12_run_media_inference.py` | Run inference on media files |

**Utility Scripts (unnumbered):**
| Script | Purpose |
|--------|---------|
| `audit_autogen_labels.py` | Audit auto-generated labels |
| `cleanup_autogen_labels.py` | Clean up auto-generated labels |
| `prune_massive_synth_boxes.py` | Prune oversized synthetic bounding boxes |
| `visualize_synth_boxes.py` | Visualize synthetic bounding boxes |

### Run Directory Conventions (Important)
- **Per-run isolation is the preferred pattern.** Each run under `runs/<run_name>/` has its own `data/`, `manifests/`, and `lists/`.
- Do not mix root and run-scoped artifacts in the same pipeline execution.
- `scripts/04_synthesize_vehicle_edits.py` supports `--run-dir` directly.
- `scripts/05_prelabel_boxes.py` does not have `--run-dir`; use explicit paths:
  - `--manifest runs/<run_name>/manifests/images.csv`
  - `--out-dir runs/<run_name>/data/labels_autogen`
- `web_ui/app.py` supports run scoping via `--run-dir` and directory switching endpoints (`/api/directories/current`, `/api/directories/set`).

### Run Directory Internal Structure
```
runs/<run_name>/
├── data/
│   ├── images_original/       # Downloaded Street View images
│   ├── images_synth/          # Synthetic vehicle edits
│   ├── labels_autogen/        # Auto-generated labels
│   ├── labels_final/          # Manually reviewed labels
│   └── splits/                # Train/val/test splits
├── manifests/
│   ├── points.csv             # Input locations
│   └── images.csv             # Image manifest
└── lists/                     # Processing lists (empty candidates, valid road, etc.)
```

### Key Configuration Files

| File | Purpose |
|------|---------|
| `dataset.yaml` | YOLO dataset configuration (train/val/test paths, class names) |
| `holdout_new/holdout.yaml` | Holdout dataset configuration for testing |
| `Makefile` | Pipeline command shortcuts |
| `.env` | API keys (GSV_API_KEY, GEMINI_API_KEY) |
| `.env` | API keys (GSV_API_KEY, GEMINI_API_KEY) |

---

## The Complete Pipeline

The YOLO training pipeline follows a structured 9-step process. Each step is idempotent and can be run independently (with dependencies noted).

### Pipeline Flowchart
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  1. MAKE POINTS │ -> │  2. FETCH GSV   │ -> │  2b. CROP       │
│  (Location CSV) │    │  (Download)     │    │  (Remove Overlay)│
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                 │
┌─────────────────┐    ┌─────────────────┐      ▼
│  3. EMPTY DET.  │ <- │  4. SYNTHESIZE  │    ┌─────────────────┐
│  (Find No-Veh)  │    │  (Add Vehicles) │    │  5. PRELABEL    │
└─────────────────┘    └─────────────────┘    │  (Auto-Labels)  │
    │              ▲        │                 └─────────────────┘
    ▼              │        ▼                          │
┌─────────────────┐    ┌─────────────────┐      ┌─────▼─────┐
│  6. LABELER UI  │    │  7. SPLIT       │      │  MANUAL   │
│  (Review)       │    │  (70/20/10)     │      │  REVIEW   │
└───────┬─────────┘    └────────┬────────┘      └───────────┘
        │                       │
        └───────────────────────▼
              ┌──────────────────┐
              │  8. TRAIN YOLO   │
              │  9. EVALUATE     │
              └──────────────────┘
```

### Step-by-Step Execution

#### Step 1: Make Points Template
**Script:** `scripts/01_make_points_template.py`
**Command:** `python scripts/01_make_points_template.py`

Generates a CSV template for defining Street View locations:
```bash
python scripts/01_make_points_template.py -o manifests/points.csv -n 2
```

**CSV Format:**
```csv
street,label,location,heading,pitch,fov,radius
downtown_main,downtown,"40.7128,-74.0060",180,0,80,50
```

**Key Columns:**
- `location`: Lat/Lng coordinate (e.g., "40.7128,-74.0060")
- `heading`: 0-360° camera angle (0,45,90,135,180,225,270,315)
- `fov`: Field of view (30-120)
- `radius`: Search radius in meters

**Area Expansion Option:**
```bash
# Auto-generate points within a bounding box
python scripts/01b_expand_points_by_area.py \
  --points_csv manifests/points.csv \
  --bbox "40.5,-74.1,40.9,-73.7" \
  --target_panos 200
```

---

#### Step 2: Fetch Street View Images
**Script:** `scripts/02_fetch_streetview.py`
**Command:** `make fetch`

Downloads Google Street View static images for all locations in `points.csv`.

**Outputs:**
- `data/images_original/{street}/{image_id}.jpg`
- `manifests/images.csv` (metadata for all downloaded images)

**CLI Options:**
```bash
python scripts/02_fetch_streetview.py \
  --api-key YOUR_KEY \
  --resume  # Continue interrupted downloads
```

**Note:** Requires `GSV_API_KEY` in `.env` file.

---

#### Step 2b: Crop Bottom Overlay
**Script:** `scripts/02b_crop_bottom.py`
**Command:** `make crop`

Removes bottom 30px overlay (copyright watermarks, UI elements).

**Outputs:**
- Cropped images with `_bc30` suffix (e.g., `imageid_bc30.jpg`)
- Updated `manifests/images.csv` with new file paths

**CLI Options:**
```bash
python scripts/02b_crop_bottom.py --crop-px 30 --dry-run
```

**Critical:** If cropping is used, apply the same crop at inference time.

---

#### Step 3: Detect Empty Frames
**Script:** `scripts/03_detect_empty_frames.py`
**Command:** `make empty`

Uses pretrained YOLO11s (COCO weights) to detect images with no vehicles.

**Outputs:**
- `lists/empty_candidates.txt`
- `lists/valid_road_candidates.txt` (preferred for synthesis)
- `lists/non_road_candidates.txt`
- `lists/non_street_candidates.txt`

**CLI Options:**
```bash
python scripts/03_detect_empty_frames.py \
  --conf 0.25 \          # Lower threshold (0.15-0.25) to avoid false negatives
  --device cpu \
  --nuke  # Reset detection artifacts (preserves original images)
```

**Quality Safeguard:**
Use a **low confidence threshold (0.15-0.25)**. If you miss a partially occluded vehicle, you'll insert a synthetic vehicle into a frame that already has one, creating confusing labels.

---

#### Step 4: Synthesize Vehicle Edits
**Script:** `scripts/04_synthesize_vehicle_edits.py`
**Command:** `make synth`

Uses Google Gemini API to insert vehicles into empty frames.

**Generation Strategy:**
For each empty frame, generate:
1. **1 random_vehicle** (Class 0)
2. **1 enforcement_vehicle** (Class 1)
3. **1 police_old** (Class 2)
4. **1 police_new** (Class 3)

**CLI Options:**
```bash
python scripts/04_synthesize_vehicle_edits.py \
  --enforcement-rate 0.2 \
  --seed 42 \
  --model gemini-1.5-pro
```

**Cost:** ~$6.23 for 2,658 images
**Time:** ~1 hour 11 minutes

**Requirements:** `GEMINI_API_KEY` in `.env`

---

#### Step 5: Prelabel Boxes
**Script:** `scripts/05_prelabel_boxes.py`
**Command:** `make prelabel`

Auto-generates YOLO format bounding box labels.

**Method:**
- **Synthetic images:** Uses image differencing to find inserted vehicle
- **Original images:** Uses pretrained YOLO to detect vehicles

**Outputs:**
- `data/labels_autogen/{image_id}.txt`
- `reports/prelabel_vehicle_count_hist.png`

**CLI Options:**
```bash
python scripts/05_prelabel_boxes.py \
  --conf 0.25 \
  --qa-threshold-area 0.15 \
  --qa-skip  # Skip QA checks
```

**Rejected-image behavior:**
- Default excludes rows where `review_status=rejected`
- Override with `--include-rejected`

---

#### Step 6: Manual Label Review
**Script:** `web_ui/app.py` (FastAPI backend)
**Command:** `make labeler`

Web UI for reviewing and correcting auto-generated labels.

**Access:** Open http://localhost:8000 in browser

**Rejected-image behavior:**
- Labeler queues hide rows with `review_status=rejected` by default
- Start UI with `--labeler-include-rejected` to include them

**Review Process:**
1. Review bounding box accuracy
2. Delete nonsensical synthetic images
3. Correct mislabeled vehicles
4. Save final labels to `data/labels_final/`

---

#### Step 7: Split Dataset
**Script:** `scripts/06_split_dataset.py`
**Command:** `make split`

Group-aware splitting by `pano_id` to prevent data leakage.

**Split Ratios:**
- Train: 70%
- Val: 20%
- Test: 10%

**Thresholds (enforcement classes):**
| Class | Val Minimum | Test Minimum |
|-------|-------------|--------------|
| enforcement_vehicle | 50 instances | 30 instances |
| police_old | 10 instances | 5 instances |
| police_new | 10 instances | 5 instances |

**CLI Options:**
```bash
python scripts/06_split_dataset.py \
  --train-ratio 0.7 \
  --val-ratio 0.2 \
  --test-ratio 0.1 \
  --min-enforcement-val 50 \
  --min-enforcement-test 30
```

**Rejected-image behavior:**
- Default excludes rows where `review_status=rejected`
- Override with `--include-rejected`
- Split output directories are rebuilt each run, so stale rejected files are removed from `data/splits/{train,val,test}`

**Outputs:**
```
data/splits/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

---

#### Step 8: Train YOLO11s
**Script:** `scripts/07_train_yolo.py`
**Command:** `make train`

Fine-tunes YOLO11s on custom dataset.

**CLI Options:**
```bash
python scripts/07_train_yolo.py \
  --data dataset.yaml \
  --manifest manifests/images.csv \
  --model yolo11s.pt \
  --epochs 50 \
  --imgsz 640 \
  --batch 16 \
  --device 0 \  # GPU device
  --name parkopticon_vehicle_enforcement
```

**Key Parameters:**
- `--model`: Pretrained weights (`yolo11s.pt`) or config (`yolo11s.yaml`)
- `--device`: CPU or GPU (e.g., `0` for first GPU)
- `--resume`: Resume from last checkpoint

**Rejected-image behavior:**
- Training performs rejection safety checks by default:
  - Blocks if rejected rows are still assigned to `train`/`val`/`test` in manifest
  - Blocks if rejected image IDs are still physically present in split image directories
- Override with `--include-rejected`

---

#### Step 9: Evaluate Model
**Script:** `scripts/08_eval_report.py`
**Command:** `make eval`

Evaluates model on test set and generates metrics report.

**CLI Options:**
```bash
python scripts/08_eval_report.py \
  --weights runs/detect/parkopticon_vehicle_enforcement/weights/best.pt \
  --data dataset.yaml
```

**Outputs:**
- `reports/eval_report.md`
- `reports/eval_report.json`

---

## Runs Directory & Experiment Tracking

### Directory Structure
Ultralytics YOLO automatically creates a structured runs directory for each experiment.

**Default Location:** `runs/detect/`

**Experiment Structure:**
```
runs/detect/
└── parkopticon_vehicle_enforcement/
    ├── weights/
    │   ├── best.pt      # Best checkpoint
    │   └── last.pt       # Final checkpoint
    ├── args.yaml        # Full configuration used
    ├── results.png      # Training curves
    ├── results.csv      # Per-epoch metrics
    ├── confusion_matrix.png
    ├── val_batchX_labels.jpg
    └── ...
```

### Custom Run Names
Specify custom names to manage multiple experiments:

```bash
# Experiment 1: Baseline
python scripts/07_train_yolo.py --name baseline_v1

# Experiment 2: Different learning rate
python scripts/07_train_yolo.py --name experiment_lr_adjust

# Experiment 3: Different image size
python scripts/07_train_yolo.py --name experiment_imgsz_416 --imgsz 416
```

### Experiment Comparison
Compare different runs by comparing:
- `results.png` (training/validation curves)
- `results.csv` (per-epoch metrics)
- `confusion_matrix.png`
- `eval_report.md` from evaluation step

### Automatic Logging
Ultralytics automatically logs:
- TensorBoard logs (in `runs/detect/runs/.../events.out.tfevents.*`)
- Training metadata in `metadata.json`
- Checkpoints every `save_period` epochs (default: 10)

---

## Reverting Changes & Experimentation

### Option 1: Create New Run Directory (Recommended)
Best for benchmarking different parameters without affecting existing data.

**Steps:**
```bash
# 1. Create fresh benchmark directory
mkdir -p Benchmark_Run_002/data Benchmark_Run_002/manifests Benchmark_Run_002/lists

# 2. Copy original images (no synth, no excluded)
cp -r data/images_original Benchmark_Run_002/data/

# 3. Copy starting points
cp manifests/points.csv Benchmark_Run_002/manifests/

# 4. Run pipeline in new directory
cd Benchmark_Run_002
python ../scripts/02_fetch_streetview.py --run-dir .
python ../scripts/03_detect_empty_frames.py --run-dir .
# ... continue with other stages
```

### Option 2: Clean In-Place (Destructive)
Reset pipeline state while keeping original images.

**Warning:** This removes generated artifacts. Backup first if needed.

**Steps:**
```bash
# 1. Clear all list files
rm -f lists/*.txt

# 2. Remove generated boxes
rm -f manifests/boxes_autogen*.jsonl

# 3. Remove synthetic images
rm -rf data/images_synth/*

# 4. Remove auto-generated labels
rm -rf data/labels_autogen/*

# 5. Remove final labels (backup first if needed)
rm -rf data/labels_final/*

# 6. Move excluded images back to original location
mv data/images_excluded/non_road/* data/images_original/ 2>/dev/null
mv data/images_excluded/non_street/* data/images_original/ 2>/dev/null
rm -rf data/images_excluded/*

# 7. Reset manifest processing columns
# Requires backup of images.csv from BEFORE processing
cp manifests/images_post_fetch_backup.csv manifests/images.csv
```

**Quick Clean Script:**
```bash
# Use --nuke flag on detection stage
python scripts/03_detect_empty_frames.py --nuke
```

### Experimenting with Different Parameters

**1. Different Confidence Thresholds:**
```bash
# Empty detection with lower threshold (more conservative)
python scripts/03_detect_empty_frames.py --conf 0.15

# Empty detection with higher threshold (more aggressive)
python scripts/03_detect_empty_frames.py --conf 0.30
```

**2. Different Image Sizes:**
```bash
# Smaller image size (faster training)
python scripts/07_train_yolo.py --imgsz 416 --name experiment_416

# Larger image size (more detail)
python scripts/07_train_yolo.py --imgsz 800 --name experiment_800
```

**3. Different Batch Sizes:**
```bash
# Auto batch size (recommended)
python scripts/07_train_yolo.py --batch -1

# Manual batch size
python scripts/07_train_yolo.py --batch 8 --device 0
```

**4. Different Epochs:**
```bash
# Short training (quick test)
python scripts/07_train_yolo.py --epochs 10 --name short_test

# Long training (full convergence)
python scripts/07_train_yolo.py --epochs 100 --name long_train
```

### Manifest Management

**Critical:** The `manifests/images.csv` file accumulates processing columns:
- `num_boxes_autogen`
- `route_category`
- `street_scene_valid`, `road_scene_valid`
- `gemini_gate_json`
- `qa_passed`, `qa_failure_reason`

**Best Practice - Backup Manifest:**
```bash
# After make fetch (or python scripts/02_fetch_streetview.py)
cp manifests/images.csv manifests/images_post_fetch_backup.csv
```

**Reset Manifest:**
```bash
# To re-run from scratch, restore backup
cp manifests/images_post_fetch_backup.csv manifests/images.csv
```

---

## Configuration & CLI Options

### Global Configuration Files

**dataset.yaml:**
```yaml
path: data/splits
train: train/images
val: val/images
test: test/images

names:
  0: vehicle
  1: enforcement_vehicle
  2: police_old
  3: police_new
  4: lookalike_negative

nc: 5
```

### Makefile Commands

| Command | Script | Purpose |
|---------|--------|---------|
| `make fetch` | `02_fetch_streetview.py` | Download Street View images |
| `make crop` | `02b_crop_bottom.py` | Remove bottom overlay |
| `make empty` | `03_detect_empty_frames.py` | Find empty frames |
| `make synth` | `04_synthesize_vehicle_edits.py` | Generate synthetic images |
| `make prelabel` | `05_prelabel_boxes.py` | Auto-generate labels |
| `make labeler` | `web_ui/app.py` | Launch labeling UI |
| `make split` | `06_split_dataset.py` | Create train/val/test split |
| `make train` | `07_train_yolo.py` | Train YOLO model |
| `make eval` | `08_eval_report.py` | Evaluate model |
| `make clean` | (Makefile) | Clean generated files |

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data` | `dataset.yaml` | Dataset configuration file |
| `--model` | `yolo11s.pt` | Pretrained model or config |
| `--epochs` | 50 | Number of training epochs |
| `--imgsz` | 640 | Input image size |
| `--batch` | None (auto) | Batch size (`-1` for auto) |
| `--device` | `cpu` | Device (`cpu`, `0`, `0,1`, etc.) |
| `--name` | `parkopticon_vehicle_enforcement` | Run name |
| `--resume` | False | Resume from last checkpoint |
| `--project` | `runs/detect` | Output directory |

### Common Training Parameters (Ultralytics)

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `lr0` | 0.01 | 0.0001-0.1 | Initial learning rate |
| `lrf` | 0.01 | - | Final learning rate (lr0 * lrf) |
| `momentum` | 0.937 | 0.8-0.99 | SGD momentum |
| `weight_decay` | 0.0005 | 0.0-0.01 | Weight decay |
| `warmup_epochs` | 3.0 | 0-10 | Warmup epochs |
| `mosaic` | 1.0 | 0.0-1.0 | Mosaic augmentation |
| `mixup` | 0.0 | 0.0-1.0 | Mixup augmentation |
| `iou` | 0.7 | 0.3-0.9 | NMS IoU threshold |
| `patience` | 100 | 10-1000 | Early stopping patience |

### Inference Parameters

**Live Inference (`scripts/11_live_inference.py`):**
```bash
python scripts/11_live_inference.py \
  --weights runs/detect/parkopticon_final_v1/weights/best.pt \
  --source 0 \  # Webcam
  --conf 0.25 \
  --smooth  # Temporal smoothing
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--weights` | Default trained model | Path to trained weights |
| `--source` | `0` | Webcam (0) or video file path |
| `--conf` | 0.25 | Confidence threshold |
| `--iou` | 0.45 | NMS IoU threshold |
| `--smooth` | False | Temporal smoothing with tracker |
| `--max-det` | 50 | Max detections per frame |
| `--save` | False | Save output video |

---

## Troubleshooting & Best Practices

### Common Issues

**1. "No empty candidates found"**
- **Cause:** All images contain vehicles or detector threshold too high
- **Fix:** Lower confidence threshold: `--conf 0.15`
- **Fix:** Try different locations with fewer vehicles

**2. "Synthesis failed"**
- **Cause:** Gemini API quota/rate limits
- **Fix:** Check API key in `.env`
- **Fix:** Try different model: `--model gemini-1.5-pro`

**3. "Out of memory during training"**
- **Cause:** Batch size too large
- **Fix:** Use auto batch: `--batch -1`
- **Fix:** Reduce batch size: `--batch 8`
- **Fix:** Reduce image size: `--imgsz 416`

**4. "Labels don't match vehicles"**
- **Cause:** Image differencing failed on subtle insertions
- **Fix:** Use labeler UI to manually correct
- **Fix:** Increase QA threshold or skip QA: `--qa-skip`

**5. Training loss decreases but val loss increases (overfitting)**
- **Fix:** Increase augmentation: `mixup=0.1`, `mosaic=1.0`
- **Fix:** Reduce model complexity (use smaller model)
- **Fix:** Add more training data

### Quality Safeguards

**1. Empty Frame Detection:**
- Always use **low confidence threshold (0.15-0.25)**
- Why: Prevents missing vehicles that would create bad synthetic labels

**2. Data Leakage Prevention:**
- Always split by `pano_id`, never by individual image
- Why: 8 headings per panorama are near-duplicates
- Script handles this automatically

**3. Sibling Management:**
- Original empty + synthetic siblings must stay in same split
- Why: Same background in train/test creates inflated metrics

**4. Consistent Preprocessing:**
- If cropping during training, apply same crop at inference
- Why: Mismatched preprocessing degrades performance

### Performance Optimization

**Training Speed:**
- Use GPU: `--device 0`
- Auto batch size: `--batch -1`
- Smaller image size: `--imgsz 416` (faster but less detail)

**Memory Usage:**
- Reduce batch size manually: `--batch 4`
- Use mixed precision (automatic in YOLO)
- Gradient accumulation (advanced)

### Validation Metrics

**Per-Class AP (Average Precision):**
- Class 0: `vehicle` (regular traffic)
- Class 1: `enforcement_vehicle` (parking enforcement)
- Class 2: `police_old` (old livery)
- Class 3: `police_new` (new livery)

**Enforcement-Present Metric:**
- Binary metric: Does image contain enforcement vehicle?
- Calculated from per-class AP scores
- Target: >0.90 precision and recall

### Model Export (Deployment)

**Export to different formats:**
```python
from ultralytics import YOLO

model = YOLO("runs/detect/parkopticon_vehicle_enforcement/weights/best.pt")

# Export to ONNX
model.export(format="onnx")

# Export to TensorRT (NVIDIA GPU)
model.export(format="engine", imgsz=640, half=True)

# Export to TFLite (mobile)
model.export(format="tflite")
```

---

## References

- **Ultralytics Documentation:** https://docs.ultralytics.com/
- **YOLO11 Architecture:** https://github.com/ultralytics/ultralytics
- **Project README:** `README.md`
- **Onboarding Guide:** `docs/project_docs/ONBOARDING.md`
- **Implementation Details:** `docs/project_docs/IMPLEMENTATION_DETAILS.md`

---

*Document generated for ParkOpticon YOLO11s Bootstrap project.*
*Last Updated: April 5, 2026*
