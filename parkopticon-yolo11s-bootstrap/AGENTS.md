# ParkOpticon YOLO11s Project Skills

This file defines project-specific skills and knowledge for the ParkOpticon YOLO11s Bootstrap project.

## Overview
This project involves training a YOLO11s object detector using 100% synthetic data generated from Google Street View backgrounds and Gemini API vehicle insertion.

## Project Structure Knowledge

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
- `dataset.yaml` - YOLO dataset configuration (train/val/test paths, class names)
- `Makefile` - Pipeline command shortcuts
- `.env` - API keys (GSV_API_KEY, GEMINI_API_KEY)

## Pipeline Skills

### Step 1: Make Points Template
**Script:** `scripts/01_make_points_template.py`
**Command:** `python scripts/01_make_points_template.py`
**Purpose:** Generate CSV template for locations

**Key Knowledge:**
- Creates `manifests/points.csv` template
- User fills in location coordinates

### Step 2: Fetch Street View Images
**Script:** `scripts/02_fetch_streetview.py`
**Command:** `make fetch`
**Purpose:** Downloads Google Street View images for locations in `manifests/points.csv`

**Key Knowledge:**
- Requires `GSV_API_KEY` in `.env`
- Creates `manifests/images.csv` with metadata
- Images stored in `data/images_original/{street}/{image_id}.jpg`
- Idempotent: Use `--resume` to continue interrupted downloads

### Step 2b: Image Preprocessing
**Script:** `scripts/02b_crop_bottom.py`
**Command:** `make crop`
**Purpose:** Removes bottom 30px overlay (copyright watermarks, UI elements)

**Key Knowledge:**
- Creates cropped siblings with `_bc30` suffix
- Updates manifest with new file paths
- Critical: Apply same crop at inference time

### Step 3: Empty Frame Detection
**Script:** `scripts/03_detect_empty_frames.py`
**Command:** `make empty`
**Purpose:** Finds images without vehicles using pretrained YOLO

**Key Knowledge:**
- Uses COCO-pretrained YOLO
- **Critical:** Use low confidence threshold (0.15-0.25) to avoid false negatives
- Outputs: `lists/empty_candidates.txt`, `lists/valid_road_candidates.txt`
- `--nuke` flag resets artifacts but preserves original images

### Step 4: Synthetic Data Generation
**Script:** `scripts/04_synthesize_vehicle_edits.py`
**Command:** `make synth`
**Purpose:** Generates synthetic vehicle edits using Google Gemini API

**Key Knowledge:**
- For each empty frame, generates 4 variants:
  1. Random vehicle (Class 0)
  2. Enforcement vehicle (Class 1)
  3. Police old livery (Class 2)
  4. Police new livery (Class 3)
- Requires `GEMINI_API_KEY` in `.env`
- Cost: ~$6.23 per 2,658 images
- Siblings must stay in same train/val/test split

### Step 5: Auto-Labeling
**Script:** `scripts/05_prelabel_boxes.py`
**Command:** `make prelabel`
**Purpose:** Auto-generates YOLO format bounding box labels

**Key Knowledge:**
- Synthetic images: Uses image differencing to find inserted vehicle
- Original images: Uses pretrained YOLO detection
- Outputs: `data/labels_autogen/{image_id}.txt`
- QA checks available: `--qa-threshold-area`, `--qa-skip`
- Default excludes rows with `review_status=rejected`
- Override: `--include-rejected`

### Step 6: Manual Label Review
**Script:** `web_ui/app.py`
**Command:** `make labeler`
**Purpose:** Web UI for reviewing and correcting auto-generated labels

**Key Knowledge:**
- Access: http://localhost:8000
- Review bounding boxes, delete bad synthetics, save to `data/labels_final/`
- Labeler hides `review_status=rejected` rows by default
- Override at startup: `--labeler-include-rejected`
- Includes Batch Run Manager page at `/batch-run-manager`
- Includes Batch Run Manager page at `/batch-run-manager`

### Step 7: Dataset Splitting
**Script:** `scripts/06_split_dataset.py`
**Command:** `make split`
**Purpose:** Group-aware train/val/test split to prevent data leakage

**Key Knowledge:**
- Splits by `pano_id` (panorama ID) - critical for preventing leakage
- Each panorama has 8 headings; all must stay in same split
- Ratios: 70% train, 20% val, 10% test
- Enforces minimum enforcement instances in val/test
- Siblings (synthetics) inherit parent's group and split
- Default excludes rows with `review_status=rejected`
- Override: `--include-rejected`
- Rebuilds split output directories each run to avoid stale artifacts

### Step 8: Model Training
**Script:** `scripts/07_train_yolo.py`
**Command:** `make train`
**Purpose:** Fine-tunes YOLO11s on custom dataset

**Key Knowledge:**
- Default: 50 epochs, 640x640 images
- Outputs to `runs/detect/{name}/`
- Checkpoints: `weights/best.pt`, `weights/last.pt`
- Use `--device 0` for GPU, `--batch -1` for auto batch size
- By default, blocks training if rejected rows are assigned to splits or still present in split dirs
- Provide `--manifest` for run-scoped safety checks
- Override: `--include-rejected`

### Step 9: Evaluation
**Script:** `scripts/08_eval_report.py`
**Command:** `make eval`
**Purpose:** Evaluates model on test set

**Key Knowledge:**
- Requires trained weights: `runs/detect/{name}/weights/best.pt`
- Generates `reports/eval_report.md` and `reports/eval_report.json`

## Data Leakage Prevention

### Critical Rule
**Split by panorama group, NOT by individual image.**

Each panorama has 8 heading angles (0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°) showing the same intersection. If these end up in different splits, the model memorizes the location instead of learning to detect vehicles.

### Group Structure
```
pano_id: abc123
├── auto_abc123_h0.jpg
├── auto_abc123_h45.jpg
├── auto_abc123_h90.jpg
├── ...
└── auto_abc123_h315.jpg
    └── Siblings (synthetics) inherit same pano_id
```

### Enforcement Minimums
- **Val split:** ≥50 enforcement instances total
- **Test split:** ≥30 enforcement instances total
- Script automatically rebalances if thresholds not met

## Class Definitions

| Class ID | Name | Description |
|----------|------|-------------|
| 0 | `vehicle` | Regular vehicles (cars, trucks, buses, motorcycles) |
| 1 | `enforcement_vehicle` | Parking enforcement vehicles |
| 2 | `police_old` | Ottawa Police cruisers with old livery |
| 3 | `police_new` | Ottawa Police cruisers with new livery |
| 4 | `lookalike_negative` | Non-enforcement vehicles that resemble enforcement |

## Common Commands Reference

### Setup
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env  # Add API keys
```

### Pipeline Execution
```bash
make fetch      # Download Street View images
make crop       # Remove bottom overlay
make empty      # Find empty frames
make synth      # Generate synthetic images
make prelabel   # Auto-generate labels
make labeler    # Launch labeling UI
make split      # Create train/val/test split
make train      # Train YOLO11s
make eval       # Evaluate on test set
```

### Run-Scoped Execution (Preferred)
```bash
# Example: synthesize into an isolated run directory
python scripts/04_synthesize_vehicle_edits.py --run-dir runs/Benchmark_Run_002

# Prelabel for that same run (explicit manifest/output paths)
python scripts/05_prelabel_boxes.py \
  --manifest runs/Benchmark_Run_002/manifests/images.csv \
  --out-dir runs/Benchmark_Run_002/data/labels_autogen

# Launch UI scoped to that run (default hides rejected)
python web_ui/app.py --run-dir runs/Benchmark_Run_002

# Optional: include rejected rows in labeler queues
python web_ui/app.py --run-dir runs/Benchmark_Run_002 --labeler-include-rejected

# Split run-scoped dataset (default excludes rejected)
python scripts/06_split_dataset.py \
  --manifest runs/Benchmark_Run_002/manifests/images.csv \
  --labels-dir runs/Benchmark_Run_002/data/labels_autogen \
  --labels-final-dir runs/Benchmark_Run_002/data/labels_final \
  --out-dir runs/Benchmark_Run_002/data/splits

# Train run-scoped model with rejection safety checks
python scripts/07_train_yolo.py \
  --data runs/Benchmark_Run_002/data/splits/data.yaml \
  --manifest runs/Benchmark_Run_002/manifests/images.csv
```

### Experimentation
```bash
# Different confidence thresholds
python scripts/03_detect_empty_frames.py --conf 0.15

# Different image sizes
python scripts/07_train_yolo.py --imgsz 416 --name experiment_416

# Resume training
python scripts/07_train_yolo.py --resume

# Clean run (reset artifacts)
python scripts/03_detect_empty_frames.py --nuke
```

## Troubleshooting

### "No empty candidates found"
- Lower confidence threshold: `--conf 0.15`
- Check if images actually contain vehicles

### "Synthesis failed"
- Check Gemini API quota and key in `.env`
- Try different model: `--model gemini-1.5-pro`

### "Out of memory during training"
- Use auto batch: `--batch -1`
- Reduce batch size: `--batch 8`
- Reduce image size: `--imgsz 416`

### "Labels don't match vehicles"
- Use labeler UI to manually correct
- Increase QA threshold or skip QA: `--qa-skip`

## Performance Optimization

### Training Speed
- **GPU:** Always use GPU if available: `--device 0`
- **Auto Batch:** `--batch -1` (automatically determines optimal batch size)
- **Image Size:** Smaller = faster: `--imgsz 416` (vs default 640)

### Memory Management
- Gradient accumulation (advanced)
- Mixed precision (automatic in YOLO)
- Reduce batch size manually if OOM

## Project-Specific Tips

### 1. Quality Safeguards
- **Empty Detection:** Always use low threshold (0.15-0.25) to avoid false negatives
- **Consistent Preprocessing:** Apply same crop at inference as during training
- **Sibling Management:** Keep original + synthetic siblings in same split

### 2. Data Quality
- **Minimum Box Size:** Ignore tiny far-away detections (add noise)
- **Enforcement Variety:** Add normal vehicles alongside enforcement vehicles in some synth images
- **Hard Negatives:** Include empty scenes, sidewalks, shoulders (~20-30% of training)

### 3. Model Deployment
- Export to ONNX, TensorRT, or TFLite for deployment
- Use same preprocessing (crop, resize) at inference

## Script Quick Reference

| Script | Purpose | Key Args |
|--------|---------|----------|
| `01_make_points_template.py` | Generate location template | `-o`, `-n` |
| `01b_expand_points_by_area.py` | Auto-generate points | `--bbox`, `--target_panos` |
| `02_fetch_streetview.py` | Download images | `--api-key`, `--resume` |
| `02b_crop_bottom.py` | Remove overlay | `--crop-px`, `--dry-run` |
| `02c_rebuild_manifest.py` | Rebuild manifest | — |
| `03_detect_empty_frames.py` | Find empty frames | `--conf`, `--device`, `--nuke` |
| `03a_yolo_pass.py` | YOLO detection pass | — |
| `03b_gemini_gate.py` | Gemini scene gating | — |
| `04_synthesize_vehicle_edits.py` | Generate synthetics | `--enforcement-rate`, `--seed`, `--run-dir` |
| `04b_generate_cardinal_references.py` | Cardinal reference images | — |
| `04c_augment_synth_images.py` | Augment synthetics | — |
| `05_prelabel_boxes.py` | Auto-label | `--conf`, `--qa-threshold-area`, `--include-rejected` |
| `05b_sweep_prelabel_confidence.py` | Sweep confidence thresholds | — |
| `06_split_dataset.py` | Create splits | `--train-ratio`, `--val-ratio`, `--test-ratio`, `--include-rejected` |
| `07_train_yolo.py` | Train model | `--epochs`, `--imgsz`, `--batch`, `--device`, `--manifest`, `--include-rejected` |
| `07b_batch_run_manager.py` | Batch split+train | `--config` |
| `08_eval_report.py` | Evaluate model | `--weights`, `--data` |
| `09_analyze_dataset.py` | Dataset analysis | — |
| `09_count_boxes.py` | Count boxes | — |
| `09_detect_duplicates.py` | Detect duplicates | — |
| `09_visualize_boxes.py` | Visualize boxes | — |
| `10_collect_hard_negatives.py` | Collect hard negatives | — |
| `11_live_inference.py` | Live inference | — |
| `12_run_media_inference.py` | Media inference | — |
| `09_analyze_dataset.py` | Dataset analysis | — |
| `09_count_boxes.py` | Count boxes | — |
| `09_detect_duplicates.py` | Detect duplicates | — |
| `09_visualize_boxes.py` | Visualize boxes | — |
| `10_collect_hard_negatives.py` | Collect hard negatives | — |
| `11_live_inference.py` | Live inference | — |
| `12_run_media_inference.py` | Media inference | — |

## API Keys Required

| Key | Source | Purpose |
|-----|--------|---------|
| `GSV_API_KEY` | Google Cloud Console | Street View Static API |
| `GEMINI_API_KEY` | Google AI Studio | Synthetic image generation |

Add to `.env`:
```
GSV_API_KEY=your_key_here
GEMINI_API_KEY=your_key_here
```

---
*Last Updated: April 5, 2026*
