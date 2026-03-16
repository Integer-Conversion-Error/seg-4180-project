# ParkOpticon YOLO11s Project Skills

This file defines project-specific skills and knowledge for the ParkOpticon YOLO11s Bootstrap project.

## Overview
This project involves training a YOLO11s object detector using 100% synthetic data generated from Google Street View backgrounds and Gemini API vehicle insertion.

## Project Structure Knowledge

### Key Directories
- `scripts/` - Pipeline automation scripts (numbered by execution order)
- `data/` - Dataset storage
  - `images_original/` - Downloaded Street View images
  - `images_synth/` - Synthetic vehicle edits
  - `labels_autogen/` - Auto-generated labels
  - `labels_final/` - Manually reviewed labels
  - `splits/` - Train/Val/Test splits
- `manifests/` - CSV manifests for tracking images and processing status
- `runs/` - YOLO training outputs (automatically generated)
- `labeler/` - Web UI for label review

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

### Step 6: Manual Label Review
**Script:** `labeler/app.py`
**Command:** `make labeler`
**Purpose:** Web UI for reviewing and correcting auto-generated labels

**Key Knowledge:**
- Access: http://localhost:8000
- Review bounding boxes, delete bad synthetics, save to `data/labels_final/`

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

### Step 8: Model Training
**Script:** `scripts/07_train_yolo.py`
**Command:** `make train`
**Purpose:** Fine-tunes YOLO11s on custom dataset

**Key Knowledge:**
- Default: 50 epochs, 640x640 images
- Outputs to `runs/detect/{name}/`
- Checkpoints: `weights/best.pt`, `weights/last.pt`
- Use `--device 0` for GPU, `--batch -1` for auto batch size

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
| `03_detect_empty_frames.py` | Find empty frames | `--conf`, `--device`, `--nuke` |
| `04_synthesize_vehicle_edits.py` | Generate synthetics | `--enforcement-rate`, `--seed` |
| `05_prelabel_boxes.py` | Auto-label | `--conf`, `--qa-threshold-area` |
| `06_split_dataset.py` | Create splits | `--ratios`, `--min-enforcement-val` |
| `07_train_yolo.py` | Train model | `--epochs`, `--imgsz`, `--batch`, `--device` |
| `08_eval_report.py` | Evaluate model | `--weights`, `--data` |

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
*Last Updated: March 12, 2026*
