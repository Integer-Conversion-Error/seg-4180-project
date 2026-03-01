# ParkOpticon YOLO11s Bootstrap

A complete pipeline for training a 2-class YOLO11s vehicle detector with synthetic data generation using Google Street View and Gemini image editing.

## Classes

- **Class 0**: `vehicle` - Regular vehicles (cars, trucks, buses, motorcycles)
- **Class 1**: `enforcement_vehicle` - Parking enforcement vehicles

## Quickstart

### 1. Setup

```bash
# Clone and enter the project
cd parkopticon-yolo11s-bootstrap

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env
# Edit .env with your API keys:
#   GSV_API_KEY=your_google_street_view_api_key
#   GEMINI_API_KEY=your_gemini_api_key
```

### 2. Create Points Template

```bash
python scripts/01_make_points_template.py -o manifests/points.csv -n 2
# Edit manifests/points.csv with your desired locations
```

Example `manifests/points.csv` format:
```csv
street,label,location,heading,pitch,fov,radius
downtown_main,downtown,"40.7128,-74.0060",180,0,80,50
parking_lot_a,lot_a,"40.7580,-73.9855",90,-5,90,50
```

### 3. Run Pipeline

```bash
# Step 1: Download Street View images
make fetch

# Step 2: (Optional, recommended) Crop bottom of images
make crop

# Step 3: Detect empty frames (no vehicles)
make empty

# Step 4: Generate synthetic vehicle edits (requires Gemini API)
make synth

# Step 5: Auto-generate bounding box labels
make prelabel

# Step 6: Review labels in the UI
make labeler
# Open http://localhost:8000 in your browser

# Step 7: Create train/val/test split
make split

# Step 8: Train YOLO11s
make train

# Step 9: Evaluate on test set
make eval
```

## Pipeline Details

| Step | Command | Description |
|------|---------|-------------|
| Template | `01_make_points_template.py` | Generate CSV template for locations |
| Fetch | `02_fetch_streetview.py` | Download Street View images + metadata |
| Crop | `02b_crop_bottom.py` | Bottom-crop downloaded images and update manifest |
| Sample | `01b_expand_points_by_area.py` | Expand points by area sampling |
| Detect Empty | `03_detect_empty_frames.py` | Find images without vehicles |
| Synthesize | `04_synthesize_vehicle_edits.py` | Create synthetic variants with Gemini |
| Pre-label | `05_prelabel_boxes.py` | Auto-generate YOLO format labels |
| Labeler | `labeler/app.py` | Web UI for manual review |
| Split | `06_split_dataset.py` | Group-aware train/val/test split |
| Train | `07_train_yolo.py` | Train YOLO11s detector |
| Eval | `08_eval_report.py` | Evaluate and generate report |

## Project Structure

```
parkopticon-yolo11s-bootstrap/
├── scripts/                    # Pipeline scripts
│   ├── 01_make_points_template.py
│   ├── 01b_expand_points_by_area.py
│   ├── 02_fetch_streetview.py
│   ├── 02b_crop_bottom.py
│   ├── 03_detect_empty_frames.py
│   ├── 04_synthesize_vehicle_edits.py
│   ├── 05_prelabel_boxes.py
│   ├── 06_split_dataset.py
│   ├── 07_train_yolo.py
│   └── 08_eval_report.py
├── labeler/                    # Labeling UI
│   ├── app.py                 # FastAPI backend
│   └── static/
│       ├── index.html
│       └── app.js
├── data/
│   ├── images_original/       # Downloaded Street View images
│   ├── images_synth/          # Synthetic edits
│   ├── labels_autogen/        # Auto-generated labels
│   ├── labels_final/          # Reviewed labels
│   └── splits/                # Train/val/test
├── manifests/
│   ├── points.csv             # Input locations
│   └── images.csv             # Image manifest
├── reports/                   # Training/eval reports
├── utils/                     # Shared utility modules
│   └── area_sampler_utils.py
├── dataset.yaml               # YOLO dataset config
├── requirements.txt
├── Makefile
└── .env.example
```

## Area Sampling (Auto-Expanding Points)

Instead of manually adding locations to `points.csv`, you can use the area sampler to automatically find valid Street View panoramas within a bounding box or polygon.

### How It Works

1. Randomly sample locations within a user-defined area (bbox or polygon)
2. For each sample, call the Street View **metadata endpoint** to check if imagery exists
3. Deduplicate by `pano_id` to avoid adding the same panorama twice
4. Append new rows to `points.csv` with 8 headings per panorama (0, 45, 90, etc.)

### Why `pano_id` Matters

The script uses **pano_id** (the unique panorama identifier from Google) to:
- **Deduplicate**: If a panorama already exists in your points.csv, it won't be added again
- **Prevent leakage**: When splitting data for training/testing, ensure the same physical location doesn't appear in both sets by grouping on `pano_id`
- **Idempotency**: Rerunning the sampler won't create duplicates

### Usage Examples

```bash
# Sample 200 panoramas within a bounding box (NYC area)
python scripts/01b_expand_points_by_area.py \
  --points_csv manifests/points.csv \
  --bbox "40.5,-74.1,40.9,-73.7" \
  --target_panos 200 \
  --radius 50

# Sample using a GeoJSON polygon (more precise area definition)
python scripts/01b_expand_points_by_area.py \
  --points_csv manifests/points.csv \
  --polygon_geojson data/my_area.geojson \
  --target_panos 100 \
  --min_spacing_m 40 \
  --seed 456

# Dry run - see what would be added without writing
python scripts/01b_expand_points_by_area.py \
  --points_csv manifests/points.csv \
  --bbox "40.5,-74.1,40.9,-73.7" \
  --target_panos 50 \
  --dry_run

# Custom headings and pitch
python scripts/01b_expand_points_by_area.py \
  --points_csv manifests/points.csv \
  --bbox "40.5,-74.1,40.9,-73.7" \
  --headings "0,90,180,270" \
  --pitch -10 \
  --fov 90
```

### CSV Schema

The sampler appends **enrichment columns** to the right of your existing schema:

| Column | Description |
|--------|-------------|
| `street` | Unique identifier (e.g., `auto_101`) |
| `label` | Human-readable label (e.g., `auto_panoid_h90`) |
| `location` | Snapped pano lat,lng |
| `heading` | Compass heading (0-360) |
| `pitch` | Pitch angle (-90 to 90) |
| `fov` | Field of view (30-120) |
| `radius` | Search radius in meters |
| `bottom_crop` | Crop amount from bottom |
| `source` | "auto_sampler" (vs "manual") |
| `sample_location` | Original sampled lat,lng before snapping |
| `pano_id` | Google panorama ID (for dedup) |
| `pano_lat` | Snapped latitude |
| `pano_lng` | Snapped longitude |
| `pano_date` | Image capture date (if available) |
| `status` | `ok`, `no_imagery`, `failed`, `out_of_area`, or `too_close` |
| `group_id` | Set to pano_id for stable grouping |
| `created_at` | ISO timestamp |

**Backward Compatibility**: Existing code that reads only the original 8 columns will continue to work.

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--points_csv` | manifests/points.csv | Input/output CSV path |
| `--bbox` | - | Bounding box: min_lat,min_lng,max_lat,max_lng |
| `--polygon_geojson` | - | Path to GeoJSON polygon file |
| `--target_panos` | 200 | Number of unique panoramas to add |
| `--radius` | 50 | Search radius in meters |
| `--min_spacing_m` | - | Min distance between panos (meters) |
| `--seed` | 123 | Random seed for reproducibility |
| `--max_attempts` | 10000 | Max random samples to try |
| `--headings` | 0,45,90,135,180,225,270,315 | Comma-separated headings |
| `--pitch` | 0 | Pitch angle for all views |
| `--fov` | 80 | Field of view |
| `--bottom_crop` | (auto) | Crop from bottom; uses most common existing value |
| `--label_prefix` | auto | Prefix for auto-generated labels |
| `--dry_run` | False | Preview without writing |
| `--api_key_env_var` | GSV_API_KEY | Env var for API key |
| `--api_key` | - | API key value (overrides env var) |

## Bottom Crop Script (`02b_crop_bottom.py`)

Use this after fetch to generate cropped copies with a deterministic suffix and avoid re-cropping:

```bash
# Crop bottom 30px (default), update manifests/images.csv to new file paths
python scripts/02b_crop_bottom.py

# Custom crop amount
python scripts/02b_crop_bottom.py --crop-px 40

# Preview only
python scripts/02b_crop_bottom.py --dry-run
```

What it does:
- Creates sibling files named like `imageid_bc30.jpg`
- Leaves original downloaded files untouched
- Updates `manifests/images.csv` `file_path` to cropped files
- Skips already-cropped files on rerun (idempotent behavior)

## Expected Outputs

After running each stage:

1. **After `make fetch`**: Images in `data/images_original/{street}/{image_id}.jpg`
2. **After `make crop`**: Cropped siblings like `*_bc30.jpg`, manifest paths updated
3. **After `make empty`**: `lists/empty_candidates.txt` with image IDs
4. **After `make synth`**: Synthetic images in `data/images_synth/{edit_type}/`
5. **After `make prelabel`**: Labels in `data/labels_autogen/{image_id}.txt`
6. **After `make split`**: Train/val/test in `data/splits/{split}/`
7. **After `make train`**: Model weights in `runs/detect/parkopticon_vehicle_enforcement/weights/`
8. **After `make eval`**: Reports in `reports/eval_report.{md,json}`

## Configuration

All scripts support CLI arguments. Common options:

```bash
# Street View fetching
python scripts/02_fetch_streetview.py --api-key KEY --resume

# Bottom-crop fetched images
python scripts/02b_crop_bottom.py --crop-px 30

# Vehicle detection
python scripts/03_detect_empty_frames.py --conf 0.25 --device cpu

# Synthetic generation
python scripts/04_synthesize_vehicle_edits.py --enforcement-rate 0.2 --seed 42

# Training
python scripts/07_train_yolo.py --epochs 50 --batch 16 --device cpu
```

## Troubleshooting

### API Key Errors

- **GSV_API_KEY**: Get from [Google Cloud Console](https://console.cloud.google.com/). Enable "Street View Static API".
- **GEMINI_API_KEY**: Get from [Google AI Studio](https://aistudio.google.com/app/apikey).

### Empty Candidates

If no empty candidates found:
- Try different locations with fewer vehicles
- Lower the confidence threshold: `--conf 0.15`

### Synthesis Failures

- Check Gemini API quota and rate limits
- Try a different model: `--model gemini-1.5-pro`

### Training Issues

- GPU recommended: `--device 0`
- If out of memory: reduce batch size `--batch 8`

## Smoke Test (Empty Dataset)

To verify the pipeline works without real data:

1. Create a dummy image:
   ```bash
   python -c "from PIL import Image; Image.new('RGB', (640, 640), 'gray').save('data/images_original/test/img1.jpg')"
   ```

2. Manually add to manifest with basic fields

3. Continue with subsequent steps

The pipeline is idempotent - reruns skip already-processed items.

## License

MIT
