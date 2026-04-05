# ParkOpticon YOLO11s Bootstrap

A complete pipeline for training a 4-class YOLO11s vehicle detector using synthetic data generation with Google Street View backgrounds and high-fidelity image editing.

## Classes

- **Class 0**: `vehicle` - Regular vehicles (cars, trucks, buses, motorcycles)
- **Class 1**: `enforcement_vehicle` - Parking enforcement vehicles
- **Class 2**: `police_old` - Ottawa Police cruisers with old livery
- **Class 3**: `police_new` - Ottawa Police cruisers with new livery (facelift in progress)

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

# Step 3b (new): this now also performs Gemini JSON scene gating,
# routes non-street/non-road images to traceable stash directories,
# and writes lists/valid_road_candidates.txt

# Step 4: Generate synthetic vehicle edits (requires Gemini API)
make synth

# Step 5: Auto-generate bounding box labels
make prelabel

# Step 5b (new): prelabel now emits a histogram PNG of vehicle box counts
# (excluding 0-box, non-road, non-street, and excluded images)

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

### Rejected Image Policy (Prelabel, Labeler, Split, Train)

Rows marked `review_status=rejected` in `manifests/images.csv` are now excluded by default across labeling and training prep. Rejected rows are not deleted.

- **Prelabel (`05_prelabel_boxes.py`)**: skips rejected rows by default; override with `--include-rejected`
- **Labeler (`web_ui/app.py`)**: rejected rows are hidden by default; override with startup flag `--labeler-include-rejected`
- **Split (`06_split_dataset.py`)**: rejected rows are excluded by default; override with `--include-rejected`
- **Train (`07_train_yolo.py`)**: blocks training if rejected rows are assigned to splits or still present in split image dirs; override with `--include-rejected`

Run-scoped examples:

```bash
# Prelabel (default excludes rejected)
python scripts/05_prelabel_boxes.py \
  --manifest runs/Benchmark_Run_002/manifests/images.csv \
  --out-dir runs/Benchmark_Run_002/data/labels_autogen

# Launch UI (default hides rejected in labeler)
python web_ui/app.py --run-dir runs/Benchmark_Run_002

# Optional: show rejected in labeler queues
python web_ui/app.py --run-dir runs/Benchmark_Run_002 --labeler-include-rejected

# Split (default excludes rejected)
python scripts/06_split_dataset.py \
  --manifest runs/Benchmark_Run_002/manifests/images.csv \
  --labels-dir runs/Benchmark_Run_002/data/labels_autogen \
  --labels-final-dir runs/Benchmark_Run_002/data/labels_final \
  --out-dir runs/Benchmark_Run_002/data/splits

# Train with rejection safety checks (default)
python scripts/07_train_yolo.py \
  --data runs/Benchmark_Run_002/data/splits/data.yaml \
  --manifest runs/Benchmark_Run_002/manifests/images.csv
```

### Batch Run Manager (Sequential Split + Train)

Use `scripts/07b_batch_run_manager.py` to run multiple experiments back-to-back from one run directory. Each job can define its own split settings and train settings (model, epochs, imgsz, batch, device, and extra args).

Quick start:

```bash
# 1) Copy and edit the example plan
cp manifests/batch_run_plan.example.json manifests/batch_run_plan.json

# 2) Run the batch manager
python scripts/07b_batch_run_manager.py --config manifests/batch_run_plan.json
```

Notes:
- Run names are timestamped automatically to prevent collisions.
- Each job writes to its own split output folder under the configured `split_root`.
- Train a smaller model by setting `"model_size": "n"` (or `"s"`, `"m"`, `"l"`, `"x"`) in a job, or set `"model"` directly to a custom checkpoint path.
- Configure checkpoint cadence per job with `"save_period"` (default `10`).
- The web UI now includes a **Batch Run Manager** page at `/batch-run-manager` for creating/saving plans and launching jobs.

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
| Batch Train | `07b_batch_run_manager.py` | Run multiple sequential split+train jobs |
| Eval | `08_eval_report.py` | Evaluate and generate report |

## Synthetic Data Strategy

### Why Synthetic?

All training data is synthetically generated. The image generation model produces edits that are practically indistinguishable from real photographs, enabling:

- Full control over class balance and scene composition
- No privacy concerns from real vehicle imagery
- Rapid iteration on model improvements
- Consistent labeling without manual annotation errors

### How Many Synthetic Images to Generate

Target based on **bounding box count**, not image count.

**Recommended target:** ~500-800 synthetic boxes per enforcement class (`enforcement_vehicle`, `police_old`, `police_new`).

A concrete plan with K = 600 base empty frames:

For each base empty frame, generate:
1. **One synthetic normal vehicle** version
2. **One synthetic enforcement class** version (rotate through enforcement_vehicle, police_old, police_new)

Keep the original empty frame too.

This gives you:
- **600 normal vehicle images** (negative examples for enforcement classes)
- **~200 images per enforcement class** (600 total enforcement boxes)
- **600 original empties** (true negatives)

### Keeping Original Empty Frames

**Yes — keep the original empty frame.** It's valuable as a negative example.

**Critical rule: Keep "siblings" together in the same split.**

For a given base empty frame, these must all land in the same split:
- The original empty frame
- Its synthetic normal vehicle version
- Its synthetic enforcement vehicle version(s)

If you don't, the same background leaks into train and test, producing inflated metrics.

### Avoiding "Enforcement Always Has 1 Box" Shortcuts

If enforcement vehicles only appear as "one added car to an empty scene," the model can learn:
- Enforcement vehicles are always alone
- Enforcement vehicles always appear in wide open roads
- Enforcement is always centered / same scale

**Fixes (apply 1-2):**

1. **For ~20-30% of enforcement synth images**, prompt: "Add one enforcement vehicle **and** one normal vehicle farther away."
2. **For ~10-20% of enforcement synth**, add enforcement **smaller / farther** (varied scale).

### Minimum Box Size

Ignore tiny far-away detections during labeling — they add noise without useful signal.

### Quality Safeguards

**Confidence threshold for "empty" detection:**

When selecting empty frames for synthetic insertion, use a **lower confidence threshold** (e.g., 0.15–0.25) for the pretrained YOLO detector.

Why: If the detector misses a car at higher confidence, you'll inject a synthetic vehicle into a frame that already has one — creating confusing labels.

**Consistent preprocessing:**

If you crop the bottom of images to remove overlays (watermarks, UI elements), apply the same crop at inference time. Mismatched preprocessing degrades model performance.

---

## Dataset Splitting (Leakage-Safe)

### The Unit of Splitting

**Split by panorama group, not by image.**

- Primary group key: **`pano_id`**
- Each pano group contains up to 8 headings (orthogonal views) plus synthetic siblings

**Rule:** One pano group can only be in **train OR val OR test** — never more than one.

Why: Your 8 headings are near-duplicates of the same scene. Random splitting causes the model to memorize the intersection.

### Step-by-Step Split Procedure

#### Step 1 — Build Groups

For every image, assign:
- `group_id = pano_id` (preferred)

If `pano_id` is missing, approximate with:
- `rounded pano_lat/lng + heading bucket`

#### Step 2 — Create "Families" for Synthetic

For each synthetic image:
- `parent_image_id` points to the original empty frame
- Inherit the same `group_id` as the parent

#### Step 3 — Choose Split Ratios

Use:
- **Train: 70%**
- **Val: 20%**
- **Test: 10%**

#### Step 4 — Allocate Synthetic Bases AFTER Splitting Groups

**Do not** randomly pick synthetic bases from the whole dataset.

**Do:**
- Pick K bases *within train panos*
- Pick ~0.2K bases within val panos
- Pick ~0.1K bases within test panos

This ensures synthetic siblings stay in the same split as their parent.

#### Step 5 — Ensure Enforcement Appears in Val/Test

After labels exist, verify:
- Does val have at least **~50 enforcement instances total**?
- Does test have at least **~30 enforcement instances total**?

If not, move a few enforcement-positive pano groups from train into val/test (group-level move only).

#### Step 6 — Materialize the Splits

Copy/symlink images + labels into:
```
splits/train/images, splits/train/labels
splits/val/...
splits/test/...
```

Empty images should have **empty label files** (valid negatives).
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
4. **After `make empty`**:
   - `lists/valid_road_candidates.txt` (preferred synthesis input)
   - `lists/non_road_candidates.txt`
   - `lists/non_street_candidates.txt`
   - stashed excluded images under `data/images_excluded/non_road` and `data/images_excluded/non_street`
5. **After `make synth`**: Synthetic images in `data/images_synth/{edit_type}/`
   - plus parameterized retention lists for non-road hard negatives (`lists/non_road_keep.txt`, `lists/non_road_drop.txt`)
6. **After `make prelabel`**: Labels in `data/labels_autogen/{image_id}.txt` and histogram at `reports/prelabel_vehicle_count_hist.png`
7. **After `make split`**: Train/val/test in `data/splits/{split}/`
8. **After `make train`**: Model weights in `runs/detect/parkopticon_vehicle_enforcement/weights/`
9. **After `make eval`**: Reports in `reports/eval_report.{md,json}`

## Clean Run Prerequisites

To do a **fresh re-run** while keeping existing downloaded images (e.g., for benchmarking different gating prompts or synthesis parameters), you must reset several directories and files:

### Option 1: Create a New Run Directory (Recommended)

```bash
# Create fresh benchmark directory
mkdir -p Benchmark_Run_002/data Benchmark_Run_002/manifests Benchmark_Run_002/lists

# Copy only original images (no synth, no excluded)
cp -r data/images_original Benchmark_Run_002/data/

# Copy starting points
# Option A: Use original points template
cp manifests/points.csv Benchmark_Run_002/manifests/

# Option B: Use a backup of images.csv from BEFORE any processing
cp manifests/images_backup.csv Benchmark_Run_002/manifests/images.csv

# Run pipeline in new directory
cd Benchmark_Run_002
python ../scripts/02_fetch_streetview.py --run-dir .
python ../scripts/03_detect_empty_frames.py --run-dir .
# ... continue with other stages
```

### Option 2: Clean In-Place (Destructive)

```bash
# 1. Clear all list files
rm -f lists/*.txt

# 2. Remove generated boxes
rm -f manifests/boxes_autogen*.jsonl

# 3. Remove synthetic images
rm -rf data/images_synth/*

# 4. Remove auto-generated labels
rm -rf data/labels_autogen/*

# 5. Remove final labels (or backup first)
rm -rf data/labels_final/*

# 6. Move excluded images back to original location
mv data/images_excluded/non_road/* data/images_original/ 2>/dev/null
mv data/images_excluded/non_street/* data/images_original/ 2>/dev/null
rm -rf data/images_excluded/*

# 7. Reset manifest processing columns
# You need a backup of images.csv from before processing, OR re-run fetch
# to regenerate it from points.csv
```

### Critical: Manifest State

The `manifests/images.csv` file accumulates processing columns:
- `num_boxes_autogen`
- `route_category`
- `street_scene_valid`, `road_scene_valid`
- `gemini_gate_json`
- etc.

**The scripts use these columns to skip already-processed items.** If you don't reset the manifest, re-running will skip everything.

**Best practice:** Keep a backup of `images.csv` after fetch but before any processing:

```bash
# After make fetch (or python scripts/02_fetch_streetview.py)
cp manifests/images.csv manifests/images_post_fetch_backup.csv
```

### Quick Clean Script

For convenience, you can use the `--nuke` flag on the detection stage:

```bash
# This wipes run artifacts but preserves original images
python scripts/03_detect_empty_frames.py --nuke
```

Note: `--nuke` resets detection/gating artifacts but does NOT remove synthetic images or labels.

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

# Include rejected rows explicitly (override default safety policy)
python scripts/05_prelabel_boxes.py --include-rejected
python scripts/06_split_dataset.py --include-rejected
python scripts/07_train_yolo.py --include-rejected --manifest manifests/images.csv
```

## Troubleshooting

### API Key Errors

- **GSV_API_KEY**: Get from [Google Cloud Console](https://console.cloud.google.com/). Enable "Street View Static API".
- **GEMINI_API_KEY**: Get from [Google AI Studio](https://aistudio.google.com/app/apikey).

### Empty Candidates

If no empty candidates found:
- Try different locations with fewer vehicles
- Lower the confidence threshold: `--conf 0.15`

**Warning:** A missed vehicle during empty detection means you'll inject a synthetic car into a frame that already has one. Always use a lower confidence threshold (0.15–0.25) for this step, even if it means fewer "empty" frames.
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
