# ParkOpticon YOLO11s Bootstrap

Web-UI-first pipeline for building and training a 4-class YOLO11s vehicle detector from Street View backgrounds and synthetic vehicle edits.

The Jupyter notebook in this repo is historical and mostly outdated. Use the web UI for day-to-day setup, review, and batch execution.

## What the pipeline does

1. Collect Street View panoramas
2. Review empty frames and synthesize vehicles
3. Auto-label and manually correct boxes
4. Split by panorama group to avoid leakage
5. Train, evaluate, and run inference

## Classes

- `0 vehicle` - regular vehicles
- `1 enforcement_vehicle` - parking enforcement vehicles
- `2 police_old` - Ottawa Police old livery
- `3 police_new` - Ottawa Police new livery
- `4 lookalike_negative` - non-enforcement vehicles that resemble enforcement

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Add API keys to `.env`:

```env
GSV_API_KEY=your_google_street_view_api_key
GEMINI_API_KEY=your_gemini_api_key
```

## Start the web UI

```bash
make ui
```

Or:

```bash
python web_ui/app.py
```

Open: `http://localhost:8000`

## Recommended workflow

### 1) Open the UI and choose a run

Use the directory selector at the top of the web UI to scope the session to an existing run directory.

This is the preferred way to work. The only missing piece is creating a brand-new run directory from inside the UI.

If you want to start the server already scoped to a run, you can still pass `--run-dir`, but it is optional.

### 2) Add points

Use **Point Manager** (`/point-manager`) to create or review Street View sample points.

### 3) Review empty frames

Use **Empty Review** (`/synth-review-empty`) to exclude bad backgrounds before synthesis.

### 4) Generate synthetic images

Use **Synthetic Generation** (`/synth-gen`) to create vehicle edits.

Optional cleanup tools:

- **Touchup Playground** (`/touchup-playground`)
- **Synthetic Review Buckets** (`/synth-review-buckets`)
- **Synthetic Cleanup** (`/synth-cleanup`)

### 5) Review labels

Use **Labeler** (`/labeler`) to fix auto-generated boxes and approve training data.

Rejected rows in `manifests/images.csv` are hidden by default. Start the UI with:

```bash
python web_ui/app.py --labeler-include-rejected
```

### 6) Split and train

Use **Batch Run Manager** (`/batch-run-manager`) for sequential split + train jobs.

Use **Training Viewer** (`/training-viewer`) to inspect runs, weights, and plots.

### 7) Run inference

Use **Inference Runner** (`/inference-runner`) to test models on images or video.

## Web UI pages

- `/labeler` - review and correct labels
- `/batch-run-manager` - build and launch split+train plans
- `/point-manager` - manage Street View points
- `/synth-gen` - generate synthetic edits
- `/synth-review-empty` - filter empty frames
- `/synth-review-buckets` - bucketed synthetic QA
- `/synth-cleanup` - delete poor synthetic outputs
- `/training-viewer` - inspect training runs
- `/inference-runner` - run inference on media
- `/lookalike-tracker` - review lookalike negatives
- `/oversized-box-audit` - audit oversized synthetic boxes

## Core command-line fallback

The UI is the preferred path, but the scripts remain available:

```bash
make fetch
make crop
make empty
make synth
make prelabel
make split
make train
make eval
```

For run-scoped execution, pass explicit paths to the scripts that need them.

## Rejected image policy

Rows marked `review_status=rejected` are excluded by default from prelabeling, splitting, and training prep.

- Prelabel: `--include-rejected`
- UI labeler: `--labeler-include-rejected`
- Split: `--include-rejected`
- Train: `--include-rejected`

## Run directory layout

```text
runs/<run_name>/
├── data/
│   ├── images_original/
│   ├── images_synth/
│   ├── labels_autogen/
│   ├── labels_final/
│   └── splits/
├── manifests/
│   ├── points.csv
│   └── images.csv
└── lists/
```

Create new run folders manually for now; the UI can switch between existing runs.

## Notes

- Split by `pano_id`, not by image.
- Keep synthetic siblings in the same split as their source frame.
- Apply the same preprocessing at inference time that you used during training.
- The legacy notebook is not the source of truth for current workflows.
