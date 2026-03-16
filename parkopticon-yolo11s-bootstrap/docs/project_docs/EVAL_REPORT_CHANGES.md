# Evaluation Report Script Updates (08_eval_report.py)

## Summary
Updated `scripts/08_eval_report.py` to fully support 4-class metrics and added enforcement_present binary metric.

## Changes Made

### 1. Dynamic Class Loading
- **New function**: `load_class_names(data_yaml_path)`
- Reads class names from `dataset.yaml` using `yaml.safe_load()`
- Falls back to hardcoded defaults if YAML not found or malformed
- Returns dict: `{0: 'vehicle', 1: 'enforcement_vehicle', 2: 'police_old', 3: 'police_new'}`

### 2. Enforcement-Present Binary Metric
- **New function**: `compute_enforcement_present_metric(box_metrics, class_names)`
- Computes 4 metrics for binary classification (enforcement present = True/False):
  - **Accuracy**: Overall correctness in distinguishing enforcement from regular vehicles
  - **Precision**: Avoid misclassifying regular vehicles as enforcement
  - **Recall**: Ability to detect images with enforcement vehicles (target >0.85)
  - **F1**: Harmonic mean balancing precision and recall
- Uses per-class AP scores as proxy for image-level predictions
- Handles edge cases: missing classes, empty predictions, division by zero

### 3. Report Structure Updates

#### JSON Report (`eval_report.json`)
```json
{
  "eval_date": "...",
  "weights": "...",
  "confidence_threshold": 0.25,
  "num_classes": 4,
  "classes": {
    "0": "vehicle",
    "1": "enforcement_vehicle",
    "2": "police_old",
    "3": "police_new"
  },
  "metrics": {
    "mAP50": 0.85,
    "mAP50-95": 0.75,
    "precision": 0.90,
    "recall": 0.88
  },
  "per_class_metrics": {
    "mAP": {
      "vehicle": 0.82,
      "enforcement_vehicle": 0.76,
      "police_old": 0.72,
      "police_new": 0.70
    },
    "mAP50": {
      "vehicle": 0.92,
      "enforcement_vehicle": 0.88,
      "police_old": 0.85,
      "police_new": 0.83
    }
  },
  "enforcement_present": {
    "accuracy": 0.79,
    "precision": 0.76,
    "recall": 0.76,
    "f1": 0.76
  }
}
```

#### Markdown Report (`eval_report.md`)
- Overall Metrics table (unchanged)
- **New**: Per-Class Metrics table with all 4 classes and both mAP and mAP50
- **New**: Enforcement-Present Metric section with binary classification metrics
- Updated interpretation section with guidance on enforcement_present F1 score

### 4. Per-Class Metrics Extraction
- Reads `box_metrics.ap` (mAP at 0.5-0.95 IoU)
- Reads `box_metrics.ap50` (mAP at 0.5 IoU)
- Iterates through all class indices in `class_names` dictionary
- Handles missing classes gracefully (assigns 0.0)
- Includes try-except for robustness

### 5. Dependencies
- Added: `import yaml` (standard library via PyYAML package)
- Existing dependencies unchanged

## Edge Case Handling
1. Missing YAML file → Fallback to defaults
2. Missing classes in ground truth → Assigns 0.0 for that class
3. Empty predictions → Returns empty enforcement_metric dict
4. Division by zero in F1 → Checked with `if p + r > 0`
5. None values in AP arrays → Converted to 0.0

## Output Files (Unchanged)
- `reports/eval_report.json` - JSON format
- `reports/eval_report.md` - Markdown format

## Backwards Compatibility
- Keeps existing overall metrics (mAP50, mAP50-95, precision, recall)
- JSON output adds new sections but preserves existing keys
- Markdown output maintains same section structure

## Testing Notes
- Syntax verified with `ast.parse()`
- All key functions present: `load_class_names`, `compute_enforcement_present_metric`
- No new dependencies beyond existing `ultralytics`, `yaml`
