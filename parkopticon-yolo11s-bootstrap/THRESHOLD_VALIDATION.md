# Dataset Threshold Validation & Rebalancing

## Overview

`scripts/06_split_dataset.py` now validates and rebalances train/val/test splits to ensure minimum enforcement class representation.

## Features Added

### 1. Threshold Configuration

```python
MIN_INSTANCES = {
    'enforcement_vehicle': {'val': 50, 'test': 30},
    'police_old': {'val': 10, 'test': 5},
    'police_new': {'val': 10, 'test': 5}
}

MIN_PANO_GROUPS = {'val': 15, 'test': 8}
```

- Per-class minimum instance counts for val/test splits
- Per-class minimum unique pano-group counts
- Override via CLI arguments

### 2. Validation Functions

**`validate_split_thresholds(manifest, split_name) → list`**
- Counts instances of each enforcement class in split
- Counts unique pano groups containing each class
- Returns list of violations (empty if all thresholds met)
- INFO/WARNING logging for each class check

**`count_class_instances(manifest, split_name, class_name) → int`**
- Counts instances of a specific class in a split

**`count_pano_groups_with_class(manifest, split_name, class_name) → int`**
- Counts unique pano groups containing a class in a split

### 3. Rebalancing Function

**`rebalance_splits(manifest, violations) → int`**
- Identifies enforcement-positive groups in train split
- Moves whole groups to val/test to meet thresholds
- Respects group integrity (no splitting across splits)
- Logs all moves with source/dest and instance count
- Returns number of groups moved

### 4. Workflow Integration

After initial split creation:

1. Save initial manifest with splits
2. Validate val and test splits against thresholds
3. If violations detected:
   - Extract enforcement groups from train
   - Move groups to val/test until thresholds met
   - Recount splits and save updated manifest
4. Continue with file copying

## CLI Arguments

### New Arguments

```bash
--skip-threshold-check
  Skip validation and rebalancing (default: False)
  Use: python scripts/06_split_dataset.py --skip-threshold-check

--min-enforcement-val (int, default: 50)
  Minimum enforcement_vehicle instances in val set
  Use: python scripts/06_split_dataset.py --min-enforcement-val 60

--min-enforcement-test (int, default: 30)
  Minimum enforcement_vehicle instances in test set
  Use: python scripts/06_split_dataset.py --min-enforcement-test 40
```

## Example Usage

```bash
# Run with defaults (val: 50 enforcement, test: 30 enforcement)
python scripts/06_split_dataset.py

# Skip threshold check (original behavior)
python scripts/06_split_dataset.py --skip-threshold-check

# Custom enforcement minimums
python scripts/06_split_dataset.py --min-enforcement-val 75 --min-enforcement-test 50

# With custom split ratios and threshold check
python scripts/06_split_dataset.py \
  --train-ratio 0.65 \
  --val-ratio 0.25 \
  --test-ratio 0.1 \
  --min-enforcement-val 60
```

## Log Output

### Success Case

```
========================= THRESHOLD VALIDATION =========================
Validating val split thresholds...
  ✓ enforcement_vehicle: 55 instances >= 50
  ✓ police_old: 12 instances >= 10
  ✓ police_new: 11 instances >= 10
  ✓ enforcement_vehicle_pano_groups: 18 groups >= 15
  ✓ police_old_pano_groups: 16 groups >= 15
  ✓ police_new_pano_groups: 15 groups >= 15
Validating test split thresholds...
  ✓ enforcement_vehicle: 35 instances >= 30
  ...

All threshold validation checks passed!
```

### Violation & Rebalancing Case

```
========================= THRESHOLD VALIDATION =========================
Validating val split thresholds...
  VIOLATION: enforcement_vehicle: 25 instances < 50 required
  ...

Total violations detected: 2

======================================================================
REBALANCING: 2 threshold violations detected
======================================================================
Found 150 enforcement groups in train split
Attempting to fix 1 violations in val...
  Moved group pano:abc123xyz (enforcement_vehicle) with 8 images from train → val
  enforcement_vehicle threshold now met in val
Rechecking violations after rebalancing...
Updated split counts after rebalancing:
  train: 1200
  val: 280
  test: 120

Rebalancing complete: Moved 1 groups
======================================================================
```

## Backward Compatibility

- **Enabled by default**: Thresholds are enforced unless `--skip-threshold-check` is used
- **Original algorithm preserved**: Core splitting logic unchanged
- **Safe fallback**: If thresholds can't be met, script warns but continues
- **Manifest updated**: Split assignments persisted to `manifests/images.csv`

## Key Design Decisions

1. **Group-level moves**: Entire pano groups move together, preventing leakage
2. **Train-only source**: Only takes groups from train split (preserves initial ratios)
3. **Pano-group thresholds**: Ensures diversity, not just instance count
4. **Per-class thresholds**: Each enforcement class validated independently
5. **Clear logging**: Every validation check and move logged at INFO/WARNING level

## Troubleshooting

### Insufficient Training Data

If rebalancing can't meet thresholds:

```
VIOLATION: enforcement_vehicle: 20 instances < 50 required
```

Options:
1. **Reduce thresholds**: `--min-enforcement-val 40`
2. **Skip check**: `--skip-threshold-check` (not recommended)
3. **More data**: Generate more synthetic images

### Split Imbalance

If 70/20/10 ratio severely disrupted after rebalancing:

```
Rebalancing moved groups: 15
Updated split counts: train: 900, val: 280, test: 120
```

Consider:
1. Generating more enforcement data
2. Adjusting initial split ratios: `--train-ratio 0.6 --val-ratio 0.25 --test-ratio 0.15`

## Technical Details

### Violation Detection Logic

```python
# Instance check (per class, per split)
if count_class_instances(manifest, 'val', 'enforcement_vehicle') < 50:
    violation = "enforcement_vehicle: X instances < 50 required"

# Pano group check (per class, per split)
if count_pano_groups_with_class(manifest, 'val', 'enforcement_vehicle') < 15:
    violation = "enforcement_vehicle_pano_groups: X groups < 15 required"
```

### Rebalancing Logic

1. For each split (val, test):
   - Get current violations
   - For each violation (non-pano_groups):
     - Get all enforcement groups from train
     - For each group (sorted by key):
       - Skip if spans multiple splits
       - Move group to target split
       - Recount violations
       - Break if fixed

2. Save updated manifest

## Future Enhancements

- Per-split ratio targets (maintain 70/20/10 during rebalancing)
- Genetic algorithm for optimal group selection
- Support for `police_old` and `police_new` specific thresholds
- Dry-run mode to preview moves without applying
