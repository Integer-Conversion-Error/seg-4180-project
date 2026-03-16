# Synthetic QA Checks Implementation Summary

## Overview
Added comprehensive quality assurance checks for synthetic vehicle edits in `scripts/05_prelabel_boxes.py`. The implementation validates diff masks between original and edited images to identify problematic synthetic edits.

## Changes Made

### 1. New Function: `analyze_diff_mask()`
**Purpose**: Analyzes the difference mask between original and edited images.

**Signature**:
```python
def analyze_diff_mask(
    original_path: Path,
    edited_path: Path,
    threshold: int = 30,
    area_ratio_threshold: float = 0.15
) -> dict
```

**Returns**:
```python
{
    'valid': bool,                    # Passed all QA checks
    'mask_area_ratio': float,         # Ratio of changed pixels to total
    'touches_border': bool,           # Change region touches image edge
    'num_large_components': int,      # Count of connected components > 100px
    'bbox': tuple | None,             # (x1, y1, x2, y2) of change region
    'failure_reason': str             # Reason for failure (if any)
}
```

**QA Checks**:
1. **`edit_too_large`**: `mask_area_ratio > area_ratio_threshold` (default: 0.15)
   - Detects excessively large edits indicating poor quality
   
2. **`vehicle_cut_off`**: `touches_border = True`
   - Detects vehicles edited at image boundaries (incomplete)
   
3. **`multiple_edits`**: `num_large_components > 1`
   - Detects multiple separate edits (unwanted multiple insertions)

### 2. CLI Arguments Added

```bash
python scripts/05_prelabel_boxes.py \
  --qa-threshold-area 0.15 \    # Configurable mask area threshold
  --qa-skip                       # Disable QA checks entirely
```

| Argument | Type | Default | Purpose |
|----------|------|---------|---------|
| `--qa-threshold-area` | float | 0.15 | Max allowed mask area ratio |
| `--qa-skip` | flag | False | Skip all QA validation |

### 3. Manifest Enhancement

Two new columns added to the image manifest:

| Column | Values | Purpose |
|--------|--------|---------|
| `qa_passed` | '0', '1' | Whether image passed QA checks |
| `qa_failure_reason` | string | Specific failure reason if `qa_passed='0'` |

Example failure reasons:
- `edit_too_large` - Edit region covers >15% of image
- `vehicle_cut_off` - Vehicle extends to image border
- `multiple_edits` - Multiple separate edits detected
- `parent_not_found` - Parent original image missing
- `image_load_failed` - Unable to read image file
- `exception: ...` - Other errors

### 4. QA Processing Loop Integration

During pre-labeling, for each synthetic image:

```python
if is_synthetic and expected_class != 'none' and not args.qa_skip:
    qa_result = analyze_diff_mask(parent_path, file_path, 
                                   area_ratio_threshold=args.qa_threshold_area)
    qa_passed = qa_result['valid']
    
    if not qa_passed:
        bad_synthetic_ids.append(image_id)  # Track failed IDs
        row['needs_review'] = '1'            # Flag for manual review
```

### 5. Bad Synthetics List

Creates `lists/bad_synthetics.txt` containing image IDs that failed QA:

```
image_001
image_042
image_157
...
```

Used for downstream filtering (e.g., excluding from training data).

## Technical Implementation

### Image Differencing Algorithm

1. Load original and edited images
2. Convert to grayscale and compute absolute difference
3. Threshold at 30 (configurable via `threshold` param)
4. Morphological closing with 15×15 kernel
5. Border checking via dilation + edge detection
6. Connected component labeling using `scipy.ndimage.label()`
7. Calculate mask coverage ratio

### Border Detection

Dilates the mask by 5 iterations and checks:
- Top edge: `mask[0, :]`
- Bottom edge: `mask[-1, :]`
- Left edge: `mask[:, 0]`
- Right edge: `mask[:, -1]`

Any non-zero pixel on edges = `touches_border = True`

### Large Component Detection

Uses `scipy.ndimage.label()` to find connected components:
- Label each contiguous region
- Count components with area > 100 pixels
- Multiple large components indicate multiple edits

## Dependencies

- `cv2` (OpenCV) - Image reading, differencing, morphology
- `scipy.ndimage` - Connected component labeling
- `numpy` - Array operations
- Existing: `argparse`, `csv`, `pathlib`, `logging`

**Removed unused import**: `from skimage import measure`

## Usage Examples

### Run with default QA settings (0.15 area threshold)
```bash
python scripts/05_prelabel_boxes.py \
  --manifest manifests/images.csv \
  --out-dir data/labels_autogen
```

### Use stricter threshold (only 5% area change allowed)
```bash
python scripts/05_prelabel_boxes.py \
  --qa-threshold-area 0.05
```

### Skip QA checks (process all synthetics)
```bash
python scripts/05_prelabel_boxes.py \
  --qa-skip
```

## Output

Logged information:
```
Labels saved to data/labels_autogen
QA failed synthetics: 12
Bad synthetics list: lists/bad_synthetics.txt
```

## Backward Compatibility

- All existing functionality preserved
- YOLO label format unchanged
- Manifest CSV structure extended (new columns appended)
- QA checks are only applied to synthetic images
- Original empty frame behavior unchanged

## Testing Recommendations

1. **Visual inspection** of `lists/bad_synthetics.txt` results
2. **Threshold tuning** - adjust `--qa-threshold-area` based on dataset
3. **Spot check** - verify failed images via image viewer
4. **Compare with/without QA** - measure impact on model training

## Future Enhancements

- Configurable component size threshold (currently hardcoded at 100px)
- Visualization of failed QA checks (save marked-up images)
- QA metrics dashboard
- Per-failure-type statistics
- Multi-channel diff analysis (RGB instead of grayscale)
