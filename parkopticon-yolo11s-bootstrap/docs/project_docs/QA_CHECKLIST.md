# QA Implementation Checklist

## Requirements Met

### 1. `analyze_diff_mask()` Function ✓
- [x] Loads both original and edited images
- [x] Computes absolute difference between images
- [x] Applies threshold (default 30) and morphological closing
- [x] Finds connected components using scipy.ndimage
- [x] Returns dict with all required fields:
  - `valid`: bool indicating if image passed QA
  - `mask_area_ratio`: ratio of changed pixels
  - `touches_border`: boolean for border detection
  - `num_large_components`: count of components > 100px
  - `bbox`: bounding box of change region
  - `failure_reason`: description of failure (if any)

### 2. QA Thresholds ✓
- [x] `mask_area_ratio > 0.15` = `edit_too_large` (configurable)
- [x] `touches_border = True` = `vehicle_cut_off`
- [x] `num_large_components > 1` = `multiple_edits`
- [x] Threshold is configurable via `--qa-threshold-area`

### 3. Integration into Processing Loop ✓
- [x] Called after computing change region
- [x] Added to manifest with two new columns:
  - `qa_passed`: '0' or '1'
  - `qa_failure_reason`: specific failure reason
- [x] Sets `needs_review = '1'` for failed images
- [x] Only applied to synthetic images (not empty originals)

### 4. Bad Synthetics List ✓
- [x] Creates `lists/bad_synthetics.txt`
- [x] Contains image IDs of failed synthetics
- [x] One ID per line for easy filtering
- [x] Created in `lists/` directory (auto-created if missing)

### 5. CLI Arguments ✓
- [x] `--qa-threshold-area FLOAT` (default: 0.15)
  - Configurable mask area ratio threshold
  - Used in `analyze_diff_mask()` call
  
- [x] `--qa-skip` FLAG
  - Disables all QA checks when set
  - Allows processing all synthetics without validation

### 6. Dependencies ✓
- [x] Uses OpenCV (cv2) for image processing
- [x] Uses scipy.ndimage for component labeling
- [x] No new dependencies beyond cv2/scipy (already required)
- [x] Removed unused skimage import

## Code Quality

- [x] Syntax valid (py_compile passes)
- [x] All 9 functions present and accounted for
- [x] Proper error handling in analyze_diff_mask()
- [x] Backward compatible (no breaking changes)
- [x] Existing functionality preserved
- [x] YOLO label format unchanged
- [x] Comments and docstrings present

## Integration Points

### Pre-labeling Loop
```python
# QA checks for synthetic images
if is_synthetic and expected_class != 'none' and not args.qa_skip:
    qa_result = analyze_diff_mask(parent_path, file_path,
                                   area_ratio_threshold=args.qa_threshold_area)
    qa_passed = qa_result['valid']
    qa_failure_reason = qa_result['failure_reason']
    
    if not qa_passed:
        bad_synthetic_ids.append(image_id)
        row['needs_review'] = '1'

# Add QA columns to manifest
row['qa_passed'] = '1' if qa_passed else '0'
if qa_failure_reason:
    row['qa_failure_reason'] = qa_failure_reason
```

### Output
- Logs summary: "QA failed synthetics: N"
- Creates filtered list: "Bad synthetics list: lists/bad_synthetics.txt"
- Updates manifest with QA status columns

## File Structure

```
scripts/05_prelabel_boxes.py
├── Imports (cv2, numpy, scipy.ndimage, etc.)
├── Logging setup
├── detect_vehicles()
├── compute_change_region()
├── analyze_diff_mask()          [NEW]
├── box_iou()
├── convert_to_yolo()
├── write_yolo_labels()
├── load_manifest()
├── save_manifest()
└── main()
    ├── Parse args (including --qa-threshold-area, --qa-skip)
    ├── Load manifest
    ├── Load YOLO model
    ├── Process each image
    │   ├── Detect vehicles
    │   ├── Run QA checks for synthetics
    │   ├── Track bad_synthetic_ids
    │   ├── Add QA columns to manifest
    │   ├── Process detected boxes
    │   └── Write YOLO labels
    ├── Save manifest with QA columns
    └── Write lists/bad_synthetics.txt

lists/
├── empty_candidates.txt         [EXISTING]
├── excluded_from_synth.txt      [EXISTING]
└── bad_synthetics.txt           [NEW]
```

## Verification

- Run: `python scripts/05_prelabel_boxes.py --help`
  - Should show `--qa-threshold-area` and `--qa-skip` options
  
- Check: `lists/bad_synthetics.txt` after first run
  - Should contain image IDs of failed QA checks
  
- Inspect: Updated `manifests/images.csv`
  - Should have new `qa_passed` and `qa_failure_reason` columns

