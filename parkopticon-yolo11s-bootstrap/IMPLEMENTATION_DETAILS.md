# QA Implementation - Technical Details

## File Modified
- `scripts/05_prelabel_boxes.py` (423 lines, +9 functions)

## Code Changes Summary

### 1. Import Cleanup
**Removed**: `from skimage import measure` (unused)
**Kept**: All required dependencies (cv2, numpy, scipy.ndimage)

### 2. New Function: `analyze_diff_mask()` (Lines 88-199)

Core algorithm:
```
1. Load original and edited images
2. Resize to same dimensions if needed
3. Convert to grayscale
4. Compute absolute difference
5. Threshold at 30 (pixels > 30 are "changed")
6. Morphological closing with 15x15 kernel to connect regions
7. Calculate mask_area_ratio = changed_pixels / total_pixels
8. Dilate mask and check if touches image borders
9. Label connected components using scipy.ndimage.label()
10. Count large components (area > 100 pixels)
11. Run QA checks and set failure_reason
12. Return result dict
```

QA Check Logic (Priority Order):
1. If `mask_area_ratio > area_ratio_threshold` → FAIL: `edit_too_large`
2. Elif `touches_border` → FAIL: `vehicle_cut_off`
3. Elif `num_large_components > 1` → FAIL: `multiple_edits`
4. Else → PASS: `valid = True`

### 3. CLI Arguments Added (Lines 281-283)

```python
parser.add_argument(
    "--qa-threshold-area",
    type=float,
    default=0.15,
    help="QA threshold for mask area ratio"
)
parser.add_argument(
    "--qa-skip",
    action="store_true",
    help="Skip QA checks"
)
```

### 4. Main Function Enhancements

#### Initialization (Lines 290-304)
```python
lists_dir = Path("lists")
lists_dir.mkdir(exist_ok=True)
bad_synthetics_path = lists_dir / "bad_synthetics.txt"
bad_synthetic_ids = []  # Collect failed IDs
```

#### QA Check Integration (Lines 318-346)
```python
if is_synthetic and expected_class != 'none' and not args.qa_skip:
    parent_row = next((r for r in manifest if r.get("image_id") == parent_id), None)
    if parent_row:
        parent_path = Path(parent_row.get("file_path", ""))
        if parent_path.exists():
            qa_result = analyze_diff_mask(
                parent_path,
                file_path,
                area_ratio_threshold=args.qa_threshold_area
            )
            qa_passed = qa_result['valid']
            qa_failure_reason = qa_result['failure_reason']
            
            if not qa_passed:
                bad_synthetic_ids.append(image_id)
                row['needs_review'] = '1'
```

#### Manifest Enhancement (Lines 348-351)
```python
row['qa_passed'] = '1' if qa_passed else '0'
if qa_failure_reason:
    row['qa_failure_reason'] = qa_failure_reason
```

#### Output Generation (Lines 412-419)
```python
with open(bad_synthetics_path, 'w') as f:
    for img_id in bad_synthetic_ids:
        f.write(f"{img_id}\n")

logger.info(f"QA failed synthetics: {len(bad_synthetic_ids)}")
if bad_synthetic_ids:
    logger.info(f"Bad synthetics list: {bad_synthetics_path}")
```

## Algorithm Details

### Border Detection
Dilates mask by 5 iterations, then checks:
- `touches_top = np.any(border_check[0, :] > 0)`
- `touches_bottom = np.any(border_check[-1, :] > 0)`
- `touches_left = np.any(border_check[:, 0] > 0)`
- `touches_right = np.any(border_check[:, -1] > 0)`

Result: `touches_border = OR of all 4 checks`

### Connected Components
```python
labeled, num_features = ndimage.label(closed)
component_sizes = np.bincount(labeled.ravel())
large_components = np.sum(component_sizes[1:] > 100)  # Skip label 0 (background)
```

### Mask Area Calculation
```python
mask_pixels = np.count_nonzero(closed)  # Count white pixels
total_pixels = closed.shape[0] * closed.shape[1]
mask_area_ratio = mask_pixels / total_pixels
```

## Control Flow

```
main()
  ├─ Parse arguments (including qa-threshold-area, qa-skip)
  ├─ Load manifest
  ├─ Load YOLO model
  └─ For each image in manifest:
      ├─ Skip if not ok status or file missing
      ├─ Run vehicle detection
      │
      ├─ IF is_synthetic AND has expected_class AND NOT qa_skip:
      │   ├─ Find parent image
      │   └─ IF parent exists:
      │       ├─ Call analyze_diff_mask()
      │       ├─ Track result in qa_passed, qa_failure_reason
      │       └─ IF failed:
      │           ├─ Add image_id to bad_synthetic_ids
      │           └─ Set needs_review = '1'
      │
      ├─ Add qa_passed and qa_failure_reason to manifest row
      │
      ├─ IF is_synthetic AND has expected_class:
      │   ├─ Get change bbox from compute_change_region()
      │   └─ Match detected boxes to change bbox
      │
      ├─ Convert boxes to YOLO format
      └─ Write YOLO labels
  
  ├─ Save updated manifest (with QA columns)
  ├─ Write lists/bad_synthetics.txt
  └─ Log summary stats
```

## Data Flow

### Input
- `manifests/images.csv` (manifest with image metadata)
- Original and synthetic images on disk

### Processing
- Extract image paths and synthetic flags
- Run QA analysis on each synthetic image
- Collect bad synthetic IDs

### Output
- Updated `manifests/images.csv` (new columns: qa_passed, qa_failure_reason)
- `lists/bad_synthetics.txt` (one image ID per line)
- YOLO labels in `data/labels_autogen/`

## Performance Considerations

### Time Complexity
- Per image: O(W*H*N) where W=width, H=height, N=image operations
- Dominates: Image I/O and morphological operations
- CC labeling: O(W*H) using union-find

### Space Complexity
- Per image: O(W*H*3) for RGB + O(W*H) for processing arrays
- Total: Linear with image count and size

### Optimization Opportunities
- Skip QA for small batch with `--qa-skip`
- Parallel processing (futures/multiprocessing)
- Lower resolution diff masks
- GPU acceleration (CUDA morphology)

## Error Handling

```python
try:
    # All QA logic
    ...
    return result  # valid=True/False based on checks
except Exception as e:
    logger.error(f"QA analysis failed: {e}")
    result['valid'] = False
    result['failure_reason'] = f'exception: {str(e)[:50]}'
    return result
```

- Catches and logs all exceptions
- Returns safe failure result
- Continues processing other images

## Testing Strategy

### Unit Tests (Manual)
1. Test with known good synthetic edit (should pass)
2. Test with oversized edit (should fail: edit_too_large)
3. Test with edge-touching edit (should fail: vehicle_cut_off)
4. Test with multiple vehicles (should fail: multiple_edits)

### Integration Tests
1. Run with real manifest and images
2. Verify bad_synthetics.txt created
3. Verify manifest columns added
4. Compare --qa-skip vs --qa-threshold-area results

### Validation
1. Visual spot-check failed images
2. Verify no false positives
3. Tune threshold based on results
4. Measure impact on model training

