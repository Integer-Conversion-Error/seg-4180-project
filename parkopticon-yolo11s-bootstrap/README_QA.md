# Synthetic QA Checks - Implementation Guide

## Quick Links

- **Main Script**: `scripts/05_prelabel_boxes.py`
- **Summary**: `DELIVERY_SUMMARY.txt` (read this first!)
- **Features**: `QA_IMPLEMENTATION_SUMMARY.md`
- **Checklist**: `QA_CHECKLIST.md`
- **Details**: `IMPLEMENTATION_DETAILS.md`

## What Was Added?

A new `analyze_diff_mask()` function that automatically validates synthetic vehicle edits by:

1. **Comparing images** - Original vs. edited using pixel differences
2. **Detecting problems**:
   - Oversized edits (>15% of image by default)
   - Vehicles touching image borders (cut off)
   - Multiple separate edits (suggests unwanted insertions)
3. **Creating output**:
   - Manifest columns: `qa_passed`, `qa_failure_reason`
   - Output file: `lists/bad_synthetics.txt`

## Quick Start

```bash
# Default: uses 0.15 area threshold
python scripts/05_prelabel_boxes.py

# Stricter: only allow 5% coverage
python scripts/05_prelabel_boxes.py --qa-threshold-area 0.05

# Skip QA checks
python scripts/05_prelabel_boxes.py --qa-skip
```

## Output

After running, check:

```bash
# See which synthetics failed QA
cat lists/bad_synthetics.txt

# See QA columns in manifest
grep "qa_passed\|qa_failure_reason" manifests/images.csv
```

## Failure Reasons

| Reason | Meaning | What to do |
|--------|---------|-----------|
| `edit_too_large` | >15% of image changed | Adjust `--qa-threshold-area` down |
| `vehicle_cut_off` | Vehicle extends to edge | Image is incomplete |
| `multiple_edits` | Multiple separate changes | Gemini added multiple objects |
| `parent_not_found` | Original image missing | Check manifest consistency |
| `image_load_failed` | Cannot read file | Check file paths |

## Configuration

### Threshold Tuning

Default is 0.15 (15% of image area). Adjust based on your needs:

- `0.05` - Very strict (5% max change)
- `0.10` - Strict (10% max change)
- `0.15` - Standard (15% max change)
- `0.20` - Lenient (20% max change)

### Skip All Checks

Use `--qa-skip` to process all synthetics without validation:

```bash
python scripts/05_prelabel_boxes.py --qa-skip
```

## How It Works

### Algorithm

1. Load original and synthetic images
2. Compute absolute difference between them
3. Threshold the difference (pixels > 30 intensity)
4. Morphological closing to connect nearby changes
5. Check if coverage exceeds threshold
6. Dilate mask and check if touches image borders
7. Find connected components and count large ones
8. Return pass/fail decision

### Image Processing Libraries

- **cv2** (OpenCV): Image loading, differencing, morphology
- **scipy.ndimage**: Connected component labeling
- **numpy**: Array operations

All are already dependencies of the project.

## Integration

The QA checks run automatically during pre-labeling:

```python
# For each synthetic image:
if is_synthetic and not args.qa_skip:
    qa_result = analyze_diff_mask(parent_img, synthetic_img)
    
    if qa_result['valid']:
        # Image passed - process normally
    else:
        # Image failed - flag for review
        row['needs_review'] = '1'
        bad_synthetic_ids.append(image_id)
```

## Downstream Usage

The `lists/bad_synthetics.txt` file can be used to filter training data:

```bash
# Exclude bad synthetics from training
comm -23 <(sort all_images.txt) <(sort lists/bad_synthetics.txt) > good_images.txt
```

## Verification

To verify the implementation is working:

1. Check help shows new arguments:
   ```bash
   python scripts/05_prelabel_boxes.py --help | grep -E "qa-"
   ```

2. Run with test data and inspect output:
   ```bash
   python scripts/05_prelabel_boxes.py
   ls -l lists/bad_synthetics.txt
   ```

3. Check manifest has new columns:
   ```bash
   head -1 manifests/images.csv | tr ',' '\n' | grep qa
   ```

## Advanced Usage

### Combine With Other Options

```bash
python scripts/05_prelabel_boxes.py \
  --manifest manifests/custom.csv \
  --out-dir data/labels_custom \
  --model yolo11s.pt \
  --conf 0.25 \
  --qa-threshold-area 0.12
```

### Post-Processing

```bash
# Count failures by reason
grep "qa_failure_reason" manifests/images.csv | \
  cut -d',' -f2 | sort | uniq -c

# Visual inspection of failed images
while read img_id; do
  echo "Image: $img_id"
  # ... display or process image
done < lists/bad_synthetics.txt
```

## Troubleshooting

### Too Many Failures

If many synthetics are failing QA:

1. Check threshold is reasonable:
   ```bash
   python scripts/05_prelabel_boxes.py --qa-threshold-area 0.20
   ```

2. Inspect a few failed images manually
3. Adjust threshold or QA rules if needed

### Too Few Failures

If no synthetics are failing:

1. Use stricter threshold:
   ```bash
   python scripts/05_prelabel_boxes.py --qa-threshold-area 0.05
   ```

2. Check if edits are actually reasonable
3. May need to regenerate synthetics with better prompts

### Missing Output File

If `lists/bad_synthetics.txt` doesn't exist:

1. Check script ran successfully (check logs)
2. Verify no errors in image loading
3. Ensure manifest has synthetic images

## Performance

- Processing time: ~100-500ms per synthetic image
- Dominated by: Image I/O and morphological operations
- Memory: ~O(image_width × image_height)

For 1000 synthetics: ~2-10 minutes on CPU

## Future Enhancements

Ideas for improvement:

1. **Visualization**: Save marked-up images showing QA failures
2. **Metrics Dashboard**: Track QA statistics over batches
3. **Multi-channel**: Analyze RGB differences, not just grayscale
4. **Component Size**: Make threshold configurable
5. **Parallelization**: Process multiple images in parallel

## Documentation Files

- `DELIVERY_SUMMARY.txt` - Executive summary
- `QA_IMPLEMENTATION_SUMMARY.md` - Feature overview
- `QA_CHECKLIST.md` - Requirements verification
- `IMPLEMENTATION_DETAILS.md` - Technical deep dive
- `README_QA.md` - This file

## Support

For issues or questions:

1. Check the documentation files above
2. Review inline code comments in `scripts/05_prelabel_boxes.py`
3. Test with `--qa-skip` to isolate QA-related issues
4. Inspect `lists/bad_synthetics.txt` for pattern analysis

