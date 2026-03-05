"""
Audit autogen labels for quality issues.

Scans every .txt file in data/labels_autogen/ and flags:
  1. FULL_IMAGE  — box covers >= 90% of the image (w*h >= 0.9)
  2. HUGE_BOX    — box area > 50% of image but < 90%
  3. TINY_BOX    — box area < 0.05% of image (w*h < 0.0005)
  4. OOB         — center or edges outside [0, 1]
  5. DEGENERATE  — width or height <= 0
  6. BAD_CLASS   — class ID not in {0,1,2,3,4}
  7. OVERLAP     — two boxes in same file overlap >= 80% IoU (near-duplicates)
  8. EXTREME_AR  — aspect ratio > 8:1 or < 1:8

Outputs a summary to stdout and writes detailed CSV to data/audit_autogen_report.csv.

Usage:
    python scripts/audit_autogen_labels.py
"""

import csv
import os
from pathlib import Path

LABELS_DIR = Path("data/labels_autogen")
REPORT_PATH = Path("data/audit_autogen_report.csv")

VALID_CLASSES = {0, 1, 2, 3, 4}

# Thresholds
FULL_IMAGE_AREA = 0.90
HUGE_BOX_AREA = 0.50
TINY_BOX_AREA = 0.0005
OVERLAP_IOU_THRESH = 0.80
EXTREME_AR_THRESH = 8.0


def parse_label(path: Path) -> list[tuple[int, float, float, float, float]]:
    """Parse YOLO label file. Returns list of (cls, xc, yc, w, h)."""
    boxes = []
    text = path.read_text().strip()
    if not text:
        return boxes
    for line_no, line in enumerate(text.splitlines(), 1):
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            cls = int(parts[0])
            xc, yc, w, h = (
                float(parts[1]),
                float(parts[2]),
                float(parts[3]),
                float(parts[4]),
            )
            boxes.append((cls, xc, yc, w, h))
        except (ValueError, IndexError):
            pass
    return boxes


def iou(box_a, box_b):
    """Compute IoU between two YOLO boxes (cls, xc, yc, w, h)."""
    _, axc, ayc, aw, ah = box_a
    _, bxc, byc, bw, bh = box_b

    ax1, ay1, ax2, ay2 = axc - aw / 2, ayc - ah / 2, axc + aw / 2, ayc + ah / 2
    bx1, by1, bx2, by2 = bxc - bw / 2, byc - bh / 2, bxc + bw / 2, byc + bh / 2

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area_a = aw * ah
    area_b = bw * bh
    union = area_a + area_b - inter_area

    if union <= 0:
        return 0.0
    return inter_area / union


def audit_file(path: Path) -> list[dict]:
    """Audit a single label file. Returns list of issue dicts."""
    issues = []
    boxes = parse_label(path)
    fname = path.name

    for i, (cls, xc, yc, w, h) in enumerate(boxes):
        box_str = f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"

        # BAD_CLASS
        if cls not in VALID_CLASSES:
            issues.append(
                {
                    "file": fname,
                    "box_idx": i,
                    "issue": "BAD_CLASS",
                    "detail": f"class={cls}",
                    "box": box_str,
                }
            )

        # DEGENERATE
        if w <= 0 or h <= 0:
            issues.append(
                {
                    "file": fname,
                    "box_idx": i,
                    "issue": "DEGENERATE",
                    "detail": f"w={w:.6f} h={h:.6f}",
                    "box": box_str,
                }
            )
            continue  # skip further checks on degenerate box

        area = w * h

        # OOB — check if box edges extend outside [0, 1]
        x1, y1 = xc - w / 2, yc - h / 2
        x2, y2 = xc + w / 2, yc + h / 2
        if x1 < -0.01 or y1 < -0.01 or x2 > 1.01 or y2 > 1.01:
            issues.append(
                {
                    "file": fname,
                    "box_idx": i,
                    "issue": "OOB",
                    "detail": f"corners=[{x1:.4f},{y1:.4f},{x2:.4f},{y2:.4f}]",
                    "box": box_str,
                }
            )

        # FULL_IMAGE
        if area >= FULL_IMAGE_AREA:
            issues.append(
                {
                    "file": fname,
                    "box_idx": i,
                    "issue": "FULL_IMAGE",
                    "detail": f"area={area:.4f}",
                    "box": box_str,
                }
            )
        elif area >= HUGE_BOX_AREA:
            issues.append(
                {
                    "file": fname,
                    "box_idx": i,
                    "issue": "HUGE_BOX",
                    "detail": f"area={area:.4f}",
                    "box": box_str,
                }
            )

        # TINY_BOX
        if area < TINY_BOX_AREA:
            issues.append(
                {
                    "file": fname,
                    "box_idx": i,
                    "issue": "TINY_BOX",
                    "detail": f"area={area:.8f}",
                    "box": box_str,
                }
            )

        # EXTREME_AR
        ar = w / h if h > 0 else float("inf")
        if ar > EXTREME_AR_THRESH or ar < 1.0 / EXTREME_AR_THRESH:
            issues.append(
                {
                    "file": fname,
                    "box_idx": i,
                    "issue": "EXTREME_AR",
                    "detail": f"ar={ar:.2f} (w={w:.4f} h={h:.4f})",
                    "box": box_str,
                }
            )

    # OVERLAP — pairwise IoU check
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            iou_val = iou(boxes[i], boxes[j])
            if iou_val >= OVERLAP_IOU_THRESH:
                issues.append(
                    {
                        "file": fname,
                        "box_idx": f"{i},{j}",
                        "issue": "OVERLAP",
                        "detail": f"IoU={iou_val:.4f} classes={boxes[i][0]},{boxes[j][0]}",
                        "box": f"box{i} vs box{j}",
                    }
                )

    return issues


def main():
    if not LABELS_DIR.is_dir():
        print(f"ERROR: {LABELS_DIR} not found")
        return

    all_files = sorted(LABELS_DIR.glob("*.txt"))
    print(f"Scanning {len(all_files)} label files in {LABELS_DIR}/\n")

    all_issues: list[dict] = []
    files_with_issues = set()

    for path in all_files:
        issues = audit_file(path)
        if issues:
            all_issues.extend(issues)
            files_with_issues.add(path.name)

    # Summary by issue type
    issue_counts: dict[str, int] = {}
    for iss in all_issues:
        issue_counts[iss["issue"]] = issue_counts.get(iss["issue"], 0) + 1

    print("=" * 60)
    print("AUTOGEN LABEL AUDIT SUMMARY")
    print("=" * 60)
    print(f"Total files scanned:      {len(all_files)}")
    print(f"Files with issues:        {len(files_with_issues)}")
    print(f"Total issues found:       {len(all_issues)}")
    print()

    if issue_counts:
        print("Issue breakdown:")
        for issue_type in sorted(issue_counts, key=issue_counts.get, reverse=True):
            print(f"  {issue_type:20s}  {issue_counts[issue_type]:>6d}")
        print()

        # Show first few examples of each type
        shown_types: dict[str, int] = {}
        print("Sample issues (first 5 per type):")
        print("-" * 60)
        for iss in all_issues:
            itype = iss["issue"]
            shown_types[itype] = shown_types.get(itype, 0) + 1
            if shown_types[itype] <= 5:
                print(
                    f"  [{itype}] {iss['file']}  box={iss['box_idx']}  {iss['detail']}"
                )
                print(f"          {iss['box']}")
        print("-" * 60)
    else:
        print("No issues found! All labels look clean.")

    # Write CSV report
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["file", "box_idx", "issue", "detail", "box"]
        )
        writer.writeheader()
        writer.writerows(all_issues)

    print(f"\nDetailed report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
