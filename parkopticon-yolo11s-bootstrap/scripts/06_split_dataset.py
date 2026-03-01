#!/usr/bin/env python3
"""
Split dataset into train/val/test with group-aware splitting.
Groups by pano_id or location+heading to prevent data leakage.
"""

import argparse
import csv
import logging
import random
import shutil
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_manifest(manifest_path: Path) -> list:
    with open(manifest_path, 'r') as f:
        return list(csv.DictReader(f))


def save_manifest(manifest: list, manifest_path: Path):
    if not manifest:
        return
    fieldnames = list(manifest[0].keys())
    with open(manifest_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest)


def get_group_key(row: dict) -> str:
    pano_id = row.get('pano_id', '')
    if pano_id:
        return f"pano:{pano_id}"
    
    lat = row.get('pano_lat', '')
    lng = row.get('pano_lng', '')
    heading = row.get('heading', '')
    
    try:
        lat_rounded = round(float(lat), 4) if lat else 0
        lng_rounded = round(float(lng), 4) if lng else 0
        heading_bucket = (int(heading) // 90) * 90 if heading else 0
        return f"loc:{lat_rounded},{lng_rounded},h{heading_bucket}"
    except (ValueError, TypeError):
        return f"loc:{lat},{lng}"


def main():
    parser = argparse.ArgumentParser(description="Split dataset with group awareness")
    parser.add_argument("--manifest", "-m", default="manifests/images.csv", help="Input manifest")
    parser.add_argument("--out-dir", "-o", default="data/splits", help="Output directory")
    parser.add_argument("--labels-dir", "-l", default="data/labels_autogen", help="Labels directory")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Train ratio")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Val ratio")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    manifest_path = Path(args.manifest)
    out_dir = Path(args.out_dir)
    labels_dir = Path(args.labels_dir)
    
    if not manifest_path.exists():
        logger.error(f"Manifest not found: {manifest_path}")
        return
    
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 0.01:
        logger.error("Ratios must sum to 1.0")
        return
    
    manifest = load_manifest(manifest_path)
    
    valid_images = [
        row for row in manifest 
        if row.get('status') == 'ok' 
        and Path(row.get('file_path', '')).exists()
    ]
    
    groups = defaultdict(list)
    for row in valid_images:
        group_key = get_group_key(row)
        groups[group_key].append(row)
    
    logger.info(f"Found {len(valid_images)} valid images in {len(groups)} groups")
    
    random.seed(args.seed)
    group_keys = list(groups.keys())
    random.shuffle(group_keys)
    
    n_groups = len(group_keys)
    n_train = int(n_groups * args.train_ratio)
    n_val = int(n_groups * args.val_ratio)
    
    train_groups = set(group_keys[:n_train])
    val_groups = set(group_keys[n_train:n_train + n_val])
    test_groups = set(group_keys[n_train + n_val:])
    
    split_map = {}
    for g in train_groups:
        for row in groups[g]:
            split_map[row['image_id']] = 'train'
    for g in val_groups:
        for row in groups[g]:
            split_map[row['image_id']] = 'val'
    for g in test_groups:
        for row in groups[g]:
            split_map[row['image_id']] = 'test'
    
    split_counts = defaultdict(int)
    enforcement_in_test = 0
    
    for row in valid_images:
        split = split_map.get(row['image_id'], 'train')
        row['split'] = split
        split_counts[split] += 1
        
        if split == 'test' and row.get('expected_inserted_class') == 'enforcement_vehicle':
            enforcement_in_test += 1
    
    save_manifest(manifest, manifest_path)
    
    for split in ['train', 'val', 'test']:
        img_dir = out_dir / split / 'images'
        lbl_dir = out_dir / split / 'labels'
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
    
    for row in tqdm(valid_images, desc="Copying files"):
        split = row['split']
        image_id = row['image_id']
        
        src_img = Path(row['file_path'])
        dst_img = out_dir / split / 'images' / f"{image_id}.jpg"
        
        if not dst_img.exists():
            shutil.copy2(src_img, dst_img)
        
        src_lbl = labels_dir / f"{image_id}.txt"
        dst_lbl = out_dir / split / 'labels' / f"{image_id}.txt"
        
        if src_lbl.exists() and not dst_lbl.exists():
            shutil.copy2(src_lbl, dst_lbl)
    
    logger.info(f"Split complete:")
    for split, count in split_counts.items():
        logger.info(f"  {split}: {count}")
    
    if enforcement_in_test == 0:
        logger.warning("=" * 60)
        logger.warning("WARNING: No enforcement_vehicle examples in test set!")
        logger.warning("Consider rebalancing or increasing enforcement rate.")
        logger.warning("=" * 60)
    else:
        logger.info(f"Test set has {enforcement_in_test} enforcement_vehicle examples")


if __name__ == "__main__":
    main()
