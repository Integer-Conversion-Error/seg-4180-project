#!/usr/bin/env python3
"""
Collect Hard Negatives for ParkOpticon Vehicle Detection

Hard negatives are images that SHOULD NOT contain vehicles but might cause
false positives in the detector. Examples:
  - Empty sidewalks and shoulders
  - Grass medians and green spaces
  - Bike lanes without cyclists
  - Driveway entrances and parking lot edges
  - Parking spaces without vehicles
  - Street furniture and poles

This script provides a template and utilities for collecting hard negatives
from Google Street View to reduce false positive detections.

Why Hard Negatives Matter:
  - The model learns from what it DOESN'T see
  - False positives on empty spaces hurt real-world performance
  - Hard negatives teach the model to distinguish signal from noise
  - Examples: shadows on pavement, manhole covers, road markings

Usage:
    python scripts/10_collect_hard_negatives.py \\
        --points-csv manifests/points.csv \\
        --output-dir data/images_hard_negatives \\
        --num-samples 100

Future Enhancement:
    - Street View API integration for automated collection
    - Region-of-Interest (ROI) selection for specific scene areas
    - Automatic ground truth verification
"""

import os
import sys
import csv
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class HardNegativeCollector:
    """
    Placeholder for hard negatives collection.

    Future implementation will:
    1. Use Street View API to fetch scene images
    2. Detect and extract hard negative regions
    3. Validate they don't contain vehicles
    4. Create labels indicating no vehicles
    """

    def __init__(self, output_dir: str):
        """
        Initialize hard negatives collector.

        Args:
            output_dir: Directory to store collected hard negatives
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.images_dir = self.output_dir / "images"
        self.images_dir.mkdir(exist_ok=True)

        self.metadata_file = self.output_dir / "hard_negatives_manifest.jsonl"
        self.log_file = self.output_dir / "collection_log.txt"

        logger.info(f"Hard negatives output directory: {self.output_dir}")

    def log_collection_info(self, message: str) -> None:
        """Log collection information to file."""
        with open(self.log_file, "a") as f:
            f.write(f"[{datetime.now().isoformat()}] {message}\n")

    def collect_from_points_csv(
        self,
        points_csv: str,
        num_samples_per_point: int = 1,
        sample_type: str = "sidewalk",
    ) -> int:
        """
        Placeholder: Collect hard negatives from points in CSV.

        Future implementation will:
        1. Read points.csv with location metadata
        2. For each point, fetch Street View images
        3. Extract hard negative regions (sidewalks, shoulders, medians)
        4. Validate no vehicles using pretrained detector
        5. Save images and metadata

        Args:
            points_csv: Path to points.csv manifest
            num_samples_per_point: Number of hard negatives per location
            sample_type: Type of hard negative to collect
                        (sidewalk, shoulder, median, bike_lane, parking_space)

        Returns:
            Number of hard negatives collected
        """
        logger.info(f"[PLACEHOLDER] Would collect hard negatives from {points_csv}")
        logger.info(f"  Sample type: {sample_type}")
        logger.info(f"  Samples per point: {num_samples_per_point}")

        self.log_collection_info(
            f"Placeholder: Would collect {sample_type} negatives "
            f"({num_samples_per_point} per point)"
        )

        return 0

    def collect_from_directory(
        self, source_dir: str, filters: Optional[Dict] = None
    ) -> int:
        """
        Placeholder: Collect hard negatives from existing image directory.

        Future implementation will:
        1. Scan source directory for images
        2. Apply optional filters (size, date, location, etc.)
        3. Run vehicle detector to verify emptiness
        4. Copy verified hard negatives to output directory
        5. Create metadata records

        Args:
            source_dir: Directory containing source images
            filters: Optional dict with filter criteria
                     {
                       'min_confidence': 0.1,  # Vehicle detection threshold
                       'min_width': 640,
                       'min_height': 480,
                     }

        Returns:
            Number of hard negatives collected
        """
        logger.info(f"[PLACEHOLDER] Would collect hard negatives from {source_dir}")
        if filters:
            logger.info(f"  Filters: {filters}")

        self.log_collection_info(
            f"Placeholder: Would collect from directory {source_dir}"
        )

        return 0

    def validate_hard_negative(
        self, image_path: str, vehicle_confidence_threshold: float = 0.15
    ) -> Tuple[bool, Optional[float]]:
        """
        Placeholder: Validate that image contains no vehicles.

        Future implementation will:
        1. Load pretrained YOLO detector
        2. Run inference on image
        3. Return True if max confidence < threshold
        4. Return False + max_confidence if vehicles detected

        Args:
            image_path: Path to image to validate
            vehicle_confidence_threshold: Confidence threshold for vehicles

        Returns:
            (is_valid, max_vehicle_confidence)
            - is_valid: True if no vehicles detected
            - max_vehicle_confidence: Highest confidence score found
        """
        # Placeholder: pretend validation passes
        return True, None

    def create_empty_label(self, image_id: str) -> None:
        """
        Create empty YOLO label file for hard negative.

        Hard negatives have no objects, so label files are empty.

        Args:
            image_id: Image identifier (without extension)
        """
        label_path = self.output_dir / "labels" / f"{image_id}.txt"
        label_path.parent.mkdir(parents=True, exist_ok=True)

        # Empty file indicates no objects in this image
        label_path.touch()
        logger.debug(f"Created empty label: {label_path}")

    def save_metadata(self, image_id: str, image_path: str, metadata: Dict) -> None:
        """
        Save metadata for hard negative image as JSONL.

        Args:
            image_id: Unique image identifier
            image_path: Path to image file
            metadata: Additional metadata dict
        """
        record = {
            "image_id": image_id,
            "image_path": str(image_path),
            "timestamp": datetime.now().isoformat(),
            "type": "hard_negative",
            "has_vehicles": False,
            **metadata,
        }

        with open(self.metadata_file, "a") as f:
            f.write(json.dumps(record) + "\n")

    def print_examples(self) -> None:
        """Print examples of hard negative types."""
        examples = {
            "sidewalk": "Empty sidewalk without pedestrians or vehicles",
            "shoulder": "Road shoulder or edge without vehicles",
            "median": "Grass or vegetation median between lanes",
            "bike_lane": "Bike lane without cyclists or vehicles",
            "parking_space": "Empty parking space or parking lot",
            "grass": "Grassy area or park without vehicles",
            "building_wall": "Building facade or wall (static scene)",
            "street_furniture": "Poles, signs, bollards (scene elements, no vehicles)",
        }

        logger.info("\nExamples of hard negatives to collect:")
        logger.info("=" * 60)
        for neg_type, description in examples.items():
            logger.info(f"  {neg_type:20} - {description}")
        logger.info("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Collect hard negatives for vehicle detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show examples of hard negatives
  python scripts/10_collect_hard_negatives.py --show-examples
  
  # Placeholder for future collection from points CSV
  python scripts/10_collect_hard_negatives.py \\
    --points-csv manifests/points.csv \\
    --sample-type sidewalk \\
    --num-samples 100
  
  # Placeholder for validation from existing directory
  python scripts/10_collect_hard_negatives.py \\
    --source-dir data/images_original \\
    --validate-only
        """,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/images_hard_negatives",
        help="Output directory for hard negatives",
    )

    parser.add_argument(
        "--points-csv",
        type=str,
        default="manifests/points.csv",
        help="Path to points CSV manifest",
    )

    parser.add_argument(
        "--source-dir", type=str, help="Source directory of images to validate"
    )

    parser.add_argument(
        "--sample-type",
        type=str,
        choices=[
            "sidewalk",
            "shoulder",
            "median",
            "bike_lane",
            "parking_space",
            "grass",
            "building_wall",
            "street_furniture",
        ],
        default="sidewalk",
        help="Type of hard negative to collect",
    )

    parser.add_argument(
        "--num-samples", type=int, default=100, help="Number of samples to collect"
    )

    parser.add_argument(
        "--vehicle-threshold",
        type=float,
        default=0.15,
        help="Confidence threshold for vehicle detection (lower = stricter)",
    )

    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate existing images, don't collect new ones",
    )

    parser.add_argument(
        "--show-examples",
        action="store_true",
        help="Show examples of hard negative types and exit",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )

    args = parser.parse_args()

    # Initialize collector
    collector = HardNegativeCollector(args.output_dir)

    # Show examples if requested
    if args.show_examples:
        collector.print_examples()
        return 0

    logger.info("\n" + "=" * 70)
    logger.info("ParkOpticon Hard Negatives Collection - Placeholder")
    logger.info("=" * 70)
    logger.info("\nThis script is a placeholder for hard negatives collection.")
    logger.info("Future implementation will integrate with Google Street View API.\n")

    # Log parameters
    collector.log_collection_info(f"Started collection with parameters: {vars(args)}")

    # Handle different collection modes
    if args.validate_only and args.source_dir:
        logger.info(f"Validation mode: {args.source_dir}")
        logger.info("[PLACEHOLDER] Would validate images from source directory")
        count = collector.collect_from_directory(args.source_dir)
    else:
        logger.info(f"Collection mode: {args.sample_type}")
        logger.info(f"Target samples: {args.num_samples}")
        count = collector.collect_from_points_csv(
            args.points_csv,
            num_samples_per_point=args.num_samples,
            sample_type=args.sample_type,
        )

    logger.info(f"\nCollection complete: {count} hard negatives collected")
    logger.info(f"Output directory: {collector.output_dir}")

    collector.log_collection_info(
        f"Collection completed: {count} hard negatives collected"
    )

    # Print next steps
    logger.info("\nNext steps:")
    logger.info("  1. Implement Street View API integration")
    logger.info("  2. Add region-of-interest (ROI) selection")
    logger.info("  3. Integrate with existing pipeline")
    logger.info("  4. Use hard negatives in training to reduce false positives")

    return 0


if __name__ == "__main__":
    sys.exit(main())
