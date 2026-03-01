#!/usr/bin/env python3
"""
Generate a CSV template for Street View points.
This script creates a points.csv template that users fill in with their desired locations.
"""

import argparse
import csv
import os
from pathlib import Path


def generate_template(output_path: str, example_count: int = 1):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    examples = [
        {
            "street": "example_street_1",
            "label": "downtown_main_st",
            "location": "40.7128,-74.0060",
            "heading": "180",
            "pitch": "0",
            "fov": "80",
            "radius": "50"
        },
        {
            "street": "example_street_2", 
            "label": "parking_lot_a",
            "location": "40.7580,-73.9855",
            "heading": "90",
            "pitch": "-5",
            "fov": "90",
            "radius": "50"
        }
    ]
    
    fieldnames = ["street", "label", "location", "heading", "pitch", "fov", "radius"]
    
    if output_path.exists():
        print(f"Template already exists at {output_path}")
        response = input("Overwrite? (y/N): ").strip().lower()
        if response != 'y':
            print("Aborted.")
            return
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, example in enumerate(examples[:example_count]):
            writer.writerow(example)
    
    print(f"Template created at {output_path}")
    print(f"Columns: {', '.join(fieldnames)}")
    print(f"Rows: {example_count} (example data)")
    print("\nEdit this file to add your own Street View locations.")
    print("Format:")
    print("  - location: lat,lng OR address string")
    print("  - heading: 0-360 (0=North, 90=East, 180=South, 270=West)")
    print("  - pitch: -90 to 90 (negative looks up, positive looks down)")
    print("  - fov: 10-120 (field of view)")
    print("  - radius: 10-500 (search radius in meters)")


def main():
    parser = argparse.ArgumentParser(description="Generate Street View points template")
    parser.add_argument("--out", "-o", default="manifests/points.csv", help="Output CSV path")
    parser.add_argument("--examples", "-n", type=int, default=2, help="Number of example rows")
    args = parser.parse_args()
    
    generate_template(args.out, args.examples)


if __name__ == "__main__":
    main()
