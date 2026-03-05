#!/usr/bin/env python3
"""
Count bounding boxes by class per split.

Reads YOLO format labels from splits/{split}/labels/ and counts boxes by class.
Outputs results to console (table) and JSON file.

Classes:
  0 - vehicle
  1 - enforcement_vehicle
  2 - police_old
  3 - police_new
  4 - lookalike_negative
"""

import argparse
import csv
import json
import logging
from collections import defaultdict
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

CLASS_NAMES = {
    0: "vehicle",
    1: "enforcement_vehicle",
    2: "police_old",
    3: "police_new",
    4: "lookalike_negative",
}
