from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from utils.dataset_exclusion import (
    clear_dataset_exclusion_cache,
    load_dataset_excluded_ids,
    resolve_exclusion_path,
)


class DatasetExclusionTests(unittest.TestCase):
    def setUp(self) -> None:
        clear_dataset_exclusion_cache()

    def tearDown(self) -> None:
        clear_dataset_exclusion_cache()

    def test_resolve_exclusion_path_prefers_nearest_lists_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "runs" / "demo" / "manifests" / "images.csv"
            manifest.parent.mkdir(parents=True)
            manifest.write_text("image_id\n", encoding="utf-8")

            expected = root / "runs" / "demo" / "lists" / "excluded_from_synth.txt"
            expected.parent.mkdir(parents=True)
            expected.write_text("rooted\n", encoding="utf-8")

            self.assertEqual(resolve_exclusion_path(manifest), expected)

    def test_load_dataset_excluded_ids_closes_over_parent_image_descendants(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "runs" / "demo" / "manifests" / "images.csv"
            manifest.parent.mkdir(parents=True)
            exclusion = root / "runs" / "demo" / "lists" / "excluded_from_synth.txt"
            exclusion.parent.mkdir(parents=True)
            exclusion.write_text("parent\n", encoding="utf-8")

            with manifest.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["image_id", "parent_image_id"])
                writer.writeheader()
                writer.writerows(
                    [
                        {"image_id": "parent", "parent_image_id": ""},
                        {"image_id": "child", "parent_image_id": "parent"},
                        {"image_id": "grandchild", "parent_image_id": "child"},
                        {"image_id": "unrelated", "parent_image_id": ""},
                    ]
                )

            excluded = load_dataset_excluded_ids(manifest)
            self.assertEqual(excluded, {"parent", "child", "grandchild"})


if __name__ == "__main__":
    unittest.main()
