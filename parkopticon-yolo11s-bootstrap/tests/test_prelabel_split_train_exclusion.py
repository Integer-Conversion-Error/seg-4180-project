from __future__ import annotations

import csv
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

os.environ.setdefault("MPLBACKEND", "Agg")


def load_module(relative_path: str, name: str):
    root = Path(__file__).resolve().parents[1]
    path = root / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


prelabel = load_module("scripts/05_prelabel_boxes.py", "prelabel_boxes_test")
splitter = load_module("scripts/06_split_dataset.py", "split_dataset_test")
trainer = load_module("scripts/07_train_yolo.py", "train_yolo_test")


class ExclusionFlowTests(unittest.TestCase):
    def _write_manifest(self, path: Path, rows: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def test_prelabel_skips_rejected_and_excluded_descendants(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifests" / "images.csv"
            image = root / "img.jpg"
            Image.new("RGB", (10, 10), color="white").save(image)
            self._write_manifest(
                manifest,
                [
                    {
                        "image_id": "allowed",
                        "file_path": str(image),
                        "status": "ok",
                        "review_status": "",
                        "parent_image_id": "",
                        "is_synthetic": "0",
                        "expected_inserted_class": "none",
                    },
                    {
                        "image_id": "rejected",
                        "file_path": str(image),
                        "status": "ok",
                        "review_status": "rejected",
                        "parent_image_id": "",
                        "is_synthetic": "0",
                        "expected_inserted_class": "none",
                    },
                    {
                        "image_id": "excluded_parent",
                        "file_path": str(image),
                        "status": "ok",
                        "review_status": "",
                        "parent_image_id": "",
                        "is_synthetic": "0",
                        "expected_inserted_class": "none",
                    },
                    {
                        "image_id": "excluded_child",
                        "file_path": str(image),
                        "status": "ok",
                        "review_status": "",
                        "parent_image_id": "excluded_parent",
                        "is_synthetic": "1",
                        "expected_inserted_class": "vehicle",
                    },
                ],
            )

            saved_manifest = {}

            def capture_manifest(manifest_rows):
                saved_manifest["rows"] = [dict(row) for row in manifest_rows]

            with (
                patch.object(
                    prelabel,
                    "load_manifest",
                    return_value=[
                        {
                            "image_id": "allowed",
                            "file_path": str(image),
                            "status": "ok",
                            "review_status": "",
                            "parent_image_id": "",
                            "is_synthetic": "0",
                            "expected_inserted_class": "none",
                        },
                        {
                            "image_id": "rejected",
                            "file_path": str(image),
                            "status": "ok",
                            "review_status": "rejected",
                            "parent_image_id": "",
                            "is_synthetic": "0",
                            "expected_inserted_class": "none",
                        },
                        {
                            "image_id": "excluded_parent",
                            "file_path": str(image),
                            "status": "ok",
                            "review_status": "",
                            "parent_image_id": "",
                            "is_synthetic": "0",
                            "expected_inserted_class": "none",
                        },
                        {
                            "image_id": "excluded_child",
                            "file_path": str(image),
                            "status": "ok",
                            "review_status": "",
                            "parent_image_id": "excluded_parent",
                            "is_synthetic": "1",
                            "expected_inserted_class": "vehicle",
                        },
                    ],
                ),
                patch.object(
                    prelabel,
                    "load_dataset_excluded_ids",
                    return_value={"excluded_parent", "excluded_child"},
                ),
                patch.object(
                    prelabel,
                    "detect_vehicles",
                    new=lambda model, image_path, conf=0.25: [
                        {
                            "x1": 1.0,
                            "y1": 1.0,
                            "x2": 5.0,
                            "y2": 5.0,
                            "conf": 0.9,
                            "class": 0,
                        }
                    ],
                ),
                patch.object(
                    prelabel,
                    "importlib",
                    new=type(
                        "StubImportLib",
                        (),
                        {
                            "import_module": staticmethod(
                                lambda name: type(
                                    "StubUltralytics",
                                    (),
                                    {"YOLO": staticmethod(lambda model_name: object())},
                                )()
                            )
                        },
                    )(),
                ),
                patch.object(
                    prelabel,
                    "save_manifest",
                    side_effect=lambda manifest, path: capture_manifest(manifest),
                ),
                patch.object(
                    prelabel,
                    "analyze_diff_mask",
                    return_value={"valid": True, "failure_reason": ""},
                ),
                patch.object(
                    prelabel.cv2,
                    "imread",
                    return_value=np.zeros((10, 10, 3), dtype=np.uint8),
                ),
                patch.object(prelabel, "tqdm", new=lambda x, **kwargs: x),
                patch(
                    "sys.argv",
                    [
                        "prog",
                        "--manifest",
                        str(manifest),
                        "--out-dir",
                        str(root / "labels"),
                        "--histogram-out",
                        str(root / "prelabel_hist.png"),
                    ],
                ),
            ):
                prelabel.main()

            rows_by_id = {row["image_id"]: row for row in saved_manifest["rows"]}
            self.assertEqual(rows_by_id["allowed"].get("num_boxes_autogen"), "1")
            self.assertEqual(rows_by_id["rejected"].get("num_boxes_autogen"), None)
            self.assertEqual(
                rows_by_id["excluded_child"].get("num_boxes_autogen"), None
            )

    def test_split_excludes_rejected_and_dataset_excluded_rows_and_blanks_manifest(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifests" / "images.csv"
            img = root / "img.jpg"
            img.write_bytes(b"dummy")
            rows = [
                {
                    "image_id": "allowed",
                    "file_path": str(img),
                    "status": "ok",
                    "review_status": "",
                    "parent_image_id": "",
                    "pano_id": "g1",
                    "expected_inserted_class": "vehicle",
                    "split": "",
                },
                {
                    "image_id": "rejected",
                    "file_path": str(img),
                    "status": "ok",
                    "review_status": "rejected",
                    "parent_image_id": "",
                    "pano_id": "g2",
                    "expected_inserted_class": "vehicle",
                    "split": "",
                },
                {
                    "image_id": "excluded_parent",
                    "file_path": str(img),
                    "status": "ok",
                    "review_status": "",
                    "parent_image_id": "",
                    "pano_id": "g3",
                    "expected_inserted_class": "vehicle",
                    "split": "",
                },
                {
                    "image_id": "excluded_child",
                    "file_path": str(img),
                    "status": "ok",
                    "review_status": "",
                    "parent_image_id": "excluded_parent",
                    "pano_id": "g3",
                    "expected_inserted_class": "vehicle",
                    "split": "",
                },
            ]
            self._write_manifest(manifest, rows)
            labels = root / "labels"
            labels_final = root / "labels_final"
            out_dir = root / "splits"
            for d in [labels, labels_final]:
                d.mkdir()
                (d / "allowed.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

            with (
                patch.object(
                    splitter,
                    "load_dataset_excluded_ids",
                    return_value={"excluded_parent", "excluded_child"},
                ),
                patch.object(splitter.random, "shuffle", lambda x: None),
                patch.object(
                    splitter.shutil,
                    "copy2",
                    lambda src, dst: Path(dst).write_bytes(Path(src).read_bytes()),
                ),
                patch.object(splitter.shutil, "rmtree", lambda p: None),
                patch.object(splitter, "tqdm", new=lambda x, **kwargs: x),
                patch(
                    "sys.argv",
                    [
                        "prog",
                        "--manifest",
                        str(manifest),
                        "--labels-dir",
                        str(labels),
                        "--labels-final-dir",
                        str(labels_final),
                        "--out-dir",
                        str(out_dir),
                        "--skip-threshold-check",
                    ],
                ),
            ):
                splitter.main()

            with manifest.open(newline="", encoding="utf-8") as f:
                out_manifest = list(csv.DictReader(f))
            by_id = {row["image_id"]: row for row in out_manifest}
            self.assertNotEqual(by_id["allowed"]["split"], "")
            self.assertEqual(by_id["rejected"]["split"], "")
            self.assertEqual(by_id["excluded_parent"]["split"], "")
            self.assertEqual(by_id["excluded_child"]["split"], "")
            self.assertTrue(
                (
                    out_dir / by_id["allowed"]["split"] / "images" / "allowed.jpg"
                ).exists()
            )
            self.assertFalse((out_dir / "train" / "images" / "rejected.jpg").exists())

    def test_train_blocks_rejected_and_excluded_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_yaml = root / "dataset.yaml"
            data_yaml.write_text(
                "path: .\ntrain: train/images\nval: val/images\ntest: test/images\n",
                encoding="utf-8",
            )
            manifest = root / "manifests" / "images.csv"
            self._write_manifest(
                manifest,
                [
                    {
                        "image_id": "rejected",
                        "status": "ok",
                        "review_status": "rejected",
                        "split": "train",
                        "file_path": "x",
                        "parent_image_id": "",
                    },
                    {
                        "image_id": "excluded",
                        "status": "ok",
                        "review_status": "",
                        "split": "train",
                        "file_path": "x",
                        "parent_image_id": "",
                    },
                ],
            )

            with (
                patch.object(
                    trainer,
                    "_find_dataset_excluded_assigned_to_splits",
                    return_value=["excluded"],
                ),
                patch.object(
                    trainer,
                    "_find_dataset_excluded_present_in_split_dirs",
                    return_value=["excluded"],
                ),
                patch.object(
                    trainer,
                    "_find_rejected_assigned_to_splits",
                    return_value=["rejected"],
                ),
                patch.object(
                    trainer, "_find_rejected_present_in_split_dirs", return_value=[]
                ),
                patch.object(
                    trainer, "load_dataset_excluded_ids", return_value={"excluded"}
                ),
                patch(
                    "sys.argv",
                    ["prog", "--data", str(data_yaml), "--manifest", str(manifest)],
                ),
            ):
                self.assertEqual(trainer.main(), 1)

            with (
                patch.object(
                    trainer,
                    "_find_dataset_excluded_assigned_to_splits",
                    return_value=["excluded"],
                ),
                patch.object(
                    trainer,
                    "_find_dataset_excluded_present_in_split_dirs",
                    return_value=[],
                ),
                patch.object(
                    trainer, "_find_rejected_assigned_to_splits", return_value=[]
                ),
                patch.object(
                    trainer, "_find_rejected_present_in_split_dirs", return_value=[]
                ),
                patch.object(
                    trainer, "load_dataset_excluded_ids", return_value={"excluded"}
                ),
                patch(
                    "sys.argv",
                    [
                        "prog",
                        "--data",
                        str(data_yaml),
                        "--manifest",
                        str(manifest),
                        "--include-rejected",
                    ],
                ),
            ):
                self.assertEqual(trainer.main(), 1)


if __name__ == "__main__":
    unittest.main()
