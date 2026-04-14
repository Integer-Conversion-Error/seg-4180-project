from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


def load_module(relative_path: str, name: str):
    root = Path(__file__).resolve().parents[1]
    path = root / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


synth = load_module("scripts/04_synthesize_vehicle_edits.py", "synth_vehicle_test")


class GeminiBatchHelperTests(unittest.TestCase):
    def test_build_gemini_batch_request_uses_file_data_and_image_modalities(
        self,
    ) -> None:
        req = synth.build_gemini_batch_request(
            request_id="abc",
            source_file_uri="files/source.jpg",
            source_mime_type="image/jpeg",
            prompt="insert vehicle",
            reference_file_uris=["files/ref1.jpg", "files/ref2.jpg"],
            reference_mime_types=["image/jpeg", "image/png"],
        )

        self.assertEqual(req["key"], "abc")
        self.assertEqual(
            req["request"]["generation_config"]["responseModalities"], ["TEXT", "IMAGE"]
        )
        self.assertEqual(
            req["request"]["contents"][0]["parts"][0]["file_data"]["mime_type"],
            "image/jpeg",
        )
        parts = req["request"]["contents"][0]["parts"]
        self.assertEqual(parts[0]["file_data"]["file_uri"], "files/source.jpg")
        self.assertEqual(parts[0]["file_data"]["mime_type"], "image/jpeg")
        self.assertEqual(parts[1]["file_data"]["file_uri"], "files/ref1.jpg")
        self.assertEqual(parts[1]["file_data"]["mime_type"], "image/jpeg")
        self.assertEqual(parts[2]["file_data"]["file_uri"], "files/ref2.jpg")
        self.assertEqual(parts[2]["file_data"]["mime_type"], "image/png")
        self.assertEqual(parts[3]["text"], "insert vehicle")

    def test_decode_gemini_batch_result_line_reads_inline_data(self) -> None:
        decoded = synth.decode_gemini_batch_result_line(
            '{"request_id":"abc","response":{"candidates":[{"content":{"parts":[{"inlineData":{"data":"aGVsbG8="}}]}}]}}'
        )

        self.assertEqual(decoded, b"hello")


if __name__ == "__main__":
    unittest.main()
