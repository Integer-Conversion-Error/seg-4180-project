#!/usr/bin/env python3

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stage 03 orchestrator (03a YOLO pass -> 03b Gemini gate)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-dir", default=".")
    parser.add_argument("--manifest", "-m", default="manifests/images.csv")
    parser.add_argument("--out-manifest", "-o", default="manifests/images.csv")
    parser.add_argument("--boxes-out", "-b", default="manifests/boxes_autogen.jsonl")
    parser.add_argument("--empty-out", "-e", default="lists/empty_candidates.txt")
    parser.add_argument("--valid-road-out", default="lists/valid_road_candidates.txt")
    parser.add_argument("--non-road-out", default="lists/non_road_candidates.txt")
    parser.add_argument("--non-street-out", default="lists/non_street_candidates.txt")
    parser.add_argument("--yolo-empty-out", default="lists/yolo_empty_all.txt")
    parser.add_argument("--yolo-empty-in", default="lists/yolo_empty_all.txt")
    parser.add_argument("--stash-non-road-dir", default="data/images_excluded/non_road")
    parser.add_argument(
        "--stash-non-street-dir", default="data/images_excluded/non_street"
    )
    parser.add_argument("--model", default="yolo11s.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="0")
    parser.add_argument("--gemini-gate-model", default="gemini-3.1-pro-preview")
    parser.add_argument("--gemini-api-key", default=None)
    parser.add_argument("--gemini-retries", type=int, default=3)
    parser.add_argument("--gemini-workers", type=int, default=60)
    parser.add_argument("--min-street-confidence", type=float, default=0.75)
    parser.add_argument("--min-road-confidence", type=float, default=0.70)
    parser.add_argument("--skip-gemini-gate", action="store_true")
    parser.add_argument("--no-fresh-reset", action="store_true")
    parser.add_argument("--nuke", action="store_true")
    parser.add_argument("--soft-gate", action="store_true")
    parser.add_argument("--soft-gate-dir", default="data/gating_snapshots")
    args = parser.parse_args()

    scripts_dir = Path(__file__).resolve().parent
    stage03a = scripts_dir / "03a_yolo_pass.py"
    stage03b = scripts_dir / "03b_gemini_gate.py"

    stage03a_cmd = [
        sys.executable,
        str(stage03a),
        "--run-dir",
        args.run_dir,
        "--manifest",
        args.manifest,
        "--out-manifest",
        args.out_manifest,
        "--boxes-out",
        args.boxes_out,
        "--empty-out",
        args.empty_out,
        "--valid-road-out",
        args.valid_road_out,
        "--non-road-out",
        args.non_road_out,
        "--non-street-out",
        args.non_street_out,
        "--yolo-empty-out",
        args.yolo_empty_out,
        "--stash-non-road-dir",
        args.stash_non_road_dir,
        "--stash-non-street-dir",
        args.stash_non_street_dir,
        "--model",
        args.model,
        "--conf",
        f"{args.conf:.3f}",
        "--device",
        args.device,
    ]

    if args.no_fresh_reset:
        stage03a_cmd.append("--no-fresh-reset")
    if args.nuke:
        stage03a_cmd.append("--nuke")

    res_a = subprocess.run(stage03a_cmd)
    if res_a.returncode != 0:
        return res_a.returncode

    stage03b_cmd = [
        sys.executable,
        str(stage03b),
        "--run-dir",
        args.run_dir,
        "--manifest",
        args.manifest,
        "--out-manifest",
        args.out_manifest,
        "--empty-out",
        args.empty_out,
        "--valid-road-out",
        args.valid_road_out,
        "--non-road-out",
        args.non_road_out,
        "--non-street-out",
        args.non_street_out,
        "--yolo-empty-in",
        args.yolo_empty_in,
        "--stash-non-road-dir",
        args.stash_non_road_dir,
        "--stash-non-street-dir",
        args.stash_non_street_dir,
        "--gemini-gate-model",
        args.gemini_gate_model,
        "--gemini-retries",
        str(args.gemini_retries),
        "--min-street-confidence",
        f"{args.min_street_confidence:.3f}",
        "--min-road-confidence",
        f"{args.min_road_confidence:.3f}",
        "--gemini-workers",
        str(args.gemini_workers),
    ]

    if args.gemini_api_key:
        stage03b_cmd.extend(["--gemini-api-key", args.gemini_api_key])
    if args.skip_gemini_gate:
        stage03b_cmd.append("--skip-gemini-gate")
    if args.soft_gate:
        stage03b_cmd.extend(["--soft-gate", "--soft-gate-dir", args.soft_gate_dir])

    res_b = subprocess.run(stage03b_cmd)
    return res_b.returncode


if __name__ == "__main__":
    raise SystemExit(main())
