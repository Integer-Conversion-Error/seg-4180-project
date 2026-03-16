#!/usr/bin/env python3
"""
Live inference script for webcam or video file using a trained YOLO model.
Displays bounding boxes and class labels in real-time.
"""

import argparse
import sys
from pathlib import Path
from collections import deque

import cv2
import numpy as np
from ultralytics import YOLO


# Class names for the 4-class vehicle detector
CLASS_NAMES = {
    0: "vehicle",
    1: "enforcement_vehicle",
    2: "police_old",
    3: "police_new",
}

# Color scheme for each class (BGR format for OpenCV)
CLASS_COLORS = {
    0: (128, 128, 128),  # Gray - regular vehicle
    1: (0, 165, 255),  # Orange - enforcement vehicle
    2: (255, 100, 100),  # Light blue - police old
    3: (0, 100, 255),  # Red/orange - police new
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Live inference with trained YOLO model"
    )
    parser.add_argument(
        "--weights",
        "-w",
        default="runs/detect/runs/detect/parkopticon_final_v1/weights/best.pt",
        help="Path to trained YOLO weights (best.pt)",
    )
    parser.add_argument(
        "--source",
        "-s",
        default="0",
        help="Video source: '0' for webcam, '1' for second camera, or path to video file",
    )
    parser.add_argument(
        "--conf", type=float, default=0.25, help="Confidence threshold for detections"
    )
    parser.add_argument("--iou", type=float, default=0.45, help="IOU threshold for NMS")
    parser.add_argument(
        "--smooth",
        action="store_true",
        help="Enable temporal smoothing using built-in tracker",
    )
    parser.add_argument(
        "--max-det", type=int, default=50, help="Maximum detections per frame"
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Don't display video (useful for headless testing)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save output video to runs/live_inference_output/",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load the model
    print(f"Loading model from: {args.weights}")
    model = YOLO(args.weights)
    print(f"Model loaded successfully!")

    # Setup video source
    if args.source.isdigit():
        source = int(args.source)
        print(f"Opening webcam device {source}...")
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"ERROR: Could not open webcam device {source}")
            sys.exit(1)
    else:
        source_path = Path(args.source)
        if not source_path.exists():
            print(f"ERROR: Video file not found: {args.source}")
            sys.exit(1)
        print(f"Opening video file: {args.source}")
        cap = cv2.VideoCapture(str(source_path))

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

    print(f"Video: {width}x{height} @ {fps} FPS")
    print("\nControls:")
    print("  - Press 'q' or ESC to quit")
    print("  - Press 's' to save current frame as screenshot")
    print("  - Press 'p' to pause/resume")
    print("-" * 50)

    # Setup video writer if saving
    video_writer = None
    if args.save:
        output_dir = Path("runs/live_inference_output")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"inference_{Path(args.source).stem or 'webcam'}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        print(f"Saving output to: {output_path}")

    paused = False
    frame_count = 0

    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    if not isinstance(source, int):
                        print("\nVideo ended. Press 'r' to replay or 'q' to quit.")
                    else:
                        print("\nLost connection to webcam.")
                    break

                frame_count += 1

                # Run inference (with tracking if smoothing is enabled)
                if args.smooth:
                    results = model.track(
                        frame,
                        persist=True,
                        conf=args.conf,
                        iou=args.iou,
                        max_det=args.max_det,
                        verbose=False,
                        tracker="botsort.yaml",  # Standard YOLO tracker
                    )
                else:
                    results = model.predict(
                        frame,
                        conf=args.conf,
                        iou=args.iou,
                        max_det=args.max_det,
                        verbose=False,
                    )

                # Draw detections
                if results and len(results) > 0:
                    result = results[0]
                    if result.boxes is not None:
                        for box in result.boxes:
                            # Get box coordinates
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            conf = float(box.conf[0])
                            cls = int(box.cls[0])
                            track_id = int(box.id[0]) if box.id is not None else None

                            # Get class info
                            class_name = CLASS_NAMES.get(cls, f"class_{cls}")
                            color = CLASS_COLORS.get(cls, (0, 255, 0))

                            # Draw bounding box
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                            # Draw label with confidence and track ID
                            label = f"{class_name} {conf:.2f}"
                            if track_id is not None:
                                label = f"ID:{track_id} {label}"

                            label_size, _ = cv2.getTextSize(
                                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                            )
                            label_y = max(y1 - 10, label_size[1])
                            cv2.rectangle(
                                frame,
                                (x1, label_y - label_size[1] - 5),
                                (x1 + label_size[0], label_y),
                                color,
                                -1,
                            )
                            cv2.putText(
                                frame,
                                label,
                                (x1, label_y - 3),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (255, 255, 255),
                                2,
                            )

                # Add frame info
                cv2.putText(
                    frame,
                    f"Frame: {frame_count} | Press 'q' to quit",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

                # Save frame if recording
                if video_writer:
                    video_writer.write(frame)

            # Display frame
            if not args.no_display:
                cv2.imshow("YOLO Live Inference", frame)

            # Handle keyboard input
            key = cv2.waitKey(1 if not paused else 100) & 0xFF

            if key == ord("q") or key == 27:  # q or ESC
                print("\nQuitting...")
                break
            elif key == ord("s"):
                screenshot_path = Path("runs/live_inference_output")
                screenshot_path.mkdir(parents=True, exist_ok=True)
                screenshot_path /= f"frame_{frame_count}.jpg"
                cv2.imwrite(str(screenshot_path), frame)
                print(f"Saved screenshot: {screenshot_path}")
            elif key == ord("p"):
                paused = not paused
                print("Paused" if paused else "Resumed")
            elif key == ord("r") and not isinstance(source, int):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                print("Replaying video...")

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    finally:
        cap.release()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()
        print("Done.")


if __name__ == "__main__":
    main()
