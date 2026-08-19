"""
demo_count.py — Vehicle Counting Demo Script (CLI).

Runs classical-CV localization + classification on an image or video,
draws labeled colored boxes and running counts per class, and saves the output.

Usage:
    py -3.11 part1_foundations/demo_count.py --input path/to/image.jpg --output output.jpg
    py -3.11 part1_foundations/demo_count.py --input path/to/video.mp4 --output output.mp4
"""

import argparse
import sys
from pathlib import Path

SITE_PKGS = Path(__file__).parent.parent / "site_pkgs"
if SITE_PKGS.exists() and str(SITE_PKGS) not in sys.path:
    sys.path.insert(0, str(SITE_PKGS))

import cv2
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from part1_foundations.localize import VehiclePipeline
from part1_foundations.train_scratch import build_model

CKPT_DIR = Path(__file__).parent.parent / "checkpoints"


def load_best_classifier(device):
    """Load scratch model if available, else pretrained baseline, else uninitialized with warning."""
    scratch_path = CKPT_DIR / "scratch_best.pth"
    pretrained_path = CKPT_DIR / "pretrained_best.pth"

    model = build_model(simclr_backbone_path=None).to(device)

    if scratch_path.exists():
        print(f"Loading trained from-scratch checkpoint: {scratch_path}")
        ckpt = torch.load(scratch_path, map_location=device)
        model.load_state_dict(ckpt["model"])
    elif pretrained_path.exists():
        print(f"Loading pretrained baseline checkpoint: {pretrained_path}")
        ckpt = torch.load(pretrained_path, map_location=device)
        model.load_state_dict(ckpt["model"])
    else:
        print("[WARNING] No trained model checkpoint found in checkpoints/. Running with initialized backbone for testing.")

    model.eval()
    return model


def process_image(input_path: str, output_path: str, device):
    frame = cv2.imread(input_path)
    if frame is None:
        print(f"Error: Could not read image at {input_path}")
        return

    model = load_best_classifier(device)
    pipeline = VehiclePipeline(model=model, device=device, mode="still")

    annotated, detections, counts = pipeline.process_frame(frame)

    cv2.imwrite(output_path, annotated)
    print(f"\nSuccessfully processed image: {input_path}")
    print(f"Annotated output saved to: {output_path}")
    print("\nVehicle counts detected:")
    for cls_name, count in counts.items():
        print(f"  - {cls_name.capitalize():<12}: {count}")


def process_video(input_path: str, output_path: str, device):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video at {input_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    model = load_best_classifier(device)
    pipeline = VehiclePipeline(model=model, device=device, mode="video")

    frame_idx = 0
    print(f"Processing video ({total_frames} frames)...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        annotated, detections, counts = pipeline.process_frame(frame)
        out.write(annotated)
        frame_idx += 1

        if frame_idx % 30 == 0 or frame_idx == total_frames:
            print(f"Processed frame {frame_idx}/{total_frames} | Counts: {counts}")

    cap.release()
    out.release()
    print(f"\nProcessing complete! Output saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vehicle Counting Demo")
    parser.add_argument("--input", required=True, help="Path to input image or video")
    parser.add_argument("--output", required=True, help="Path to save annotated output")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ext = Path(args.input).suffix.lower()
    if ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
        process_image(args.input, args.output, device)
    else:
        process_video(args.input, args.output, device)
