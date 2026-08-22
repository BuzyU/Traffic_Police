import sys
from pathlib import Path
import os
import argparse

SITE_PKGS = Path(__file__).parent / "site_pkgs"
if SITE_PKGS.exists() and str(SITE_PKGS) not in sys.path:
    sys.path.insert(0, str(SITE_PKGS))

# Setting YOLO config dir to avoid AppData permissions
os.environ["YOLO_CONFIG_DIR"] = str(Path(__file__).parent / "runs" / "config")

from ultralytics import YOLO

def main(args):
    dataset_yaml = Path(args.data)
    if not dataset_yaml.exists():
        print(f"ERROR: Dataset not found at {dataset_yaml}")
        sys.exit(1)

    # 1. Load the model
    model = YOLO(args.model)
    
    # 2. Train
    print(f"Starting YOLOv8 training on {dataset_yaml}")
    model.train(
        data=str(dataset_yaml),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=0 if args.device == "cuda" else args.device,
        project="runs/detect",
        name="train_yolo",
        exist_ok=True,
    )
    
    print("\nTraining completed.")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=r"C:\Users\Umer Zingu\Desktop\Learning\Traffic_Police\Vehicles-yolo\data.yaml")
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    main(args)
