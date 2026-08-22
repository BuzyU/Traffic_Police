import yaml
from pathlib import Path
import random
import sys

def verify_dataset(yolo_root: Path):
    print(f"Verifying YOLO dataset at: {yolo_root}")
    if not yolo_root.exists():
        print("ERROR: Dataset directory not found!")
        sys.exit(1)

    yaml_path = yolo_root / "data.yaml"
    if not yaml_path.exists():
        print("ERROR: data.yaml not found!")
        sys.exit(1)

    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data.get("nc") != 4:
        print(f"ERROR: Expected 4 classes, got {data.get('nc')} in data.yaml")
        sys.exit(1)

    names = data.get("names", [])
    expected_names = ["bus", "car", "motorcycle", "truck"]
    if names != expected_names:
        print(f"ERROR: Expected classes {expected_names}, got {names}")
        sys.exit(1)

    errors = 0
    for split in ["train", "valid", "test"]:
        split_dir = yolo_root / split
        img_dir = split_dir / "images"
        lbl_dir = split_dir / "labels"

        if not split_dir.exists():
            print(f"ERROR: Split '{split}' missing!")
            errors += 1
            continue

        imgs = list(img_dir.glob("*.*"))
        lbls = list(lbl_dir.glob("*.txt"))
        print(f"[{split}] Images: {len(imgs)} | Labels: {len(lbls)}")

        if len(imgs) != len(lbls):
            print(f"  WARNING: Mismatch in {split} counts ({len(imgs)} images vs {len(lbls)} labels)")

        if lbls:
            sample = random.sample(lbls, min(20, len(lbls)))
            for lbl_file in sample:
                content = lbl_file.read_text().strip()
                if not content:
                    continue
                for line_idx, line in enumerate(content.split("\n")):
                    parts = line.strip().split()
                    if len(parts) != 5:
                        print(f"  ERROR in {lbl_file.name}:{line_idx+1}: Wrong number of parts (expected 5, got {len(parts)})")
                        errors += 1
                        continue
                    try:
                        cls_id = int(parts[0])
                        coords = [float(x) for x in parts[1:]]
                        if cls_id not in [0, 1, 2, 3]:
                            print(f"  ERROR in {lbl_file.name}:{line_idx+1}: Unknown class_id {cls_id}")
                            errors += 1
                        if not all(0.0 <= c <= 1.0 for c in coords):
                            print(f"  ERROR in {lbl_file.name}:{line_idx+1}: Coordinates out of range [0,1]")
                            errors += 1
                    except ValueError:
                        print(f"  ERROR in {lbl_file.name}:{line_idx+1}: Invalid number format")
                        errors += 1

    if errors > 0:
        print(f"\nVerification FAILED with {errors} errors.")
        sys.exit(1)
    else:
        print("\nVerification PASSED. Dataset is clean and ready for training.")

if __name__ == "__main__":
    yolo_root = Path(r"C:\Users\Umer Zingu\Desktop\Learning\Traffic_Police\Vehicles-yolo")
    verify_dataset(yolo_root)
