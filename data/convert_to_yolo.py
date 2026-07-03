"""
convert_to_yolo.py — Converts Supervisely-format BDD100K annotations
(ann/*.json with objects[].points.exterior rectangles) into YOLO-format
.txt label files, and organizes images+labels into the folder structure
the `ultralytics` library expects for training.

Run once per split (train/val). Skips 'lane' and 'drivable area' classes
since those are for segmentation (Milestone 4), not detection.
"""
import json
import os
import shutil
from pathlib import Path
from tqdm import tqdm

# Only these classes get detection boxes. Order here = YOLO class index.
DETECTION_CLASSES = [
    "person", "rider", "car", "truck", "bus",
    "train", "motor", "bike", "traffic light", "traffic sign"
]
CLASS_TO_ID = {name: idx for idx, name in enumerate(DETECTION_CLASSES)}


def convert_split(raw_root: str, split: str, out_root: str):
    """
    Convert one split (train/val/test) from Supervisely JSON to YOLO txt labels.

    Args:
        raw_root: path to data/raw/dataset (contains train/val/test folders)
        split: "train", "val", or "test"
        out_root: path to write YOLO-formatted data (e.g. data/yolo_dataset)
    """
    img_dir = Path(raw_root) / split / "img"
    ann_dir = Path(raw_root) / split / "ann"

    out_img_dir = Path(out_root) / "images" / split
    out_lbl_dir = Path(out_root) / "labels" / split
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    ann_files = list(ann_dir.glob("*.json"))
    print(f"[{split}] Found {len(ann_files)} annotation files")

    converted, skipped_no_boxes = 0, 0

    # tqdm wraps the loop and prints a live progress bar with ETA,
    # so you can see it's actually working instead of staring at a blank terminal
    for ann_path in tqdm(ann_files, desc=f"Converting {split}", unit="img"):
        with open(ann_path, "r") as f:
            data = json.load(f)

        img_height = data["size"]["height"]
        img_width = data["size"]["width"]

        img_filename = ann_path.stem  # strips ".json" -> "0000f77c-6257be58.jpg"
        src_img_path = img_dir / img_filename

        if not src_img_path.exists():
            tqdm.write(f"  WARNING: image not found for {ann_path.name}, skipping")
            continue

        yolo_lines = []
        for obj in data.get("objects", []):
            class_title = obj.get("classTitle")
            geometry_type = obj.get("geometryType")

            if class_title not in CLASS_TO_ID or geometry_type != "rectangle":
                continue

            (x1, y1), (x2, y2) = obj["points"]["exterior"]
            x_center = ((x1 + x2) / 2) / img_width
            y_center = ((y1 + y2) / 2) / img_height
            box_width = abs(x2 - x1) / img_width
            box_height = abs(y2 - y1) / img_height

            class_id = CLASS_TO_ID[class_title]
            yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}")

        if not yolo_lines:
            skipped_no_boxes += 1
            continue

        shutil.copy(src_img_path, out_img_dir / img_filename)
        label_filename = img_filename.rsplit(".", 1)[0] + ".txt"
        with open(out_lbl_dir / label_filename, "w") as f:
            f.write("\n".join(yolo_lines))
        converted += 1

    print(f"[{split}] Converted: {converted}, skipped (no boxes): {skipped_no_boxes}")


if __name__ == "__main__":
    RAW_ROOT = "data/raw/dataset"
    OUT_ROOT = "data/yolo_dataset"

    convert_split(RAW_ROOT, "train", OUT_ROOT)
    convert_split(RAW_ROOT, "val", OUT_ROOT)