import os
import json

IMAGES_DIR = "data/raw/dataset/train/img"
DET_LABELS_PATH = "data/raw/dataset/train/ann"

def inspect_images():
    files = os.listdir(IMAGES_DIR)
    print(f"FOUND {len(files)} images in {IMAGES_DIR}")
    print(f"Sample filename: {files[0]}")

def inspect_labels():
    with open(DET_LABELS_PATH, "r") as f:
        labels = json.load(f)
    print(f"Total Labeled entries {len(labels)}")
    print("Sample entry Structure:")
    print(json.dumps(labels[0], indent=2)[:1000])

if __name__ == "__main__":
    inspect_images()
    inspect_labels()
