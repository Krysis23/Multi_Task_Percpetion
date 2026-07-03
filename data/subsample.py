"""
subsample.py — Randomly samples a manageable subset of the full YOLO-converted
dataset (Milestone 2 output) into a smaller dataset sized for training on
limited hardware (RTX 3050 4GB). Copies matching image+label pairs.

Run once. Uses a fixed random seed so the sample is reproducible.
"""
import random
import shutil
from pathlib import Path
from tqdm import tqdm

random.seed(42)  # reproducibility — same "random" sample every time we run this

SRC_ROOT = "data/yolo_dataset"
DST_ROOT = "data/yolo_dataset_small"

SAMPLE_SIZES = {
    "train": 8000,
    "val": 1500,
}


def subsample_split(split: str, n_samples: int):
    """
    Randomly pick n_samples image+label pairs from a split and copy them
    into the smaller dataset folder.

    Args:
        split: "train" or "val"
        n_samples: how many image+label pairs to keep
    """
    src_img_dir = Path(SRC_ROOT) / "images" / split
    src_lbl_dir = Path(SRC_ROOT) / "labels" / split

    dst_img_dir = Path(DST_ROOT) / "images" / split
    dst_lbl_dir = Path(DST_ROOT) / "labels" / split
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    all_images = list(src_img_dir.glob("*.jpg"))
    print(f"[{split}] Source has {len(all_images)} images, sampling {n_samples}")

    n_samples = min(n_samples, len(all_images))  # don't crash if fewer images exist than requested
    sampled = random.sample(all_images, n_samples)

    for img_path in tqdm(sampled, desc=f"Sampling {split}", unit="img"):
        label_path = src_lbl_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            continue  # safety check, shouldn't happen since we only converted images with boxes

        shutil.copy(img_path, dst_img_dir / img_path.name)
        shutil.copy(label_path, dst_lbl_dir / label_path.name)

    print(f"[{split}] Copied {n_samples} image+label pairs to {dst_img_dir}")


if __name__ == "__main__":
    subsample_split("train", SAMPLE_SIZES["train"])
    subsample_split("val", SAMPLE_SIZES["val"])