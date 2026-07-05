"""
lane_dataset.py — PyTorch Dataset for lane segmentation. Reads Supervisely
annotation JSONs, extracts 'lane' objects (geometryType='line', 2-point
segments), and rasterizes them into binary pixel masks (1=lane, 0=background).
Applies Albumentations transforms to image+mask together.
"""
import json
import random
from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import Dataset


class LaneDataset(Dataset):
    """Loads BDD100K images and generates lane segmentation masks on the fly."""

    def __init__(self, raw_root: str, split: str, transform=None,
                 line_thickness: int = 6, max_samples: int = None, seed: int = 42):
        """
        Args:
            raw_root: path to data/raw/dataset (contains train/val/test folders)
            split: "train" or "val"
            transform: Albumentations Compose object (applies to image+mask together)
            line_thickness: pixel width to draw each lane line segment
            max_samples: if set, randomly subsamples the dataset to this many
                images — keeps training time manageable on limited hardware.
                None = use the full split.
            seed: random seed for reproducible subsampling
        """
        self.img_dir = Path(raw_root) / split / "img"
        self.ann_dir = Path(raw_root) / split / "ann"
        ann_files = sorted(self.ann_dir.glob("*.json"))

        if max_samples is not None and max_samples < len(ann_files):
            rng = random.Random(seed)  # local Random instance, doesn't affect global random state
            ann_files = rng.sample(ann_files, max_samples)

        self.ann_files = ann_files
        self.transform = transform
        self.line_thickness = line_thickness

    def __len__(self):
        return len(self.ann_files)

    def _build_mask(self, data: dict, height: int, width: int) -> np.ndarray:
        """Rasterize all 'lane' line objects in an annotation into a binary mask."""
        mask = np.zeros((height, width), dtype=np.uint8)
        for obj in data.get("objects", []):
            if obj.get("classTitle") != "lane":
                continue
            exterior = obj["points"]["exterior"]
            if len(exterior) < 2:
                continue
            for i in range(len(exterior) - 1):
                pt1 = tuple(map(int, exterior[i]))
                pt2 = tuple(map(int, exterior[i + 1]))
                cv2.line(mask, pt1, pt2, color=1, thickness=self.line_thickness)
        return mask

    def __getitem__(self, idx):
        ann_path = self.ann_files[idx]
        with open(ann_path, "r") as f:
            data = json.load(f)

        img_filename = ann_path.stem
        img_path = self.img_dir / img_filename
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        height, width = data["size"]["height"], data["size"]["width"]
        mask = self._build_mask(data, height, width)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]

        mask = mask.float().unsqueeze(0) if mask.dim() == 2 else mask.float()

        return image, mask