"""
test_pipeline_image.py — Runs the YOLO + U-Net perception pipeline on a
single test image and saves the annotated result. Faster and easier to
debug than full video processing — use this first to confirm both models
work together correctly before moving to video.

Run: python inference/test_pipeline_image.py
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
from inference.pipeline import PerceptionPipeline

YOLO_WEIGHTS = "runs/detect/runs/detect/yolo_bdd100k_v1/weights/best.pt"
UNET_WEIGHTS = "checkpoints/unet_best.pt"
TEST_IMAGE = "data/test.jpg" # <-- we'll pick one specific file below
OUTPUT_IMAGE = "outputs/annotated_test.png"

if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)

    # Grab the first available validation image to test on — you already
    # have plenty of these from the YOLO subsample in Milestone 2/3
    val_dir = "data/test.jpg"
    test_image_path = os.path.join(val_dir)
    print(f"Using test image: {test_image_path}")

    frame = cv2.imread(test_image_path)
    if frame is None:
        raise ValueError(f"Could not read image: {test_image_path}")

    pipeline = PerceptionPipeline(
        yolo_weights=YOLO_WEIGHTS,
        unet_weights=UNET_WEIGHTS,
        unet_imgsz=512
    )

    annotated = pipeline.process_frame(frame, conf_threshold=0.25)

    cv2.imwrite(OUTPUT_IMAGE, annotated)
    print(f"Saved annotated result to: {OUTPUT_IMAGE}")