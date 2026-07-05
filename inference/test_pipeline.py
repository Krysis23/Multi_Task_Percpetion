"""
test_pipeline.py — Runs the full YOLO + U-Net perception pipeline on a
sample video and saves an annotated output. Use this to verify both
models work together correctly before wiring up the Gradio app.

Run: python inference/test_pipeline.py
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from inference.pipeline import PerceptionPipeline

YOLO_WEIGHTS = "runs/detect/runs/detect/yolo_bdd100k_v1/weights/best.pt"
UNET_WEIGHTS = "checkpoints/unet_best.pt"
INPUT_VIDEO = "data/test.mp4"   # <-- update this to your actual test video path
OUTPUT_VIDEO = "outputs/annotated_output.mp4"

if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)

    pipeline = PerceptionPipeline(
        yolo_weights=YOLO_WEIGHTS,
        unet_weights=UNET_WEIGHTS,
        unet_imgsz=512
    )

    pipeline.process_video(
        input_path=INPUT_VIDEO,
        output_path=OUTPUT_VIDEO,
        conf_threshold=0.25
    )