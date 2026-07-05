"""
app.py — Gradio web app for the autonomous perception system. Lets users
upload an image or video and see it annotated with vehicle/pedestrian
detections and lane segmentation.

Run: python app.py
"""
import os

# Redirect Gradio's temp directory into our own project folder — avoids
# Windows permission/locking quirks with the default AppData temp path.
os.environ["GRADIO_TEMP_DIR"] = os.path.abspath("gradio_temp")
os.makedirs("gradio_temp", exist_ok=True)

import shutil
import cv2
import gradio as gr

from inference.pipeline import PerceptionPipeline

YOLO_WEIGHTS = "runs/detect/runs/detect/yolo_bdd100k_v1/weights/best.pt"
UNET_WEIGHTS = "checkpoints/unet_best.pt"

# Load the pipeline once at startup, not per-request — loading model
# weights is slow, and Gradio would otherwise reload them on every submit
print("Loading models...")
pipeline = PerceptionPipeline(
    yolo_weights=YOLO_WEIGHTS,
    unet_weights=UNET_WEIGHTS,
    unet_imgsz=512
)
print("Models loaded.")


def process_image(image, conf_threshold):
    """
    Gradio callback for image upload. Gradio passes images as RGB numpy
    arrays, but our pipeline expects BGR (OpenCV convention), so we
    convert both ways.
    """
    if image is None:
        return None

    bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    annotated_bgr = pipeline.process_frame(bgr_image, conf_threshold=conf_threshold)
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    return annotated_rgb


def process_video(video_path, conf_threshold, progress=gr.Progress()):
    """
    Gradio callback for video upload. Copies the uploaded file locally first
    (avoids Windows file-lock issues with Gradio's temp directory), processes
    it frame-by-frame, and returns both the original (safely copied) and
    annotated video paths for side-by-side preview.

    Note: video_input uses gr.File (not gr.Video) to avoid a Windows-specific
    race condition where Gradio's native video preview tries to stream the
    upload before the file is fully released, causing a PermissionError.
    """
    if video_path is None:
        return None, None

    os.makedirs("outputs", exist_ok=True)
    local_input_path = "outputs/_gradio_input_video.mp4"
    shutil.copy(video_path, local_input_path)

    cap = cv2.VideoCapture(local_input_path)
    if not cap.isOpened():
        raise gr.Error(f"Could not open video file: {local_input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_path = "outputs/gradio_annotated_output.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        annotated = pipeline.process_frame(frame, conf_threshold=conf_threshold)
        writer.write(annotated)

        frame_idx += 1
        progress(frame_idx / total_frames, desc=f"Processing frame {frame_idx}/{total_frames}")

    cap.release()
    writer.release()

    # Both files are fully written and closed by this point, so Gradio's
    # Video preview widgets can safely stream them without hitting the
    # upload-time race condition.
    return local_input_path, output_path


# Build the Gradio interface with tabs for image and video input
with gr.Blocks(title="Autonomous Perception System") as demo:
    gr.Markdown("# 🚗 Autonomous Driving Perception System")
    gr.Markdown(
        "Upload a dashcam image or video to detect vehicles/pedestrians/cyclists "
        "(YOLOv8, fine-tuned on BDD100K) and segment lane lines (U-Net, trained from scratch)."
    )

    conf_slider = gr.Slider(
        minimum=0.1, maximum=0.9, value=0.25, step=0.05,
        label="Detection Confidence Threshold",
        info="Lower = more detections (more false positives). Higher = fewer, more confident detections."
    )

    with gr.Tab("Image"):
        with gr.Row():
            image_input = gr.Image(label="Upload Image", type="numpy")
            image_output = gr.Image(label="Annotated Output")
        image_button = gr.Button("Run Detection + Segmentation", variant="primary")
        image_button.click(
            fn=process_image,
            inputs=[image_input, conf_slider],
            outputs=image_output
        )

    with gr.Tab("Video"):
        with gr.Row():
            video_input = gr.File(label="Upload Video", file_types=[".mp4", ".mov", ".avi"])
        with gr.Row():
            original_preview = gr.Video(label="Original (Preview)")
            video_output = gr.Video(label="Annotated Output")
        video_button = gr.Button("Run Detection + Segmentation", variant="primary")
        video_button.click(
            fn=process_video,
            inputs=[video_input, conf_slider],
            outputs=[original_preview, video_output]
        )

    gr.Markdown(
        "**Note:** Video processing runs frame-by-frame on a single model instance and "
        "is not real-time — expect roughly a few FPS depending on video length and hardware."
    )


if __name__ == "__main__":
    demo.launch()