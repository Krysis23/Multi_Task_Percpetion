# 🚗 Autonomous Driving Perception System

A multi-task perception pipeline for dashcam footage, combining object detection,
lane segmentation, and a from-scratch Feature Pyramid Network — built end-to-end
from data pipeline to deployed web app.

## Overview

This project detects vehicles, pedestrians, and cyclists in dashcam footage and
segments lane lines, deployed as an interactive Gradio web app. It was built as a
learning project to go deep on both applying pretrained models (transfer learning)
and implementing core architectures from scratch.

**Core components:**
- **YOLOv8n** fine-tuned on BDD100K for 10-class object detection
- **U-Net** built from scratch (encoder/decoder + skip connections) for binary lane segmentation
- **Feature Pyramid Network** built from scratch and unit-tested independently for multi-scale feature fusion
- **Gradio** web interface supporting both image and video upload, with an adjustable confidence threshold

## Demo

![Demo](assets/demo.gif)
*(Record a short screen capture of the Gradio app processing a test video — e.g. with ScreenToGif on Windows — and save it to `assets/demo.gif`)*

## Architecture

```
                    Dashcam Frame
                         │
           ┌─────────────┴─────────────┐
           ▼                           ▼
   YOLOv8n (fine-tuned)         U-Net (from scratch)
   10-class detection           Binary lane segmentation
           │                           │
           └─────────────┬─────────────┘
                         ▼
              Combined annotated output
              (boxes + lane overlay)
```

*The Feature Pyramid Network (`models/fpn.py`) was built and verified independently
with dummy multi-scale tensors as an architecture deep-dive — it demonstrates the
same multi-scale feature fusion principle used inside YOLO's neck, without being
wired into the training loop itself.*

## Results

| Model | Metric | Score |
|---|---|---|
| YOLOv8n | mAP@50 | `0.299` |
| YOLOv8n | mAP@50-95 | `0.160` |
| YOLOv8n | Precision | `0.557` |
| YOLOv8n | Recall | `0.293` |
| U-Net | Best Val Dice | `0.0599` |
| U-Net | Best Val IoU | `0.429` |

**Hardware:** Trained on a single NVIDIA RTX 3050 (4GB VRAM, laptop). Model
variants, dataset size, and training config were deliberately scoped to fit this
budget — see [Design Decisions](#design-decisions--tradeoffs) below.

Full training curves and logs: `[link to your public W&B project]`

## Design Decisions & Tradeoffs

Being upfront about constraints and the reasoning behind them — these are the
choices that mattered most for making this project feasible on limited hardware:

- **YOLOv8n over larger variants (s/m/l):** the nano variant was the right fit for
  4GB VRAM at a workable batch size. Larger variants would likely improve mAP
  given more VRAM headroom.
- **8,000 / 1,500 image train/val subset instead of the full ~70K/10K BDD100K
  set:** the full dataset was estimated at multiple days of training time on this
  hardware. A reproducible random subsample (fixed seed) kept iteration cycles
  practical while preserving enough class diversity for meaningful metrics.
- **Mixed precision training (AMP):** U-Net training initially took ~2 hours per
  epoch. Diagnosing with `nvidia-smi` showed the GPU was compute-bound at 100%
  utilization (not data-starved), so adding `torch.cuda.amp` cut this to
  ~22 minutes per epoch — roughly a 5-6x speedup — by reducing precision where
  numerically safe.
- **U-Net `base_features=32`** instead of the original paper's 64, halving
  parameter count and memory footprint at an acceptable capacity tradeoff.
- **Dice + BCE combined loss** for U-Net, rather than BCE alone — lane pixels are
  a small minority of each image (~3%), and Dice loss handles this class
  imbalance far better than BCE by itself.

## Setup

```bash
git clone <your-repo-url>
cd autonomous-perception
python -m venv venv

# Windows
.\venv\Scripts\Activate.ps1
# Mac/Linux
source venv/bin/activate

pip install -r requirements.txt
```

**Dataset:** Download the BDD100K 10K-image subset (with detection + lane
annotations) and place it under `data/raw/dataset/`, matching the structure:
```
data/raw/dataset/
├── train/{img,ann}/
├── val/{img,ann}/
├── test/{img,ann}/
└── meta.json
```
See `data/explore.py` for a quick script to inspect and verify your download.

## Usage

**1. Convert labels and subsample the dataset:**
```bash
python data/convert_to_yolo.py
python data/subsample.py
```

**2. Train YOLO (detection):**
```bash
python training/train_yolo.py
```

**3. Train U-Net (lane segmentation):**
```bash
python training/train_unet.py
```

**4. Run inference on a single image or video:**
```bash
python inference/test_pipeline_image.py
python inference/test_pipeline.py
```

**5. Launch the web app:**
```bash
python app.py
```
Then open `http://127.0.0.1:7860` in your browser. Upload an image or video,
adjust the confidence threshold, and view the annotated output.

## Project Structure

```
autonomous-perception/
├── data/
│   ├── convert_to_yolo.py   # Supervisely JSON -> YOLO label format
│   ├── subsample.py         # Reproducible dataset subsampling
│   ├── lane_dataset.py      # PyTorch Dataset for U-Net (rasterizes lane masks)
│   └── augmentations.py     # Albumentations pipeline for U-Net
├── models/
│   ├── detector.py          # YOLOv8 fine-tuning wrapper
│   ├── unet.py               # U-Net, built from scratch
│   └── fpn.py                # Feature Pyramid Network, built from scratch
├── training/
│   ├── train_yolo.py
│   ├── train_unet.py
│   ├── losses.py             # Dice + BCE combined loss
│   └── metrics.py            # IoU, Dice coefficient
├── inference/
│   └── pipeline.py           # Combined YOLO + U-Net real-time pipeline
├── utils/
│   └── visualize.py          # Box and lane overlay drawing utilities
├── app.py                    # Gradio web app
├── config.yaml                # Central hyperparameter config
└── requirements.txt
```

## Experiment Tracking

All training runs (loss curves, mAP, IoU, Dice) were logged to Weights & Biases:
`https://wandb.ai/borkutepr-shri-ramdeobaba-college-of-engineering-and-man/autonomous-perception/runs/3z85rm5v?nw=nwuserborkutepr`

## What I'd Improve With More Compute

- Train on the full BDD100K dataset rather than the subsampled version
- Use YOLOv8s or YOLOv8m for improved detection accuracy
- Increase U-Net's `base_features` back to the standard 64
- Train for significantly more epochs with a learning rate schedule
- Add multi-class lane segmentation (the dataset includes lane style/type attributes not currently used)
