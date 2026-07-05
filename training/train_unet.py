"""
train_unet.py — Trains the from-scratch U-Net for lane segmentation on
BDD100K. Unlike YOLO (handled by ultralytics), we write this training
loop ourselves since U-Net is a custom architecture with no built-in
trainer. Logs loss/IoU/Dice to W&B each epoch.

Uses automatic mixed precision (AMP) to reduce VRAM usage and speed up
training — important on a 4GB GPU where fp32 alone maxes out compute.

Run: python training/train_unet.py
"""
import sys
import os
import yaml
import torch
import wandb
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.unet import UNet
from data.lane_dataset import LaneDataset
from data.augmentations import get_train_transforms, get_val_transforms
from training.losses import DiceBCELoss
from training.metrics import iou_score, dice_coefficient


def load_config(path: str = "config.yaml") -> dict:
    """Load hyperparameters from the central config file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def run_epoch(model, loader, criterion, optimizer, scaler, device, epoch_num, total_epochs, train: bool = True):
    """
    Run one full pass over the dataset (either training or validation).
    Uses AMP (autocast + GradScaler) for faster, lower-memory computation,
    and shows a live tqdm progress bar with running loss/IoU/Dice.

    Args:
        train: if True, updates model weights; if False, just evaluates
    """
    model.train() if train else model.eval()
    total_loss, total_iou, total_dice, n_batches = 0.0, 0.0, 0.0, 0

    phase = "Train" if train else "Val"
    pbar = tqdm(loader, desc=f"Epoch {epoch_num}/{total_epochs} [{phase}]", unit="batch")

    for images, masks in pbar:
        images, masks = images.to(device), masks.to(device)

        with torch.set_grad_enabled(train):
            with autocast():  # runs forward pass in fp16 where numerically safe
                logits = model(images)
                loss = criterion(logits, masks)

            if train:
                optimizer.zero_grad()
                scaler.scale(loss).backward()   # scales loss to avoid fp16 underflow
                scaler.step(optimizer)
                scaler.update()

        batch_iou = iou_score(logits, masks)
        batch_dice = dice_coefficient(logits, masks)

        total_loss += loss.item()
        total_iou += batch_iou
        total_dice += batch_dice
        n_batches += 1

        pbar.set_postfix({
            "loss": f"{total_loss/n_batches:.4f}",
            "iou": f"{total_iou/n_batches:.4f}",
            "dice": f"{total_dice/n_batches:.4f}",
        })

    return {
        "loss": total_loss / n_batches,
        "iou": total_iou / n_batches,
        "dice": total_dice / n_batches,
    }


def main():
    config = load_config()
    unet_cfg = config["unet"]
    wandb_cfg = config["wandb"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    wandb.init(
        project=wandb_cfg["project"],
        name="unet_lane_seg_v1",
        config=unet_cfg,
        job_type="train-unet",
    )

    # Datasets — subsampled to 8000/1500 to keep training time manageable
    train_ds = LaneDataset(
        raw_root="data/raw/dataset", split="train",
        transform=get_train_transforms(unet_cfg["imgsz"]),
        max_samples=8000
    )
    val_ds = LaneDataset(
        raw_root="data/raw/dataset", split="val",
        transform=get_val_transforms(unet_cfg["imgsz"]),
        max_samples=1500
    )

    # num_workers/persistent_workers/pin_memory keep the GPU fed with data
    # in parallel instead of it sitting idle waiting on CPU preprocessing
    train_loader = DataLoader(
        train_ds, batch_size=unet_cfg["batch"], shuffle=True,
        num_workers=6, persistent_workers=True, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=unet_cfg["batch"], shuffle=False,
        num_workers=6, persistent_workers=True, pin_memory=True
    )

    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    model = UNet(in_channels=3, out_channels=1, base_features=32).to(device)
    criterion = DiceBCELoss(bce_weight=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=unet_cfg["lr"])
    scaler = GradScaler()

    best_dice = 0.0
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(unet_cfg["epochs"]):
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, scaler, device,
                                   epoch + 1, unet_cfg["epochs"], train=True)
        val_metrics = run_epoch(model, val_loader, criterion, optimizer, scaler, device,
                                 epoch + 1, unet_cfg["epochs"], train=False)

        print(f"Epoch {epoch+1}/{unet_cfg['epochs']} Summary | "
              f"Train Loss: {train_metrics['loss']:.4f} | "
              f"Val IoU: {val_metrics['iou']:.4f} | Val Dice: {val_metrics['dice']:.4f}")

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_metrics["loss"],
            "train_iou": train_metrics["iou"],
            "train_dice": train_metrics["dice"],
            "val_loss": val_metrics["loss"],
            "val_iou": val_metrics["iou"],
            "val_dice": val_metrics["dice"],
        })

        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            torch.save(model.state_dict(), "checkpoints/unet_best.pt")
            print(f"  -> New best model saved (Dice: {best_dice:.4f})")

    print(f"Training complete. Best Val Dice: {best_dice:.4f}")
    wandb.finish()


if __name__ == "__main__":
    main()