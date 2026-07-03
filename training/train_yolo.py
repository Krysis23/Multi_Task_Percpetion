import sys
import os
import yaml
import wandb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"..")))
from models.detector import VehicleDetector

def load_config(path: str="config.yaml")->dict:
    with open(path,"r")as f:
        return yaml.safe_load(f)
    


def main():
    config = load_config()
    yolo_cfg = config["yolo"]
    wandb_cfg = config["wandb"]

    wandb.init(
        project=wandb_cfg["project"],
        name=yolo_cfg["run_name"],
        config=yolo_cfg,
        job_type="train-yolo"
    )

    detector = VehicleDetector(model_path=yolo_cfg["model_variant"])

    print(f"Starting YOLOv8 fine-tuning: {yolo_cfg['epochs']} epochs, "
          f"batch={yolo_cfg['batch']}, imgsz={yolo_cfg['imgsz']}")
    
    results = detector.train(
        data_yaml=yolo_cfg["data_yaml"],
        epochs=yolo_cfg["epochs"],
        imgsz=yolo_cfg["imgsz"],
        batch=yolo_cfg["batch"],
        device=yolo_cfg["device"],
        patience=yolo_cfg["patience"],
        project=yolo_cfg["project"],
        name=yolo_cfg["run_name"],
    )

    print("Training complete. Best weights saved at:")
    print(f"{yolo_cfg['project']}/{yolo_cfg['run_name']}/weights/best.pt")

    wandb.finish()

if __name__ == "__main__":
    main()