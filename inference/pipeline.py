import sys
import os

import cv2
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.detector import VehicleDetector
from models.unet import UNet
from utils.visualize import draw_detections, overlay_lane_mask

class PerceptionPipeline:
    def __init__(self, yolo_weights: str, unet_weights:str, device: str = None, unet_imgsz: int = 512):
        self.device = device or("cuda" if torch.cuda.is_available() else "cpu")
        self.unet_imgsz = unet_imgsz

        self.detector = VehicleDetector(model_path=yolo_weights)

        self.unet = UNet(in_channels=3,out_channels=1,base_features=32)
        self.unet.load_state_dict(torch.load(unet_weights, map_location=self.device))
        self.unet.to(self.device)
        self.unet.eval()

        self.unet_transform = A.Compose([
            A.Resize(unet_imgsz, unet_imgsz),
            A.Normalize(mean=(0.485, 0.456,0.406),std=(0.229,0.224,0.225)),
            ToTensorV2(),
        ])

    def _run_unet(self,frame: np.ndarray) -> np.ndarray:
        orig_h, orig_w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

        transformed = self.unet_transform(image=rgb_frame)
        input_tensor = transformed["image"].unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.unet(input_tensor)
            probs = torch.sigmoid(logits)
            mask = (probs > 0.5).float().squeeze().cpu().numpy()

        mask_resized = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        return mask_resized.astype(np.uint8)
    
    def process_frame(self,frame: np.ndarray, conf_threshold: float = 0.25)->np.ndarray:
        results = self.detector.predict(frame,conf=conf_threshold, verbose=False)
        result = results[0]

        boxes = result.boxes.xyxy.cpu().numpy() if len(result.boxes) > 0 else []
        class_ids = result.boxes.cls.cpu().numpy() if len(result.boxes) > 0 else []
        confidences = result.boxes.conf.cpu().numpy() if len(result.boxes) > 0 else []

        lane_mask = self._run_unet(frame)

        annotated = overlay_lane_mask(frame, lane_mask)
        annotated = draw_detections(annotated, boxes, class_ids,confidences, conf_threshold)
        return annotated
    
    def process_video(self,input_path:str,output_path:str, conf_threshold: float = 0.25):
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {input_path}")
        

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path,fourcc, fps, (width,height))

        print(f"Processing {total_frames} frames at {fps:.1f} FPS...")
        

        from tqdm import tqdm
        frame_idx = 0
        with tqdm(total=total_frames,unit="frame") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                annotated = self.process_frame(frame,conf_threshold)
                writer.write(annotated)

                frame_idx += 1
                pbar.update(1)

        cap.release()
        writer.release()
        print(f"Done. Output saved to: {output_path}")




