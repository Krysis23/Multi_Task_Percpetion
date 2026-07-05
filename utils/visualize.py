import cv2
import numpy as np

CLASS_COLORS = {
    0: (0, 255, 0),     
    1: (0, 165, 255),    
    2: (255, 0, 0),      
    3: (255, 0, 255),    
    4: (0, 255, 255),    
    5: (128, 0, 128),   
    6: (0, 128, 255),   
    7: (255, 255, 0),   
    8: (0, 0, 255),      
    9: (255, 128, 0), 

}

CLASS_NAMES = [
    "person", "rider", "car", "truck", "bus",
    "train", "motor", "bike", "traffic light", "traffic sign"
]

def draw_detections(frame: np.ndarray, boxes, class_ids, confidences, conf_threshold: float = 0.25) -> np.ndarray:
    output = frame.copy()

    for box,class_id, conf in zip(boxes, class_ids, confidences):
        if conf < conf_threshold:
            continue

        x1,y1,x2,y2 = map(int,box)
        color = CLASS_COLORS.get(int(class_id), (255,255,255))
        label = f"{CLASS_NAMES[int(class_id)]} {conf: .2f}"

        cv2.rectangle(output, (x1,y1), (x2,y2),color,thickness=2)

        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5,1)
        cv2.rectangle(output, (x1,y1-text_h -6), (x1+ text_w , y1), color, thickness=-1)
        cv2.putText(output, label,(x1,y1-4), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0), thickness=1)

    return output

def overlay_lane_mask(frame:np.ndarray, mask: np.ndarray, color=(0,255,255), alpha: float = 0.4)-> np.ndarray:
    output = frame.copy()

    color_layer = np.zeros_like(frame)
    color_layer[:,:] = color

    mask_3ch = np.stack([mask] * 3, axis =-1).astype(bool)
    output[mask_3ch] = cv2.addWeighted(frame, 1 - alpha, color_layer, alpha, 0)[mask_3ch]

    return output




