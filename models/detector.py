from ultralytics import YOLO

class VehicleDetector:
    def __init__(self, model_path: str = "yolov8n.pt"):
        self.model = YOLO(model_path)

    def train(self, data_yaml: str, epochs:int, imgsz: int, batch: int, **kwargs):
        return self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            **kwargs
        )
    
    def predict(self,source, conf: float=0.25, **kwargs):
        return self.model.predict(source=source, conf=conf, **kwargs)
    

    def validate(self,data_yaml: str, **kwargs):
        return self.model.val(data=data_yaml, **kwargs)
