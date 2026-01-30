import time
from typing import Any, Dict, List, Tuple
from PIL import Image
from ultralytics import YOLO
from .config import YOLO_MODEL_NAME

class YOLODetector:
    def __init__(self, model_name: str = YOLO_MODEL_NAME):
        self.model_name = model_name
        self.model = YOLO(model_name)

    def predict(self, image: Image.Image, conf: float = 0.25) -> Tuple[List[Dict[str, Any]], int]:
        t0 = time.perf_counter()
        res = self.model.predict(image, conf=conf, verbose=False)[0]

        detections: List[Dict[str, Any]] = []
        names = res.names

        if res.boxes is not None and len(res.boxes) > 0:
            for b in res.boxes:
                cls_id = int(b.cls[0].item())
                class_name = str(names.get(cls_id, cls_id))
                conf_score = float(b.conf[0].item())
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
                detections.append({
                    "class_id": cls_id,
                    "class_name": class_name,
                    "confidence": conf_score,
                    "bbox": [x1, y1, x2, y2],
                    "mask": None,
                    "has_mask": 0,
                })

        ms = int((time.perf_counter() - t0) * 1000)
        return detections, ms
