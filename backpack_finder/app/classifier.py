import time
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image

try:
    import torch
    import torchvision
except Exception:
    torch = None
    torchvision = None


class CropClassifier:
    """Классифицирует кропы (bbox) предобученной моделью ImageNet + прикладной bag_type."""

    def __init__(self, model_name: str = "resnet50", device: Optional[str] = None):
        if torch is None or torchvision is None:
            raise RuntimeError("torch/torchvision not installed, cannot use classifier")

        self.model_name = model_name.lower().strip()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        if self.model_name == "resnet50":
            weights = torchvision.models.ResNet50_Weights.DEFAULT
            self.model = torchvision.models.resnet50(weights=weights)
        elif self.model_name == "efficientnet_b0":
            weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
            self.model = torchvision.models.efficientnet_b0(weights=weights)
        else:
            raise ValueError("model_name must be 'resnet50' or 'efficientnet_b0'")

        self.preprocess = weights.transforms()
        self.categories = weights.meta["categories"]
        self.model.eval().to(self.device)

    @staticmethod
    def _safe_crop(image: Image.Image, bbox: List[float], pad: int = 6) -> Image.Image:
        w, h = image.size
        x1, y1, x2, y2 = bbox
        x1 = int(max(0, np.floor(x1) - pad))
        y1 = int(max(0, np.floor(y1) - pad))
        x2 = int(min(w, np.ceil(x2) + pad))
        y2 = int(min(h, np.ceil(y2) + pad))
        if x2 <= x1 + 1 or y2 <= y1 + 1:
            return image.copy()
        return image.crop((x1, y1, x2, y2))

    def predict_top1(self, image_crop: Image.Image) -> Tuple[str, float]:
        t = self.preprocess(image_crop).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(t)
            probs = torch.softmax(logits, dim=1)[0]
            conf, idx = torch.max(probs, dim=0)
        label = self.categories[int(idx.item())]
        return label, float(conf.item())

    @staticmethod
    def _normalize_label(label: str) -> str:
        return (label or "").lower().replace("-", " ").strip()

    @classmethod
    def label_to_bag_type(cls, label: str) -> str:
        s = cls._normalize_label(label)

        if "backpack" in s or "rucksack" in s or "knapsack" in s:
            return "backpack"
        if "suitcase" in s or "luggage" in s or "trunk" in s:
            return "suitcase"
        if "handbag" in s or "purse" in s:
            return "handbag"
        if "duffel" in s or "duffle" in s or "tote" in s or ("bag" in s and "bagel" not in s):
            return "bag"
        return "other"

    def classify_detections(self, image: Image.Image, detections: List[Dict], pad: int = 6) -> Tuple[List[Dict], int]:
        t0 = time.perf_counter()
        for d in detections:
            crop = self._safe_crop(image, d["bbox"], pad=pad)
            label, conf = self.predict_top1(crop)
            d["crop_top1_label"] = label
            d["crop_top1_conf"] = conf
            d["bag_type"] = self.label_to_bag_type(label)
            d["bag_type_conf"] = conf
        ms = int((time.perf_counter() - t0) * 1000)
        return detections, ms
