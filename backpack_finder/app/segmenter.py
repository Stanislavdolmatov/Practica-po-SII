import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from .config import ENABLE_SEGMENTATION

try:
    import torch
    import torchvision
    from torchvision.transforms.functional import to_tensor
except Exception:
    torch = None
    torchvision = None
    to_tensor = None


class Segmenter:
    """Instance segmentation via Mask R-CNN (torchvision)."""

    def __init__(self, device: Optional[str] = None, score_thresh: float = 0.5, iou_match_thresh: float = 0.3):
        self.enabled = bool(ENABLE_SEGMENTATION)
        self.score_thresh = float(score_thresh)
        self.iou_match_thresh = float(iou_match_thresh)

        if not self.enabled:
            self.model = None
            self.device = None
            return

        if torch is None or torchvision is None:
            raise RuntimeError("ENABLE_SEGMENTATION=True but torch/torchvision not installed.")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        weights = torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=weights)
        self.model.eval().to(self.device)

    @staticmethod
    def _iou(a: np.ndarray, b: np.ndarray) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter = inter_w * inter_h
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter
        return float(inter / union) if union > 0 else 0.0

    def _predict(self, image: Image.Image) -> Tuple[dict, int]:
        t0 = time.perf_counter()
        img_t = to_tensor(image).to(self.device)
        with torch.no_grad():
            out = self.model([img_t])[0]
        ms = int((time.perf_counter() - t0) * 1000)
        return out, ms

    def add_masks(self, image: Image.Image, detections: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
        if not self.enabled or self.model is None:
            return detections, 0

        out, ms = self._predict(image)

        scores = out.get("scores", None)
        boxes = out.get("boxes", None)
        masks = out.get("masks", None)

        if scores is None or boxes is None or masks is None:
            return detections, ms

        scores = scores.detach().cpu().numpy()
        boxes = boxes.detach().cpu().numpy()
        masks = masks.detach().cpu().numpy()  # [N,1,H,W]

        keep = np.where(scores >= self.score_thresh)[0]
        boxes = boxes[keep] if len(keep) else boxes[:0]
        masks = masks[keep] if len(keep) else masks[:0]

        if boxes.shape[0] == 0:
            return detections, ms

        for det in detections:
            det_box = np.array(det["bbox"], dtype=np.float32)

            best_iou = 0.0
            best_idx = -1
            for i in range(boxes.shape[0]):
                iou = self._iou(det_box, boxes[i])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i

            if best_idx >= 0 and best_iou >= self.iou_match_thresh:
                m = masks[best_idx, 0]
                det["mask"] = (m >= 0.5).astype(np.uint8)  # (H,W)
                det["has_mask"] = 1
                det["mask_iou"] = float(best_iou)
            else:
                det["mask"] = None
                det["has_mask"] = 0
                det["mask_iou"] = float(best_iou)

        return detections, ms
