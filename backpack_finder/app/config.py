from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
OUTPUTS_DIR = DATA_DIR / "outputs"
DB_PATH = DATA_DIR / "history.sqlite3"

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# YOLO model (Ultralytics автоматически скачает веса при первом запуске)
YOLO_MODEL_NAME = "yolov8n.pt"

# Instance segmentation (Mask R-CNN). Требует torch/torchvision.
ENABLE_SEGMENTATION = True

# Classification (crop-level, ImageNet). Требует torch/torchvision.
ENABLE_CLASSIFICATION = True
CLASSIFIER_MODEL_NAME = "resnet50"  # или "efficientnet_b0"
