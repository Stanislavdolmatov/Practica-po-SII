import io
import uuid
from typing import Any, Dict, List
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import JSONResponse, FileResponse, Response

from .config import (
    UPLOADS_DIR, OUTPUTS_DIR,
    YOLO_MODEL_NAME, ENABLE_SEGMENTATION,
    ENABLE_CLASSIFICATION, CLASSIFIER_MODEL_NAME
)
from .db import init_db, insert_request, get_requests, get_detections_for_request
from .yolo_detector import YOLODetector
from .segmenter import Segmenter
from .classifier import CropClassifier
from .draw import draw_boxes, overlay_masks, draw_contours
from .reports_excel import build_excel_report
from .reports_pdf import build_pdf_report


app = FastAPI(title="Backpack Finder API", version="1.1")

detector = YOLODetector(YOLO_MODEL_NAME)
segmenter = Segmenter()
classifier = CropClassifier(CLASSIFIER_MODEL_NAME) if ENABLE_CLASSIFICATION else None


def _sanitize_for_json(detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Удаляем несериализуемые поля (np маски)."""
    out = []
    for d in detections:
        dd = dict(d)
        # маску в ответ не отдаём
        dd["mask"] = None
        # has_mask оставляем
        dd["has_mask"] = int(dd.get("has_mask", 0))
        out.append(dd)
    return out


@app.on_event("startup")
def _startup():
    init_db()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "yolo": YOLO_MODEL_NAME,
        "segmentation_enabled": bool(ENABLE_SEGMENTATION),
        "classification_enabled": bool(ENABLE_CLASSIFICATION),
        "classifier": CLASSIFIER_MODEL_NAME if ENABLE_CLASSIFICATION else None,
    }


@app.post("/infer/image")
async def infer_image(
    file: UploadFile = File(...),
    conf: float = Query(0.25, ge=0.01, le=0.99),
    only_backpack: bool = Query(False),
):
    raw = await file.read()
    img = Image.open(io.BytesIO(raw)).convert("RGB")

    uid = uuid.uuid4().hex
    (UPLOADS_DIR / f"{uid}_{file.filename}").write_bytes(raw)

    detections, det_ms = detector.predict(img, conf=conf)
    if only_backpack:
        detections = [d for d in detections if d["class_name"] == "backpack"]

    # segmentation (for visualization)
    detections, seg_ms = segmenter.add_masks(img, detections)

    # classification
    cls_ms = 0
    if ENABLE_CLASSIFICATION and classifier is not None and len(detections) > 0:
        detections, cls_ms = classifier.classify_detections(img, detections, pad=6)

    total_ms = det_ms + seg_ms + cls_ms

    # draw result (boxes only)
    out_img = draw_boxes(img, detections)
    out_name = f"{uid}_result.jpg"
    out_path = OUTPUTS_DIR / out_name
    out_img.save(out_path, quality=92)

    # store without raw masks
    dets_to_store = _sanitize_for_json(detections)

    req_id = insert_request(
        input_type="image",
        filename=file.filename,
        model_yolo=YOLO_MODEL_NAME,
        model_seg=("maskrcnn_resnet50_fpn" if ENABLE_SEGMENTATION else None),
        detections=dets_to_store,
        processing_ms=total_ms,
        output_image=out_name,
        extra={"conf": conf, "only_backpack": only_backpack, "det_ms": det_ms, "seg_ms": seg_ms, "cls_ms": cls_ms},
    )

    return {
        "request_id": req_id,
        "processing_ms": total_ms,
        "num_detections": len(dets_to_store),
        "output_image": f"/outputs/{out_name}",
        "detections": dets_to_store,
    }


@app.post("/infer/segment")
async def infer_segment(
    file: UploadFile = File(...),
    conf: float = Query(0.25, ge=0.01, le=0.99),
    mode: str = Query("mask"),  # "mask" or "contour"
    only_backpack: bool = Query(True),
):
    raw = await file.read()
    img = Image.open(io.BytesIO(raw)).convert("RGB")

    uid = uuid.uuid4().hex
    (UPLOADS_DIR / f"{uid}_{file.filename}").write_bytes(raw)

    detections, det_ms = detector.predict(img, conf=conf)
    if only_backpack:
        detections = [d for d in detections if d["class_name"] == "backpack"]

    detections, seg_ms = segmenter.add_masks(img, detections)

    cls_ms = 0
    if ENABLE_CLASSIFICATION and classifier is not None and len(detections) > 0:
        detections, cls_ms = classifier.classify_detections(img, detections, pad=6)

    total_ms = det_ms + seg_ms + cls_ms

    if mode == "contour":
        vis = draw_contours(img, detections)
    else:
        vis = overlay_masks(img, detections, alpha=0.35)

    vis = draw_boxes(vis, detections)

    out_name = f"{uid}_seg_{mode}.jpg"
    out_path = OUTPUTS_DIR / out_name
    vis.save(out_path, quality=92)

    dets_to_store = _sanitize_for_json(detections)

    req_id = insert_request(
        input_type="image",
        filename=file.filename,
        model_yolo=YOLO_MODEL_NAME,
        model_seg=("maskrcnn_resnet50_fpn" if ENABLE_SEGMENTATION else None),
        detections=dets_to_store,
        processing_ms=total_ms,
        output_image=out_name,
        extra={"conf": conf, "only_backpack": only_backpack, "mode": mode, "det_ms": det_ms, "seg_ms": seg_ms, "cls_ms": cls_ms},
    )

    return {
        "request_id": req_id,
        "processing_ms": total_ms,
        "num_detections": len(dets_to_store),
        "output_image": f"/outputs/{out_name}",
        "detections": dets_to_store,
    }


@app.get("/outputs/{filename}")
def get_output(filename: str):
    path = OUTPUTS_DIR / filename
    if not path.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(path)


@app.get("/history")
def history(limit: int = Query(50, ge=1, le=500)):
    rows = get_requests(limit=limit)
    return {"items": rows}


@app.get("/history/{request_id}")
def history_one(request_id: int):
    dets = get_detections_for_request(request_id)
    return {"request_id": request_id, "detections": dets}


@app.get("/report/excel")
def report_excel(limit: int = Query(200, ge=1, le=1000)):
    content = build_excel_report(limit=limit)
    return Response(
        content,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=backpack_report.xlsx"},
    )


@app.get("/report/pdf")
def report_pdf(limit: int = Query(200, ge=1, le=1000)):
    content = build_pdf_report(limit=limit)
    return Response(
        content,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=backpack_report.pdf"},
    )
