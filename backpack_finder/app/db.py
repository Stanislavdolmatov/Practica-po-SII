import json
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional
from .config import DB_PATH

def _connect():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con

def init_db():
    con = _connect()
    cur = con.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS requests (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT NOT NULL,
        input_type TEXT NOT NULL,
        filename TEXT,
        model_yolo TEXT,
        model_seg TEXT,
        num_detections INTEGER NOT NULL,
        processing_ms INTEGER NOT NULL,
        output_image TEXT,
        extra_json TEXT
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS detections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        request_id INTEGER NOT NULL,
        class_name TEXT NOT NULL,
        confidence REAL NOT NULL,
        x1 REAL NOT NULL, y1 REAL NOT NULL, x2 REAL NOT NULL, y2 REAL NOT NULL,
        has_mask INTEGER NOT NULL,
        FOREIGN KEY(request_id) REFERENCES requests(id)
    );
    """)

    # мягкие миграции (добавляем колонки, если их ещё нет)
    for ddl in [
        "ALTER TABLE detections ADD COLUMN crop_top1_label TEXT;",
        "ALTER TABLE detections ADD COLUMN crop_top1_conf REAL;",
        "ALTER TABLE detections ADD COLUMN bag_type TEXT;",
        "ALTER TABLE detections ADD COLUMN bag_type_conf REAL;",
    ]:
        try:
            cur.execute(ddl)
        except Exception:
            pass

    con.commit()
    con.close()

def insert_request(
    input_type: str,
    filename: Optional[str],
    model_yolo: str,
    model_seg: Optional[str],
    detections: List[Dict[str, Any]],
    processing_ms: int,
    output_image: Optional[str],
    extra: Optional[Dict[str, Any]] = None
) -> int:
    con = _connect()
    cur = con.cursor()
    ts = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    cur.execute("""
    INSERT INTO requests (ts, input_type, filename, model_yolo, model_seg, num_detections, processing_ms, output_image, extra_json)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        ts, input_type, filename, model_yolo, model_seg,
        len(detections), int(processing_ms), output_image,
        json.dumps(extra or {}, ensure_ascii=False)
    ))
    request_id = cur.lastrowid

    for d in detections:
        bbox = d["bbox"]
        has_mask = int(d.get("has_mask", d.get("mask") is not None))

        cur.execute("""
        INSERT INTO detections (
            request_id, class_name, confidence, x1, y1, x2, y2, has_mask,
            crop_top1_label, crop_top1_conf, bag_type, bag_type_conf
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            request_id,
            d["class_name"],
            float(d["confidence"]),
            float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]),
            has_mask,
            d.get("crop_top1_label", None),
            float(d.get("crop_top1_conf")) if d.get("crop_top1_conf") is not None else None,
            d.get("bag_type", None),
            float(d.get("bag_type_conf")) if d.get("bag_type_conf") is not None else None,
        ))

    con.commit()
    con.close()
    return int(request_id)

def get_requests(limit: int = 200) -> List[Dict[str, Any]]:
    con = _connect()
    cur = con.cursor()
    cur.execute("SELECT * FROM requests ORDER BY id DESC LIMIT ?", (limit,))
    rows = [dict(r) for r in cur.fetchall()]
    con.close()
    return rows

def get_detections_for_request(request_id: int) -> List[Dict[str, Any]]:
    con = _connect()
    cur = con.cursor()
    cur.execute("SELECT * FROM detections WHERE request_id=? ORDER BY id ASC", (request_id,))
    rows = [dict(r) for r in cur.fetchall()]
    con.close()
    return rows
