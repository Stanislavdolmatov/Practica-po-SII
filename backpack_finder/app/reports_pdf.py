from io import BytesIO
from collections import defaultdict
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from .db import get_requests

def build_pdf_report(limit: int = 200) -> bytes:
    reqs = get_requests(limit=limit)

    by_day = defaultdict(int)
    total_requests = len(reqs)
    total_detections = 0

    for r in reqs:
        ts = r["ts"]
        day = ts[:10]
        nd = int(r["num_detections"])
        by_day[day] += nd
        total_detections += nd

    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w, h = A4

    y = h - 60
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, "Backpack Finder — Report")
    y -= 22

    c.setFont("Helvetica", 10)
    c.drawString(50, y, f"Generated: {datetime.utcnow().isoformat(timespec='seconds')}Z")
    y -= 18

    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, f"Requests (last {min(limit, total_requests)}): {total_requests}")
    y -= 14
    c.setFont("Helvetica", 10)
    c.drawString(50, y, f"Total detections (sum): {total_detections}")
    y -= 18

    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Detections by day (sum of detections):")
    y -= 14

    c.setFont("Helvetica", 10)
    for day in sorted(by_day.keys(), reverse=True)[:30]:
        c.drawString(60, y, f"{day}: {by_day[day]}")
        y -= 12
        if y < 70:
            c.showPage()
            y = h - 60
            c.setFont("Helvetica", 10)

    c.showPage()
    c.save()
    return buf.getvalue()
