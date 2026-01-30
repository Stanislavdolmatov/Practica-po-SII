from PIL import Image, ImageDraw, ImageFont
import numpy as np

def draw_boxes(image: Image.Image, detections: list) -> Image.Image:
    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        label = f'{d["class_name"]} {d["confidence"]:.2f}'
        if d.get("bag_type"):
            label += f' | {d["bag_type"]}'
        draw.rectangle([x1, y1, x2, y2], width=3)
        draw.text((x1 + 3, max(0, y1 - 12)), label, font=font)

    return img

def overlay_masks(image: Image.Image, detections: list, alpha: float = 0.35) -> Image.Image:
    img = image.convert("RGBA").copy()
    w, h = img.size
    base = np.array(img).astype(np.uint8)
    overlay = base.copy()

    for d in detections:
        m = d.get("mask", None)
        if m is None:
            continue
        if m.shape[0] != h or m.shape[1] != w:
            continue
        mask = m.astype(bool)
        overlay[mask, 0] = 200
        overlay[mask, 1] = 200
        overlay[mask, 2] = 200
        overlay[mask, 3] = 255

    out = base.copy()
    out[..., :3] = (base[..., :3] * (1 - alpha) + overlay[..., :3] * alpha).astype(np.uint8)
    out[..., 3] = 255
    return Image.fromarray(out, mode="RGBA").convert("RGB")

def draw_contours(image: Image.Image, detections: list) -> Image.Image:
    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size

    for d in detections:
        m = d.get("mask", None)
        if m is None:
            continue
        if m.shape[0] != h or m.shape[1] != w:
            continue

        m = m.astype(np.uint8)
        up = np.roll(m, 1, axis=0)
        down = np.roll(m, -1, axis=0)
        left = np.roll(m, 1, axis=1)
        right = np.roll(m, -1, axis=1)
        eroded = m & up & down & left & right
        border = (m ^ eroded).astype(bool)

        ys, xs = np.where(border)
        for x, y in zip(xs.tolist(), ys.tolist()):
            draw.point((x, y))

    return img
