"""
Boxes on event photos.

An event's still is saved clean, because it is training data. The box is
drawn when the photo is served (/api/events/{id}/snapshot), so the
dashboard, the wall and the built-in page all show what was detected and
where, and ?overlay=0 still returns the untouched file.

New events carry their box in the sample's meta. Events recorded before
that have a crop of the subject instead, cut from the full-resolution
frame with 10% padding, so the box is recovered by finding the crop in
the (downscaled) still. The result is cached next to the still as
<stem>.box.json, so each old photo is searched once and the sample store
is never rewritten.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

Box = Tuple[float, float, float, float]  # normalised x1, y1, x2, y2

CROP_PAD = 0.10            # must match EventCollector._encode_crop
MATCH_THRESHOLD = 0.85     # normalised cross-correlation needed to trust a match
# Frame widths the crop may have been cut from. The still is downscaled to
# the collector's snapshot width (640), the crop is not.
FRAME_WIDTHS = (1920, 1280, 1536, 2304, 640)

ROUTINE = (80, 220, 100)   # BGR green
ANOMALY = (60, 60, 240)    # BGR red


def _unpad(px1: float, py1: float, px2: float, py2: float) -> Box:
    """Undo the crop's 10% padding (exact away from the frame edges)."""
    bw = (px2 - px1) / (1 + 2 * CROP_PAD)
    bh = (py2 - py1) / (1 + 2 * CROP_PAD)
    x1, y1 = px1 + CROP_PAD * bw, py1 + CROP_PAD * bh
    return (x1, y1, x1 + bw, y1 + bh)


def locate_crop(still: np.ndarray, crop: np.ndarray) -> Optional[Box]:
    """Where the crop sits in the still, as a normalised box, or None."""
    sh, sw = still.shape[:2]
    best_score, best_box = -1.0, None
    for frame_w in FRAME_WIDTHS:
        scale = sw / float(frame_w) if frame_w > sw else 1.0
        cw, ch = int(round(crop.shape[1] * scale)), int(round(crop.shape[0] * scale))
        if cw < 8 or ch < 8 or cw > sw or ch > sh:
            continue
        templ = cv2.resize(crop, (cw, ch), interpolation=cv2.INTER_AREA) if scale != 1.0 else crop
        result = cv2.matchTemplate(still, templ, cv2.TM_CCOEFF_NORMED)
        _, score, _, (x, y) = cv2.minMaxLoc(result)
        if score > best_score:
            best_score = score
            best_box = (x / sw, y / sh, (x + cw) / sw, (y + ch) / sh)
    if best_box is None or best_score < MATCH_THRESHOLD:
        return None
    return _unpad(*best_box)


def event_box(meta: dict, snapshot_dir: Path) -> Optional[Box]:
    """The subject's box for an event: stored, cached, or recovered once."""
    stored = meta.get("box")
    if stored and len(stored) == 4:
        return tuple(float(v) for v in stored)

    name, crop_name = meta.get("snapshot"), meta.get("crop")
    if not name or not crop_name:
        return None
    still_path = snapshot_dir / name
    cache = still_path.with_suffix(".box.json")
    if cache.exists():
        try:
            cached = json.loads(cache.read_text()).get("box")
            return tuple(cached) if cached else None
        except (OSError, ValueError):
            pass

    still = cv2.imread(str(still_path))
    crop = cv2.imread(str(snapshot_dir / crop_name))
    box = locate_crop(still, crop) if still is not None and crop is not None else None
    try:
        cache.write_text(json.dumps({"box": [round(v, 4) for v in box] if box else None}))
    except OSError as exc:
        logger.debug("overlay: could not cache %s: %s", cache, exc)
    return box


def draw_overlay(jpeg: bytes, box: Sequence[float], label: str, anomaly: bool) -> bytes:
    """The still with the subject boxed and labelled, as JPEG bytes."""
    img = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return jpeg
    h, w = img.shape[:2]
    color = ANOMALY if anomaly else ROUTINE
    x1, y1 = int(max(0.0, box[0]) * w), int(max(0.0, box[1]) * h)
    x2, y2 = int(min(1.0, box[2]) * w), int(min(1.0, box[3]) * h)
    thick = max(2, w // 320)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)

    scale = max(0.45, w / 1280)
    (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    ty = y1 - 6 if y1 - th - 10 > 0 else y1 + th + 8
    cv2.rectangle(img, (x1, ty - th - 4), (min(w - 1, x1 + tw + 8), ty + base), color, -1)
    cv2.putText(img, label, (x1 + 4, ty), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)

    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    return buf.tobytes() if ok else jpeg
