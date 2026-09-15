"""
Feature extraction for the ARGUS federated head.

The head does not classify pixels. YOLO already emits person / vehicle /
animal / package, and a model that reproduces that adds nothing. The job
left over is the one the detector cannot do: deciding whether a detection
is routine *for this site*. A van in the driveway at 14:00 is routine; the
same van at 03:00 is not. That judgement is site-specific, which is what
makes it worth federating.

So the input is a detection in context -- what was found, how big, where,
how it moved, how long it stayed, and what time it was.

Layout (19 dims) -- must stay in lockstep with sim/workload.py.

     0-3   class one-hot: person, vehicle, animal, package
     4-9   geometry: w, h, sqrt(area), aspect, cx, cy
    10-13  dynamics: vx, vy, speed, dwell
    14-17  temporal: hour sin/cos, weekday sin/cos
    18     detector confidence

Everything is normalised into roughly [-1, 1] at the source, so no scaler
has to be fitted, shipped, or kept in sync across nodes -- one less thing
to go wrong in a federated round.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np

N_FEATURES = 19
N_CLASSES = 5

CLASS_NAMES = ("person", "vehicle", "animal", "package")
LABEL_NAMES = (
    "routine_person",
    "routine_vehicle",
    "routine_animal",
    "routine_package",
    "anomaly",
)
ANOMALY = 4

# COCO ids that stand in for "package". COCO has no package class, and
# DetectionType has no PACKAGE member either, so these are matched before
# falling back to the DetectionType map in src/detection/__init__.py.
_PACKAGE_COCO_IDS = frozenset({24, 26, 28})  # backpack, handbag, suitcase

_DETECTION_TYPE_TO_CLASS = {
    "human": 0,
    "face": 0,
    "dangerous_person": 0,
    "vehicle": 1,
    "animal": 2,
}


def class_index(detection_type: str, coco_id: Optional[int] = None) -> Optional[int]:
    """Map a detection onto one of the four ARGUS object classes.

    Returns None for anything outside the taxonomy (MOTION, UNKNOWN), which
    the caller should skip rather than feed to the head as a zero vector.
    """
    if coco_id is not None and coco_id in _PACKAGE_COCO_IDS:
        return 3
    return _DETECTION_TYPE_TO_CLASS.get(str(detection_type).lower())


@dataclass
class Track:
    """Minimal per-track state the head needs.

    SORT already maintains track identity; this only accumulates what the
    feature vector reads. Kept separate so it can be populated from the
    tracker, from a replayed event log, or from a test.
    """

    track_id: int
    cls: int
    first_seen: float
    last_seen: float
    frames: int = 0
    cx: float = 0.5
    cy: float = 0.5
    prev_cx: Optional[float] = None
    prev_cy: Optional[float] = None
    w: float = 0.0
    h: float = 0.0
    confidence: float = 0.0

    def update(self, bbox_norm: Tuple[float, float, float, float],
               confidence: float, now: float) -> None:
        x1, y1, x2, y2 = bbox_norm
        self.prev_cx, self.prev_cy = self.cx, self.cy
        self.w = max(1e-3, x2 - x1)
        self.h = max(1e-3, y2 - y1)
        self.cx = (x1 + x2) / 2.0
        self.cy = (y1 + y2) / 2.0
        self.confidence = confidence
        self.last_seen = now
        self.frames += 1


def extract(track: Track, when: Optional[datetime] = None,
            dwell_scale_s: float = 30.0) -> np.ndarray:
    """Build the 19-dim feature vector for one track.

    dwell_scale_s bounds the dwell feature: 30 s of continuous presence
    saturates it. Loitering past that reads the same as loitering well
    past it, which is the intent -- the head should not learn to rank
    someone standing still for 5 minutes above someone standing still for
    2, it should learn that either is unusual here.
    """
    when = when or datetime.now()
    x = np.zeros(N_FEATURES, dtype=np.float32)

    if 0 <= track.cls < 4:
        x[track.cls] = 1.0

    w, h = track.w, track.h
    x[4] = w
    x[5] = h
    x[6] = math.sqrt(max(0.0, w * h))
    x[7] = min(w / max(h, 1e-3), 6.0) / 6.0
    x[8] = track.cx
    x[9] = track.cy

    if track.prev_cx is None:
        vx = vy = 0.0
    else:
        vx = track.cx - track.prev_cx
        vy = track.cy - track.prev_cy
    x[10] = math.tanh(vx * 10.0)
    x[11] = math.tanh(vy * 10.0)
    x[12] = math.tanh(math.hypot(vx, vy) * 10.0)
    x[13] = min(1.0, max(0.0, (track.last_seen - track.first_seen) / dwell_scale_s))

    ang_h = 2 * math.pi * (when.hour + when.minute / 60.0) / 24.0
    ang_d = 2 * math.pi * when.weekday() / 7.0
    x[14] = math.sin(ang_h)
    x[15] = math.cos(ang_h)
    x[16] = math.sin(ang_d)
    x[17] = math.cos(ang_d)

    x[18] = float(np.clip(track.confidence, 0.0, 1.0))
    return x


def describe(x: np.ndarray) -> Dict[str, float]:
    """Human-readable view of a feature vector, for the dashboard and logs."""
    cls = int(np.argmax(x[:4])) if x[:4].any() else -1
    return {
        "class": CLASS_NAMES[cls] if cls >= 0 else "unknown",
        "width": round(float(x[4]), 4),
        "height": round(float(x[5]), 4),
        "centre_x": round(float(x[8]), 4),
        "centre_y": round(float(x[9]), 4),
        "speed": round(float(x[12]), 4),
        "dwell": round(float(x[13]), 4),
        "confidence": round(float(x[18]), 4),
    }
