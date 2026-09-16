"""
The bridge from live detections to federated training data.

Nothing in the pipeline currently persists anything: `db.log_event` exists
and is never called, so `/api/events` returns an empty list and there is no
record for the head to learn from. This closes that gap.

Per frame it associates detections to tracks, and when a track closes it
emits exactly one sample -- one person walking through the frame produces
one training row, not thirty. That is the same track-scoping rule the
detection pipeline already uses for the expensive stages, applied to data
collection for the same reason: per-frame rows would be thirty copies of
one event, and would teach the head that whatever loiters longest matters
most.

Association is IoU + class, kept self-contained rather than reaching into
SORT, so this works identically against a live camera, a replayed video, or
a test fixture.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.federated.features import Track, class_index, extract, describe

logger = logging.getLogger(__name__)


def _iou(a: Tuple[float, float, float, float],
         b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


class EventCollector:
    """Turns a stream of per-frame detections into per-track samples."""

    def __init__(self, store, db=None, iou_threshold: float = 0.3,
                 max_idle_s: float = 2.0, min_frames: int = 3,
                 source: str = "camera", reach_factor: float = 2.5,
                 motion_min_area: float = 0.002, motion_min_s: float = 1.0,
                 motion_max_s: float = 60.0):
        self.store = store
        self.db = db
        self.iou_threshold = iou_threshold
        self.max_idle_s = max_idle_s
        self.min_frames = min_frames
        self.reach_factor = reach_factor
        self.source = source
        self._tracks: Dict[int, Track] = {}
        self._boxes: Dict[int, Tuple[float, float, float, float]] = {}
        self._next_id = 1
        self.emitted = 0
        self.dropped_short = 0
        # Motion with no classified object is still worth a record -- it is
        # how a person YOLO missed shows up at all. Sustained motion over a
        # minimum area becomes one "motion" event; flicker does not.
        self.motion_min_area = motion_min_area
        self.motion_min_s = motion_min_s
        # A scene that never goes still -- a busy street, this demo's
        # pedestrian video -- would hold one segment open forever and log
        # nothing. Long motion is cut into chunks of at most this length.
        self.motion_max_s = motion_max_s
        self._motion: Optional[dict] = None
        self.motion_events = 0

    # ------------------------------------------------------------------
    def observe(self, detections: Sequence, frame_shape: Tuple[int, int],
                when: Optional[float] = None) -> List[int]:
        """Feed one frame. Returns ids of tracks updated this frame."""
        now = when or time.time()
        h, w = frame_shape[0], frame_shape[1]
        if not h or not w:
            return []

        self._track_motion(detections, w, h, now)

        touched: List[int] = []
        used: set = set()

        for det in detections:
            cls = class_index(
                getattr(getattr(det, "type", None), "value", getattr(det, "type", "")),
                getattr(det, "coco_id", None),
            )
            if cls is None:
                continue  # MOTION / UNKNOWN are not head inputs

            # Detection.bbox is (x, y, width, height) in pixels.
            bx, by, bw, bh = det.bbox
            box = (bx / w, by / h, (bx + bw) / w, (by + bh) / h)

            best_id, best_iou = None, self.iou_threshold
            for tid, t in self._tracks.items():
                if tid in used or t.cls != cls:
                    continue
                score = _iou(box, self._boxes.get(tid, (0, 0, 0, 0)))
                if score > best_iou:
                    best_id, best_iou = tid, score

            # IoU alone loses fast movers. A car crossing the frame can
            # displace further than its own width between detections, giving
            # consecutive boxes zero overlap -- so every frame starts a new
            # one-frame track and all of them get dropped as flicker. On a
            # driveway that silently discards the main subject. Fall back to
            # centre distance, scaled by box size, so a same-class detection
            # roughly a box-width away still continues its track.
            if best_id is None:
                cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                reach = max(box[2] - box[0], box[3] - box[1]) * self.reach_factor
                best_d = reach
                for tid, t in self._tracks.items():
                    if tid in used or t.cls != cls:
                        continue
                    d = ((cx - t.cx) ** 2 + (cy - t.cy) ** 2) ** 0.5
                    if d < best_d:
                        best_id, best_d = tid, d

            if best_id is None:
                best_id = self._next_id
                self._next_id += 1
                self._tracks[best_id] = Track(
                    track_id=best_id, cls=cls, first_seen=now, last_seen=now,
                )

            used.add(best_id)
            self._tracks[best_id].update(box, float(det.confidence), now)
            self._boxes[best_id] = box
            touched.append(best_id)

        self._close_stale(now)
        return touched

    def flush(self) -> int:
        """Close every open track. Call at shutdown or end of a capture."""
        self._close_motion()
        n = 0
        for tid in list(self._tracks):
            if self._emit(tid):
                n += 1
        return n

    # ------------------------------------------------------------------
    def _track_motion(self, detections, w: int, h: int, now: float) -> None:
        area = 0.0
        for det in detections:
            kind = getattr(getattr(det, "type", None), "value", getattr(det, "type", ""))
            if str(kind).lower() == "motion":
                _, _, bw, bh = det.bbox
                area += (bw * bh) / float(w * h)
        area = min(area, 1.0)

        seg = self._motion
        if area >= self.motion_min_area:
            if seg is None:
                self._motion = {"start": now, "last": now, "frames": 1, "peak": area}
            else:
                seg["last"] = now
                seg["frames"] += 1
                seg["peak"] = max(seg["peak"], area)
                if now - seg["start"] >= self.motion_max_s:
                    self._close_motion()
        elif seg is not None and now - seg["last"] > self.max_idle_s:
            self._close_motion()

    def _close_motion(self) -> None:
        seg, self._motion = self._motion, None
        if seg is None:
            return
        duration = seg["last"] - seg["start"]
        if seg["frames"] < self.min_frames or duration < self.motion_min_s:
            return
        self.motion_events += 1
        if self.db is not None:
            try:
                self.db.log_event(
                    event_id=str(uuid.uuid4()),
                    event_type="motion",
                    source=self.source,
                    data=json.dumps({
                        "duration_s": round(duration, 2),
                        "frames": seg["frames"],
                        "peak_area": round(seg["peak"], 4),
                    }),
                )
            except Exception as exc:
                logger.warning("collector: motion log_event failed: %s", exc)

    def _close_stale(self, now: float) -> None:
        for tid, t in list(self._tracks.items()):
            if now - t.last_seen > self.max_idle_s:
                self._emit(tid)

    def _emit(self, tid: int) -> bool:
        t = self._tracks.pop(tid, None)
        self._boxes.pop(tid, None)
        if t is None:
            return False

        # A two-frame blip is a detector flicker, not an event. Recording it
        # would fill the store with noise the operator then has to label.
        if t.frames < self.min_frames:
            self.dropped_short += 1
            return False

        x = extract(t)
        meta = {
            "track_id": t.track_id,
            "frames": t.frames,
            "duration_s": round(t.last_seen - t.first_seen, 3),
            "source": self.source,
            "synthetic": False,
        }
        try:
            sid = self.store.append(x, label=None, meta=meta)
        except ValueError as exc:
            logger.warning("collector: rejected sample for track %s: %s", tid, exc)
            return False

        self.emitted += 1

        if self.db is not None:
            try:
                self.db.log_event(
                    event_id=str(uuid.uuid4()),
                    event_type="detection",
                    source=self.source,
                    data=json.dumps({
                        "sample_id": sid,
                        "features": describe(x),
                        **meta,
                    }),
                )
            except Exception as exc:
                # A logging failure must not take down the capture loop.
                logger.warning("collector: log_event failed: %s", exc)

        return True

    def status(self) -> dict:
        return {
            "open_tracks": len(self._tracks),
            "emitted": self.emitted,
            "dropped_short": self.dropped_short,
            "min_frames": self.min_frames,
            "motion_events": self.motion_events,
            "motion_open": self._motion is not None,
            "reach_factor": self.reach_factor,
            "max_idle_s": self.max_idle_s,
        }
