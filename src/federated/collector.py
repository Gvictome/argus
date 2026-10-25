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
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.federated.features import CLASS_NAMES, Track, class_index, extract, describe

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
                 motion_max_s: float = 60.0, snapshot_dir=None,
                 snapshot_width: int = 640, snapshot_quality: int = 70,
                 stationary_classes: Sequence[str] = ("vehicle", "package"),
                 stationary_iou: float = 0.8, stationary_keep_s: float = 3600.0,
                 stationary_forget_s: float = 1800.0):
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
        # One still per event, kept beside the row it belongs to. A
        # confidence number is not reviewable; a picture is. Stored at the
        # track's largest box rather than its last frame, because by the
        # time a track closes the subject has usually left.
        self.snapshot_dir = Path(snapshot_dir) if snapshot_dir else None
        self.snapshot_width = snapshot_width
        self.snapshot_quality = snapshot_quality
        self._snaps: Dict[int, Tuple[float, bytes]] = {}
        self._crops: Dict[int, Optional[bytes]] = {}
        self.snapshots_saved = 0
        self.crops_saved = 0
        # Parked objects. A parked car is re-detected every time something
        # else in the scene (palm fronds, light) opens the motion gate, and
        # each time it was a new event with a new photo: 1,754 of one night's
        # 2,140 events on Node A were the same few cars. A track that did not
        # move and sits where a known object already sits is suppressed;
        # one event per object per stationary_keep_s is kept, marked
        # stationary, so the object is still on record.
        self.stationary_classes = {CLASS_NAMES.index(c) for c in stationary_classes
                                   if c in CLASS_NAMES}
        self.stationary_iou = stationary_iou
        self.stationary_keep_s = stationary_keep_s
        self.stationary_forget_s = stationary_forget_s
        self._first_box: Dict[int, Tuple[float, float, float, float]] = {}
        self._parked: List[dict] = []
        self.suppressed_stationary = 0

    # ------------------------------------------------------------------
    def observe(self, detections: Sequence, frame_shape: Tuple[int, int],
                when: Optional[float] = None, frame=None) -> List[int]:
        """Feed one frame. Returns ids of tracks updated this frame.

        `frame` is optional so callers that only have detections still
        work; without it, events are stored without a snapshot.
        """
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
            self._first_box.setdefault(best_id, box)
            self._tracks[best_id].update(box, float(det.confidence), now)
            self._boxes[best_id] = box
            touched.append(best_id)
            if frame is not None and self.snapshot_dir is not None:
                self._keep_snapshot(best_id, frame, box)

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

    def _keep_snapshot(self, tid: int, frame, box) -> None:
        """Hold the best still seen for this track, encoded, not raw.

        Encoding here costs a few ms and only when the subject gets
        meaningfully bigger; holding raw frames for every open track would
        cost megabytes each instead.
        """
        area = max(0.0, (box[2] - box[0])) * max(0.0, (box[3] - box[1]))
        best = self._snaps.get(tid)
        if best is not None and area < best[0] * 1.15:
            return
        try:
            import cv2

            img = frame
            h, w = img.shape[:2]
            # The crop is the subject on its own, which is what an image
            # model would train on later. Captured now so switching to one
            # never means re-recording everything.
            self._crops[tid] = self._encode_crop(cv2, frame, box, h, w)
            if w > self.snapshot_width:
                scale = self.snapshot_width / float(w)
                img = cv2.resize(img, (self.snapshot_width, max(1, int(h * scale))),
                                 interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(
                ".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(self.snapshot_quality)])
            if ok:
                # The box travels with the still, so the photo can carry its
                # overlay later without re-running the detector.
                self._snaps[tid] = (area, buf.tobytes(), tuple(float(v) for v in box))
        except Exception as exc:
            logger.warning("collector: snapshot encode failed: %s", exc)

    def _encode_crop(self, cv2, frame, box, h: int, w: int):
        """The detection box, padded a little, as JPEG bytes."""
        pad = 0.10
        bw, bh = box[2] - box[0], box[3] - box[1]
        x1 = max(0, int((box[0] - bw * pad) * w))
        y1 = max(0, int((box[1] - bh * pad) * h))
        x2 = min(w, int((box[2] + bw * pad) * w))
        y2 = min(h, int((box[3] + bh * pad) * h))
        if x2 - x1 < 8 or y2 - y1 < 8:
            return None
        ok, buf = cv2.imencode(".jpg", frame[y1:y2, x1:x2],
                               [int(cv2.IMWRITE_JPEG_QUALITY), int(self.snapshot_quality)])
        return buf.tobytes() if ok else None

    def _write_snapshot(self, tid: int) -> Tuple[Optional[str], Optional[str], Optional[list]]:
        """Write the still and the crop. Returns both filenames and the box.

        The crop's name is returned rather than inferred from the still's:
        anything reading the store later should not have to know the
        naming convention to find it.
        """
        best = self._snaps.pop(tid, None)
        if best is None or self.snapshot_dir is None:
            self._crops.pop(tid, None)
            return None, None, None
        try:
            self.snapshot_dir.mkdir(parents=True, exist_ok=True)
            stem = uuid.uuid4().hex
            name = f"{stem}.jpg"
            (self.snapshot_dir / name).write_bytes(best[1])
            self.snapshots_saved += 1

            crop_name = None
            crop = self._crops.pop(tid, None)
            if crop:
                crop_name = f"{stem}_crop.jpg"
                (self.snapshot_dir / crop_name).write_bytes(crop)
                self.crops_saved += 1
            # Normalised xyxy of the subject in the still at the moment it
            # was taken (the still is resized, the fractions are not).
            box = [round(v, 4) for v in best[2]] if len(best) > 2 else None
            return name, crop_name, box
        except Exception as exc:
            logger.warning("collector: snapshot write failed: %s", exc)
            return None, None, None

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

    def _stationary(self, t: Track, first, last) -> Optional[bool]:
        """None = not a stationary-class track that sat still. Otherwise
        True if it should be suppressed, False if kept as the record."""
        if t.cls not in self.stationary_classes or first is None or last is None:
            return None
        if _iou(first, last) < self.stationary_iou:
            return None  # it moved during the track: arriving, leaving
        now = t.last_seen
        self._parked = [p for p in self._parked if now - p["last_seen"] <= self.stationary_forget_s]
        for p in self._parked:
            if p["cls"] == t.cls and _iou(p["box"], last) >= self.stationary_iou:
                p["last_seen"] = now
                p["box"] = last
                if now - p["last_kept"] < self.stationary_keep_s:
                    return True
                p["last_kept"] = now
                return False
        self._parked.append({"cls": t.cls, "box": last, "last_seen": now, "last_kept": now})
        return False

    def _emit(self, tid: int) -> bool:
        t = self._tracks.pop(tid, None)
        last_box = self._boxes.pop(tid, None)
        first_box = self._first_box.pop(tid, None)
        if t is None:
            self._snaps.pop(tid, None)
            self._crops.pop(tid, None)
            return False

        # A two-frame blip is a detector flicker, not an event. Recording it
        # would fill the store with noise the operator then has to label.
        if t.frames < self.min_frames:
            self.dropped_short += 1
            self._snaps.pop(tid, None)
            self._crops.pop(tid, None)
            return False

        stationary = self._stationary(t, first_box, last_box)
        if stationary:
            self.suppressed_stationary += 1
            self._snaps.pop(tid, None)
            self._crops.pop(tid, None)
            return False

        x = extract(t)
        snapshot, crop, box = self._write_snapshot(tid)
        meta = {
            "snapshot": snapshot,
            # Where the subject is in the still, for the photo's overlay.
            "box": box,
            # The subject on its own. Kept so an image model can be trained
            # later without recapturing any of this.
            "crop": crop,
            "track_id": t.track_id,
            "frames": t.frames,
            "duration_s": round(t.last_seen - t.first_seen, 3),
            "source": self.source,
            "synthetic": False,
            # Kept as the record of an object that is not going anywhere.
            "stationary": stationary is False,
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
                    media_path=(str(self.snapshot_dir / snapshot)
                                if snapshot else None),
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
            "snapshots_saved": self.snapshots_saved,
            "crops_saved": self.crops_saved,
            "suppressed_stationary": self.suppressed_stationary,
            "parked_objects": len(self._parked),
            "motion_open": self._motion is not None,
            "reach_factor": self.reach_factor,
            "max_idle_s": self.max_idle_s,
        }
