"""
Event-triggered recording.

The cameras run 24/7; disk does not. Recording continuously fills a
demo unit's SD card in hours and buries the ten seconds anyone actually
wants to see. This records only around detections.

Three behaviours that matter more than they look:

**Pre-roll.** A clip that starts when detection fires has already missed
the approach -- the most useful part of the footage. A rolling in-memory
buffer of the last few seconds is prepended, so the clip starts *before*
the event.

**Post-roll.** Recording continues for a few seconds after the last
detection, so someone walking out of frame does not truncate the clip.

**Cooldown.** Without one, a person loitering in frame produces hundreds
of near-identical clips. Continued detection extends the current clip
instead of starting a new one.
"""

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Deque, List, Optional, Sequence, Set

import numpy as np

from src.detection import Detection, DetectionType

logger = logging.getLogger(__name__)

# Detection types worth keeping footage of. MOTION is deliberately
# excluded: it fires on a cloud crossing the sun or a branch moving, and
# recording on raw motion is how you end up back at 24/7 recording with
# extra steps.
DEFAULT_TRIGGERS = frozenset({
    DetectionType.HUMAN,
    DetectionType.DANGEROUS_PERSON,
    DetectionType.FACE,
    DetectionType.VEHICLE,
})


@dataclass
class RecorderConfig:
    pre_roll_s: float = 5.0
    post_roll_s: float = 10.0
    max_clip_s: float = 120.0        # hard stop; a stuck trigger must not
                                     # record until the disk fills
    fps: int = 15                    # clip fps, independent of capture
    triggers: Set[DetectionType] = field(default_factory=lambda: set(DEFAULT_TRIGGERS))
    min_confidence: float = 0.5
    output_dir: Optional[Path] = None


@dataclass
class ClipRecord:
    """One saved clip, for the events table and the dashboard."""
    path: str
    started_at: str
    ended_at: str
    duration_s: float
    trigger: str
    peak_confidence: float
    frames: int

    def as_dict(self) -> dict:
        return {
            "path": self.path,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_s": round(self.duration_s, 2),
            "trigger": self.trigger,
            "peak_confidence": round(self.peak_confidence, 3),
            "frames": self.frames,
        }


class EventRecorder:
    """
    Turns a stream of frames plus detections into event clips.

    Call `process(frame, detections)` once per processed frame. The
    recorder decides on its own when to open and close a clip.
    """

    def __init__(self, config: Optional[RecorderConfig] = None, camera_name: str = "camera0"):
        self.config = config or RecorderConfig()
        self.camera_name = camera_name

        buffer_frames = max(1, int(self.config.pre_roll_s * self.config.fps))
        self._pre_roll: Deque[np.ndarray] = deque(maxlen=buffer_frames)

        self._writer = None
        self._lock = threading.Lock()
        self._clip_path: Optional[Path] = None
        self._clip_started: Optional[float] = None
        self._last_trigger_at: Optional[float] = None
        self._trigger_label = ""
        self._peak_confidence = 0.0
        self._frames_written = 0
        self._clips: List[ClipRecord] = []

    # -- state ----------------------------------------------------------

    @property
    def is_recording(self) -> bool:
        return self._writer is not None

    def recent_clips(self, limit: int = 20) -> List[dict]:
        return [c.as_dict() for c in self._clips[-limit:][::-1]]

    def status(self) -> dict:
        return {
            "camera": self.camera_name,
            "recording": self.is_recording,
            "current_clip": str(self._clip_path) if self._clip_path else None,
            "clips_saved": len(self._clips),
            "pre_roll_s": self.config.pre_roll_s,
            "post_roll_s": self.config.post_roll_s,
            "triggers": sorted(t.value for t in self.config.triggers),
        }

    # -- the decision ----------------------------------------------------

    def _triggering(self, detections: Sequence[Detection]) -> Optional[Detection]:
        """
        The detection that justifies recording, if any.

        Highest confidence among triggering types, so the clip is
        labelled by the most certain thing in it rather than the first.
        """
        candidates = [
            d for d in detections
            if d.type in self.config.triggers and d.confidence >= self.config.min_confidence
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda d: d.confidence)

    def process(
        self,
        frame: np.ndarray,
        detections: Sequence[Detection],
        now: Optional[float] = None,
    ) -> None:
        """
        Feed one frame and its detections.

        Cheap when nothing is happening: the frame goes into the pre-roll
        ring buffer and nothing else occurs.
        """
        if frame is None:
            return
        now = now if now is not None else time.time()

        with self._lock:
            trigger = self._triggering(detections)

            if trigger is not None:
                self._last_trigger_at = now
                self._peak_confidence = max(self._peak_confidence, trigger.confidence)
                if not self.is_recording:
                    self._start_clip(frame, trigger, now)

            if self.is_recording:
                self._write(frame)
                if self._should_stop(now):
                    self._stop_clip(now)
            else:
                # Only buffer when idle; while recording the frames are
                # already going to disk.
                self._pre_roll.append(frame.copy())

    def _should_stop(self, now: float) -> bool:
        if self._clip_started is not None and now - self._clip_started >= self.config.max_clip_s:
            logger.info("Clip hit max length (%.0fs)", self.config.max_clip_s)
            return True
        if self._last_trigger_at is None:
            return True
        return now - self._last_trigger_at >= self.config.post_roll_s

    # -- writing ---------------------------------------------------------

    def _output_dir(self) -> Path:
        if self.config.output_dir is not None:
            return Path(self.config.output_dir)
        from src.config import settings
        return Path(settings.MEDIA_DIR) / "events"

    def _start_clip(self, frame: np.ndarray, trigger: Detection, now: float) -> None:
        import cv2

        out_dir = self._output_dir()
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            logger.error("Cannot create %s: %s", out_dir, exc)
            return

        stamp = datetime.fromtimestamp(now).strftime("%Y%m%d_%H%M%S")
        label = trigger.type.value
        path = out_dir / f"{self.camera_name}_{label}_{stamp}.mp4"

        h, w = frame.shape[:2]
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), self.config.fps, (w, h)
        )
        if not writer.isOpened():
            logger.error("Could not open video writer for %s", path)
            return

        self._writer = writer
        self._clip_path = path
        self._clip_started = now
        self._trigger_label = label
        self._peak_confidence = trigger.confidence
        self._frames_written = 0

        # Pre-roll first: this is the approach that led to the detection.
        for buffered in self._pre_roll:
            if buffered.shape[:2] == (h, w):
                writer.write(buffered)
                self._frames_written += 1
        self._pre_roll.clear()

        logger.info("Recording started: %s (trigger=%s %.2f)",
                    path.name, label, trigger.confidence)

    def _write(self, frame: np.ndarray) -> None:
        if self._writer is None:
            return
        try:
            self._writer.write(frame)
            self._frames_written += 1
        except Exception as exc:
            logger.warning("Frame write failed: %s", exc)

    def _stop_clip(self, now: float) -> None:
        if self._writer is None:
            return
        try:
            self._writer.release()
        except Exception:
            pass

        # Explicit None checks, not `or`: a clip legitimately starts at
        # t=0.0 under a monotonic clock, and 0.0 is falsy -- `started or
        # now` silently replaced the start time with the end time and
        # reported every such clip as zero-length.
        started = self._clip_started if self._clip_started is not None else now
        duration = now - started
        if self._clip_path is not None:
            record = ClipRecord(
                path=str(self._clip_path),
                started_at=datetime.fromtimestamp(started).isoformat(),
                ended_at=datetime.fromtimestamp(now).isoformat(),
                duration_s=duration,
                trigger=self._trigger_label,
                peak_confidence=self._peak_confidence,
                frames=self._frames_written,
            )
            self._clips.append(record)
            logger.info("Recording saved: %s (%.1fs, %d frames)",
                        self._clip_path.name, duration, self._frames_written)

        self._writer = None
        self._clip_path = None
        self._clip_started = None
        self._last_trigger_at = None
        self._peak_confidence = 0.0
        self._frames_written = 0

    def close(self) -> None:
        """Finalize any open clip. Call on shutdown so it is not truncated."""
        with self._lock:
            if self.is_recording:
                self._stop_clip(time.time())
