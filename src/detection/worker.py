"""
Continuous detection, decoupled from anyone watching.

Before this, `DetectionService.process_frame` had exactly one caller: the
MJPEG stream generator. Detection ran only while a browser held the stream
open, which contradicts FR-19 (detect every frame, 24/7) and meant a node
with no viewer collected no events at all.

It also serialised everything. The stream loop captured, detected, drew,
and encoded one frame at a time, so a slow detector stalled capture and a
slow encode stalled detection. This splits the work across threads:

    capture thread   camera -> latest-frame slot     (never waits on the model)
    detect thread    latest frame -> process_frame   (always works on the newest frame)
    stream           latest frame + latest boxes -> JPEG, at its own rate

When the detector is slower than the camera, frames it never sees are
dropped rather than queued, so boxes describe what is in front of the
camera now rather than several seconds ago. That trade -- fresher results
over processing every frame -- is the right one for a security feed.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Generator, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class _Rate:
    """Events per second over a sliding window of timestamps."""

    def __init__(self, window: int = 30):
        self._t: deque = deque(maxlen=window)

    def tick(self) -> None:
        self._t.append(time.monotonic())

    def value(self) -> float:
        if len(self._t) < 2:
            return 0.0
        span = self._t[-1] - self._t[0]
        return (len(self._t) - 1) / span if span > 0 else 0.0


class DetectionWorker:
    """Owns the camera and runs detection continuously in the background."""

    def __init__(self, camera, detector, recorder=None):
        self.camera = camera
        self.detector = detector
        self.recorder = recorder

        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._frame_seq = 0
        self._dets: List = []
        self._dets_seq = 0

        self._stop = threading.Event()
        self._threads: List[threading.Thread] = []
        self.running = False
        self.camera_error: Optional[str] = None

        self.capture_rate = _Rate()
        self.detect_rate = _Rate()
        self.detect_ms = 0.0
        self.frames_captured = 0
        self.frames_detected = 0

    # ------------------------------------------------------------------
    def start(self) -> None:
        if self.running:
            return
        self._stop.clear()
        self.camera_error = None
        self.running = True
        self._threads = [
            threading.Thread(target=self._capture_loop, name="argus-capture", daemon=True),
            threading.Thread(target=self._detect_loop, name="argus-detect", daemon=True),
        ]
        for t in self._threads:
            t.start()
        logger.info("DetectionWorker started")

    def stop(self, timeout: float = 3.0) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout)
        self._threads = []
        self.running = False
        logger.info("DetectionWorker stopped")

    def latest(self) -> Tuple[Optional[np.ndarray], List, int]:
        with self._lock:
            return self._frame, list(self._dets), self._frame_seq

    # ------------------------------------------------------------------
    def _capture_loop(self) -> None:
        # Opening the camera here rather than in start() keeps application
        # startup from blocking on hardware, and a missing camera from
        # stopping the API serving.
        if not self.camera.is_initialized and not self.camera.initialize():
            self.camera_error = "camera failed to initialize"
            logger.error("DetectionWorker: %s", self.camera_error)
            self.running = False
            self._stop.set()
            return

        while not self._stop.is_set():
            frame = self.camera.get_frame_array()
            if frame is None:
                time.sleep(0.01)
                continue
            with self._lock:
                self._frame = frame
                self._frame_seq += 1
                dets = self._dets
            self.frames_captured += 1
            self.capture_rate.tick()

            if self.recorder is not None:
                # Every captured frame, not just detected ones: the recorder's
                # pre-roll buffer needs continuous footage to be evidence.
                try:
                    self.recorder.process(frame, dets)
                except Exception as exc:
                    logger.warning("Event recording failed: %s", exc)

    def _detect_loop(self) -> None:
        last_seq = 0
        while not self._stop.is_set():
            with self._lock:
                seq, frame = self._frame_seq, self._frame
            if frame is None or seq == last_seq:
                time.sleep(0.002)
                continue

            t0 = time.perf_counter()
            try:
                dets = self.detector.process_frame(frame)
            except Exception as exc:
                logger.warning("Detection failed on frame %d: %s", seq, exc)
                dets = []
            ms = (time.perf_counter() - t0) * 1000.0

            with self._lock:
                self._dets = dets
                self._dets_seq = seq
            self.detect_ms = ms if self.frames_detected == 0 else 0.9 * self.detect_ms + 0.1 * ms
            self.frames_detected += 1
            self.detect_rate.tick()
            last_seq = seq

    # ------------------------------------------------------------------
    def status(self) -> dict:
        return {
            "running": self.running,
            "camera_error": self.camera_error,
            "capture_fps": round(self.capture_rate.value(), 2),
            "detect_fps": round(self.detect_rate.value(), 2),
            "detect_ms": round(self.detect_ms, 1),
            "frames_captured": self.frames_captured,
            "frames_detected": self.frames_detected,
            # Captured frames the detector skipped to stay on the newest one.
            "frames_not_detected": max(0, self.frames_captured - self.frames_detected),
        }


def stream_from_worker(
    worker: DetectionWorker,
    jpeg_quality: int = 80,
    max_fps: float = 15.0,
    stream_width: Optional[int] = 960,
) -> Generator[bytes, None, None]:
    """MJPEG of the worker's latest frame with its latest boxes drawn on.

    Never runs the detector, so a viewer costs only draw + encode. Encoding
    is downscaled to `stream_width`: a 1080p JPEG per frame is the single
    most expensive CPU stage left once detection is offloaded, and the
    dashboard never displays it at full size.
    """
    from src.detection.annotate import draw_detections
    from src.detection.stream import BOUNDARY

    params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
    period = 1.0 / max(1.0, float(max_fps))
    last_seq = -1

    while worker.running:
        t0 = time.monotonic()
        frame, dets, seq = worker.latest()
        if frame is None or seq == last_seq:
            time.sleep(0.01)
            continue
        last_seq = seq

        annotated = draw_detections(frame, dets)
        if stream_width and annotated.shape[1] > stream_width:
            h = int(annotated.shape[0] * stream_width / annotated.shape[1])
            annotated = cv2.resize(annotated, (stream_width, h), interpolation=cv2.INTER_AREA)

        ok, buf = cv2.imencode(".jpg", annotated, params)
        if ok:
            yield (
                b"--" + BOUNDARY + b"\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
            )

        spent = time.monotonic() - t0
        if spent < period:
            time.sleep(period - spent)
