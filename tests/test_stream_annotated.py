"""
Tests for the annotated MJPEG generator (showcase P0-2 / G-2).

The generator is the demo's primary surface: it is what a judge looks at
for the full two minutes.  It must survive a dropped frame, must not run
the detector on every frame, and must stop when the camera stops.
"""

import numpy as np
import pytest

from src.detection import Detection, DetectionType
from src.detection.annotate import BOX_KNOWN
from src.detection.stream import BOUNDARY, stream_annotated_mjpeg


class FakeConfig:
    # Absurdly high so the generator's inter-frame sleep is negligible.
    framerate = 10000


class FakeCamera:
    """Stands in for CameraService, yielding a fixed number of frames."""

    def __init__(self, n_frames=6, frame=None, blank_indices=()):
        self.config = FakeConfig()
        self.is_streaming = False
        self.n_frames = n_frames
        self.served = 0
        self.blank_indices = set(blank_indices)
        self._frame = frame if frame is not None else np.zeros((120, 160, 3), dtype=np.uint8)

    def get_frame_array(self):
        if self.served >= self.n_frames:
            self.is_streaming = False
            return None
        idx = self.served
        self.served += 1
        if idx in self.blank_indices:
            return None  # simulate a dropped capture
        return self._frame.copy()


class FakeDetector:
    """Counts how often the expensive pipeline is invoked."""

    def __init__(self, detections=None):
        self.calls = 0
        self.detections = detections if detections is not None else []

    def process_frame(self, frame):
        self.calls += 1
        return self.detections


def known_face(bbox=(20, 20, 40, 40)):
    return Detection(
        type=DetectionType.FACE, confidence=0.83, bbox=bbox, label="Giovanny", face_id="f1"
    )


def collect(gen):
    return list(gen)


class TestMjpegFraming:
    def test_yields_one_chunk_per_captured_frame(self):
        cam, det = FakeCamera(n_frames=4), FakeDetector()
        chunks = collect(stream_annotated_mjpeg(cam, det))
        assert len(chunks) == 4

    def test_chunk_has_multipart_boundary_and_jpeg_content_type(self):
        cam, det = FakeCamera(n_frames=1), FakeDetector()
        chunk = collect(stream_annotated_mjpeg(cam, det))[0]
        assert chunk.startswith(b"--" + BOUNDARY + b"\r\n")
        assert b"Content-Type: image/jpeg\r\n" in chunk
        assert chunk.endswith(b"\r\n")

    def test_payload_decodes_as_a_real_jpeg(self):
        import cv2

        cam, det = FakeCamera(n_frames=1), FakeDetector()
        chunk = collect(stream_annotated_mjpeg(cam, det))[0]
        payload = chunk.split(b"\r\n\r\n", 1)[1].rstrip(b"\r\n")
        decoded = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
        assert decoded is not None, "yielded payload is not a decodable JPEG"
        assert decoded.shape == (120, 160, 3)


class TestDetectionCadence:
    def test_runs_detector_on_every_frame_when_detect_every_is_one(self):
        cam, det = FakeCamera(n_frames=5), FakeDetector()
        collect(stream_annotated_mjpeg(cam, det, detect_every=1))
        assert det.calls == 5

    def test_skips_detection_between_intervals(self):
        """6 frames at detect_every=3 means detection on frames 0 and 3 only."""
        cam, det = FakeCamera(n_frames=6), FakeDetector()
        collect(stream_annotated_mjpeg(cam, det, detect_every=3))
        assert det.calls == 2

    def test_reuses_last_detections_on_skipped_frames(self):
        """Boxes must persist between detection runs, not flicker off."""
        cam = FakeCamera(n_frames=3)
        det = FakeDetector(detections=[known_face()])
        chunks = collect(stream_annotated_mjpeg(cam, det, detect_every=3))
        assert det.calls == 1
        import cv2

        for i, chunk in enumerate(chunks):
            payload = chunk.split(b"\r\n\r\n", 1)[1].rstrip(b"\r\n")
            img = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
            green = int((img[:, :, 1] > 200).sum())
            assert green > 0, f"frame {i} lost its box on a skipped detection frame"


class TestResilience:
    def test_dropped_frame_is_skipped_without_yielding(self):
        cam, det = FakeCamera(n_frames=4, blank_indices={1}), FakeDetector()
        chunks = collect(stream_annotated_mjpeg(cam, det))
        assert len(chunks) == 3

    def test_detector_failure_still_yields_video(self):
        """A detector crash must degrade to raw video, never kill the stream."""

        class ExplodingDetector:
            calls = 0

            def process_frame(self, frame):
                ExplodingDetector.calls += 1
                raise RuntimeError("model blew up")

        cam = FakeCamera(n_frames=3)
        chunks = collect(stream_annotated_mjpeg(cam, ExplodingDetector(), detect_every=1))
        assert len(chunks) == 3, "stream died when the detector raised"

    def test_stops_when_camera_stops_streaming(self):
        cam, det = FakeCamera(n_frames=2), FakeDetector()
        collect(stream_annotated_mjpeg(cam, det))
        assert cam.is_streaming is False
