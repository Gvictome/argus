"""
DetectionService tests: model loading, the detection cascade, and status.

Both ultralytics and insightface are faked at the module boundary. The Pi
carries those wheels; a laptop or CI runner does not, and the cascade's
control flow is worth testing on every machine that runs the suite.
"""

import time

import numpy as np
import pytest

from src.detection import (
    Detection,
    DetectionConfig,
    DetectionService,
    DetectionType,
)
import src.detection as detection_module


def _frame(value: int = 0) -> np.ndarray:
    """A plain BGR frame. Uniform frames produce no motion contours."""
    return np.full((480, 640, 3), value, dtype=np.uint8)


def _moving_frame() -> np.ndarray:
    """A frame with a bright block big enough to clear min_detection_size."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[100:300, 100:300] = 255
    return frame


class _FakeYOLO:
    """Records the model path it was constructed with; returns no boxes."""

    last_path = None

    def __init__(self, path):
        _FakeYOLO.last_path = str(path)

    def __call__(self, frame, verbose=False):
        return []


@pytest.fixture
def fake_yolo(monkeypatch):
    _FakeYOLO.last_path = None
    monkeypatch.setattr(detection_module, "_ULTRALYTICS_AVAILABLE", True)
    monkeypatch.setattr(detection_module, "_YOLO", _FakeYOLO)
    return _FakeYOLO


class TestInitialize:
    """Model loading during initialize()."""

    def test_initialize_loads_object_model_when_ultralytics_available(self, fake_yolo):
        """
        Regression: BASE_DIR was referenced in initialize() but never imported,
        so the Hailo-model probe raised NameError. The bare `except Exception`
        swallowed it and left object_model as None, which silently disabled
        object detection -- and with it face recognition, since the cascade
        only reaches faces after YOLO reports a human.
        """
        service = DetectionService()
        service.initialize()

        assert service.object_model is not None, (
            "object_model is None -- initialize() swallowed a load failure"
        )
        assert fake_yolo.last_path.endswith("yolov8n.pt")


class _FakeRecognizer:
    """Stands in for FaceRecognitionService; counts recognize() calls."""

    def __init__(self, matches=None):
        self.matches = matches or []
        self.calls = 0

    def recognize(self, frame):
        self.calls += 1
        return list(self.matches)

    def list_known_faces(self):
        return []


class _FakeMatch:
    def __init__(self, face_id=None, name=None, confidence=0.0):
        self.face_id = face_id
        self.name = name
        self.confidence = confidence
        self.bbox = (10, 20, 30, 40)
        self.embedding = None


class _HumanYOLO(_FakeYOLO):
    """A YOLO stand-in that reports one person filling the frame."""

    def __call__(self, frame, verbose=False):
        class _Box:
            conf = [0.9]
            cls = [0]
            xywh = [np.array([200.0, 200.0, 100.0, 200.0])]

        class _Result:
            names = {0: "person"}
            boxes = [_Box()]

        return [_Result()]


class TestCascadeFallback:
    """Which models process_frame() reaches, and when."""

    def test_faces_run_on_motion_when_object_model_unavailable(self):
        """
        With no YOLO there are no human detections, so gating faces behind
        `if humans:` means faces never run at all. On a Pi where ultralytics
        or the Hailo model fails to load, that turns the headline feature off
        with no visible error. Motion alone must be enough to reach faces.
        """
        service = DetectionService()
        service.object_model = None
        recognizer = _FakeRecognizer([_FakeMatch(face_id="f1", name="Giovanny", confidence=0.8)])
        service.attach_face_recognizer(recognizer)

        service.process_frame(_frame(0))          # primes previous_frame
        detections = service.process_frame(_moving_frame())

        assert recognizer.calls == 1, "face recognition never ran without YOLO"
        faces = [d for d in detections if d.type is DetectionType.FACE]
        assert [f.label for f in faces] == ["Giovanny"]

    def test_faces_skipped_when_object_model_present_but_no_humans(self, fake_yolo):
        """
        The fallback must not become an unconditional bypass. When YOLO is
        working and reports nobody, the expensive ArcFace pass stays skipped.
        """
        service = DetectionService()
        service.initialize()
        recognizer = _FakeRecognizer()
        service.attach_face_recognizer(recognizer)

        service.process_frame(_frame(0))
        service.process_frame(_moving_frame())

        assert recognizer.calls == 0

    def test_faces_run_when_yolo_detects_a_human(self, monkeypatch):
        """The normal path: YOLO finds a person, faces run on that frame."""
        monkeypatch.setattr(detection_module, "_ULTRALYTICS_AVAILABLE", True)
        monkeypatch.setattr(detection_module, "_YOLO", _HumanYOLO)

        service = DetectionService()
        service.initialize()
        recognizer = _FakeRecognizer([_FakeMatch()])
        service.attach_face_recognizer(recognizer)

        service.process_frame(_frame(0))
        service.process_frame(_moving_frame())

        assert recognizer.calls == 1


class TestStatus:
    """
    status() backs GET /api/detection/status, which returned hardcoded
    placeholder values (showcase G-5 / PRD P0-2).
    """

    def test_status_reports_stopped_before_initialize(self):
        service = DetectionService()
        status = service.status()

        assert status["status"] == "stopped"
        assert status["object_detection"] is False
        assert status["face_recognition"] is False
        assert status["backend"] == "none"
        assert status["fps"] == 0.0

    def test_status_reports_running_after_initialize(self, fake_yolo):
        service = DetectionService()
        service.initialize()
        status = service.status()

        assert status["status"] == "running"
        assert status["object_detection"] is True
        assert status["motion_detection"] is True
        assert status["backend"] == "cpu"

    def test_status_reports_hailo_backend_when_hef_model_present(self, fake_yolo, tmp_path, monkeypatch):
        """The AI HAT+ path must be visible on the status endpoint, since
        'is the accelerator actually in use' is a question a judge asks."""
        (tmp_path / "yolov8n_hailo_model.hef").write_bytes(b"stub")
        monkeypatch.setattr(detection_module, "BASE_DIR", tmp_path)

        service = DetectionService()
        service.initialize()

        assert service.status()["backend"] == "hailo"

    def test_status_reports_face_recognition_and_known_face_count(self):
        service = DetectionService()

        class _Recognizer(_FakeRecognizer):
            def list_known_faces(self):
                return [{"face_id": "a", "name": "Giovanny"}, {"face_id": "b", "name": "Adam"}]

        service.attach_face_recognizer(_Recognizer())
        status = service.status()

        assert status["face_recognition"] is True
        assert status["known_faces"] == 2

    def test_fps_is_zero_with_fewer_than_two_frames(self):
        service = DetectionService()
        service._frame_times.append(1.0)

        assert service.status()["fps"] == 0.0

    def test_fps_computed_over_the_frame_time_window(self):
        """Five timestamps 0.1s apart is four intervals across 0.4s: 10 fps."""
        service = DetectionService()
        for t in (0.0, 0.1, 0.2, 0.3, 0.4):
            service._frame_times.append(t)

        assert service.status()["fps"] == pytest.approx(10.0, rel=1e-3)

    def test_process_frame_records_a_frame_timestamp(self):
        service = DetectionService()
        service.object_model = None

        service.process_frame(_frame(0))
        service.process_frame(_frame(0))

        assert len(service._frame_times) == 2


class TestMotionDownscale:
    """
    Motion detection ran cvtColor + GaussianBlur(21,21) + dilate at full
    resolution, ~9ms per frame at 1080p before any model runs. The diff only
    needs to find *where* something moved, which survives downscaling.
    """

    def _frame_with_block(self, x, y, w, h):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[y:y + h, x:x + w] = 255
        return frame

    def test_motion_bbox_is_reported_in_full_frame_coordinates(self):
        """The diff runs on a shrunken copy, so boxes must be scaled back or
        every motion overlay lands in the wrong place."""
        service = DetectionService(DetectionConfig(motion_detection_scale=4))

        service.detect_motion(_frame(0))
        detections = service.detect_motion(self._frame_with_block(200, 160, 160, 120))

        assert detections, "no motion detected"
        x, y, w, h = max(detections, key=lambda d: d.bbox[2] * d.bbox[3]).bbox

        # Generous bounds on purpose: GaussianBlur(21,21) plus two dilate
        # passes inflate every motion contour, at any scale. What this pins
        # is the coordinate space -- dropping the scale multiply would put
        # this box near (50, 40, 40, 30), nowhere near these bounds.
        assert 170 <= x <= 210, f"x={x} not near 200"
        assert 130 <= y <= 170, f"y={y} not near 160"
        assert 150 <= w <= 210, f"w={w} not near 160"
        assert 110 <= h <= 170, f"h={h} not near 120"

    def test_scale_of_one_leaves_detection_at_full_resolution(self):
        service = DetectionService(DetectionConfig(motion_detection_scale=1))

        service.detect_motion(_frame(0))
        detections = service.detect_motion(self._frame_with_block(200, 160, 160, 120))

        assert detections
        x, y, w, h = max(detections, key=lambda d: d.bbox[2] * d.bbox[3]).bbox
        assert (x, y, w, h) == pytest.approx((200, 160, 160, 120), abs=14)

    def test_small_motion_below_min_size_is_still_filtered(self):
        """Downscaling must not let sub-threshold noise through as motion."""
        service = DetectionService(DetectionConfig(motion_detection_scale=4))

        service.detect_motion(_frame(0))
        detections = service.detect_motion(self._frame_with_block(100, 100, 8, 8))

        assert detections == []
