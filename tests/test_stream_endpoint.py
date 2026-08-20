"""
Integration test for GET /api/camera/stream (showcase P0-2 / G-2).

Drives the real FastAPI route with a stubbed camera and detector to prove
overlays reach the wire, not just the drawing helper.
"""

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.detection import Detection, DetectionType


@pytest.fixture
def client():
    return TestClient(create_app())


class StubCamera:
    """Minimal CameraService stand-in serving a fixed number of frames."""

    class config:
        framerate = 10000

    def __init__(self, n_frames=2):
        self.is_initialized = True
        self.is_streaming = False
        self.n_frames = n_frames
        self.served = 0

    def get_frame_array(self):
        if self.served >= self.n_frames:
            self.is_streaming = False
            return None
        self.served += 1
        return np.zeros((120, 160, 3), dtype=np.uint8)


def test_stream_returns_503_when_camera_not_initialized(client):
    response = client.get("/api/camera/stream")
    assert response.status_code == 503


def test_stream_burns_detection_boxes_into_the_jpeg(client, monkeypatch):
    """The whole point of P0-2: a judge sees a labeled box, not a raw feed."""
    import src.camera as camera_mod
    import src.detection as detection_mod

    monkeypatch.setattr(camera_mod, "camera_service", StubCamera(n_frames=2))

    def fake_process_frame(frame):
        return [
            Detection(
                type=DetectionType.FACE,
                confidence=0.83,
                bbox=(20, 20, 50, 50),
                label="Giovanny",
                face_id="f1",
            )
        ]

    monkeypatch.setattr(
        detection_mod.detection_service, "process_frame", fake_process_frame
    )

    response = client.get("/api/camera/stream")
    assert response.status_code == 200
    assert "multipart/x-mixed-replace" in response.headers["content-type"]

    body = response.content
    assert body.count(b"Content-Type: image/jpeg") == 2, "expected 2 MJPEG parts"

    first = body.split(b"\r\n\r\n", 1)[1].split(b"\r\n--frame", 1)[0]
    img = cv2.imdecode(np.frombuffer(first, np.uint8), cv2.IMREAD_COLOR)
    assert img is not None, "stream payload is not a decodable JPEG"
    assert int((img[:, :, 1] > 200).sum()) > 0, "no green box burned into the frame"
