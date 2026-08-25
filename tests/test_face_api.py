"""
Face and detection-status HTTP endpoints.

The face routes resolve the recognizer lazily via
`src.api.app.get_face_recognizer`, so tests substitute it there rather than
standing up insightface.
"""

import pytest
from fastapi.testclient import TestClient

import src.api.app as app_module
from src.api.app import create_app


@pytest.fixture
def client():
    return TestClient(create_app())


class _StubRecognizer:
    def __init__(self, faces=None):
        self.faces = faces if faces is not None else []
        self.reset_calls = 0

    def list_known_faces(self):
        return list(self.faces)

    def reset(self):
        self.reset_calls += 1
        removed = len(self.faces)
        self.faces = []
        return removed


@pytest.fixture
def recognizer(monkeypatch):
    stub = _StubRecognizer(
        [{"face_id": "a", "name": "Giovanny"}, {"face_id": "b", "name": "Adam"}]
    )
    monkeypatch.setattr(app_module, "get_face_recognizer", lambda: stub)
    return stub


@pytest.fixture
def no_recognizer(monkeypatch):
    monkeypatch.setattr(app_module, "get_face_recognizer", lambda: None)


class TestFaceReset:
    """Showcase P1-6 / G-4: clear the known-faces database between judges."""

    def test_reset_clears_faces_and_reports_the_count(self, client, recognizer):
        response = client.post("/api/faces/reset")

        assert response.status_code == 200
        assert response.json()["removed"] == 2
        assert recognizer.reset_calls == 1
        assert client.get("/api/faces").json()["count"] == 0

    def test_reset_returns_503_when_recognizer_unavailable(self, client, no_recognizer):
        """A 200 here would let an operator believe the database was cleared
        when it never was -- the exact state P1-6 exists to prevent."""
        response = client.post("/api/faces/reset")

        assert response.status_code == 503


class TestDetectionStatus:
    """
    Showcase G-5 / PRD P0-2: the endpoint returned a hardcoded
    {"status": "stopped", ...} literal regardless of what was running.
    """

    def test_status_reports_live_service_fields(self, client):
        response = client.get("/api/detection/status")

        assert response.status_code == 200
        data = response.json()
        assert set(data) == {
            "status",
            "fps",
            "backend",
            "motion_detection",
            "object_detection",
            "face_recognition",
            "known_faces",
        }
        assert isinstance(data["fps"], (int, float))
        assert data["backend"] in {"hailo", "cpu", "none"}

    def test_status_tracks_the_detection_service(self, client, monkeypatch):
        """The response must follow the service, not a literal."""
        from src.detection import detection_service

        monkeypatch.setattr(
            detection_service,
            "status",
            lambda: {
                "status": "running",
                "fps": 21.5,
                "backend": "hailo",
                "motion_detection": True,
                "object_detection": True,
                "face_recognition": True,
                "known_faces": 3,
            },
        )

        data = client.get("/api/detection/status").json()

        assert data["fps"] == 21.5
        assert data["backend"] == "hailo"
        assert data["known_faces"] == 3
