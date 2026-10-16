"""
The FL idle gate must see real activity.

FLScheduler.notify_motion() had no caller, so is_idle() was always True and
a round could train while the camera was busy. The detection service now
notifies an attached listener; app startup wires the scheduler to it.
"""

from datetime import datetime

import numpy as np
import pytest

from src.detection import DetectionConfig, DetectionService
from src.federated.scheduler import FLScheduler


def _frame(value: int = 0) -> np.ndarray:
    return np.full((480, 640, 3), value, dtype=np.uint8)


def _moving_frame() -> np.ndarray:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[100:300, 100:300] = 255
    return frame


class _Config:
    FL_ENABLED = True
    FL_SERVER_URL = "localhost:8080"
    FL_ROUND_HOUR = 2
    FL_ROUND_INTERVAL_DAYS = 14


@pytest.fixture
def scheduler(tmp_path):
    return FLScheduler(None, _Config(), state_path=tmp_path / "s.json")


@pytest.fixture
def service():
    svc = DetectionService(DetectionConfig(faces_enabled=False))
    svc.object_model = None
    return svc


def test_motion_marks_scheduler_not_idle(service, scheduler):
    service.attach_motion_listener(scheduler.notify_motion)
    assert scheduler.is_idle()

    service.process_frame(_frame(0))
    service.process_frame(_moving_frame())

    assert not scheduler.is_idle()


def test_no_motion_leaves_scheduler_idle(service, scheduler):
    service.attach_motion_listener(scheduler.notify_motion)

    service.process_frame(_frame(0))
    service.process_frame(_frame(0))

    assert scheduler.is_idle()


def test_a_failing_listener_never_breaks_detection(service):
    def boom():
        raise RuntimeError("listener fault")

    service.attach_motion_listener(boom)
    service.process_frame(_frame(0))

    service.process_frame(_moving_frame())  # must not raise


def test_detaching_stops_notifications(service, scheduler):
    service.attach_motion_listener(scheduler.notify_motion)
    service.attach_motion_listener(None)

    service.process_frame(_frame(0))
    service.process_frame(_moving_frame())

    assert scheduler.is_idle()


def test_app_wires_scheduler_to_detection(monkeypatch, tmp_path):
    """FL_ENABLED=true startup connects the live scheduler to the service."""
    from fastapi.testclient import TestClient
    from src.api.app import create_app
    from src.config import settings
    import src.detection as det

    monkeypatch.setattr(settings, "FL_ENABLED", True)
    monkeypatch.setattr(settings, "AUTH_REQUIRED", False)
    monkeypatch.setattr(settings, "DB_PATH", tmp_path / "m.db")
    monkeypatch.setattr(settings, "FL_TRAINING_DIR", tmp_path / "training")
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)

    with TestClient(create_app()) as client:
        sched = client.app.state.fl_scheduler
        assert det.detection_service.motion_listener == sched.notify_motion
        assert sched.is_idle()
        det.detection_service.process_frame(_frame(0))
        det.detection_service.process_frame(_moving_frame())
        assert not sched.is_idle()
    assert det.detection_service.motion_listener is None
