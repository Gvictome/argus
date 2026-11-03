"""
Gate B fix F2: /api/camera/record must not write outside MEDIA_DIR.

`filename` was joined straight onto MEDIA_DIR, so "../../x" or an absolute
path wrote (and could overwrite) a file anywhere the process could write.
"""

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.camera import camera_service
from src.config import settings


@pytest.fixture
def client(monkeypatch, tmp_path):
    started = []
    monkeypatch.setattr(settings, "AUTH_REQUIRED", False)
    monkeypatch.setattr(settings, "DB_PATH", tmp_path / "c.db")
    monkeypatch.setattr(settings, "FL_TRAINING_DIR", tmp_path / "training")
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    monkeypatch.setattr(settings, "MEDIA_DIR", tmp_path / "media")
    with TestClient(create_app()) as c:
        # Fake an open camera only for the request; startup may reset it.
        monkeypatch.setattr(camera_service, "is_initialized", True)
        monkeypatch.setattr(
            camera_service, "start_recording",
            lambda path: started.append(path) or True,
        )
        c.started = started
        c.media = tmp_path / "media"
        yield c


@pytest.mark.parametrize("name", [
    "../evil.mp4",
    "../../etc/cron.d/evil.mp4",
    "..\\..\\evil.mp4",
    "/tmp/evil.mp4",
    "C:\\Windows\\evil.mp4",
    "sub/dir.mp4",
    "clip.avi",
    "clip.mp4.sh",
    "clip",
    ".mp4",
    "a b.mp4",
    "x" * 65 + ".mp4",
])
def test_unsafe_filenames_are_rejected(client, name):
    r = client.post("/api/camera/record", params={"start": True, "filename": name})
    assert r.status_code == 400
    assert client.started == []


@pytest.mark.parametrize("name", ["clip.mp4", "front-door_01.mp4", "Clip2.MP4"])
def test_safe_filename_records_inside_media_dir(client, name):
    r = client.post("/api/camera/record", params={"start": True, "filename": name})
    assert r.status_code == 200
    assert len(client.started) == 1
    path = client.started[0]
    assert path.parent.resolve() == client.media.resolve()
    assert path.name == name


def test_default_filename_still_works(client):
    r = client.post("/api/camera/record", params={"start": True})
    assert r.status_code == 200
    path = client.started[0]
    assert path.parent.resolve() == client.media.resolve()
    assert path.suffix == ".mp4"
