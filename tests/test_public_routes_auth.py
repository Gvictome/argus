"""
Auth on the read-only routes that were left open.

With AUTH_REQUIRED=true these used to answer anonymous callers through the
tunnel: the event feed (what the cameras saw), camera and detection status,
federated status, device and automation lists. Only health, static pages and
the auth endpoints stay public.
"""

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.config import settings

NEWLY_GUARDED = [
    ("GET", "/api/events"),
    ("GET", "/api/events/1"),
    ("GET", "/api/cameras"),
    ("GET", "/api/cameras/main/status"),
    ("GET", "/api/camera/status"),
    ("GET", "/api/detection/status"),
    ("GET", "/api/federated/status"),
    ("GET", "/api/devices"),
    ("GET", "/api/devices/x"),
    ("GET", "/api/automations"),
    ("GET", "/api/agents/status"),
]

STILL_PUBLIC = [
    ("GET", "/"),
    ("GET", "/api/status"),
    ("GET", "/api/dashboard/status"),
    ("GET", "/api/auth/me"),
]


@pytest.fixture
def auth_on(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "AUTH_REQUIRED", True)
    monkeypatch.setattr(settings, "DB_PATH", tmp_path / "a.db")
    monkeypatch.setattr(settings, "FL_TRAINING_DIR", tmp_path / "training")
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    monkeypatch.setenv("ARGUS_ADMIN_PASSWORD", "correct-horse-battery")
    with TestClient(create_app()) as client:
        yield client


@pytest.fixture
def auth_off(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "AUTH_REQUIRED", False)
    monkeypatch.setattr(settings, "DB_PATH", tmp_path / "b.db")
    with TestClient(create_app()) as client:
        yield client


def _token(client):
    r = client.post("/api/auth/login",
                    json={"username": "admin", "password": "correct-horse-battery"})
    return r.json()["token"]


@pytest.mark.parametrize("method,path", NEWLY_GUARDED)
def test_rejected_without_token(auth_on, method, path):
    assert auth_on.request(method, path).status_code == 401


@pytest.mark.parametrize("method,path", NEWLY_GUARDED)
def test_rejected_with_bad_token(auth_on, method, path):
    r = auth_on.request(method, path, headers={"Authorization": "Bearer nope"})
    assert r.status_code == 401


@pytest.mark.parametrize("method,path", NEWLY_GUARDED)
def test_passes_gate_with_valid_token(auth_on, method, path):
    r = auth_on.request(method, path,
                        headers={"Authorization": f"Bearer {_token(auth_on)}"})
    assert r.status_code not in (401, 403)


@pytest.mark.parametrize("method,path", NEWLY_GUARDED)
def test_unchanged_when_auth_disabled(auth_off, method, path):
    assert auth_off.request(method, path).status_code != 401


@pytest.mark.parametrize("method,path", STILL_PUBLIC)
def test_public_routes_stay_public(auth_on, method, path):
    assert auth_on.request(method, path).status_code == 200
