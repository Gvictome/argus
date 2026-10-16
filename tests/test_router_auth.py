"""
Auth coverage for the trial and dashboard routers.

routes.py guarded its endpoints with require_auth, but trial_router and
dashboard_router were mounted bare, so with AUTH_REQUIRED=true the
federated and labelling endpoints were open through the tunnel. These tests
walk every route the app mounts so a future router cannot repeat that.
"""

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from src.api.app import create_app
from src.api.auth import require_auth
from src.config import settings

# Deliberately public. Login must be (it mints the token); the dashboard
# status alias mirrors /api/status, which routes.py already leaves open.
PUBLIC = {
    ("POST", "/api/login"),
    ("GET", "/api/dashboard/status"),
}

GUARDED = [
    ("GET", "/api/federated/samples"),
    ("POST", "/api/federated/samples/1/label"),
    ("POST", "/api/federated/samples/bootstrap"),
    ("GET", "/api/federated/head"),
    ("GET", "/api/federated/head/versions"),
    ("POST", "/api/federated/head/rollback"),
    ("POST", "/api/federated/head/train"),
    ("POST", "/api/federated/predict"),
    ("POST", "/api/federated/round"),
    ("GET", "/api/federated/round"),
    ("POST", "/api/events/1/label"),
    # routes.py: auto_fix executes remediation, so it must not be open.
    ("POST", "/api/errors/analyze"),
    ("POST", "/api/errors/fix"),
    ("GET", "/api/errors/history"),
]


def _login(client):
    r = client.post("/api/auth/login",
                    json={"username": "admin", "password": "correct-horse-battery"})
    return r.json()["token"]


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


@pytest.mark.parametrize("method,path", GUARDED)
def test_rejected_without_token(auth_on, method, path):
    assert auth_on.request(method, path, json={}).status_code == 401


@pytest.mark.parametrize("method,path", GUARDED)
def test_rejected_with_bad_token(auth_on, method, path):
    r = auth_on.request(method, path, json={},
                        headers={"Authorization": "Bearer nope"})
    assert r.status_code == 401


@pytest.mark.parametrize("method,path", GUARDED)
def test_passes_gate_with_valid_token(auth_on, method, path):
    token = _login(auth_on)
    r = auth_on.request(method, path, json={},
                        headers={"Authorization": f"Bearer {token}"})
    assert r.status_code not in (401, 403)


def test_authenticated_read_is_200(auth_on):
    token = _login(auth_on)
    r = auth_on.get("/api/federated/head",
                    headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200


@pytest.mark.parametrize("method,path", GUARDED)
def test_unchanged_when_auth_disabled(auth_off, method, path):
    assert auth_off.request(method, path, json={}).status_code != 401


def test_public_endpoints_stay_public(auth_on):
    assert auth_on.get("/api/dashboard/status").status_code == 200
    r = auth_on.post("/api/login", json={"username": "admin", "password": "x"})
    assert r.status_code == 401  # bad creds, but reached the handler


def test_events_websocket_requires_token(auth_on):
    with pytest.raises(WebSocketDisconnect):
        with auth_on.websocket_connect("/ws/events"):
            pass
    with pytest.raises(WebSocketDisconnect):
        with auth_on.websocket_connect("/ws/events?token=nope"):
            pass


def test_events_websocket_accepts_valid_token(auth_on):
    token = _login(auth_on)
    with auth_on.websocket_connect(f"/ws/events?token={token}"):
        pass


def test_events_websocket_open_when_auth_disabled(auth_off):
    with auth_off.websocket_connect("/ws/events"):
        pass


def test_every_mounted_http_route_is_guarded_or_allowlisted(auth_on):
    """Fail closed: a new unguarded route breaks this until it is decided."""
    from fastapi.routing import APIRoute

    # Read-only status and listing endpoints that routes.py already leaves
    # open (a decision for the owner, flagged in the hardening notes), plus
    # the auth endpoints themselves.
    open_paths = {
        "/", "/api/status", "/api/detection/status", "/api/auth/login",
        "/api/auth/me", "/api/auth/logout", "/api/login",
        "/api/dashboard/status", "/api/camera/status", "/api/cameras",
        "/api/cameras/{name}/status", "/api/federated/status",
        "/api/devices", "/api/devices/{device_id}", "/api/automations",
        "/api/events", "/api/events/{event_id}", "/api/agents/status",
    }
    leaks = []
    for route in auth_on.app.routes:
        if not isinstance(route, APIRoute) or not route.path.startswith("/api"):
            continue
        if route.path in open_paths:
            continue
        deps = {d.call for d in route.dependant.dependencies}
        if require_auth not in deps:
            leaks.append(route.path)
    assert leaks == []
