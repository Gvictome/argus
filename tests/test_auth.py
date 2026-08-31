"""
Authentication tests.

Every endpoint was open before this, including the live camera stream and
face deletion. These lock the gate shut, and equally lock in that turning
auth off restores the previous LAN behavior exactly.
"""

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.config import settings


@pytest.fixture
def auth_on(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "AUTH_REQUIRED", True)
    monkeypatch.setattr(settings, "DB_PATH", tmp_path / "auth.db")
    monkeypatch.setenv("ARGUS_ADMIN_PASSWORD", "correct-horse-battery")
    with TestClient(create_app()) as client:
        yield client


@pytest.fixture
def auth_off(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "AUTH_REQUIRED", False)
    monkeypatch.setattr(settings, "DB_PATH", tmp_path / "noauth.db")
    with TestClient(create_app()) as client:
        yield client


def _login(client, username="admin", password="correct-horse-battery"):
    return client.post("/api/auth/login", json={"username": username, "password": password})


class TestLogin:
    def test_valid_credentials_return_a_token(self, auth_on):
        r = _login(auth_on)

        assert r.status_code == 200
        body = r.json()
        assert body["token"]
        assert body["token_type"] == "bearer"
        assert body["role"] == "admin"

    def test_wrong_password_is_rejected(self, auth_on):
        assert _login(auth_on, password="hunter2").status_code == 401

    def test_unknown_user_is_rejected(self, auth_on):
        assert _login(auth_on, username="nobody").status_code == 401

    def test_unknown_user_and_wrong_password_are_indistinguishable(self, auth_on):
        """
        Different messages would confirm which usernames exist, turning
        the login form into a user-enumeration oracle.
        """
        bad_user = _login(auth_on, username="nobody")
        bad_pass = _login(auth_on, password="hunter2")

        assert bad_user.status_code == bad_pass.status_code
        assert bad_user.json()["detail"] == bad_pass.json()["detail"]

    def test_password_is_not_echoed(self, auth_on):
        assert "correct-horse-battery" not in _login(auth_on).text


class TestProtectedEndpoints:
    """The endpoints that read the camera or mutate state."""

    PROTECTED = [
        ("get", "/api/faces"),
        ("post", "/api/faces?name=x"),
        ("post", "/api/faces/reset"),
        ("delete", "/api/faces/abc"),
        ("get", "/api/camera/snapshot"),
        ("post", "/api/camera/init"),
        ("post", "/api/camera/shutdown"),
        ("get", "/api/logs"),
    ]

    @pytest.mark.parametrize("method,path", PROTECTED)
    def test_rejected_without_a_token(self, auth_on, method, path):
        assert getattr(auth_on, method)(path).status_code == 401

    @pytest.mark.parametrize("method,path", PROTECTED)
    def test_not_401_with_a_valid_token(self, auth_on, method, path):
        """
        A valid token must get past the gate. The endpoint may still fail
        for its own reasons (no camera on a test runner) -- what matters
        is that it is no longer an auth failure.
        """
        token = _login(auth_on).json()["token"]
        r = getattr(auth_on, method)(path, headers={"Authorization": f"Bearer {token}"})

        assert r.status_code != 401

    def test_invalid_token_is_rejected(self, auth_on):
        r = auth_on.get("/api/faces", headers={"Authorization": "Bearer not-a-real-token"})

        assert r.status_code == 401

    def test_stream_accepts_a_query_token(self, auth_on):
        """
        A browser <img> cannot set an Authorization header, so the stream
        also accepts ?token=. Without this the dashboard cannot show video.
        """
        token = _login(auth_on).json()["token"]

        assert auth_on.get(f"/api/camera/snapshot?token={token}").status_code != 401

    def test_stream_rejects_a_bad_query_token(self, auth_on):
        assert auth_on.get("/api/camera/snapshot?token=nope").status_code == 401


class TestOpenEndpoints:
    """Health and status stay open: they leak nothing and back the demo UI."""

    @pytest.mark.parametrize("path", ["/", "/api/status", "/api/detection/status"])
    def test_reachable_without_a_token(self, auth_on, path):
        assert auth_on.get(path).status_code == 200


class TestLogout:
    def test_logout_revokes_the_token(self, auth_on):
        token = _login(auth_on).json()["token"]
        headers = {"Authorization": f"Bearer {token}"}
        assert auth_on.get("/api/faces", headers=headers).status_code != 401

        auth_on.post("/api/auth/logout", headers=headers)

        assert auth_on.get("/api/faces", headers=headers).status_code == 401


class TestAuthDisabled:
    """
    AUTH_REQUIRED=false must restore the previous behavior exactly. This
    is the demo's escape hatch and Gate A depends on it.
    """

    @pytest.mark.parametrize("path", ["/api/faces", "/api/detection/status"])
    def test_no_token_needed(self, auth_off, path):
        assert auth_off.get(path).status_code != 401

    def test_me_reports_auth_is_off(self, auth_off):
        body = auth_off.get("/api/auth/me").json()

        assert body["auth_required"] is False
        assert body["authenticated"] is False


class TestPasswordHashing:
    def test_passwords_are_not_stored_in_plain_text(self, auth_on, tmp_path):
        from src.database import Database

        db = Database(settings.DB_PATH)
        db.initialize()
        user = db.get_user_by_username("admin")
        db.shutdown()

        assert user is not None
        assert "correct-horse-battery" not in user["password_hash"]
        assert "$" in user["password_hash"], "expected salt$hash from PBKDF2"

    def test_same_password_hashes_differently(self):
        """Per-password salt: identical passwords must not collide."""
        from src.security import SecurityService

        s = SecurityService(secret_key="k")
        assert s.hash_password("same") != s.hash_password("same")
        assert s.verify_password("same", s.hash_password("same"))
