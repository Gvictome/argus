"""Login brute-force protection: N failures in a window -> 429."""

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.rate_limit import LoginRateLimiter, reset_login_limiter
from src.config import settings

GOOD = "correct-horse-battery"


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "AUTH_REQUIRED", True)
    monkeypatch.setattr(settings, "DB_PATH", tmp_path / "a.db")
    monkeypatch.setattr(settings, "FL_TRAINING_DIR", tmp_path / "training")
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    monkeypatch.setattr(settings, "LOGIN_MAX_ATTEMPTS", 5)
    monkeypatch.setattr(settings, "LOGIN_WINDOW_SECONDS", 300)
    monkeypatch.setenv("ARGUS_ADMIN_PASSWORD", GOOD)
    reset_login_limiter()
    with TestClient(create_app()) as c:
        yield c


def _login(c, user="admin", pw="wrong", path="/api/auth/login"):
    return c.post(path, json={"username": user, "password": pw})


def test_five_failures_then_429(client):
    for _ in range(5):
        assert _login(client).status_code == 401
    r = _login(client)
    assert r.status_code == 429
    assert int(r.headers["Retry-After"]) > 0


def test_correct_password_blocked_while_locked(client):
    for _ in range(5):
        _login(client)
    assert _login(client, pw=GOOD).status_code == 429


def test_dashboard_alias_shares_the_limit(client):
    for _ in range(5):
        _login(client, path="/api/login")
    assert _login(client, path="/api/login").status_code == 429
    assert _login(client).status_code == 429


def test_success_resets_username_counter(client):
    for _ in range(4):
        _login(client)
    assert _login(client, pw=GOOD).status_code == 200
    from src.api.rate_limit import login_limiter

    # Username counter cleared; the IP counter (4 failures) is not.
    assert login_limiter.check("other-ip", "admin", 5, 300) == 0
    assert login_limiter.check("testclient", "nobody", 5, 300) == 0


def test_per_username_limit_across_ips(client):
    from src.api.rate_limit import login_limiter

    for i in range(5):
        login_limiter.record_failure(f"10.0.0.{i}", "admin")
    # A fresh IP is still locked out for the targeted username.
    assert _login(client, user="admin", pw=GOOD).status_code == 429
    assert _login(client, user="someone-else").status_code == 401


def test_per_ip_limit_across_usernames(client):
    for i in range(5):
        assert _login(client, user=f"user{i}").status_code == 401
    assert _login(client, user="user99").status_code == 429


def test_limits_are_configurable(client, monkeypatch):
    monkeypatch.setattr(settings, "LOGIN_MAX_ATTEMPTS", 2)
    assert _login(client).status_code == 401
    assert _login(client).status_code == 401
    assert _login(client).status_code == 429


def test_reset_clears_state(client):
    for _ in range(5):
        _login(client)
    reset_login_limiter()
    assert _login(client, pw=GOOD).status_code == 200


def test_window_expiry_unlocks():
    now = [1000.0]
    lim = LoginRateLimiter(clock=lambda: now[0])
    for _ in range(5):
        lim.record_failure("1.2.3.4", "bob")
    assert lim.check("1.2.3.4", "bob", 5, 300) > 0
    now[0] += 301
    assert lim.check("1.2.3.4", "bob", 5, 300) == 0


def test_username_is_case_insensitive():
    lim = LoginRateLimiter()
    for _ in range(5):
        lim.record_failure("1.1.1.1", "Admin")
    assert lim.check("2.2.2.2", "ADMIN", 5, 300) > 0
