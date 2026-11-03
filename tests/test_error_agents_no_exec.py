"""
Gate B fix F1: the error agents must never run a command built from posted text.

POST /api/errors/fix used to pass "pip install <module>" to a shell, with the
module name taken straight from the error text, so
"No module named 'x; echo hi'" ran `echo hi`. The route is gone and the
agents only run fixed argument lists with shell=False.
"""

import subprocess
import sys

import pytest
from fastapi.testclient import TestClient

from src.agents import error_router as error_router_module
from src.agents.dependency_agent import DependencyAgent
from src.agents.error_router import BaseErrorAgent, ErrorRouter
from src.agents.network_agent import NetworkAgent
from src.api.app import create_app
from src.config import settings

INJECTED = "ModuleNotFoundError: No module named 'x; echo hi'"


@pytest.fixture
def recorded_runs(monkeypatch):
    """Replace subprocess.run in the agents with a recorder that runs nothing."""
    calls = []

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(error_router_module.subprocess, "run", fake_run)
    return calls


def test_injected_module_name_is_not_auto_fixable():
    report = ErrorRouter().analyze(INJECTED)
    assert report.subcategory == "missing_module"
    assert report.auto_fixable is False
    assert "echo" not in (report.suggested_fix or "")


def test_injected_module_name_never_runs_a_command(recorded_runs):
    router = ErrorRouter()
    router.fix(INJECTED)
    router.fix(INJECTED, auto_approve=True)
    DependencyAgent().execute_fix(router.analyze(INJECTED))
    assert recorded_runs == []


def test_valid_module_runs_as_argument_list_without_shell(recorded_runs):
    report = ErrorRouter().analyze("No module named 'requests'")
    result = DependencyAgent().execute_fix(report)
    assert result.success
    assert len(recorded_runs) == 1
    args, kwargs = recorded_runs[0]
    assert args == [sys.executable, "-m", "pip", "install", "requests"]
    assert kwargs.get("shell", False) is False


def test_missing_module_needs_explicit_approval(recorded_runs):
    result = ErrorRouter().fix("No module named 'requests'")
    assert result.success is False
    assert recorded_runs == []


@pytest.mark.parametrize("name", [
    "x; echo hi", "a b", "$(id)", "`id`", "x|y", "x&&y", "", "a" * 65, "-e",
])
def test_module_allowlist_rejects_unsafe_names(name):
    assert DependencyAgent.is_safe_module_name(name) is False


@pytest.mark.parametrize("name", ["requests", "opencv-python", "ruamel.yaml", "a_b"])
def test_module_allowlist_accepts_normal_names(name):
    assert DependencyAgent.is_safe_module_name(name) is True


def test_run_command_refuses_a_string(recorded_runs):
    ok, msg = BaseErrorAgent(None).run_command("echo hi")
    assert ok is False
    assert recorded_runs == []


def test_port_conflict_never_kills_a_process(recorded_runs):
    text = "OSError: [Errno 98] address already in use: port 22"
    report = ErrorRouter().analyze(text)
    assert report.auto_fixable is False
    NetworkAgent().execute_fix(report)
    ErrorRouter().fix(text, auto_approve=True)
    for args, kwargs in recorded_runs:
        assert isinstance(args, list)
        assert kwargs.get("shell", False) is False
        assert not {"kill", "taskkill"} & {str(a).lower() for a in args}


@pytest.mark.parametrize("auth_required", [False, True])
def test_errors_fix_route_is_gone(monkeypatch, tmp_path, recorded_runs, auth_required):
    monkeypatch.setattr(settings, "AUTH_REQUIRED", auth_required)
    monkeypatch.setattr(settings, "DB_PATH", tmp_path / "e.db")
    monkeypatch.setattr(settings, "FL_TRAINING_DIR", tmp_path / "training")
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    monkeypatch.setenv("ARGUS_ADMIN_PASSWORD", "correct-horse-battery")
    with TestClient(create_app()) as client:
        headers = {}
        if auth_required:
            token = client.post(
                "/api/auth/login",
                json={"username": "admin", "password": "correct-horse-battery"},
            ).json()["token"]
            headers = {"Authorization": f"Bearer {token}"}
        r = client.post(
            "/api/errors/fix",
            json={"error_message": INJECTED, "auto_fix": True},
            headers=headers,
        )
        assert r.status_code in (404, 405)
        # analyze still works and only suggests
        r = client.post(
            "/api/errors/analyze", json={"error_message": INJECTED}, headers=headers
        )
        assert r.status_code == 200
        assert r.json()["auto_fixable"] is False
    assert recorded_runs == []
