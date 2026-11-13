"""Cloud Connect: node uploader against the real central events server."""

import importlib
import json
import os
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pytest

uvicorn = pytest.importorskip("uvicorn")

from src.cloud.uploader import CloudUploader  # noqa: E402

TOKEN = "test-token"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("ARGUS_EVENTS_DIR", str(tmp_path / "events"))
    monkeypatch.setenv("ARGUS_SAVED_DIR", str(tmp_path / "saved"))
    monkeypatch.setenv("ARGUS_EVENTS_STATE", str(tmp_path / "state"))
    monkeypatch.setenv("ARGUS_EVENTS_TOKEN", TOKEN)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "central_server"))
    import events_server
    events_server = importlib.reload(events_server)
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(events_server.app, host="127.0.0.1",
                                           port=port, log_level="warning"))
    up = CloudUploader(f"http://127.0.0.1:{port}", "Node C", tmp_path / "queue",
                       token=TOKEN, pre_roll_s=5)
    up.ffmpeg = None    # no transcode in CI: the raw file uploads as is
    yield SimpleNamespace(tmp=tmp_path, server=server, up=up, base=f"http://127.0.0.1:{port}",
                          mod=events_server)
    server.should_exit = True


def _start(server):
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(50):
        if server.started:
            return
        time.sleep(0.1)
    raise RuntimeError("server did not start")


def _clip(tmp, name, started, trigger="human"):
    raw = tmp / "raw"
    raw.mkdir(exist_ok=True)
    p = raw / f"{name}.mp4"
    p.write_bytes(os.urandom(20000))
    return SimpleNamespace(path=str(p), started_at=started, ended_at=started,
                           duration_s=12.0, trigger=trigger, peak_confidence=0.9)


def _req(base, path, method="GET", data=None, headers=None):
    r = urllib.request.Request(base + path, method=method, data=data, headers=headers or {})
    with urllib.request.urlopen(r) as resp:
        return resp.status, resp.headers, resp.read()


def test_queue_survives_server_down_and_keeper_delete(env):
    rec = _clip(env.tmp, "a", "2026-11-12T22:40:01")
    env.up.enqueue(rec)
    os.remove(rec.path)                 # the clip keeper deletes its copy
    env.up.run_once()                   # server not running yet
    assert env.up.status()["queued"] == 1
    assert env.up.last_error

    _start(env.server)
    env.up._retry_at.clear()
    env.up.run_once()
    assert env.up.status()["queued"] == 0
    assert env.up.uploaded == 1

    _, _, body = _req(env.base, "/api/events")
    events = json.loads(body)["events"]
    assert [e["node"] for e in events] == ["Node C"]
    eid = events[0]["id"]
    clip = env.tmp / "events" / "Node C" / "2026-11-12" / f"{eid}.mp4"
    assert clip.exists() and (clip.with_suffix(".json")).exists()

    status, headers, part = _req(env.base, f"/api/events/{eid}/clip", headers={"Range": "bytes=0-99"})
    assert status == 206 and len(part) == 100


def test_vehicle_skipped_unless_anomaly(env):
    env.up.enqueue(_clip(env.tmp, "v", "2026-11-12T10:00:00", trigger="vehicle"))
    assert env.up.status()["queued"] == 0
    p = Path(_clip(env.tmp, "v2", "2026-11-12T10:00:00", trigger="vehicle").path)
    env.up.enqueue_anomaly(p, time.time() - 20, time.time(), "vehicle", 0.7)
    assert env.up.status()["queued"] == 1
    meta = json.loads(next((env.tmp / "queue").glob("*/meta.json")).read_text())
    assert meta["event_type"] == "anomaly" and meta["trigger"] == "vehicle"


def test_save_retention_and_auth(env):
    _start(env.server)
    for name, day in (("new", "2026-11-12T08:00:00"), ("old", "2026-01-01T08:00:00"),
                      ("old2", "2026-01-02T08:00:00")):
        env.up.enqueue(_clip(env.tmp, name, day))
    env.up.run_once()
    _, _, body = _req(env.base, "/api/events")
    ids = {e["started_at"][:10]: e["id"] for e in json.loads(body)["events"]}

    # Saving moves the files out of the retention folder.
    _req(env.base, f"/api/events/{ids['2026-01-02']}/save", "POST")
    assert list((env.tmp / "saved" / "Node C" / "2026-01-02").glob("*.mp4"))

    _, _, body = _req(env.base, "/api/settings", "PUT",
                      json.dumps({"retention_days": 30}).encode(),
                      {"Content-Type": "application/json"})
    assert json.loads(body)["deleted_now"] == 1      # old, unsaved

    _, _, body = _req(env.base, "/api/events")
    left = {e["started_at"][:10]: e["saved"] for e in json.loads(body)["events"]}
    assert left == {"2026-11-12": False, "2026-01-02": True}

    # Index rebuilds from the Dropbox sidecars.
    os.remove(env.tmp / "state" / "events.db")
    env.mod.init_db()
    _, _, body = _req(env.base, "/api/events")
    assert json.loads(body)["total"] == 2

    with pytest.raises(urllib.error.HTTPError) as err:
        _req(env.base, "/api/ingest/node-c-x/thumb", "PUT", b"x")
    assert err.value.code == 401


def test_upload_types_set_on_console_reach_node_without_restart(env):
    _start(env.server)
    env.up.known_types = ["anomaly", "human", "vehicle"]
    env.up.refresh_config(force=True)
    assert env.up.config_source == "server"
    assert not env.up.wants("vehicle")                       # default: off

    _, _, body = _req(env.base, "/api/settings")
    node = json.loads(body)["nodes"][0]
    assert node["node"] == "Node C"
    assert node["known_types"] == ["anomaly", "human", "vehicle"]

    _req(env.base, "/api/settings", "PUT",
         json.dumps({"upload_types": {"Node C": ["vehicle"]}}).encode(),
         {"Content-Type": "application/json"})
    env.up.refresh_config(force=True)
    assert env.up.wants("vehicle") and not env.up.wants("human")

    env.up.enqueue(_clip(env.tmp, "h", "2026-11-12T10:00:00", trigger="human"))
    assert env.up.status()["queued"] == 0
    p = Path(_clip(env.tmp, "a", "2026-11-12T10:00:00", trigger="human").path)
    env.up.enqueue_anomaly(p, time.time() - 20, time.time(), "human", 0.7)
    assert env.up.status()["queued"] == 0                    # anomaly off too
    env.up.enqueue(_clip(env.tmp, "v", "2026-11-12T10:00:00", trigger="vehicle"))
    assert env.up.status()["queued"] == 1

    # The list survives a node restart with the server down.
    from src.cloud.uploader import CloudUploader
    again = CloudUploader("http://127.0.0.1:9", "Node C", env.tmp / "queue", token=TOKEN)
    assert again.config_source == "cache" and again.wants("vehicle")
