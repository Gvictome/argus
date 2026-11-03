"""
Gate B fix F3: /api/federated/samples/bootstrap must not block the event loop
and must cap `n`.

st.bootstrap() is synchronous and took ~110 s at the default n, freezing the
live stream and every other route. `n` had no upper bound, so a large value
could also fill the SD card.
"""

import asyncio

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.config import settings


class FakeStore:
    def __init__(self):
        self.calls = []

    def bootstrap(self, site, n, seed=0):
        try:
            asyncio.get_running_loop()
            on_loop = True
        except RuntimeError:
            on_loop = False
        self.calls.append({"site": site, "n": n, "seed": seed, "on_loop": on_loop})
        return n

    def stats(self):
        return {"total": sum(c["n"] for c in self.calls)}


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "AUTH_REQUIRED", False)
    monkeypatch.setattr(settings, "DB_PATH", tmp_path / "f.db")
    monkeypatch.setattr(settings, "FL_TRAINING_DIR", tmp_path / "training")
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    with TestClient(create_app()) as c:
        c.store = FakeStore()
        c.app.state.fl_store = c.store
        yield c


def test_bootstrap_runs_off_the_event_loop(client):
    r = client.post("/api/federated/samples/bootstrap", json={"n": 10})
    assert r.status_code == 200
    assert r.json()["added"] == 10
    assert client.store.calls == [
        {"site": "driveway", "n": 10, "seed": 0, "on_loop": False}
    ]


def test_bootstrap_accepts_the_cap(client):
    r = client.post("/api/federated/samples/bootstrap", json={"n": 20000})
    assert r.status_code == 200


@pytest.mark.parametrize("n", [20001, 10**9, 0, -5])
def test_bootstrap_rejects_out_of_range_n(client, n):
    r = client.post("/api/federated/samples/bootstrap", json={"n": n})
    assert r.status_code == 422
    assert client.store.calls == []
