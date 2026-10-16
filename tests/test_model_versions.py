"""
Model versions: the node keeps every accepted head and can roll back; the
aggregator never reuses a version number.

The gate stops a bad candidate it can measure. These cover the case it
cannot: a version that passed the gate and then behaved badly in the
field. Without an archive the only way back was to wait for the next
round, fourteen days later.
"""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def store(tmp_path):
    from src.federated.store import SampleStore

    s = SampleStore(tmp_path / "samples.jsonl")
    s.bootstrap("driveway", 1500, seed=3)
    return s


@pytest.fixture
def node(tmp_path, monkeypatch, store):
    from src.federated import model_state
    from src.federated.head import FederatedHead

    monkeypatch.setattr(model_state.settings, "DATA_DIR", tmp_path)
    app = SimpleNamespace(state=SimpleNamespace(
        fl_head=FederatedHead(seed=0), fl_store=store, fl_model_state=None))
    return app, model_state


def _trained(store, tmp_path, name, epochs=6):
    from src.federated.head import FederatedHead
    from src.federated.store import holdout_split

    X, y = store.labelled()
    Xt, yt, _, _ = holdout_split(X, y, seed=0)
    head = FederatedHead(seed=0)
    head.fit(Xt, yt, epochs=epochs)
    path = tmp_path / name
    head.save(path)
    return head, path


def _same(a, b):
    return all(np.allclose(x, y) for x, y in zip(a.get_weights(), b.get_weights()))


def test_every_accepted_version_is_archived(node, store, tmp_path):
    app, ms = node
    _, path = _trained(store, tmp_path, "cand.npz")
    ms.accept_candidate(app, path, "federated")
    ms.accept_candidate(app, path, "federated")  # level on both metrics: accepted

    assert (ms.versions_dir() / "fl-r1.npz").exists()
    assert (ms.versions_dir() / "fl-r2.npz").exists()
    labels = [v["label"] for v in ms.list_versions(app)]
    assert labels == ["fl-r2", "fl-r1"]  # newest first
    assert ms.list_versions(app)[0]["live"] is True


def test_rejected_candidate_is_not_archived(node, store, tmp_path):
    from src.federated.head import FederatedHead

    app, ms = node
    _, good = _trained(store, tmp_path, "good.npz")
    ms.accept_candidate(app, good, "federated")
    bad = tmp_path / "bad.npz"
    FederatedHead(seed=9).save(bad)
    decision = ms.accept_candidate(app, bad, "federated")

    assert decision["accepted"] is False
    assert [v["label"] for v in ms.list_versions(app)] == ["fl-r1"]


def test_rollback_restores_weights_as_a_new_version(node, store, tmp_path):
    from src.federated.head import FederatedHead

    app, ms = node
    first, p1 = _trained(store, tmp_path, "a.npz", epochs=6)
    ms.accept_candidate(app, p1, "federated")
    _, p2 = _trained(store, tmp_path, "b.npz", epochs=6)
    ms.accept_candidate(app, p2, "local-train")
    assert ms.model_version(app) == "local-r2"

    decision = ms.rollback(app, "fl-r1")

    assert decision["label"] == "rollback-r3"
    assert decision["restored_from"] == "fl-r1"
    assert decision["replaced"] == "local-r2"
    assert ms.model_version(app) == "rollback-r3"
    assert _same(app.state.fl_head, first)

    # It survives a restart: the live file and the state both say so.
    reloaded = FederatedHead(seed=1)
    reloaded.load(ms.head_path())
    assert _same(reloaded, first)
    assert ms._read_state()["label"] == "rollback-r3"
    assert (ms.versions_dir() / "rollback-r3.npz").exists()


def test_rollback_to_an_unknown_version_is_refused(node, store, tmp_path):
    app, ms = node
    _, p = _trained(store, tmp_path, "a.npz")
    ms.accept_candidate(app, p, "federated")
    for label in ("fl-r9", "", "../fl_head", "..\\fl_head"):
        with pytest.raises(KeyError):
            ms.rollback(app, label)
    assert ms.model_version(app) == "fl-r1"


def test_archive_keeps_only_the_newest(node, store, tmp_path, monkeypatch):
    import os

    app, ms = node
    monkeypatch.setattr(ms, "MAX_ARCHIVED", 2)
    _, p = _trained(store, tmp_path, "a.npz")
    for i in range(4):
        ms.accept_candidate(app, p, "federated")
        # mtime decides what is oldest; make it unambiguous on coarse clocks.
        f = ms.versions_dir() / f"fl-r{i + 1}.npz"
        os.utime(f, (1_000_000 + i, 1_000_000 + i))

    left = sorted(q.stem for q in ms.versions_dir().glob("*.npz"))
    assert left == ["fl-r3", "fl-r4"]


def test_rollback_route(node, store, tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.api.trial_routes import router

    app_ns, ms = node
    _, p = _trained(store, tmp_path, "a.npz")
    api = FastAPI()
    api.include_router(router)
    api.state.fl_head = app_ns.state.fl_head
    api.state.fl_store = store
    api.state.fl_model_state = None
    ms.accept_candidate(api, p, "federated")
    ms.accept_candidate(api, p, "federated")

    client = TestClient(api)
    listed = client.get("/api/federated/head/versions").json()
    assert listed["live"] == "fl-r2"
    assert [v["label"] for v in listed["versions"]] == ["fl-r2", "fl-r1"]

    assert client.post("/api/federated/head/rollback", json={"label": "nope"}).status_code == 404
    r = client.post("/api/federated/head/rollback", json={"label": "fl-r1"})
    assert r.status_code == 200
    assert r.json()["label"] == "rollback-r3"


# ---------------------------------------------------------------------------
# The aggregator
# ---------------------------------------------------------------------------

def _aggregator():
    """Import central_server/fl_aggregator.py once per session: it registers
    Prometheus metrics at import, and a second import would collide."""
    pytest.importorskip("flask")
    pytest.importorskip("flask_cors")
    pytest.importorskip("prometheus_client")
    name = "argus_fl_aggregator_under_test"
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).resolve().parents[1] / "central_server" / "fl_aggregator.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_aggregator_resumes_its_version_counter(tmp_path, monkeypatch):
    agg = _aggregator()
    monkeypatch.setattr(agg, "checkpoint_dir", tmp_path)
    monkeypatch.setattr(agg, "_model_version", 0)
    w = [np.zeros(3)]
    np.savez(tmp_path / "model_round_0001_v6.npz", w0=w[0])
    np.savez(tmp_path / "model_round_0002_v7.npz", w0=w[0])

    assert agg.resume_model_version() == 7
    agg.save_checkpoint(w, 1, {})

    # A restarted server continues at v8 instead of overwriting round 1.
    assert (tmp_path / "model_round_0001_v8.npz").exists()
    assert (tmp_path / "model_round_0001_v6.npz").exists()


def test_aggregator_version_never_moves_backwards(tmp_path, monkeypatch):
    agg = _aggregator()
    monkeypatch.setattr(agg, "checkpoint_dir", tmp_path)
    monkeypatch.setattr(agg, "_model_version", 12)
    np.savez(tmp_path / "model_round_0001_v3.npz", w0=np.zeros(1))
    assert agg.resume_model_version(5) == 12


# --- anomaly-recall gate -----------------------------------------------------

def test_scores_include_anomaly_recall(node, store, tmp_path):
    app, ms = node
    _, path = _trained(store, tmp_path, "cand.npz")

    decision = ms.accept_candidate(app, path, "federated")

    assert 0.0 <= decision["candidate"]["anomaly_recall"] <= 1.0
    assert "anomaly_recall" in decision["current"]


def test_candidate_below_anomaly_recall_floor_is_rejected(node, store, tmp_path):
    app, ms = node
    _, good = _trained(store, tmp_path, "good.npz")
    ms.accept_candidate(app, good, "federated")

    decision = ms.accept_candidate(app, good, "federated",
                                   min_anomaly_recall=1.01)

    assert decision["accepted"] is False
    assert "anomaly recall" in decision["reason"]


def test_first_model_must_also_clear_the_floor(node, store, tmp_path):
    app, ms = node
    _, good = _trained(store, tmp_path, "good.npz")

    decision = ms.accept_candidate(app, good, "federated",
                                   min_anomaly_recall=1.01)

    assert decision["accepted"] is False
    assert not ms.head_path().exists()


def test_default_floor_still_passes_a_normally_trained_model(node, store, tmp_path):
    app, ms = node
    _, good = _trained(store, tmp_path, "good.npz")

    decision = ms.accept_candidate(app, good, "federated")

    assert decision["accepted"] is True
    assert decision["candidate"]["anomaly_recall"] >= ms.settings.FL_MIN_ANOMALY_RECALL


def test_floor_is_skipped_when_holdout_has_no_anomalies(node, tmp_path, monkeypatch):
    """No anomaly rows means recall is unmeasurable, not zero."""
    from src.federated.head import FederatedHead

    head = FederatedHead(seed=0)
    X = np.random.default_rng(0).normal(size=(20, 19)).astype("float32")
    y = np.zeros(20, dtype=int)

    assert head.anomaly_recall(X, y) is None
