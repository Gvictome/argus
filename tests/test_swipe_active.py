"""
Swipe console support: active learning scores, confident auto-labels,
swipe and staged provenance.
"""

import numpy as np
import pytest

from src.federated.active import auto_label, score_rows
from src.federated.features import ANOMALY, LABEL_NAMES, N_FEATURES
from src.federated.store import (
    ReviewedLabelConflict, SampleStore, human_prov, is_reviewed, machine_prov, provenance,
)


def vec(cls=0, dwell=0.1):
    x = np.zeros(N_FEATURES, dtype=np.float32)
    x[cls] = 1.0
    x[13] = dwell
    return x


class RowHead:
    """Returns one probability row per event, in store order."""

    def __init__(self, rows):
        self.rows = np.array(rows, dtype=np.float32)

    def predict_proba(self, X):
        return self.rows[: len(X)]


def store_with(tmp_path, n=3):
    store = SampleStore(tmp_path / "samples.jsonl")
    for _ in range(n):
        store.append(vec())
    return store


SURE = [0.97, 0.01, 0.01, 0.0, 0.01]
UNSURE = [0.5, 0.0, 0.0, 0.0, 0.5]
MID = [0.9, 0.02, 0.03, 0.0, 0.05]


def test_scores_report_entropy_and_margin(tmp_path):
    store = store_with(tmp_path)
    out = score_rows(store, RowHead([SURE, UNSURE, MID]), "fl-r1")
    by_n = {s["n"]: s for s in out["scores"]}
    assert by_n[2]["entropy"] > by_n[3]["entropy"] > by_n[1]["entropy"]
    assert by_n[2]["margin"] == pytest.approx(0.0)
    assert by_n[1]["model_label"] == "routine_person"
    assert by_n[2]["p_anomaly"] == pytest.approx(0.5)
    assert len(by_n[1]["x"]) == N_FEATURES


def test_scores_without_model_have_no_probabilities(tmp_path):
    store = store_with(tmp_path)
    out = score_rows(store, RowHead([SURE] * 3), None)
    assert out["count"] == 3
    assert "confidence" not in out["scores"][0]


def test_scores_filter_by_time(tmp_path):
    store = store_with(tmp_path)
    for i, r in enumerate(store._rows):
        r["t"] = 100.0 + i
    out = score_rows(store, RowHead([SURE] * 3), "fl-r1", since=101, until=102)
    assert [s["n"] for s in out["scores"]] == [2]


def test_auto_label_only_confident_unlabelled_and_marks_machine(tmp_path):
    store = store_with(tmp_path)
    out = auto_label(store, RowHead([SURE, UNSURE, MID]), "fl-r1", 0.95)
    assert out["labelled"] == 1
    row = store.by_n(1)
    assert row["y"] == 0
    prov = provenance(row)
    assert prov["origin"] == "machine" and prov["reviewed"] is False
    assert prov["source"] == "auto:model:fl-r1@0.95"
    assert store.by_n(2)["y"] is None and store.by_n(3)["y"] is None
    # Never counted as human.
    X, _ = store.labelled(human_only=True)
    assert len(X) == 0
    assert store.provenance_counts()["machine_auto"] == 1


def test_auto_label_skips_excluded_and_dry_run_writes_nothing(tmp_path):
    store = store_with(tmp_path, 2)
    out = auto_label(store, RowHead([SURE, SURE]), "fl-r1", 0.95, exclude=[1], dry_run=True)
    assert out["labelled"] == 1 and out["events"][0]["n"] == 2
    assert all(r["y"] is None for r in store._rows)


def test_auto_label_needs_a_model(tmp_path):
    store = store_with(tmp_path)
    assert auto_label(store, RowHead([SURE] * 3), None, 0.95)["labelled"] == 0


def test_auto_label_never_touches_reviewed_or_labelled(tmp_path):
    store = store_with(tmp_path, 2)
    store.set_label_by_n(1, ANOMALY, prov=human_prov(ANOMALY, "blind", source="swipe"))
    out = auto_label(store, RowHead([SURE]), "fl-r1", 0.95)
    assert out["considered"] == 1
    assert store.by_n(1)["y"] == ANOMALY


def test_machine_write_cannot_replace_reviewed(tmp_path):
    store = store_with(tmp_path, 1)
    store.set_label_by_n(1, 0, prov=human_prov(0, "blind", source="swipe"))
    with pytest.raises(ReviewedLabelConflict):
        store.set_label_by_n(1, ANOMALY, prov=machine_prov("auto:x"))


def test_swipe_replaces_auto_label(tmp_path):
    store = store_with(tmp_path, 1)
    auto_label(store, RowHead([SURE]), "fl-r1", 0.95)
    store.set_label_by_n(1, ANOMALY, prov=human_prov(ANOMALY, "blind", source="swipe",
                                                     reviewer="Gio", via="swipe:uncertain"))
    prov = provenance(store.by_n(1))
    assert prov["source"] == "swipe" and prov["reviewer"] == "Gio"
    assert prov["origin"] == "human" and is_reviewed(store.by_n(1))


def test_swipe_accept_keeps_machine_origin(tmp_path):
    prov = human_prov(0, "accept", 0, "model:fl-r1", via="swipe", reviewer="Gio", source="swipe")
    assert prov["origin"] == "machine" and prov["reviewed"] is True
    assert prov["source"] == "model:fl-r1" and prov["reviewer"] == "Gio"


def test_staged_is_anomaly_only():
    prov = human_prov(ANOMALY, "staged", via="staging:abc", reviewer="Gio")
    assert prov["source"] == "staged" and prov["review"] == "staged"
    with pytest.raises(ValueError):
        human_prov(0, "staged")


def test_unknown_source_rejected():
    with pytest.raises(ValueError):
        human_prov(0, "blind", source="script")


def test_staged_counted_separately(tmp_path):
    store = store_with(tmp_path, 1)
    store.set_label_by_n(1, ANOMALY, prov=human_prov(ANOMALY, "staged"))
    counts = store.provenance_counts()
    assert counts["reviewed_staged"] == 1 and counts["reviewed_anomalies"] == 1


def test_label_route_records_swipe_reviewer(tmp_path, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from src.api import dashboard_routes

    app = FastAPI()
    app.include_router(dashboard_routes.router)
    store = store_with(tmp_path, 1)
    app.state.fl_store = store
    app.state.fl_head = None
    monkeypatch.setattr(dashboard_routes, "_event_from_row",
                        lambda row, head: {"status": "member", "cls": "person"})
    client = TestClient(app)
    res = client.post("/api/events/1/label", json={
        "action": "correct", "label": "anomaly",
        "review": {"kind": "blind", "via": "swipe:uncertain", "reviewer": "Gio",
                   "source": "swipe"}})
    assert res.status_code == 200, res.text
    prov = res.json()["provenance"]
    assert prov["source"] == "swipe" and prov["reviewer"] == "Gio"
    res = client.post("/api/events/1/label", json={
        "action": "correct", "label": "anomaly",
        "review": {"kind": "staged", "via": "staging:s1"}})
    assert res.status_code == 200 and res.json()["provenance"]["source"] == "staged"
