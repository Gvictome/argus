"""
Label provenance, machine suggestions and human-only training.

The first live federated run scored about 98% on labels derived from dwell
while dwell was also an input. These keep that from being repeated quietly:
a label has to say who decided it, a suggestion can never become a label on
its own, and a human-only run sees nothing a person did not review.
"""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from src.federated.features import ANOMALY, LABEL_NAMES, N_FEATURES
from src.federated.store import (
    ReviewedLabelConflict, SampleStore, human_prov, is_reviewed, machine_prov, provenance,
)
from src.federated.suggest import (
    RULE_SOURCE, SuggestionStore, generate_suggestions, rule_label,
)


def vec(cls=0, dwell=0.1):
    x = np.zeros(N_FEATURES, dtype=np.float32)
    x[cls] = 1.0
    x[13] = dwell
    return x


def legacy_file(tmp_path, labels):
    path = tmp_path / "samples.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for i, label in enumerate(labels, start=1):
            f.write(json.dumps({"id": f"s{i}", "n": i, "t": float(i),
                                "x": [float(v) for v in vec()], "y": label,
                                "meta": {}}) + "\n")
    return path


class FixedHead:
    """Calls the given probabilities for every event."""

    def __init__(self, probs):
        self.probs = np.array(probs, dtype=np.float32)

    def predict_proba(self, X):
        return np.tile(self.probs, (len(X), 1))


# --- Backward compatibility --------------------------------------------------

def test_existing_rows_without_provenance_are_unreviewed_machine(tmp_path):
    store = SampleStore(legacy_file(tmp_path, [0, 4, None]))
    labelled, unlabelled = store._rows[0], store._rows[2]
    prov = provenance(labelled)
    assert prov["origin"] == "machine"
    assert prov["reviewed"] is False
    assert prov["source"].startswith("legacy")
    assert provenance(unlabelled) is None


def test_old_rows_keep_working_for_ordinary_training_and_stay_unchanged_on_disk(tmp_path):
    path = legacy_file(tmp_path, [0, 4, None])
    before = path.read_text(encoding="utf-8")
    store = SampleStore(path)
    X, y = store.labelled()
    assert len(X) == 2
    assert store.stats()["provenance"]["machine_unreviewed"] == 2
    # Loading and reading must not rewrite anyone's data file.
    assert path.read_text(encoding="utf-8") == before


def test_human_only_excludes_legacy_rows(tmp_path):
    store = SampleStore(legacy_file(tmp_path, [0, 4, None]))
    X, _ = store.labelled(human_only=True)
    assert len(X) == 0


# --- Attribution -------------------------------------------------------------

def test_write_without_provenance_is_machine_unattributed(tmp_path):
    store = SampleStore(legacy_file(tmp_path, [None]))
    assert store.set_label_by_n(1, 0)
    prov = provenance(store.by_n(1))
    assert (prov["origin"], prov["reviewed"]) == ("machine", False)
    assert prov["source"] == "api:unattributed"


def test_unattributed_write_cannot_replace_a_review(tmp_path):
    store = SampleStore(legacy_file(tmp_path, [None]))
    store.set_label_by_n(1, ANOMALY, prov=human_prov(ANOMALY, "blind"))
    with pytest.raises(ReviewedLabelConflict):
        store.set_label_by_n(1, 0)
    with pytest.raises(ReviewedLabelConflict):
        store.set_label("s1", 0)
    assert store.by_n(1)["y"] == ANOMALY


def test_a_later_person_can_change_a_review(tmp_path):
    store = SampleStore(legacy_file(tmp_path, [None]))
    store.set_label_by_n(1, ANOMALY, prov=human_prov(ANOMALY, "blind"))
    store.set_label_by_n(1, 0, prov=human_prov(0, "blind"))
    assert store.by_n(1)["y"] == 0


def test_accepted_suggestion_keeps_its_machine_source(tmp_path):
    prov = human_prov(ANOMALY, "accept", ANOMALY, "rule:x", via="review")
    assert prov["origin"] == "machine"
    assert prov["source"] == "rule:x"
    assert prov["reviewed"] is True


def test_correction_is_human_and_records_what_was_shown():
    prov = human_prov(0, "correct", ANOMALY, "model:fl-r1")
    assert (prov["origin"], prov["source"], prov["reviewed"]) == ("human", "operator", True)
    assert prov["suggested_label"] == "anomaly"


@pytest.mark.parametrize("args", [
    (0, "accept", ANOMALY, "rule:x"),    # accepting something else
    (0, "correct", 0, "rule:x"),         # "correcting" to the same label
    (0, "accept", None, None),           # accepting nothing
    (0, "looked", None, None),           # not a review kind
])
def test_inconsistent_reviews_are_refused(args):
    with pytest.raises(ValueError):
        human_prov(*args)


def test_provenance_survives_a_restart(tmp_path):
    path = legacy_file(tmp_path, [None])
    SampleStore(path).set_label_by_n(1, 0, prov=human_prov(0, "blind"))
    assert is_reviewed(SampleStore(path).by_n(1))


def test_counts_separate_how_a_review_was_made(tmp_path):
    store = SampleStore(legacy_file(tmp_path, [None] * 4 + [0]))
    store.set_label_by_n(1, 0, prov=human_prov(0, "blind"))
    store.set_label_by_n(2, ANOMALY, prov=human_prov(ANOMALY, "correct", 0, "rule:x"))
    store.set_label_by_n(3, 0, prov=human_prov(0, "accept", 0, "rule:x"))
    assert store.provenance_counts() == {
        "reviewed": 3, "reviewed_blind": 1, "reviewed_corrected": 1,
        "reviewed_accepted": 1, "reviewed_anomalies": 1, "machine_unreviewed": 1,
    }


# --- Suggestions --------------------------------------------------------------

def stored(tmp_path, rows):
    store = SampleStore(tmp_path / "samples.jsonl")
    for x, label in rows:
        store.append(x, label=label, label_source="test")
    return store, SuggestionStore(tmp_path / "suggestions.jsonl")


def test_rule_flags_only_long_dwell_people():
    assert rule_label(vec(0, 0.95)) == ANOMALY
    assert rule_label(vec(0, 0.5)) == 0
    assert rule_label(vec(1, 0.95)) == 1


def test_suggestions_are_stored_apart_and_never_become_labels(tmp_path):
    store, sug = stored(tmp_path, [(vec(0, 1.0), None), (vec(1, 0.1), None)])
    before = store.path.read_text(encoding="utf-8")
    generate_suggestions(store, sug, None, None, set())
    assert sug.path.exists() and sug.path != store.path
    assert store.path.read_text(encoding="utf-8") == before
    assert store.stats()["labelled"] == 0
    X, _ = store.labelled()
    assert len(X) == 0


def test_rule_only_suggestion_has_no_invented_confidence(tmp_path):
    store, sug = stored(tmp_path, [(vec(0, 1.0), None)])
    generate_suggestions(store, sug, None, None, set())
    item = sug.current(store)["suggestions"][0]
    assert item["label"] == "anomaly"
    assert item["source"] == RULE_SOURCE
    assert item["confidence"] is None and item["uncertainty"] is None


def test_either_proposer_saying_anomaly_makes_it_anomaly_and_marks_disagreement(tmp_path):
    store, sug = stored(tmp_path, [(vec(0, 0.2), None)])
    head = FixedHead([0.1, 0.0, 0.0, 0.0, 0.9])     # rule says routine
    generate_suggestions(store, sug, head, "fl-r1", set())
    item = sug.current(store)["suggestions"][0]
    assert item["label"] == "anomaly" and item["disagree"] is True
    assert item["source"] == "model:fl-r1"
    assert item["uncertainty"] == 1.0


def test_agreement_lists_both_sources(tmp_path):
    store, sug = stored(tmp_path, [(vec(0, 0.2), None)])
    head = FixedHead([0.8, 0.05, 0.05, 0.05, 0.05])
    generate_suggestions(store, sug, head, "fl-r1", set())
    item = sug.current(store)["suggestions"][0]
    assert item["source"] == f"model:fl-r1+{RULE_SOURCE}"
    assert item["uncertainty"] == pytest.approx(0.2)


def test_test_set_events_are_never_suggested_and_earlier_ones_are_dropped(tmp_path):
    store, sug = stored(tmp_path, [(vec(), None), (vec(), None)])
    generate_suggestions(store, sug, None, None, set())
    assert sug.current(store)["count"] == 2
    generate_suggestions(store, sug, None, None, {1})
    assert [s["n"] for s in sug.current(store)["suggestions"]] == [2]


def test_reviewed_events_are_not_suggested_but_legacy_labels_are(tmp_path):
    store, sug = stored(tmp_path, [(vec(), None), (vec(), None), (vec(), 0)])
    store.set_label_by_n(1, 0, prov=human_prov(0, "blind"))
    generate_suggestions(store, sug, None, None, set())
    assert sorted(s["n"] for s in sug.current(store)["suggestions"]) == [2, 3]


def test_a_suggestion_disappears_once_its_event_is_reviewed(tmp_path):
    store, sug = stored(tmp_path, [(vec(), None)])
    generate_suggestions(store, sug, None, None, set())
    store.set_label_by_n(1, 0, prov=human_prov(0, "blind"))
    assert sug.current(store)["count"] == 0


def test_synthetic_rows_get_no_suggestion(tmp_path):
    store, sug = stored(tmp_path, [])
    store.append(vec(), meta={"synthetic": True})
    generate_suggestions(store, sug, None, None, set())
    assert sug.current(store)["count"] == 0


def test_suggestion_file_does_not_match_a_labelled_row_shape(tmp_path):
    store, sug = stored(tmp_path, [(vec(), None)])
    generate_suggestions(store, sug, None, None, set())
    row = json.loads(sug.path.read_text(encoding="utf-8").splitlines()[0])
    assert "y" not in row and "x" not in row


# --- API ----------------------------------------------------------------------

@pytest.fixture
def api(tmp_path, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.api import dashboard_routes, trial_routes
    from src.federated import model_state
    from src.federated.head import FederatedHead

    monkeypatch.setattr(model_state.settings, "DATA_DIR", tmp_path)
    app = FastAPI()
    app.include_router(dashboard_routes.router)
    app.include_router(trial_routes.router)
    app.state.fl_store = SampleStore(tmp_path / "samples.jsonl")
    app.state.fl_head = FederatedHead(seed=0)
    app.state.fl_model_state = None
    for dwell in (1.0, 0.1, 0.2):
        app.state.fl_store.append(vec(0, dwell))
    return TestClient(app), app


def test_api_review_records_a_person(api):
    client, app = api
    body = {"action": "correct", "label": "routine_person", "review": {"kind": "blind", "via": "single"}}
    res = client.post("/api/events/1/label", json=body).json()
    assert res["label"] == "routine_person"
    assert res["provenance"]["reviewed"] is True
    assert res["provenance"]["origin"] == "human"
    listed = client.get("/api/federated/samples?limit=10").json()
    assert listed["stats"]["provenance"]["reviewed"] == 1
    row = next(s for s in listed["samples"] if s["n"] == 1)
    assert row["provenance"]["reviewed"] is True


def test_api_old_console_request_is_recorded_as_machine(api):
    client, _ = api
    res = client.post("/api/events/2/label", json={"action": "correct", "label": "routine_person"}).json()
    assert res["ok"] is True
    assert res["provenance"]["origin"] == "machine"
    assert res["provenance"]["reviewed"] is False


def test_api_confirm_cannot_be_a_review(api):
    client, _ = api
    body = {"action": "confirm", "review": {"kind": "blind"}}
    assert client.post("/api/events/1/label", json=body).status_code == 422


def test_api_free_text_cannot_be_a_review(api):
    client, _ = api
    body = {"action": "correct", "label": "delivery driver", "review": {"kind": "blind"}}
    assert client.post("/api/events/1/label", json=body).status_code == 422


def test_api_false_acceptance_is_refused(api):
    client, _ = api
    body = {"action": "correct", "label": "routine_person",
            "review": {"kind": "accept", "suggested_label": "anomaly", "suggested_source": "rule:x"}}
    assert client.post("/api/events/1/label", json=body).status_code == 422


def test_api_unattributed_write_over_a_review_is_refused(api):
    client, app = api
    client.post("/api/events/1/label", json={"action": "correct", "label": "anomaly", "review": {"kind": "blind"}})
    res = client.post("/api/events/1/label", json={"action": "correct", "label": "routine_person"})
    assert res.status_code == 409
    assert app.state.fl_store.by_n(1)["y"] == ANOMALY


def test_api_generate_excludes_test_events_and_writes_no_labels(api):
    client, app = api
    res = client.post("/api/federated/suggestions/generate", json={"exclude": [2]}).json()
    assert res["generated"] == 2 and res["excluded"] == 1
    listed = client.get("/api/federated/suggestions").json()
    assert sorted(s["n"] for s in listed["suggestions"]) == [1, 3]
    assert app.state.fl_store.stats()["labelled"] == 0


def test_api_untrained_head_gives_rule_suggestions_only(api):
    client, _ = api
    client.post("/api/federated/suggestions/generate", json={})
    listed = client.get("/api/federated/suggestions").json()
    assert {s["source"] for s in listed["suggestions"]} == {RULE_SOURCE}
    assert all(s["model_label"] is None for s in listed["suggestions"])


def test_api_human_only_training_is_refused_without_enough_reviews(api):
    client, app = api
    for n in (1, 2, 3):
        app.state.fl_store.set_label_by_n(n, 0)
    res = client.post("/api/federated/head/train", json={"min_samples": 3, "human_only": True})
    assert res.status_code == 409
    assert "human-reviewed" in res.json()["detail"]


# --- Human-only training and the gate ----------------------------------------

def labelled_store(tmp_path, reviewed, machine):
    store = SampleStore(tmp_path / "samples.jsonl")
    rng = np.random.default_rng(0)
    for i in range(reviewed + machine):
        x = vec(0, float(rng.random()))
        label = ANOMALY if x[13] > 0.8 else 0
        store.append(x)
        store.set_label_by_n(store._rows[-1]["n"], label,
                             prov=human_prov(label, "blind") if i < reviewed else None)
    return store


def test_human_only_returns_exactly_the_reviewed_rows(tmp_path):
    store = labelled_store(tmp_path, 30, 70)
    assert len(store.labelled()[0]) == 100
    assert len(store.labelled(human_only=True)[0]) == 30


def test_gate_scores_on_reviewed_rows_when_human_only(tmp_path, monkeypatch):
    from src.federated import model_state
    from src.federated.head import FederatedHead

    monkeypatch.setattr(model_state.settings, "DATA_DIR", tmp_path)
    store = labelled_store(tmp_path, 30, 70)
    head = FederatedHead(seed=0)
    app = SimpleNamespace(state=SimpleNamespace(fl_head=head, fl_store=store, fl_model_state=None))
    candidate = tmp_path / "c.npz"
    head.save(candidate)

    decision = model_state.accept_candidate(app, candidate, "local-train", human_only=True)
    assert decision["human_only"] is True
    assert decision["val_samples"] == 6  # 20% of 30, not of 100

    decision = model_state.accept_candidate(app, candidate, "local-train")
    assert decision["human_only"] is False
    assert decision["val_samples"] == 20


def test_node_client_trains_only_on_reviewed_rows(tmp_path):
    pytest.importorskip("flwr")
    from src.federated.node_client import NodeHeadClient

    store = labelled_store(tmp_path, 30, 70)
    client = NodeHeadClient(store.path, None, tmp_path / "out", 1, 0, human_only=True)
    assert len(client.Xt) + len(client.Xv) == 30
    ready = json.loads((tmp_path / "out" / "events.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert ready["human_only"] is True
    assert ready["labels_reviewed"] == 30 and ready["labels_machine"] == 70


def test_scheduled_round_refuses_when_reviews_are_short(tmp_path):
    from src.federated.rounds import run_round_blocking

    store = labelled_store(tmp_path, 30, 70)
    app = SimpleNamespace(state=SimpleNamespace(fl_head=object(), fl_store=store))
    result = run_round_blocking(app, min_samples=50, human_only=True)
    assert result["completed"] is False
    assert "human-reviewed" in result["reason"]


def test_every_label_name_round_trips_through_review(tmp_path):
    for i, name in enumerate(LABEL_NAMES):
        prov = human_prov(i, "correct", (i + 1) % len(LABEL_NAMES), "rule:x")
        assert prov["suggested_label"] != name


def test_machine_prov_is_never_reviewed():
    assert machine_prov("rule:x")["reviewed"] is False
