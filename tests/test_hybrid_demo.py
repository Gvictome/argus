"""
Single-node hybrid demo: one real (or stand-in) node plus simulated MEVA
peers in live federated rounds.

Pins the pieces the launcher is built from: the poisoned-peer attacks, the
server-side evaluation that makes each round's balanced accuracy and
per-class recall visible, the camera discovery, the launch plan (which must
grow from one real node to two by adding only a count), and the console
rendering of a round.
"""

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "sim"))
sys.path.insert(0, str(REPO))

import metrics  # noqa: E402
import poison  # noqa: E402
from src.federated.store import SampleStore  # noqa: E402


def _load_demo():
    spec = importlib.util.spec_from_file_location(
        "demo_hybrid", REPO / "scripts" / "demo_hybrid.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["demo_hybrid"] = mod
    spec.loader.exec_module(mod)
    return mod


# ---- poison.py ----------------------------------------------------------
def _pair():
    rng = np.random.default_rng(0)
    g = [rng.normal(size=(4, 3)).astype(np.float32), rng.normal(size=3).astype(np.float32)]
    trained = [w + 0.05 * np.ones_like(w) for w in g]
    return g, trained


def _flat(ws):
    return np.concatenate([w.ravel() for w in ws])


def test_none_returns_the_honest_update_unchanged():
    g, t = _pair()
    out = poison.poison_weights("none", g, t)
    assert all(np.array_equal(a, b) for a, b in zip(out, t))


def test_signflip_points_against_the_honest_direction_and_is_boosted():
    g, t = _pair()
    honest = _flat(t) - _flat(g)
    bad = _flat(poison.poison_weights("signflip", g, t)) - _flat(g)
    cos = bad @ honest / (np.linalg.norm(bad) * np.linalg.norm(honest))
    assert cos == pytest.approx(-1.0, abs=1e-4)
    assert np.linalg.norm(bad) > 3 * np.linalg.norm(honest)


def test_nan_attack_is_not_finite_and_inputs_are_not_mutated():
    g, t = _pair()
    before = [w.copy() for w in t]
    out = poison.poison_weights("nan", g, t)
    assert not np.isfinite(_flat(out)).all()
    assert all(np.array_equal(a, b) for a, b in zip(t, before))


def test_unknown_mode_is_rejected():
    g, t = _pair()
    with pytest.raises(ValueError):
        poison.poison_weights("bogus", g, t)


# ---- metrics.py ---------------------------------------------------------
def test_per_class_recall_marks_absent_classes_none():
    y = np.array([0, 0, 1, 4])
    pred = np.array([0, 1, 1, 0])
    r = metrics.per_class_recall(pred, y, 5)
    assert r[0] == pytest.approx(0.5) and r[1] == 1.0 and r[4] == 0.0
    assert r[2] is None and r[3] is None


def test_balanced_accuracy_ignores_absent_classes():
    y = np.array([0, 0, 0, 0, 4])
    pred = np.array([0, 0, 0, 0, 0])
    assert metrics.balanced_accuracy(pred, y, 5) == pytest.approx(0.5)


def _write_store(root: Path, name: str, n_train=30, n_hold=10, shift=0.0):
    d = root / name
    d.mkdir(parents=True)
    rng = np.random.default_rng(abs(hash(name)) % 1000)
    s = SampleStore(d / "samples.jsonl")
    for i in range(n_train + n_hold):
        y = int(i % 2) * 4 if i % 5 else 0
        x = rng.normal(size=19).astype(np.float32) + shift
        x[0] = 1.0 if y == 0 else 0.0
        s.append(x, label=y, meta={"split": "holdout" if i >= n_train else "train"})
    return d


def test_load_eval_sets_reads_only_holdout_rows(tmp_path):
    a = _write_store(tmp_path, "G1")
    b = _write_store(tmp_path, "G2")
    sets = metrics.load_eval_sets([a, b])
    assert [s.name for s in sets] == ["G1", "G2"]
    assert all(len(s.y) == 10 for s in sets)


def test_make_evaluate_fn_reports_balanced_accuracy_and_recall(tmp_path):
    sets = metrics.load_eval_sets([_write_store(tmp_path, "G1"), _write_store(tmp_path, "G2")])
    from head import FederatedHead
    head = FederatedHead(seed=0)
    fn = metrics.make_evaluate_fn(sets)
    loss, out = fn(1, head.get_weights(), {})
    assert loss > 0
    assert 0.0 <= out["balanced_accuracy"] <= 1.0
    assert set(out["per_class_recall"]) <= {"routine_person", "routine_vehicle",
                                            "routine_animal", "routine_package", "anomaly"}
    assert set(out["per_camera_balanced_accuracy"]) == {"G1", "G2"}


def test_make_evaluate_fn_with_no_sets_is_none():
    assert metrics.make_evaluate_fn([]) is None


# ---- demo_hybrid.py -----------------------------------------------------
def test_discover_cameras_ranks_by_size_and_skips_tiny(tmp_path):
    demo = _load_demo()
    _write_store(tmp_path, "BIG", n_train=80, n_hold=20)
    _write_store(tmp_path, "MID", n_train=40, n_hold=10)
    _write_store(tmp_path, "TINY", n_train=3, n_hold=1)
    cams = demo.discover_cameras(tmp_path, min_events=20)
    assert [c.name for c in cams] == ["BIG", "MID"]


def test_plan_total_clients_grows_only_by_real_node_count(tmp_path):
    demo = _load_demo()
    for n in ("A", "B", "C", "D", "E"):
        _write_store(tmp_path, n)
    one = demo.build_plan(events_root=tmp_path, peers=4, rounds=3, epochs=2,
                          real_nodes=1, poison="none", port=18080)
    two = demo.build_plan(events_root=tmp_path, peers=4, rounds=3, epochs=2,
                          real_nodes=2, poison="none", port=18080)
    assert one.total_clients == 5 and two.total_clients == 6
    assert len(one.peer_cmds) == len(two.peer_cmds) == 4
    assert one.server_cmd != two.server_cmd  # only --clients changes
    diff = [i for i, (a, b) in enumerate(zip(one.server_cmd, two.server_cmd)) if a != b]
    assert len(diff) == 1


def test_plan_poison_adds_one_extra_attacker_client(tmp_path):
    demo = _load_demo()
    for n in ("A", "B", "C", "D", "E"):
        _write_store(tmp_path, n)
    clean = demo.build_plan(events_root=tmp_path, peers=4, rounds=3, epochs=2,
                            real_nodes=1, poison="none", port=18080)
    bad = demo.build_plan(events_root=tmp_path, peers=4, rounds=3, epochs=2,
                          real_nodes=1, poison="signflip", port=18080)
    assert bad.total_clients == clean.total_clients + 1
    assert any("--poison" in c and "signflip" in c for c in bad.peer_cmds)
    assert not any("--poison" in c for c in clean.peer_cmds)


def test_plan_rejects_more_peers_than_cameras(tmp_path):
    demo = _load_demo()
    _write_store(tmp_path, "A")
    with pytest.raises(SystemExit):
        demo.build_plan(events_root=tmp_path, peers=4, rounds=3, epochs=2,
                        real_nodes=1, poison="none", port=18080)


def test_format_round_shows_accuracy_recall_and_rejections():
    demo = _load_demo()
    ev = {"kind": "round_eval", "round": 2, "balanced_accuracy": 0.8123,
          "loss": 0.4, "per_class_recall": {"routine_person": 0.9, "anomaly": 0.5,
                                            "routine_vehicle": None}}
    agg = {"kind": "aggregation", "round": 2, "accepted": ["G299", "node-A"],
           "rejected": ["evil"], "consensus_applied": True,
           "detail": [{"node_id": "evil", "accepted": False,
                           "reason": "direction: cosine -1.00 < 0.0"}]}
    text = demo.format_round(ev, agg)
    assert "round 2" in text.lower() and "0.812" in text
    assert "routine_person 0.90" in text and "anomaly 0.50" in text
    assert "routine_vehicle" not in text
    assert "REJECTED evil" in text and "direction" in text
    assert "accepted 2" in text


def test_format_round_without_aggregation_report_still_renders():
    demo = _load_demo()
    text = demo.format_round({"kind": "round_eval", "round": 1, "balanced_accuracy": 0.5,
                              "loss": 1.0, "per_class_recall": {}}, None)
    assert "round 1" in text.lower()
