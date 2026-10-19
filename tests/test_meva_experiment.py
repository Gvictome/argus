"""
Helpers behind sim/meva_experiment.py: clip-level resplit, the FedAvg
average, the confidence interval, and a tiny end-to-end seed run.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "sim"))
sys.path.insert(0, str(REPO))

import meva_experiment as mx  # noqa: E402
from src.federated.store import SampleStore  # noqa: E402


def _rows(n_clips=10, per_clip=6):
    rng = np.random.default_rng(0)
    X, y, clips = [], [], []
    for c in range(n_clips):
        for i in range(per_clip):
            X.append(rng.normal(size=19).astype(np.float32))
            y.append(0 if i % 3 else 4)
            clips.append(f"clip{c}")
    return np.array(X), np.array(y), clips


def test_resplit_never_splits_a_clip_and_holds_out_about_a_fifth():
    X, y, clips = _rows()
    cam = mx.resplit("G1", X, y, clips, seed=3)
    assert len(cam.yv) == 12 and len(cam.yt) == 48
    # Every holdout row's clip contributes no row to the training set.
    hold_rows = {tuple(r) for r in cam.Xv}
    assert not hold_rows & {tuple(r) for r in cam.Xt}


def test_resplit_differs_by_seed_and_is_deterministic():
    X, y, clips = _rows()
    a = mx.resplit("G1", X, y, clips, seed=1)
    b = mx.resplit("G1", X, y, clips, seed=1)
    c = mx.resplit("G1", X, y, clips, seed=2)
    assert np.array_equal(a.Xv, b.Xv)
    assert not np.array_equal(a.Xv, c.Xv)


def test_resplit_caps_training_rows():
    X, y, clips = _rows()
    cam = mx.resplit("G1", X, y, clips, seed=1, cap=20)
    assert len(cam.yt) == 20


def test_fedavg_weights_is_sample_weighted():
    a = [np.zeros((2, 2), np.float32)]
    b = [np.ones((2, 2), np.float32) * 4]
    out = mx.fedavg_weights([(a, 3), (b, 1)])
    assert np.allclose(out[0], 1.0)


def test_mean_ci_known_values():
    m, hw = mx.mean_ci([1.0, 1.0, 1.0])
    assert m == 1.0 and hw == 0.0
    m, hw = mx.mean_ci([0.0, 1.0])
    assert m == 0.5 and hw == pytest.approx(12.706 * np.std([0, 1], ddof=1) / np.sqrt(2))


def test_mean_ci_single_value_has_no_interval():
    m, hw = mx.mean_ci([0.7])
    assert m == 0.7 and np.isnan(hw)


@pytest.mark.slow
def test_run_seed_end_to_end_on_tiny_cameras(tmp_path):
    for name in ("A", "B", "C"):
        X, y, clips = _rows(n_clips=10, per_clip=8)
        s = SampleStore(tmp_path / name / "samples.jsonl")
        for xi, yi, ci in zip(X, y, clips):
            s.append(xi, label=int(yi), meta={"clip": ci, "split": "train"})
    res = mx.run_seed((str(tmp_path), ["A", "B", "C"], 0, True))
    for arm in ("untrained", "local@3", "fedavg@8", "robust@5", "robust+finetune@8",
                "fedavg+finetune@8", "cold-start@8"):
        assert 0.0 <= res[arm]["macro_bal_acc"] <= 1.0
    summary = mx.summarise([res, res])
    assert "robust@8" in summary
    assert "| robust@8 |" in mx.to_markdown([res, res])
    json.dumps(res)
