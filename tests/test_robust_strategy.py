"""
The Flower strategy that connects robust.py to the aggregators.

test_robust_aggregation.py proves the filters work on their own. These
prove the server actually runs them: an attacker that robust.py rejects
must also be left out of the model a Flower round produces, and the
fedavg switch must reproduce plain FedAvg exactly, since it is the
baseline arm of the comparison study.
"""

import json

import numpy as np
import pytest

flwr = pytest.importorskip("flwr")
from flwr.common import Code, FitRes, Status, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg

from src.federated.robust import ClientUpdate, RobustAggregator
from src.federated.robust_strategy import RobustFedAvg


class _Proxy:
    """aggregate_fit only reads the connection id."""

    def __init__(self, cid):
        self.cid = cid


def _global(n=50):
    rng = np.random.default_rng(0)
    return [rng.normal(0, 0.1, (n,)), rng.normal(0, 0.1, (n, 2))]


def _honest(g, seed, scale=0.01):
    rng = np.random.default_rng(seed)
    return [w + scale * (np.ones_like(w) + 0.1 * rng.normal(size=w.shape)) for w in g]


def _sign_flip(g, scale=0.05):
    return [w - scale * np.ones_like(w) for w in g]


def _res(weights, node_id=None, samples=100, **metrics):
    if node_id is not None:
        metrics["node_id"] = node_id
    return FitRes(
        status=Status(code=Code.OK, message=""),
        parameters=ndarrays_to_parameters(weights),
        num_examples=samples,
        metrics=metrics,
    )


def _results(g, n_honest=4, attacker=True):
    out = [(_Proxy(f"c{i}"), _res(_honest(g, i), f"node{i}")) for i in range(n_honest)]
    if attacker:
        out.append((_Proxy("cx"), _res(_sign_flip(g), "attacker")))
    return out


def _strategy(g, **kw):
    return RobustFedAvg(initial_parameters=ndarrays_to_parameters(g), **kw)


def test_sign_flip_attacker_is_left_out_of_the_round():
    g = _global()
    strategy = _strategy(g)
    params, metrics = strategy.aggregate_fit(1, _results(g), [])

    assert "attacker" in strategy.last_report.rejected
    assert metrics["robust_rejected"] == 1
    assert metrics["robust_accepted"] == 4
    assert metrics["robust_consensus"] == 1
    assert json.loads(metrics["robust_report"])["rejected"] == ["attacker"]

    # The round's model is exactly what robust.py produces from the honest
    # four, so the strategy adds no averaging of its own.
    honest = [ClientUpdate(f"node{i}", _honest(g, i), 100) for i in range(4)]
    expected, _ = RobustAggregator().aggregate(g, honest)
    for got, want in zip(parameters_to_ndarrays(params), expected):
        np.testing.assert_allclose(got, want)


def test_fedavg_switch_reproduces_plain_fedavg():
    g = _global()
    results = _results(g)
    ours, _ = _strategy(g, aggregation="fedavg").aggregate_fit(1, results, [])
    theirs, _ = FedAvg(initial_parameters=ndarrays_to_parameters(g)).aggregate_fit(1, results, [])
    for a, b in zip(parameters_to_ndarrays(ours), parameters_to_ndarrays(theirs)):
        np.testing.assert_allclose(a, b)


def test_the_attack_moves_fedavg_but_not_robust():
    """The counterfactual: the same round, the two arms, different outcomes."""
    g = _global()
    clean, _ = _strategy(g).aggregate_fit(1, _results(g, attacker=False), [])
    robust, _ = _strategy(g).aggregate_fit(1, _results(g), [])
    plain, _ = _strategy(g, aggregation="fedavg").aggregate_fit(1, _results(g), [])

    def dist(p):
        return float(sum(np.linalg.norm(a - b) for a, b in
                         zip(parameters_to_ndarrays(p), parameters_to_ndarrays(clean))))

    assert dist(robust) < dist(plain)


def test_nothing_survives_keeps_the_current_model():
    g = _global()
    broken = [(_Proxy(f"c{i}"), _res([np.full_like(w, np.nan) for w in g], f"n{i}"))
              for i in range(3)]
    params, metrics = _strategy(g).aggregate_fit(1, broken, [])
    assert params is None  # Flower keeps the current global model on None
    assert metrics["robust_accepted"] == 0
    assert metrics["robust_rejected"] == 3


def test_reference_is_the_model_sent_out_this_round():
    g = _global()
    sent = [w + 1.0 for w in g]

    class _Manager:
        def num_available(self):
            return 0

        def sample(self, num_clients, min_num_clients=None, criterion=None):
            return []

        def wait_for(self, num_clients, timeout=86400):
            return True

    strategy = _strategy(g)
    strategy.configure_fit(2, ndarrays_to_parameters(sent), _Manager())
    for got, want in zip(strategy._global, sent):
        np.testing.assert_allclose(got, want)


def test_two_nodes_with_the_same_name_are_both_screened():
    g = _global()
    results = [(_Proxy("c0"), _res(_honest(g, 0), "node")),
               (_Proxy("c1"), _res(_honest(g, 1), "node"))]
    strategy = _strategy(g)
    strategy.aggregate_fit(1, results, [])
    assert sorted(strategy.last_report.accepted) == ["node", "node#1"]


def test_connection_id_names_a_client_that_sends_no_name():
    g = _global()
    results = [(_Proxy(f"c{i}"), _res(_honest(g, i))) for i in range(3)]
    strategy = _strategy(g)
    strategy.aggregate_fit(1, results, [])
    assert sorted(strategy.last_report.accepted) == ["c0", "c1", "c2"]


def test_client_metrics_are_averaged_over_kept_updates_only():
    g = _global()
    results = [(_Proxy(f"c{i}"), _res(_honest(g, i), f"node{i}", accuracy=0.9))
               for i in range(4)]
    results.append((_Proxy("cx"), _res(_sign_flip(g), "attacker", accuracy=0.0)))
    seen = []

    def agg(metrics):
        seen.extend(m["node_id"] for _, m in metrics)
        return {"accuracy": sum(m["accuracy"] for _, m in metrics) / len(metrics)}

    _, metrics = _strategy(g, fit_metrics_aggregation_fn=agg).aggregate_fit(1, results, [])
    assert "attacker" not in seen
    assert metrics["accuracy"] == pytest.approx(0.9)


def test_report_callback_failure_does_not_break_the_round():
    g = _global()

    def boom(rnd, report):
        raise RuntimeError("dashboard down")

    params, _ = _strategy(g, on_report=boom).aggregate_fit(1, _results(g), [])
    assert params is not None


def test_unknown_aggregation_is_refused():
    with pytest.raises(ValueError):
        RobustFedAvg(aggregation="median")


def test_central_server_reads_aggregation_from_the_environment(monkeypatch):
    pytest.importorskip("flask")
    pytest.importorskip("flask_cors")
    pytest.importorskip("prometheus_client")
    import importlib.util
    import sys
    from pathlib import Path

    # Imported once per session: the module registers Prometheus metrics at
    # import, and test_model_versions.py loads it under the same name.
    name = "argus_fl_aggregator_under_test"
    module = sys.modules.get(name)
    if module is None:
        path = Path(__file__).resolve().parents[1] / "central_server" / "fl_aggregator.py"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)

    assert module.load_config()["aggregation"] == "robust"
    monkeypatch.setenv("ARGUS_AGGREGATION", "fedavg")
    assert module.load_config()["aggregation"] == "fedavg"
