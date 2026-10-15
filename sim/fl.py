"""
Real Flower server and client, as separate OS processes.

Not the ray-backed simulation backend: a genuine gRPC server on
127.0.0.1:8080, which is what settings.FL_SERVER_URL already points at.
Separate processes mean the trainer competes with the detection loop for
cores through the OS scheduler, which is the contention being measured.

    python sim/fl.py serve  --rounds 5 --clients 1
    python sim/fl.py serve  --rounds 5 --clients 3 --aggregation fedavg
    python sim/fl.py client --site driveway --epochs 4
    python sim/fl.py client --store data/datasets/meva/events/G336 --epochs 3
"""

import os

# MUST precede any numpy import. numpy's BLAS is multithreaded by default, so
# a single "CPU stage" would otherwise saturate every core -- and with the
# trainer process doing the same, the two oversubscribe the machine and both
# collapse. One BLAS thread per process means one stage occupies one core,
# which is the thing being modelled.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import sys
import time
import types
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import flwr as fl
import numpy as np

import workload
from head import FederatedHead
from src.federated.robust_strategy import AGGREGATIONS, RobustFedAvg
from src.federated.store import SampleStore

EVENTS = Path(__file__).resolve().parent / "_run" / "fl_events.jsonl"


def emit(**kw):
    EVENTS.parent.mkdir(parents=True, exist_ok=True)
    kw["t"] = time.time()
    with open(EVENTS, "a", encoding="utf-8") as f:
        f.write(json.dumps(kw) + "\n")


def load_store_split(store_path, seed=1):
    """Train/validation arrays from a SampleStore (e.g. a MEVA camera).

    `store_path` is a samples.jsonl or the directory holding it. Rows whose
    meta.split is "holdout" form the validation set, so whole clips are held
    out (scripts/meva_to_events.py). A store without split tags falls back to
    the random split the synthetic sites use.
    """
    path = Path(store_path)
    if path.is_dir():
        path = path / "samples.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"no sample store at {path}")
    rows = [r for r in SampleStore(path)._rows if r.get("y") is not None]
    if not rows:
        raise ValueError(f"{path} has no labelled samples")
    X = np.array([r["x"] for r in rows], dtype=np.float32)
    y = np.array([r["y"] for r in rows], dtype=np.int64)
    hold = np.array([r.get("meta", {}).get("split") == "holdout" for r in rows])
    if hold.any() and (~hold).any():
        return X[~hold], y[~hold], X[hold], y[hold]
    return workload.split(X, y, seed=seed)


class ArgusHeadClient(fl.client.NumPyClient):
    """Mirrors src/federated/client.py, with the head in place of YOLO.

    Data comes from a synthetic site (`site_key`) or, when `store` is given,
    from a SampleStore on disk such as a MEVA camera's events.
    """

    def __init__(self, site_key, samples, epochs, seed, store=None):
        if store:
            name = Path(store).resolve()
            name = (name.parent if name.suffix else name).name
            self.site = types.SimpleNamespace(key=name)
            self.Xt, self.yt, self.Xv, self.yv = load_store_split(store, seed=seed)
        else:
            self.site = workload.SITES[site_key]
            X, y = workload.generate(self.site, samples, seed=seed)
            self.Xt, self.yt, self.Xv, self.yv = workload.split(X, y, seed=seed)
        self.head = FederatedHead(seed=0)
        self.epochs = epochs
        emit(kind="client_ready", site=self.site.key, train=len(self.Xt),
             val=len(self.Xv), params=self.head.n_params)

    def get_parameters(self, config):
        return self.head.get_weights()

    def fit(self, parameters, config):
        self.head.set_weights(parameters)
        t = time.perf_counter()
        m = self.head.fit(self.Xt, self.yt, epochs=self.epochs)
        dt = time.perf_counter() - t
        loss, acc = self.head.evaluate(self.Xv, self.yv)
        bal = self.head.balanced_accuracy(self.Xv, self.yv)
        emit(kind="fit", site=self.site.key, secs=round(dt, 3),
             loss=round(loss, 4), acc=round(acc, 4), bal_acc=round(bal, 4))
        return self.head.get_weights(), len(self.Xt), {
            "train_loss": float(m["loss"]),
            "fit_seconds": float(dt),
        }

    def evaluate(self, parameters, config):
        self.head.set_weights(parameters)
        loss, acc = self.head.evaluate(self.Xv, self.yv)
        bal = self.head.balanced_accuracy(self.Xv, self.yv)
        emit(kind="evaluate", site=self.site.key, loss=round(loss, 4),
             acc=round(acc, 4), bal_acc=round(bal, 4))
        return float(loss), len(self.Xv), {
            "accuracy": float(acc),
            "balanced_accuracy": float(bal),
        }


def _agg(metrics):
    tot = sum(n for n, _ in metrics)
    out = {}
    for key in ("accuracy", "balanced_accuracy"):
        if metrics and key in metrics[0][1]:
            out[key] = sum(n * m[key] for n, m in metrics) / tot
    return out


def serve(args):
    def report(server_round, rep):
        emit(kind="aggregation", round=server_round, **rep.as_dict())

    strategy = RobustFedAvg(
        aggregation=args.aggregation,
        on_report=report,
        min_fit_clients=args.clients,
        min_evaluate_clients=args.clients,
        min_available_clients=args.clients,
        evaluate_metrics_aggregation_fn=_agg,
    )
    emit(kind="server_start", rounds=args.rounds, clients=args.clients,
         aggregation=args.aggregation)
    hist = fl.server.start_server(
        server_address=args.address,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )
    dist = dict(hist.metrics_distributed or {})
    emit(
        kind="server_done",
        accuracy=[[r, float(v)] for r, v in dist.get("accuracy", [])],
        balanced_accuracy=[[r, float(v)] for r, v in dist.get("balanced_accuracy", [])],
        losses=[[r, float(v)] for r, v in (hist.losses_distributed or [])],
    )


def client(args):
    c = ArgusHeadClient(args.site, args.samples, args.epochs, args.seed,
                        store=args.store)
    fl.client.start_client(server_address=args.address, client=c.to_client())
    emit(kind="client_done", site=c.site.key)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--rounds", type=int, default=5)
    s.add_argument("--clients", type=int, default=1)
    s.add_argument("--address", default="127.0.0.1:8080")
    s.add_argument("--aggregation", choices=AGGREGATIONS, default="robust",
                   help="robust screens updates first; fedavg is the unscreened baseline")
    c = sub.add_parser("client")
    c.add_argument("--site", default="driveway")
    c.add_argument("--store", default=None,
                   help="train from a SampleStore (samples.jsonl or its directory), "
                        "e.g. data/datasets/meva/events/G336; overrides --site")
    c.add_argument("--samples", type=int, default=6000)
    c.add_argument("--epochs", type=int, default=4)
    c.add_argument("--seed", type=int, default=1)
    c.add_argument("--address", default="127.0.0.1:8080")
    a = p.parse_args()
    (serve if a.cmd == "serve" else client)(a)
