"""
Real Flower server and client, as separate OS processes.

Not the ray-backed simulation backend: a genuine gRPC server on
127.0.0.1:8080, which is what settings.FL_SERVER_URL already points at.
Separate processes mean the trainer competes with the detection loop for
cores through the OS scheduler, which is the contention being measured.

    python sim/fl.py serve  --rounds 5 --clients 1
    python sim/fl.py client --site driveway --epochs 4
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
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import flwr as fl

import workload
from head import FederatedHead

EVENTS = Path(__file__).resolve().parent / "_run" / "fl_events.jsonl"


def emit(**kw):
    EVENTS.parent.mkdir(parents=True, exist_ok=True)
    kw["t"] = time.time()
    with open(EVENTS, "a", encoding="utf-8") as f:
        f.write(json.dumps(kw) + "\n")


class ArgusHeadClient(fl.client.NumPyClient):
    """Mirrors src/federated/client.py, with the head in place of YOLO."""

    def __init__(self, site_key, samples, epochs, seed):
        self.site = workload.SITES[site_key]
        X, y = workload.generate(self.site, samples, seed=seed)
        self.Xt, self.yt, self.Xv, self.yv = workload.split(X, y, seed=seed)
        self.head = FederatedHead(seed=0)
        self.epochs = epochs
        emit(kind="client_ready", site=site_key, train=len(self.Xt),
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
    strategy = fl.server.strategy.FedAvg(
        min_fit_clients=args.clients,
        min_evaluate_clients=args.clients,
        min_available_clients=args.clients,
        evaluate_metrics_aggregation_fn=_agg,
    )
    emit(kind="server_start", rounds=args.rounds, clients=args.clients)
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
    c = ArgusHeadClient(args.site, args.samples, args.epochs, args.seed)
    fl.client.start_client(server_address=args.address, client=c.to_client())
    emit(kind="client_done", site=args.site)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--rounds", type=int, default=5)
    s.add_argument("--clients", type=int, default=1)
    s.add_argument("--address", default="127.0.0.1:8080")
    c = sub.add_parser("client")
    c.add_argument("--site", default="driveway")
    c.add_argument("--samples", type=int, default=6000)
    c.add_argument("--epochs", type=int, default=4)
    c.add_argument("--seed", type=int, default=1)
    c.add_argument("--address", default="127.0.0.1:8080")
    a = p.parse_args()
    (serve if a.cmd == "serve" else client)(a)
