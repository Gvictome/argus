"""
Flower client for the live node.

The trial route previously launched `sim/fl.py client`, which trains a
fresh head on *synthetic* data and throws the result away: nothing the
camera captured was trained on, and the averaged model never reached the
node. This client closes both gaps.

  - It trains on the node's own SampleStore (camera events plus any
    labelled bootstrap rows), starting from the node's current weights.
  - Every aggregated global model the server sends back is written to
    `<out>/global.npz`. Flower sends the aggregate to `evaluate()` after
    each round, so when the run ends that file holds the final global
    model -- the update coming back. The API process validates it and
    hot-swaps it in; this process never touches the live head.

Runs as its own OS process so training competes with detection through
the scheduler rather than the GIL, and so a crash here cannot take the
camera down. The simulator measured that arrangement at a 0.1% FPS cost.

    python -m src.federated.node_client --store data/training/samples.jsonl \
        --init data/fl_head.npz --out data/fl_round --server 127.0.0.1:8080
"""

import os

# BLAS threads must be pinned before numpy loads. One thread leaves the
# other cores to detection; override with ARGUS_TRAIN_THREADS.
_threads = os.environ.get("ARGUS_TRAIN_THREADS", "1")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, _threads)

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import flwr as fl  # noqa: E402

from src.federated.head import FederatedHead  # noqa: E402
from src.federated.store import SampleStore, holdout_split  # noqa: E402


class NodeHeadClient(fl.client.NumPyClient):
    def __init__(self, store_path: Path, init_weights: Path, out_dir: Path,
                 epochs: int, seed: int):
        X, y = SampleStore(store_path).labelled()
        if len(X) == 0:
            raise SystemExit("no labelled samples in the store")
        self.Xt, self.yt, self.Xv, self.yv = holdout_split(X, y, seed)
        self.head = FederatedHead(seed=0)
        if init_weights and Path(init_weights).exists():
            self.head.load(init_weights)
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.epochs = epochs
        self.round = 0
        self._log(kind="ready", train=len(self.Xt), val=len(self.Xv),
                  params=self.head.n_params,
                  warm_start=bool(init_weights and Path(init_weights).exists()))

    def _log(self, **kw):
        kw["t"] = time.time()
        line = json.dumps(kw)
        with open(self.out / "events.jsonl", "a", encoding="utf-8") as f:
            f.write(line + "\n")
        print(line, flush=True)

    def get_parameters(self, config):
        return self.head.get_weights()

    def fit(self, parameters, config):
        self.head.set_weights(parameters)
        self.round += 1
        t0 = time.perf_counter()
        m = self.head.fit(self.Xt, self.yt, epochs=self.epochs, seed=self.round)
        loss, acc = self.head.evaluate(self.Xv, self.yv)
        bal = self.head.balanced_accuracy(self.Xv, self.yv)
        self._log(kind="fit", round=self.round, secs=round(time.perf_counter() - t0, 3),
                  loss=round(loss, 4), acc=round(acc, 4), bal_acc=round(bal, 4))
        return self.head.get_weights(), len(self.Xt), {"train_loss": float(m["loss"])}

    def evaluate(self, parameters, config):
        # These are the server's *aggregated* weights, not ours. Persisting
        # them here is what makes the update come back to the node.
        self.head.set_weights(parameters)
        loss, acc = self.head.evaluate(self.Xv, self.yv)
        bal = self.head.balanced_accuracy(self.Xv, self.yv)
        self.head.save(self.out / "global.npz")
        self._log(kind="global", round=self.round, loss=round(loss, 4),
                  acc=round(acc, 4), bal_acc=round(bal, 4))
        return float(loss), len(self.Xv), {"accuracy": float(acc),
                                           "balanced_accuracy": float(bal)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--init", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--server", default="127.0.0.1:8080")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    client = NodeHeadClient(Path(a.store), Path(a.init) if a.init else None,
                            Path(a.out), a.epochs, a.seed)
    fl.client.start_client(server_address=a.server, client=client.to_client())
    client._log(kind="done", rounds=client.round)
    return 0


if __name__ == "__main__":
    sys.exit(main())
