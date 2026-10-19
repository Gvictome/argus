"""
Experiments behind sim/report/ARGUS_Simulation_Results.ipynb.

Three questions the sponsor asked, each answered with the simulator that
already exists in sim/ (nothing here changes the sim's behaviour):

  1. Does federated learning beat a node training alone?     fl_vs_local, cold_start
  2. Does a node keep detecting while it trains?             run_fps
  3. Can one bad node poison the fleet, and does robust.py
     stop it?                                                robustness

Federated training here is in-process FedAvg, the same loop as
sim/compare_taxonomy.py. sim/fl.py measures the gRPC transport; this
measures the learning, which is identical maths without the sockets.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import importlib.util
import json
import logging
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

SIM = Path(__file__).resolve().parents[1]
REPO = SIM.parent
RUN = SIM / "_run"
sys.path.insert(0, str(SIM))

import compare_taxonomy as cmp  # noqa: E402
import workload  # noqa: E402
from head import FederatedHead  # noqa: E402


def _load_robust():
    """Import src/federated/robust.py without src/federated/__init__ (pulls in Flower)."""
    spec = importlib.util.spec_from_file_location("argus_robust", REPO / "src/federated/robust.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


robust = _load_robust()
# robust.py logs every rejection; the notebook reports them as counts instead
logging.getLogger(robust.__name__).setLevel(logging.ERROR)

ANOMALY = 4
Weights = List[np.ndarray]
SiteSplit = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]  # Xt, yt, Xv, yv

# ---------------------------------------------------------------- data + metrics


def site_data(n_train: int, seed: int, n_val: int = 1500,
              sites: Sequence[str] = tuple(workload.SITES)) -> Dict[str, SiteSplit]:
    """Coarse (deployed, 19-feature) events per site: n_train to learn from, n_val to score."""
    out = {}
    for i, key in enumerate(sites):
        X, y = workload.generate(workload.SITES[key], n_train + n_val, seed=seed * 1000 + i)
        out[key] = (X[:n_train], y[:n_train], X[n_train:], y[n_train:])
    return out


def balanced_acc(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean([(pred[y == c] == c).mean() for c in np.unique(y)]))


def anomaly_recall(pred: np.ndarray, y: np.ndarray) -> float:
    """Of the events that really were anomalies, how many did the head flag."""
    m = y == ANOMALY
    return float((pred[m] == ANOMALY).mean()) if m.any() else float("nan")


def epochs_for(n: int, min_steps: int = 40, batch: int = 64) -> int:
    """Epochs so every client takes at least `min_steps` optimizer steps per round.

    Without this a 50-event site takes one step per epoch and simply never
    trains, which would make any small-data comparison meaningless.
    """
    return max(2, math.ceil(min_steps / math.ceil(n / batch)))


def _fresh(weights: Optional[Weights] = None) -> FederatedHead:
    h = FederatedHead(seed=0)
    if weights is not None:
        h.set_weights(weights)
    return h


def _local_fit(weights: Weights, X, y, epochs: int, seed: int) -> Weights:
    h = _fresh(weights)
    h.fit(X, y, epochs=epochs, seed=seed)
    return h.get_weights()


def fedavg(global_w: Weights, updates: List[Tuple[Weights, int]]) -> Weights:
    total = sum(n for _, n in updates)
    return [sum(w[i] * (n / total) for w, n in updates) for i in range(len(global_w))]


# ---------------------------------------------------------------- 1. FL vs local


def train_federated(data: Dict[str, SiteSplit], rounds: int, epochs: int, seed: int,
                    init: Optional[Weights] = None) -> Weights:
    g = init if init is not None else _fresh().get_weights()
    for r in range(rounds):
        # A distinct shuffle seed per node: identical orders would correlate honest updates
        ups = [(_local_fit(g, Xt, yt, epochs, seed * 1000 + r * 10 + i), len(yt))
               for i, (Xt, yt, _, _) in enumerate(data.values())]
        g = fedavg(g, ups)
    return g


def fl_vs_local(sizes: Sequence[int], seeds: Sequence[int], rounds: int = 6) -> List[dict]:
    """Every site has n events. Compare four ways of getting a head for each site."""
    rows = []
    for n in sizes:
        ep = epochs_for(n)
        for seed in seeds:
            data = site_data(n, seed)
            g = train_federated(data, rounds, ep, seed)
            untrained = _fresh().get_weights()
            for key, (Xt, yt, Xv, yv) in data.items():
                arms = {
                    "untrained": untrained,
                    # (rounds + 1) * ep: the same local epochs the fine-tune arm ends up with
                    "local only": _local_fit(untrained, Xt, yt, (rounds + 1) * ep, seed),
                    "federated": g,
                    "federated + local fine-tune": _local_fit(g, Xt, yt, ep, seed),
                }
                for arm, w in arms.items():
                    p = _fresh(w).predict(Xv)
                    rows.append({"n_per_site": n, "seed": seed, "site": key, "arm": arm,
                                 "balanced_acc": balanced_acc(p, yv),
                                 "anomaly_recall": anomaly_recall(p, yv)})
    return rows


def cold_start(local_sizes: Sequence[int], seeds: Sequence[int], fleet_n: int = 3000,
               rounds: int = 6) -> List[dict]:
    """Leave-one-site-out: a new camera joins a fleet that has already trained.

    The fleet model never saw the new site. The new node has only
    `local n` events of its own (0 = installed today).
    """
    rows = []
    sites = list(workload.SITES)
    for seed in seeds:
        full = site_data(max(max(local_sizes), fleet_n), seed)
        for new in sites:
            fleet = {k: (v[0][:fleet_n], v[1][:fleet_n], v[2], v[3]) for k, v in full.items() if k != new}
            g = train_federated(fleet, rounds, epochs_for(fleet_n), seed)
            Xt_all, yt_all, Xv, yv = full[new]
            for n in local_sizes:
                Xt, yt = Xt_all[:n], yt_all[:n]
                arms = {"untrained": _fresh().get_weights(), "fleet model as-is": g}
                if n > 0:
                    ep = epochs_for(n)
                    arms["local only"] = _local_fit(_fresh().get_weights(), Xt, yt, (rounds + 1) * ep, seed)
                    arms["fleet model + local fine-tune"] = _local_fit(g, Xt, yt, ep, seed)
                for arm, w in arms.items():
                    p = _fresh(w).predict(Xv)
                    rows.append({"new_site": new, "local_n": n, "seed": seed, "arm": arm,
                                 "balanced_acc": balanced_acc(p, yv),
                                 "anomaly_recall": anomaly_recall(p, yv)})
    return rows


# ---------------------------------------------------------------- taxonomy


def taxonomy_with_finetune(accs: Sequence[float], seeds: Sequence[int], samples: int = 6000,
                           rounds: int = 6, epochs: int = 2) -> List[dict]:
    """compare_taxonomy.py's coarse vs fine sweep, plus a federated + fine-tune arm.

    Local-only trains rounds + 1 rounds' worth of epochs, matching the fine-tune arm's
    local epochs. `n` is the number of validation events in each slice.
    """
    rows = []
    for acc, seed in [(a, s) for a in accs for s in seeds]:
        data = cmp.build(acc, samples, seed)
        for tax in ("coarse", "fine"):
            fed = cmp.train_fedavg(data, tax, rounds, epochs)
            tuned = {}
            for key, d in data.items():
                h = cmp._new_head(tax)
                h.set_weights(fed(key).get_weights())
                h.fit(cmp._view(d["Xt"], tax), d["yt"], epochs=epochs, seed=seed)
                tuned[key] = h
            arms = {"local only": cmp.train_local(data, tax, rounds + 1, epochs),
                    "federated": fed,
                    "federated + local fine-tune": lambda k, t=tuned: t[k]}
            for arm, head_for in arms.items():
                s = cmp.score(data, tax, head_for)
                rows.append({"classifier_acc": acc, "seed": seed, "taxonomy": tax, "arm": arm,
                             **{k: v for k, v in s.items() if not k.startswith("_")},
                             **{f"n: {k}": v for k, v in s["_n"].items()}})
    return rows


# ---------------------------------------------------------------- 2. frame rate

_PHASE = re.compile(r"^\s+(baseline|fl|recovery)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)")
_STAGE = re.compile(r"^\s{4}(\w+)\s+(CPU|HAILO|LIVE)\s+([\d.]+) ms")


def parse_run_node(text: str) -> dict:
    phases, stages = {}, {}
    for line in text.splitlines():
        m = _PHASE.match(line)
        if m:
            phases[m.group(1)] = {"frames": int(m.group(2)), "fps": float(m.group(3)),
                                  "p50": float(m.group(4)), "p95": float(m.group(5)),
                                  "p99": float(m.group(6))}
        m = _STAGE.match(line)
        if m:
            stages[m.group(1)] = {"unit": m.group(2), "ms": float(m.group(3))}
    if set(phases) != {"baseline", "fl", "recovery"}:
        raise RuntimeError("run_node.py output had no complete FRAME RATE table:\n" + text[-2000:])
    return {"phases": phases, "stages": stages}


def run_fps(name: str, args: Sequence[str], refresh: bool = False) -> dict:
    """Run sim/run_node.py once (a real Flower round in separate processes), cached in sim/_run."""
    cache = RUN / f"fps_{name}.json"
    if cache.exists() and not refresh:
        return json.loads(cache.read_text(encoding="utf-8"))
    cmd = [sys.executable, str(SIM / "run_node.py"), "--clients", "1", "--cores", "4",
           "--port", "0", *args]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900, cwd=REPO)
    if proc.returncode != 0:
        raise RuntimeError(f"run_node.py {args} failed:\n{proc.stderr[-2000:]}")
    result = {"name": name, "args": list(args), **parse_run_node(proc.stdout)}
    RUN.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(result, indent=2), encoding="utf-8")
    (RUN / f"fps_{name}.txt").write_text(proc.stdout, encoding="utf-8")
    return result


# ---------------------------------------------------------------- 3. robustness


def _hide_anomalies(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Targeted label flip: every anomaly relabelled as the routine version of its class."""
    routine = X[:, :4].argmax(axis=1)
    return np.where(y == ANOMALY, routine, y)


def attacker_update(attack: str, g: Weights, Xt, yt, epochs: int, seed: int) -> Weights:
    if attack == "label flip (hide anomalies)":
        return _local_fit(g, Xt, _hide_anomalies(Xt, yt), epochs, seed)
    if attack.startswith("label flip, boosted x"):
        # x6 is a blunt model-replacement attempt; x2.5 is tuned to stay under
        # robust.py's 3x-median norm limit, which is what a careful attacker would do
        boost = float(attack.rsplit("x", 1)[1].split()[0])
        w = _local_fit(g, Xt, _hide_anomalies(Xt, yt), epochs, seed)
        return [gi + boost * (wi - gi) for gi, wi in zip(g, w)]
    honest = _local_fit(g, Xt, yt, epochs, seed)
    if attack == "sign flip x4":
        return [gi - 4.0 * (wi - gi) for gi, wi in zip(g, honest)]
    if attack == "random noise":
        rng = np.random.default_rng(seed)
        return [gi + rng.normal(0, 0.3, gi.shape).astype(np.float32) for gi in g]
    if attack == "NaN (diverged node)":
        return [np.full_like(gi, np.nan) for gi in g]
    raise ValueError(attack)


def trimmed_mean(g: Weights, ws: List[Weights], k: int = 1) -> Weights:
    """Coordinate-wise trimmed mean, no screening. The textbook FedTrimmedAvg baseline.

    Survives one NaN node only because np.sort puts NaN last and k=1 trims it;
    two NaN nodes would poison it.
    """
    out = []
    for i in range(len(g)):
        s = np.sort(np.stack([w[i] for w in ws]), axis=0)
        out.append(s[k:len(ws) - k].mean(axis=0).astype(g[i].dtype))
    return out


def _robust_step(g, ws, ns, ids, gate_eval: Optional[Callable[[Weights], float]]):
    agg = robust.RobustAggregator()  # shipped defaults, so this tests what would be deployed
    ups = [robust.ClientUpdate(i, w, n) for i, w, n in zip(ids, ws, ns)]
    cand, rep = agg.aggregate(g, ups)
    if cand is None:
        return g, rep.rejected, False
    if gate_eval is not None:
        ok, _ = robust.validate_candidate(gate_eval(cand), gate_eval(g))
        if not ok:
            return g, rep.rejected, False
    return cand, rep.rejected, True


def robustness(attacks: Sequence[str], seeds: Sequence[int], nodes_per_site: int = 2,
               n: int = 1600, rounds: int = 6, attacker_site: str = "driveway") -> List[dict]:
    """6-node fleet (3 sites x 2 cameras); one camera is broken or hostile every round."""
    strategies = ("FedAvg", "trimmed mean", "robust.py", "robust.py + validation gate")
    rows = []
    for seed in seeds:
        gate_n = 200
        base = site_data(n * nodes_per_site + gate_n, seed, n_val=600)
        nodes = {f"{k}-{j}": (Xt[j * n:(j + 1) * n], yt[j * n:(j + 1) * n])
                 for k, (Xt, yt, _, _) in base.items() for j in range(nodes_per_site)}
        bad = f"{attacker_site}-0"
        Xv = np.concatenate([v[2] for v in base.values()])
        yv = np.concatenate([v[3] for v in base.values()])
        # The server's own labelled holdout, used only by the gate and disjoint
        # from both the nodes' training data and the scoring set
        hold_X = np.concatenate([v[0][-gate_n:] for v in base.values()])
        hold_y = np.concatenate([v[1][-gate_n:] for v in base.values()])
        gate = lambda w: balanced_acc(_fresh(w).predict(hold_X), hold_y)  # noqa: E731
        ep = epochs_for(n)
        for attack in attacks:
            is_attacked = attack != "none"
            for strat in strategies:
                g, caught, false_rej = _fresh().get_weights(), 0, 0
                for r in range(rounds):
                    ids, ws, ns = [], [], []
                    for i, (nid, (Xt, yt)) in enumerate(nodes.items()):
                        s = seed * 1000 + r * 10 + i
                        is_bad = nid == bad and attack != "none"
                        ws.append(attacker_update(attack, g, Xt, yt, ep, s) if is_bad
                                  else _local_fit(g, Xt, yt, ep, s))
                        ids.append(nid)
                        ns.append(len(yt))
                    if strat == "FedAvg":
                        g = fedavg(g, list(zip(ws, ns)))
                    elif strat == "trimmed mean":
                        g = trimmed_mean(g, ws)
                    else:
                        g, rejected, _ = _robust_step(
                            g, ws, ns, ids, gate if strat.endswith("gate") else None)
                        caught += is_attacked and bad in rejected
                        false_rej += sum(1 for x in rejected if x != bad or not is_attacked)
                ok = all(np.all(np.isfinite(w)) for w in g)
                p = _fresh(g).predict(Xv) if ok else np.zeros_like(yv)
                rows.append({"seed": seed, "attack": attack, "strategy": strat,
                             "balanced_acc": balanced_acc(p, yv) if ok else 0.0,
                             "anomaly_recall": anomaly_recall(p, yv) if ok else 0.0,
                             "model_destroyed": not ok,
                             "attacker_rejected_rounds": caught if strat.startswith("robust") else None,
                             "honest_rejections": false_rej if strat.startswith("robust") else None})
    return rows
