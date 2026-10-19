"""
Federated head vs base, local-only, FedAvg and FedAvg + fine-tune on MEVA.

In-process (no gRPC): the same RobustAggregator and the same numpy head the
Flower server and clients use, driven round by round, so many seeds are
affordable. Every arm sees the same clip-level holdout within a seed.

    python sim/meva_experiment.py --seeds 10 --out data/benchmarks/meva_fl_experiment.json
    python sim/meva_experiment.py --markdown data/benchmarks/meva_fl_experiment.json

Arms, each scored at 3, 5 and 8 rounds (local-only gets the same number of
local epochs, so every arm has the same training budget):

    untrained         the shared random initial head (the "base")
    local-only        each camera trains alone
    fedavg            plain sample-weighted average
    robust            RobustAggregator, then average survivors
    robust+finetune   robust model, then 3 local epochs per camera
    fedavg+finetune   FedAvg model, then 3 local epochs per camera
    cold-start        robust model trained WITHOUT camera c, scored on c

Metrics are balanced accuracy and per-class recall, never plain accuracy.
Reported over seeds with 95% t-intervals; a seed changes the head's initial
weights, shuffling, AND which whole clips are held out.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from head import FederatedHead  # noqa: E402
from metrics import balanced_accuracy, per_class_recall  # noqa: E402
from src.federated.features import LABEL_NAMES, N_CLASSES  # noqa: E402
from src.federated.robust import ClientUpdate, RobustAggregator  # noqa: E402
from src.federated.store import SampleStore  # noqa: E402

CHECKPOINTS = (3, 5, 8)
EPOCHS_PER_ROUND = 3
FINETUNE_EPOCHS = 3
HOLDOUT_FRAC = 0.2
TRAIN_CAP = 4000  # per camera: a node trains on its most recent events, not millions


@dataclass
class CameraData:
    name: str
    Xt: np.ndarray
    yt: np.ndarray
    Xv: np.ndarray
    yv: np.ndarray


# ---- data ---------------------------------------------------------------
def load_rows(store_dir: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    rows = [r for r in SampleStore(Path(store_dir) / "samples.jsonl")._rows
            if r.get("y") is not None]
    X = np.array([r["x"] for r in rows], dtype=np.float32)
    y = np.array([r["y"] for r in rows], dtype=np.int64)
    clips = [r.get("meta", {}).get("clip", "") for r in rows]
    return X, y, clips


def resplit(name: str, X: np.ndarray, y: np.ndarray, clips: Sequence[str],
            seed: int, frac: float = HOLDOUT_FRAC, cap: int = TRAIN_CAP) -> CameraData:
    """Hold out whole clips (seeded), then cap the training rows.

    Never splits a clip across train and holdout: neighbouring events from
    one clip are near-duplicates and would leak.
    """
    uniq = sorted(set(clips))
    rng = np.random.default_rng(seed * 1009 + 17)
    order = rng.permutation(len(uniq))
    n_hold = max(1, int(round(len(uniq) * frac))) if len(uniq) > 1 else 0
    hold = {uniq[i] for i in order[:n_hold]}
    is_hold = np.array([c in hold for c in clips])
    Xt, yt, Xv, yv = X[~is_hold], y[~is_hold], X[is_hold], y[is_hold]
    if len(yt) > cap:
        keep = np.sort(rng.choice(len(yt), cap, replace=False))
        Xt, yt = Xt[keep], yt[keep]
    return CameraData(name, Xt, yt, Xv, yv)


# ---- training arms --------------------------------------------------------
def _client_heads(cams: Sequence[CameraData]) -> List[FederatedHead]:
    return [FederatedHead(seed=0, n_features=int(c.Xt.shape[1])) for c in cams]


def fedavg_weights(updates: Sequence[Tuple[List[np.ndarray], int]]) -> List[np.ndarray]:
    total = float(sum(n for _, n in updates))
    return [
        sum(w[i].astype(np.float64) * (n / total) for w, n in updates).astype(np.float32)
        for i in range(len(updates[0][0]))
    ]


def run_federation(cams: Sequence[CameraData], init: List[np.ndarray], rounds: int,
                   aggregation: str, seed: int) -> Tuple[Dict[int, List[np.ndarray]], List[dict]]:
    """Run `rounds` rounds; return the global model at each checkpoint round
    and the per-round accept/reject log (robust only)."""
    heads = _client_heads(cams)
    agg = RobustAggregator()
    global_w = [np.array(w) for w in init]
    snaps: Dict[int, List[np.ndarray]] = {}
    log: List[dict] = []
    for r in range(1, rounds + 1):
        updates = []
        for i, (h, c) in enumerate(zip(heads, cams)):
            h.set_weights(global_w)
            h.fit(c.Xt, c.yt, epochs=EPOCHS_PER_ROUND, seed=seed * 7919 + r * 131 + i)
            updates.append((c.name, [np.array(w) for w in h.get_weights()], len(c.yt)))
        if aggregation == "fedavg":
            global_w = fedavg_weights([(w, n) for _, w, n in updates])
        else:
            new, rep = agg.aggregate(global_w, [
                ClientUpdate(node_id=nm, weights=w, num_samples=n) for nm, w, n in updates])
            log.append({"round": r, "accepted": len(rep.accepted),
                        "rejected": list(rep.rejected)})
            if new is not None:
                global_w = new
        if r in CHECKPOINTS:
            snaps[r] = [np.array(w) for w in global_w]
    return snaps, log


def train_local(c: CameraData, init: List[np.ndarray], rounds_list: Sequence[int],
                seed: int, cam_idx: int) -> Dict[int, List[np.ndarray]]:
    """Local-only head at the epoch budget of each checkpoint."""
    h = FederatedHead(seed=0, n_features=int(c.Xt.shape[1]))
    h.set_weights(init)
    out: Dict[int, List[np.ndarray]] = {}
    done = 0
    for r in sorted(rounds_list):
        h.fit(c.Xt, c.yt, epochs=(r - done) * EPOCHS_PER_ROUND, seed=seed * 31 + r + cam_idx)
        done = r
        out[r] = [np.array(w) for w in h.get_weights()]
    return out


def finetune(c: CameraData, weights: List[np.ndarray], seed: int, cam_idx: int) -> List[np.ndarray]:
    h = FederatedHead(seed=0, n_features=int(c.Xt.shape[1]))
    h.set_weights(weights)
    h.fit(c.Xt, c.yt, epochs=FINETUNE_EPOCHS, seed=seed * 53 + cam_idx)
    return [np.array(w) for w in h.get_weights()]


# ---- scoring -------------------------------------------------------------
def score(per_camera_weights: Sequence[List[np.ndarray]],
          cams: Sequence[CameraData]) -> dict:
    """Score one head per camera on that camera's holdout.

    macro_bal_acc: mean over cameras of per-camera balanced accuracy.
    pooled_bal_acc / recall: every holdout event pooled, classes weighted equally.
    """
    per_cam, preds, ys = {}, [], []
    for w, c in zip(per_camera_weights, cams):
        if len(c.yv) == 0:
            continue
        h = FederatedHead(seed=0, n_features=int(c.Xv.shape[1]))
        h.set_weights(w)
        p = h.predict(c.Xv)
        per_cam[c.name] = balanced_accuracy(p, c.yv)
        preds.append(p)
        ys.append(c.yv)
    pred, y = np.concatenate(preds), np.concatenate(ys)
    rec = per_class_recall(pred, y)
    return {
        "macro_bal_acc": float(np.mean(list(per_cam.values()))),
        "pooled_bal_acc": balanced_accuracy(pred, y),
        "recall": {LABEL_NAMES[k]: v for k, v in enumerate(rec) if v is not None},
        "per_camera": per_cam,
    }


def score_detector_only(cams: Sequence[CameraData]) -> dict:
    """The system without a head: label every detection routine_<its class>.

    Features 0-3 one-hot the detector's class. This is what the node does
    when no head is installed, and it can never say "anomaly".
    """
    per_cam, preds, ys = {}, [], []
    for c in cams:
        if len(c.yv) == 0:
            continue
        p = np.where(c.Xv[:, :4].any(axis=1), c.Xv[:, :4].argmax(axis=1), 0)
        per_cam[c.name] = balanced_accuracy(p, c.yv)
        preds.append(p)
        ys.append(c.yv)
    pred, y = np.concatenate(preds), np.concatenate(ys)
    rec = per_class_recall(pred, y)
    return {"macro_bal_acc": float(np.mean(list(per_cam.values()))),
            "pooled_bal_acc": balanced_accuracy(pred, y),
            "recall": {LABEL_NAMES[k]: v for k, v in enumerate(rec) if v is not None},
            "per_camera": per_cam}


def run_seed(args: Tuple[str, List[str], int, bool]) -> dict:
    """All arms for one seed. Top level so it can run in a process pool."""
    root, names, seed, cold = args
    cams = []
    for n in names:
        X, y, clips = load_rows(Path(root) / n)
        cams.append(resplit(n, X, y, clips, seed))
    nf = int(cams[0].Xt.shape[1])
    init = FederatedHead(seed=seed, n_features=nf).get_weights()
    res: Dict[str, dict] = {}
    res["untrained"] = score([init] * len(cams), cams)
    res["detector-only"] = score_detector_only(cams)

    rob, rob_log = run_federation(cams, init, max(CHECKPOINTS), "robust", seed)
    fav, _ = run_federation(cams, init, max(CHECKPOINTS), "fedavg", seed)
    local = [train_local(c, init, CHECKPOINTS, seed, i) for i, c in enumerate(cams)]
    for r in CHECKPOINTS:
        res[f"local@{r}"] = score([l[r] for l in local], cams)
        res[f"fedavg@{r}"] = score([fav[r]] * len(cams), cams)
        res[f"robust@{r}"] = score([rob[r]] * len(cams), cams)
    last = max(CHECKPOINTS)
    res[f"robust+finetune@{last}"] = score(
        [finetune(c, rob[last], seed, i) for i, c in enumerate(cams)], cams)
    res[f"fedavg+finetune@{last}"] = score(
        [finetune(c, fav[last], seed, i) for i, c in enumerate(cams)], cams)
    res["robust_log"] = rob_log

    if cold:
        # Leave one camera out of the federation, score the model on it.
        heads = []
        for k in range(len(cams)):
            rest = [c for j, c in enumerate(cams) if j != k]
            snaps, _ = run_federation(rest, init, last, "robust", seed + 1000 + k)
            heads.append(snaps[last])
        res["cold-start@8"] = score(heads, cams)
    return res


# ---- statistics ----------------------------------------------------------
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
        8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 15: 2.145, 20: 2.093,
        30: 2.045}


def mean_ci(values: Sequence[float]) -> Tuple[float, float]:
    """Mean and 95% half-width (t-interval over seeds)."""
    v = np.asarray(values, dtype=float)
    n = len(v)
    if n < 2:
        return float(v.mean()), float("nan")
    df = n - 1
    t = _T95.get(df) or _T95[min(_T95, key=lambda k: abs(k - df))]
    return float(v.mean()), float(t * v.std(ddof=1) / math.sqrt(n))


def summarise(results: Sequence[dict]) -> dict:
    arms = [k for k in results[0] if k != "robust_log"]
    out: Dict[str, dict] = {}
    for a in arms:
        have = [r for r in results if a in r]  # cold-start runs on fewer seeds
        entry = {"n_seeds": len(have)}
        for metric in ("macro_bal_acc", "pooled_bal_acc"):
            entry[metric] = mean_ci([r[a][metric] for r in have])
        classes = sorted({c for r in have for c in r[a]["recall"]})
        entry["recall"] = {c: mean_ci([r[a]["recall"][c] for r in have if c in r[a]["recall"]])
                           for c in classes}
        out[a] = entry
    return out


def rejection_summary(results: Sequence[dict]) -> dict:
    """How often honest-only robust runs rejected anyone (all peers are honest here)."""
    rounds = [e for r in results for e in r["robust_log"]]
    bad = [e for e in rounds if e["rejected"]]
    return {"rounds": len(rounds), "rounds_with_rejections": len(bad),
            "rejected_updates": sum(len(e["rejected"]) for e in rounds)}


def paired(results: Sequence[dict], a: str, b: str, metric: str = "macro_bal_acc") -> Tuple[float, float]:
    """a minus b, paired by seed."""
    return mean_ci([r[a][metric] - r[b][metric] for r in results if a in r and b in r])


# ---- reporting -----------------------------------------------------------
def _f(m: Tuple[float, float]) -> str:
    return f"{m[0]:.3f} +/- {m[1]:.3f}"


def to_markdown(results: Sequence[dict]) -> str:
    s = summarise(results)
    lines = ["| Arm | Seeds | Macro bal. acc (per camera) | Pooled bal. acc | person | vehicle | package | anomaly |",
             "|---|---|---|---|---|---|---|---|"]
    for a, e in s.items():
        rec = e["recall"]
        cell = lambda k: _f(rec[k]) if k in rec else "n/a"  # noqa: E731
        lines.append(f"| {a} | {e['n_seeds']} | {_f(e['macro_bal_acc'])} | {_f(e['pooled_bal_acc'])} | "
                     f"{cell('routine_person')} | {cell('routine_vehicle')} | "
                     f"{cell('routine_package')} | {cell('anomaly')} |")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default="data/datasets/meva/events")
    ap.add_argument("--cameras", nargs="*", default=None)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--cold-seeds", type=int, default=5,
                    help="seeds that also run the leave-one-camera-out arm (expensive)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="data/benchmarks/meva_fl_experiment.json")
    ap.add_argument("--markdown", default=None, help="print the table for an existing results json")
    a = ap.parse_args(argv)

    if a.markdown:
        data = json.loads(Path(a.markdown).read_text())
        print(to_markdown(data["results"]))
        print(json.dumps(rejection_summary(data["results"])))
        for a, b in (("robust@8", "local@8"), ("robust+finetune@8", "local@8"),
                     ("robust@8", "fedavg@8"), ("robust+finetune@8", "fedavg+finetune@8"),
                     ("robust@8", "detector-only"), ("robust+finetune@8", "detector-only"),
                     ("cold-start@8", "untrained")):
            for m in ("macro_bal_acc", "pooled_bal_acc"):
                print(f"paired {a} - {b} [{m}]: {_f(paired(data['results'], a, b, m))}")
        return 0

    root = Path(a.root)
    names = a.cameras or sorted(d.name for d in root.iterdir() if (d / "samples.jsonl").exists())
    jobs = [(str(root), names, s, s < a.cold_seeds) for s in range(a.seeds)]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        results = []
        for i, r in enumerate(pool.map(run_seed, jobs), 1):
            results.append(r)
            print(f"seed {i}/{len(jobs)} done ({time.time() - t0:.0f}s)", flush=True)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"cameras": names, "seeds": a.seeds, "results": results},
                              indent=1), encoding="utf-8")
    print(to_markdown(results))
    print(json.dumps(rejection_summary(results)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
