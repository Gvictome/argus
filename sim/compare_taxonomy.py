"""
Does knowing the subtype make the federated head better, and how good does
the crop classifier have to be before it stops helping?

Same events, two views:
  coarse  the 19 features the deployed head sees today
  fine    those 19 plus the crop classifier's carrier / make / size

Each view is trained two ways over the four fine sites:
  fedavg  in-process FedAvg, weighted by sample count (no gRPC; fl.py
          measures the transport, this measures the learning)
  local   each site trains alone on its own data

Repeated for each classifier accuracy in --acc. Results print as a table
and land in sim/_run/taxonomy_report.{json,md}.

    python sim/compare_taxonomy.py
    python sim/compare_taxonomy.py --acc 1.0,0.9 --rounds 4 --samples 4000
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import workload
import workload_fine as wf
from head import FederatedHead

RUN = Path(__file__).resolve().parent / "_run"
DWELL = 13

SiteData = Dict[str, np.ndarray]


def _slices(site: wf.FineSite, meta: Dict[str, np.ndarray], X: np.ndarray) -> Dict[str, np.ndarray]:
    """Named groups of events the subtype is supposed to help with."""
    cls, hour = meta["cls"], meta["hour"]
    start, end = site.base.active_hours
    night = (hour < start) | (hour >= end)
    car = (cls == 1) & (meta["carrier"] == 0)
    household = car & np.isin(meta["make"], [wf.MAKES.index(m) for m in site.household_makes])
    in_window = np.zeros(len(cls), dtype=bool)
    for name, (s, e) in wf.CARRIER_WINDOWS.items():
        in_window |= (meta["carrier"] == wf.CARRIERS.index(name)) & (hour >= s) & (hour < e)
    truck = (cls == 1) & (meta["carrier"] > 0)
    pkg = cls == 3

    ev = {"cls": cls, "hour": hour, "dwell": X[:, DWELL]}
    coarse_rule = np.where(workload.anomaly_mask(site.base, ev), wf.ANOMALY, cls)

    return {
        "household car, night": household & night,
        "stranger car, night": car & ~household & night,
        "carrier truck, in window": truck & in_window,
        "carrier truck, out of window": truck & ~in_window,
        "usps regular mail": pkg & (meta["size"] == wf.PACKAGE_SIZES.index("mail")),
        "large parcel, carrier": pkg & (meta["size"] == wf.PACKAGE_SIZES.index("large"))
        & (meta["carrier"] > 0),
        "_coarse_rule": coarse_rule,
    }


def build(acc: float, samples: int, seed: int) -> Dict[str, SiteData]:
    out = {}
    for i, (key, site) in enumerate(wf.SITES.items()):
        X, y, meta = wf.generate(site, samples, seed=seed + 100 * i, classifier_acc=acc)
        slices = _slices(site, meta, X)
        idx = np.random.default_rng(seed + i).permutation(len(X))
        cut = int(len(X) * 0.8)
        tr, va = idx[:cut], idx[cut:]
        out[key] = {
            "Xt": X[tr], "yt": y[tr], "Xv": X[va], "yv": y[va],
            "slices": {k: v[va] for k, v in slices.items()},
        }
    return out


def _view(X: np.ndarray, taxonomy: str) -> np.ndarray:
    return wf.coarse_view(X) if taxonomy == "coarse" else X


def _new_head(taxonomy: str) -> FederatedHead:
    n = workload.N_FEATURES if taxonomy == "coarse" else wf.N_FEATURES
    return FederatedHead(seed=0, n_features=n)


def train_fedavg(data: Dict[str, SiteData], taxonomy: str, rounds: int,
                 epochs: int) -> Callable[[str], FederatedHead]:
    glob = _new_head(taxonomy)
    for r in range(rounds):
        updates: List[Tuple[List[np.ndarray], int]] = []
        for d in data.values():
            h = _new_head(taxonomy)
            h.set_weights(glob.get_weights())
            h.fit(_view(d["Xt"], taxonomy), d["yt"], epochs=epochs, seed=r)
            updates.append((h.get_weights(), len(d["yt"])))
        total = sum(n for _, n in updates)
        glob.set_weights([
            sum(w[i] * (n / total) for w, n in updates)
            for i in range(len(updates[0][0]))
        ])
    return lambda _key: glob


def train_local(data: Dict[str, SiteData], taxonomy: str, rounds: int,
                epochs: int) -> Callable[[str], FederatedHead]:
    heads = {}
    for key, d in data.items():
        h = _new_head(taxonomy)
        h.fit(_view(d["Xt"], taxonomy), d["yt"], epochs=rounds * epochs)
        heads[key] = h
    return lambda key: heads[key]


def score(data: Dict[str, SiteData], taxonomy: str,
          head_for: Callable[[str], FederatedHead]) -> Dict[str, float]:
    preds, ys = [], []
    hits: Dict[str, List[bool]] = {"subtype-decided events": []}
    for key, d in data.items():
        p = head_for(key).predict(_view(d["Xv"], taxonomy))
        preds.append(p)
        ys.append(d["yv"])
        correct = p == d["yv"]
        decided = d["slices"]["_coarse_rule"] != d["yv"]
        hits["subtype-decided events"].extend(correct[decided])
        for name, m in d["slices"].items():
            if not name.startswith("_"):
                hits.setdefault(name, []).extend(correct[m])
    p, y = np.concatenate(preds), np.concatenate(ys)
    per_class = [float((p[y == c] == c).mean()) for c in range(wf.N_CLASSES) if (y == c).any()]
    out = {"balanced accuracy": float(np.mean(per_class))}
    out.update({k: (float(np.mean(v)) if v else float("nan")) for k, v in hits.items()})
    out["_n"] = {k: len(v) for k, v in hits.items()}
    return out


def _md_table(results: List[dict], metrics: List[str]) -> str:
    head = "| classifier acc | training | metric | coarse | fine | delta |\n|---|---|---|---|---|---|\n"
    rows = []
    for r in results:
        for m in metrics:
            c, f = r["coarse"][m], r["fine"][m]
            rows.append(f"| {r['acc']:.2f} | {r['strategy']} | {m} | {c:.3f} | {f:.3f} | {f - c:+.3f} |")
    return head + "\n".join(rows) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acc", default="1.0,0.9,0.75,0.5",
                    help="comma-separated crop classifier accuracies to sweep")
    ap.add_argument("--samples", type=int, default=6000, help="events per site")
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=2, help="local epochs per round")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    accs = [float(a) for a in args.acc.split(",")]

    print(f"\n  sites: {', '.join(wf.SITES)}   coarse {workload.N_FEATURES} features, "
          f"fine {wf.N_FEATURES}   {args.rounds} rounds x {args.epochs} epochs\n")

    results = []
    t0 = time.perf_counter()
    for acc in accs:
        data = build(acc, args.samples, args.seed)
        for strategy, trainer in (("fedavg", train_fedavg), ("local", train_local)):
            row = {"acc": acc, "strategy": strategy}
            for taxonomy in ("coarse", "fine"):
                row[taxonomy] = score(data, taxonomy,
                                      trainer(data, taxonomy, args.rounds, args.epochs))
            results.append(row)
            c, f = row["coarse"], row["fine"]
            print(f"  acc={acc:.2f} {strategy:6}  balanced {c['balanced accuracy']:.3f} -> "
                  f"{f['balanced accuracy']:.3f}   subtype-decided "
                  f"{c['subtype-decided events']:.3f} -> {f['subtype-decided events']:.3f}")

    metrics = [k for k in results[0]["coarse"] if not k.startswith("_")]
    RUN.mkdir(parents=True, exist_ok=True)
    (RUN / "taxonomy_report.json").write_text(json.dumps({
        "args": vars(args), "results": results,
        "seconds": round(time.perf_counter() - t0, 1),
    }, indent=2), encoding="utf-8")
    (RUN / "taxonomy_report.md").write_text(
        "# Coarse vs fine taxonomy\n\n"
        f"{args.samples} events per site, sites {', '.join(wf.SITES)}, "
        f"{args.rounds} rounds x {args.epochs} epochs, seed {args.seed}.\n"
        "Slice sizes per run are in taxonomy_report.json under `_n`.\n\n"
        + _md_table(results, metrics),
        encoding="utf-8",
    )
    print(f"\n  done in {time.perf_counter() - t0:.0f}s  ->  {RUN / 'taxonomy_report.md'}\n")


if __name__ == "__main__":
    main()
