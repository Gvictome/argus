"""
Server-side evaluation for the simulator and the hybrid demo.

Per-class recall and balanced accuracy on the pooled clip-level holdout of
several MEVA cameras, so each federated round can report how the aggregated
model does on clips no node trained on. Plain accuracy is never reported:
person dominates every security dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from head import FederatedHead
from src.federated.features import LABEL_NAMES, N_CLASSES
from src.federated.store import SampleStore


@dataclass(frozen=True)
class EvalSet:
    """One camera's holdout rows."""
    name: str
    X: np.ndarray
    y: np.ndarray


def per_class_recall(pred: np.ndarray, y: np.ndarray,
                     n_classes: int = N_CLASSES) -> List[Optional[float]]:
    """Recall per class; None for a class with no examples."""
    out: List[Optional[float]] = []
    for c in range(n_classes):
        m = y == c
        out.append(float((pred[m] == c).mean()) if m.any() else None)
    return out


def balanced_accuracy(pred: np.ndarray, y: np.ndarray,
                      n_classes: int = N_CLASSES) -> float:
    """Mean recall over the classes that are present."""
    present = [r for r in per_class_recall(pred, y, n_classes) if r is not None]
    return float(np.mean(present)) if present else 0.0


def load_eval_sets(store_dirs: Sequence[Path]) -> List[EvalSet]:
    """Holdout rows (meta.split == "holdout") of each store directory."""
    sets: List[EvalSet] = []
    for d in store_dirs:
        d = Path(d)
        path = d / "samples.jsonl" if d.is_dir() else d
        rows = [r for r in SampleStore(path)._rows
                if r.get("y") is not None and r.get("meta", {}).get("split") == "holdout"]
        if not rows:
            continue
        name = (path.parent if path.suffix else path).name
        sets.append(EvalSet(
            name=name,
            X=np.array([r["x"] for r in rows], dtype=np.float32),
            y=np.array([r["y"] for r in rows], dtype=np.int64),
        ))
    return sets


def make_evaluate_fn(sets: Sequence[EvalSet]) -> Optional[Callable]:
    """Flower `evaluate_fn`: score the aggregated weights on every holdout.

    Returns None when there is nothing to evaluate on, which makes Flower
    skip centralised evaluation.
    """
    if not sets:
        return None
    X = np.concatenate([s.X for s in sets])
    y = np.concatenate([s.y for s in sets])
    head = FederatedHead(seed=0, n_features=int(X.shape[1]))

    def evaluate(server_round: int, parameters, config) -> Tuple[float, Dict]:
        head.set_weights(parameters)
        loss, _ = head.evaluate(X, y)
        pred = head.predict(X)
        recall = per_class_recall(pred, y)
        return float(loss), {
            "balanced_accuracy": balanced_accuracy(pred, y),
            "per_class_recall": {
                LABEL_NAMES[c]: r for c, r in enumerate(recall) if r is not None
            },
            "per_camera_balanced_accuracy": {
                s.name: balanced_accuracy(head.predict(s.X), s.y) for s in sets
            },
            "holdout_events": int(len(y)),
        }

    return evaluate
