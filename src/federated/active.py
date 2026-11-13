"""
Active learning scores and confident auto-labels, for the swipe console.

Scores
------
For every real event the live head's class probabilities, plus how unsure it
is (normalised entropy and the margin between its top two classes). The swipe
console shows the least sure first, so a few minutes of a person's time go
where they change the model most. The raw feature vector is included so the
console can group near-duplicates (same node, close in time, similar
features) into one card.

Auto-labels
-----------
An unlabelled event the head is at least `threshold` sure about gets a
machine label, provenance origin=machine, reviewed=False, source
"auto:<model>@<threshold>". It is never counted as human: human_only
training skips it, and a person's later swipe replaces it (a machine write
can never replace a reviewed one, see SampleStore._apply_label).

Neither touches a row a person has reviewed, a synthetic row, or an event
the console holds back for its test set.
"""

from __future__ import annotations

import logging
import math
from typing import Iterable, Optional

import numpy as np

from src.federated.features import LABEL_NAMES, N_CLASSES
from src.federated.store import AUTO_PREFIX, is_reviewed, machine_prov, provenance

logger = logging.getLogger(__name__)

_LOG_K = math.log(N_CLASSES)


def _real_rows(store, since: Optional[float], until: Optional[float]) -> list:
    out = []
    for r in list(store._rows):
        if r.get("n") is None or r.get("meta", {}).get("synthetic"):
            continue
        t = float(r.get("t", 0))
        if since is not None and t < since:
            continue
        if until is not None and t >= until:
            continue
        out.append(r)
    return out


def _uncertainty(p: np.ndarray) -> tuple:
    """(normalised entropy 0..1, margin between the top two classes)."""
    q = np.clip(p, 1e-9, 1.0)
    entropy = float(-(q * np.log(q)).sum() / _LOG_K)
    top = np.sort(p)[::-1]
    return round(entropy, 4), round(float(top[0] - top[1]), 4)


def score_rows(store, head, model_version: Optional[str],
               since: Optional[float] = None,
               until: Optional[float] = None) -> dict:
    rows = _real_rows(store, since, until)
    probas = None
    if rows and model_version:
        X = np.array([r["x"] for r in rows], dtype=np.float32)
        probas = head.predict_proba(X)

    items = []
    for i, r in enumerate(rows):
        prov = provenance(r)
        item = {
            "n": r["n"],
            "t": r["t"],
            "label": LABEL_NAMES[r["y"]] if r.get("y") is not None else None,
            "reviewed": is_reviewed(r),
            "source": prov.get("source") if prov else None,
            "x": [round(float(v), 3) for v in r["x"]],
        }
        if probas is not None:
            p = probas[i]
            best = int(p.argmax())
            entropy, margin = _uncertainty(p)
            item.update({
                "model_label": LABEL_NAMES[best],
                "confidence": round(float(p[best]), 4),
                "p_anomaly": round(float(p[-1]), 4),
                "entropy": entropy,
                "margin": margin,
            })
        items.append(item)

    return {"model": model_version, "count": len(items), "scores": items}


def auto_label(store, head, model_version: Optional[str], threshold: float,
               exclude: Iterable[int] = (), since: Optional[float] = None,
               until: Optional[float] = None, dry_run: bool = False) -> dict:
    """Machine-label unlabelled events the head is at least `threshold` sure of."""
    if not model_version:
        return {"labelled": 0, "dry_run": dry_run, "model": None,
                "reason": "no accepted model on this node", "events": []}
    if not 0.5 <= threshold < 1.0:
        raise ValueError("threshold must be in [0.5, 1.0)")

    skip = set(int(n) for n in exclude)
    rows = [r for r in _real_rows(store, since, until)
            if r.get("y") is None and r["n"] not in skip and not is_reviewed(r)]
    source = f"{AUTO_PREFIX}model:{model_version}@{threshold:g}"
    picked = []
    if rows:
        X = np.array([r["x"] for r in rows], dtype=np.float32)
        probas = head.predict_proba(X)
        for r, p in zip(rows, probas):
            best = int(p.argmax())
            if float(p[best]) >= threshold:
                picked.append((r, best, round(float(p[best]), 4)))

    if not dry_run and picked:
        with store._lock:
            for r, best, conf in picked:
                prov = machine_prov(source)
                prov["confidence"] = conf
                store._apply_label(r, best, prov)
            store._rewrite()

    per_label = {name: 0 for name in LABEL_NAMES}
    for _, best, _ in picked:
        per_label[LABEL_NAMES[best]] += 1
    logger.info("Auto-label: %d of %d unlabelled events at >= %.2f (%s)%s",
                len(picked), len(rows), threshold, source,
                " dry run" if dry_run else "")
    return {
        "labelled": len(picked),
        "considered": len(rows),
        "per_label": per_label,
        "threshold": threshold,
        "source": source,
        "dry_run": dry_run,
        "model": model_version,
        "events": [{"n": r["n"], "label": LABEL_NAMES[b], "confidence": c}
                   for r, b, c in picked],
    }

