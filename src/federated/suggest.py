"""
Machine label suggestions, stored apart from labels.

A suggestion is a guess for a person to accept or change. It lives in its
own file beside samples.jsonl and is never written to a row's `y`, so
SampleStore.labelled() cannot return one and "has a person seen this" has a
single answer: the row's provenance.

Two proposers:

  rule   a person in view for about 27 s or longer (dwell >= 0.9) is a
         candidate anomaly. Otherwise routine for the detector's class.
  model  the live head, when one has been accepted on this node.

Neither is evidence. The rule reads an input feature, and a head trained on
rule-derived labels mostly echoes the rule, so a high acceptance rate says
little until the head has been trained on reviewed labels.

If either proposer says anomaly, the suggestion is anomaly: reviewing a false
alarm costs seconds, and a missed anomaly is never seen by anyone.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from src.federated.features import ANOMALY, LABEL_NAMES
from src.federated.store import is_reviewed

logger = logging.getLogger(__name__)

# Scaled by features.extract's dwell_scale_s (30 s saturates the feature).
RULE_DWELL = 0.9
RULE_SOURCE = f"rule:dwell>={RULE_DWELL}"


def rule_label(x: np.ndarray) -> int:
    """Index of the rule's label for one 19-dim feature vector."""
    cls = int(np.argmax(x[:4])) if x[:4].any() else 0
    if cls == 0 and float(x[13]) >= RULE_DWELL:
        return ANOMALY
    return cls


def _suggest_one(row: dict, x: np.ndarray, proba: Optional[np.ndarray],
                 model_source: Optional[str], now: float) -> dict:
    rule = rule_label(x)
    model = int(proba.argmax()) if proba is not None else None
    disagree = model is not None and model != rule

    if ANOMALY in (rule, model):
        label = ANOMALY
    else:
        label = model if model is not None else rule

    sources = []
    if model == label:
        sources.append(model_source)
    if rule == label:
        sources.append(RULE_SOURCE)

    if proba is None:
        # The rule has no notion of how sure it is.
        uncertainty, confidence = None, None
    else:
        confidence = round(float(proba[label]), 4)
        uncertainty = 1.0 if disagree else round(1.0 - float(proba.max()), 4)

    return {
        "n": row["n"],
        "id": row["id"],
        "label": LABEL_NAMES[label],
        "source": "+".join(sources),
        "confidence": confidence,
        "uncertainty": uncertainty,
        "rule_label": LABEL_NAMES[rule],
        "model_label": LABEL_NAMES[model] if model is not None else None,
        "disagree": disagree,
        "at": now,
    }


class SuggestionStore:
    """The latest suggestion for each event, one JSON object per line."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def _read(self) -> list:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                out.append(json.loads(line))
            except ValueError:
                logger.warning("SuggestionStore: skipped a bad line in %s", self.path)
        return out

    def replace(self, items: Iterable[dict]) -> None:
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item) + "\n")
        tmp.replace(self.path)

    def current(self, store) -> dict:
        """Suggestions for rows that still have no person's review."""
        rows = {r.get("n"): r for r in store._rows}
        live = [s for s in self._read()
                if s.get("n") in rows and not is_reviewed(rows[s["n"]])]
        return {
            "count": len(live),
            "generated_at": max((s["at"] for s in live), default=None),
            "suggestions": live,
        }


def generate_suggestions(store, suggestions: SuggestionStore, head,
                         model_version: Optional[str], exclude: set) -> dict:
    """Recompute suggestions for every unreviewed, real, non-excluded event.

    Never writes to the sample store. Anything previously suggested for an
    event now excluded is dropped, because the file is replaced whole.
    """
    rows = [r for r in list(store._rows)
            if r.get("n") is not None
            and r["n"] not in exclude
            and not r.get("meta", {}).get("synthetic")
            and not is_reviewed(r)]

    now = time.time()
    model_source = f"model:{model_version}" if model_version else None
    items = []
    if rows:
        X = np.array([r["x"] for r in rows], dtype=np.float32)
        probas = head.predict_proba(X) if model_version else [None] * len(rows)
        for row, x, proba in zip(rows, X, probas):
            items.append(_suggest_one(row, x, proba, model_source, now))

    suggestions.replace(items)
    anomalies = sum(1 for s in items if s["label"] == LABEL_NAMES[ANOMALY])
    logger.info("Suggestions: %d events, %d suggested anomalies, model=%s",
                len(items), anomalies, model_version)
    return {
        "generated": len(items),
        "suggested_anomalies": anomalies,
        "excluded": len(exclude),
        "model": model_version,
        "generated_at": now,
    }
