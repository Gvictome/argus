"""
The node's federated model: what is running, what version it is, and
whether a new candidate is allowed to replace it.

A federated round ends with the server sending back an averaged model.
Applying that blindly would let one bad round -- a poisoned client, a
non-IID average that suits other sites but not this one -- overwrite a
working detector. So every candidate, federated or locally trained, passes
a validation gate first: it is scored on this node's own held-out labelled
samples, and replaces the live head only if balanced accuracy does not
regress by more than `tolerance`. The first model is accepted outright,
since there is nothing yet to protect.

Accepted weights are written to DATA_DIR/fl_head.npz and restored at
startup, so an update survives a restart. Every decision, accepted or not,
is kept in DATA_DIR/fl_model_state.json.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Optional

from src.config import settings
from src.federated.head import FederatedHead
from src.federated.store import holdout_split

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()


def head_path() -> Path:
    return Path(settings.DATA_DIR) / "fl_head.npz"


def state_path() -> Path:
    return Path(settings.DATA_DIR) / "fl_model_state.json"


def _read_state() -> dict:
    p = state_path()
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Unreadable %s, starting fresh: %s", p, exc)
    return {"version": 0, "history": []}


def _write_state(state: dict) -> None:
    p = state_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(p)


def load_model_state(app) -> dict:
    """Restore the last accepted head into app.state at startup."""
    state = _read_state()
    app.state.fl_model_state = state
    head = getattr(app.state, "fl_head", None)
    if head is not None and head_path().exists():
        try:
            head.load(head_path())
            logger.info("Restored federated head %s from %s",
                        model_version(app), head_path())
        except Exception as exc:
            # A shape mismatch means the architecture changed. Start fresh
            # rather than refusing to boot.
            logger.warning("Ignoring incompatible saved head: %s", exc)
    return state


def model_version(app) -> str:
    state = getattr(app.state, "fl_model_state", None) or {}
    if state.get("version", 0) > 0:
        return state.get("label") or f"r{state['version']}"
    return "untrained"


def summary(app) -> dict:
    state = getattr(app.state, "fl_model_state", None) or {}
    return {
        "version": state.get("version", 0),
        "label": model_version(app),
        "last_decision": state.get("last"),
        "weights_path": str(head_path()),
    }


def _score(head: FederatedHead, X, y) -> dict:
    loss, acc = head.evaluate(X, y)
    return {
        "loss": round(loss, 4),
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(head.balanced_accuracy(X, y), 4),
    }


def accept_candidate(app, candidate_path, source: str,
                     tolerance: float = 0.01, extra: Optional[dict] = None) -> dict:
    """Validate a candidate against the live head; swap it in if it holds up."""
    store = app.state.fl_store
    live: FederatedHead = app.state.fl_head

    X, y = store.labelled()
    if len(X) == 0:
        return {"accepted": False, "source": source,
                "reason": "no labelled samples to validate against"}
    _, _, Xv, yv = holdout_split(X, y, seed=0)

    candidate = FederatedHead(seed=0)
    candidate.load(candidate_path)

    current = _score(live, Xv, yv)
    proposed = _score(candidate, Xv, yv)
    first = not head_path().exists()
    accepted = first or (
        proposed["balanced_accuracy"] >= current["balanced_accuracy"] - tolerance
    )

    decision = {
        "accepted": accepted,
        "source": source,
        "first_model": first,
        "current": current,
        "candidate": proposed,
        "tolerance": tolerance,
        "val_samples": int(len(Xv)),
        "at": time.time(),
        **(extra or {}),
    }

    with _LOCK:
        state = getattr(app.state, "fl_model_state", None) or _read_state()
        if accepted:
            # Assign both lists in one statement so a concurrent inference
            # never sees one layer from each model for longer than a call.
            live.W, live.b = list(candidate.W), list(candidate.b)
            live.save(head_path())
            state["version"] = state.get("version", 0) + 1
            state["label"] = f"{'fl' if source == 'federated' else 'local'}-r{state['version']}"
            decision["version"] = state["version"]
            decision["label"] = state["label"]
        else:
            decision["reason"] = (
                f"candidate balanced accuracy {proposed['balanced_accuracy']} "
                f"regressed past {current['balanced_accuracy']} - {tolerance}"
            )
        state["last"] = decision
        state.setdefault("history", []).append(decision)
        state["history"] = state["history"][-50:]
        _write_state(state)
        app.state.fl_model_state = state

    logger.info("Model gate (%s): accepted=%s current=%s candidate=%s",
                source, accepted, current["balanced_accuracy"],
                proposed["balanced_accuracy"])
    return decision
