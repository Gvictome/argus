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

Balanced accuracy is coarse -- with four represented classes it moves in
steps of 0.25, so two different models routinely score the same. When it
is level, cross-entropy loss breaks the tie: a candidate whose loss is
more than `loss_tolerance` worse than the live model's is rejected even
though the headline metric did not move. Round 2 of 2026-09-20 is why --
it was accepted at 0.75 vs 0.75 while its loss went 0.1941 -> 0.2529. A
candidate that clearly improves balanced accuracy is taken regardless of
loss; the tie-break only decides rounds that the headline metric cannot.

A candidate must also reach FL_MIN_ANOMALY_RECALL on the anomaly class: a
head can hold its balanced accuracy while missing the events the product
exists to catch. This applies to the first model too.

Accepted weights are written to DATA_DIR/fl_head.npz and restored at
startup, so an update survives a restart. Every decision, accepted or not,
is kept in DATA_DIR/fl_model_state.json.

Every accepted version is also archived as DATA_DIR/fl_models/<label>.npz
(the newest MAX_ARCHIVED kept), so a bad update that got past the gate can
be rolled back without waiting for the next round. A rollback is itself a
new version (`rollback-rN`): the counter only moves forward, and the
history shows what was restored and when.
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

# Archived versions kept on the node. Each head is under 1 MB, so this is
# a cap on clutter, not on disk.
MAX_ARCHIVED = 20


def head_path() -> Path:
    return Path(settings.DATA_DIR) / "fl_head.npz"


def state_path() -> Path:
    return Path(settings.DATA_DIR) / "fl_model_state.json"


def versions_dir() -> Path:
    return Path(settings.DATA_DIR) / "fl_models"


def _archive(head: FederatedHead, label: str) -> Path:
    """Keep a copy of an accepted version, pruning the oldest past the cap."""
    path = versions_dir() / f"{label}.npz"
    head.save(path)
    archived = sorted(versions_dir().glob("*.npz"), key=lambda q: q.stat().st_mtime)
    for old in archived[:-MAX_ARCHIVED]:
        try:
            old.unlink()
        except OSError as exc:
            logger.warning("Could not prune %s: %s", old, exc)
    return path


def list_versions(app) -> list:
    """Accepted versions still on disk, newest first, with when and why."""
    state = getattr(app.state, "fl_model_state", None) or _read_state()
    on_disk = {q.stem for q in versions_dir().glob("*.npz")} if versions_dir().exists() else set()
    live = state.get("label")
    out = []
    for d in reversed(state.get("history", [])):
        label = d.get("label")
        if not d.get("accepted") or not label or label not in on_disk:
            continue
        out.append({
            "label": label,
            "version": d.get("version"),
            "source": d.get("source"),
            "at": d.get("at"),
            "balanced_accuracy": (d.get("candidate") or {}).get("balanced_accuracy"),
            "loss": (d.get("candidate") or {}).get("loss"),
            "restored_from": d.get("restored_from"),
            "live": label == live,
        })
        on_disk.discard(label)
    # Archived files whose decision has aged out of the 50-entry history.
    out.extend({"label": name, "live": name == live} for name in sorted(on_disk, reverse=True))
    return out


def rollback(app, label: str) -> dict:
    """Make an archived version live again, as a new version."""
    path = versions_dir() / f"{label}.npz"
    if not label or "/" in label or "\\" in label or not path.exists():
        raise KeyError(f"no archived version {label!r}")
    live: FederatedHead = app.state.fl_head
    restored = FederatedHead(seed=0)
    restored.load(path)

    with _LOCK:
        state = getattr(app.state, "fl_model_state", None) or _read_state()
        previous = state.get("label")
        live.W, live.b = list(restored.W), list(restored.b)
        live.save(head_path())
        state["version"] = state.get("version", 0) + 1
        state["label"] = f"rollback-r{state['version']}"
        decision = {
            "accepted": True,
            "source": "rollback",
            "restored_from": label,
            "replaced": previous,
            "version": state["version"],
            "label": state["label"],
            "at": time.time(),
        }
        _archive(live, state["label"])
        state["last"] = decision
        state.setdefault("history", []).append(decision)
        state["history"] = state["history"][-50:]
        _write_state(state)
        app.state.fl_model_state = state

    logger.warning("Model rolled back: %s -> %s (restored %s)",
                   previous, state["label"], label)
    return decision


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
        "anomaly_recall": (None if (r := head.anomaly_recall(X, y)) is None
                           else round(r, 4)),
    }


def accept_candidate(app, candidate_path, source: str,
                     tolerance: float = 0.01, extra: Optional[dict] = None,
                     loss_tolerance: float = 0.05,
                     min_anomaly_recall: Optional[float] = None) -> dict:
    """Validate a candidate against the live head; swap it in if it holds up.

    Besides the balanced-accuracy and loss checks, a candidate must reach
    `min_anomaly_recall` (default settings.FL_MIN_ANOMALY_RECALL) on the
    anomaly class, first model included. Skipped when the held-out set has
    no anomaly rows, since recall is then unmeasurable rather than zero.
    """
    if min_anomaly_recall is None:
        min_anomaly_recall = settings.FL_MIN_ANOMALY_RECALL
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

    ba_delta = proposed["balanced_accuracy"] - current["balanced_accuracy"]
    loss_delta = proposed["loss"] - current["loss"]
    # Relative to the live loss, with a floor so a near-zero loss does not
    # make the margin vanishingly small.
    loss_margin = max(loss_tolerance * current["loss"], 1e-4)

    recall = proposed.get("anomaly_recall")
    below_floor = recall is not None and recall < min_anomaly_recall
    floor_reason = (
        f"candidate anomaly recall {recall} is below the "
        f"{min_anomaly_recall} floor"
    )

    reason = None
    if first:
        accepted = not below_floor
        reason = floor_reason if below_floor else None
    elif ba_delta < -tolerance:
        accepted = False
        reason = (
            f"candidate balanced accuracy {proposed['balanced_accuracy']} "
            f"regressed past {current['balanced_accuracy']} - {tolerance}"
        )
    elif ba_delta <= tolerance and loss_delta > loss_margin:
        accepted = False
        reason = (
            f"balanced accuracy level at {proposed['balanced_accuracy']} but loss "
            f"rose {current['loss']} -> {proposed['loss']}, past the "
            f"{loss_margin:.4f} tie-break margin"
        )
    elif below_floor:
        accepted = False
        reason = floor_reason
    else:
        accepted = True

    decision = {
        "accepted": accepted,
        "source": source,
        "first_model": first,
        "current": current,
        "candidate": proposed,
        "tolerance": tolerance,
        "loss_tolerance": loss_tolerance,
        "min_anomaly_recall": min_anomaly_recall,
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
            _archive(live, state["label"])
        else:
            decision["reason"] = reason
        state["last"] = decision
        state.setdefault("history", []).append(decision)
        state["history"] = state["history"][-50:]
        _write_state(state)
        app.state.fl_model_state = state

    logger.info("Model gate (%s): accepted=%s bal_acc %s -> %s, loss %s -> %s",
                source, accepted, current["balanced_accuracy"],
                proposed["balanced_accuracy"], current["loss"], proposed["loss"])
    return decision
