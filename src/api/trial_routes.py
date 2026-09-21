"""
Endpoints for the live federated trial, plus the dashboard compatibility shim.

Two separate concerns in one module because they are wired into the app the
same way and are both scaffolding for the same demo.

1. Federated trial -- capture samples off the live camera, label them, train
   the head, run a Flower round, see the result.
The dashboard's own contract lives in src/api/dashboard_routes.py.
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from src.config import settings
from src.federated import model_state
from src.federated.features import LABEL_NAMES, N_CLASSES, N_FEATURES

logger = logging.getLogger(__name__)

router = APIRouter()
BASE_DIR = Path(__file__).resolve().parents[2]

# In-process state for the trial. Deliberately not in the database: this is
# demo scaffolding with a session lifetime, and putting it in SQLite would
# imply a durability guarantee it does not have.
_state = {
    "round_proc": None,
    "round_started": None,
    "round_log": [],
    "round_dir": None,
    "round_result": None,
    "last_metrics": None,
}


def _store(request: Request):
    s = getattr(request.app.state, "fl_store", None)
    if s is None:
        raise HTTPException(503, "Sample store not initialised (FL_ENABLED?)")
    return s


def _head(request: Request):
    h = getattr(request.app.state, "fl_head", None)
    if h is None:
        raise HTTPException(503, "Federated head not initialised")
    return h


# ============================================================================
# Samples
# ============================================================================

@router.get("/api/federated/samples", tags=["Federated"])
async def list_samples(request: Request, limit: int = 50):
    st = _store(request)
    return {"stats": st.stats(), "samples": st.recent(limit)}


class LabelRequest(BaseModel):
    label: int


@router.post("/api/federated/samples/{sample_id}/label", tags=["Federated"])
async def label_sample(sample_id: str, body: LabelRequest, request: Request):
    st = _store(request)
    if not 0 <= body.label < N_CLASSES:
        raise HTTPException(422, f"label must be 0..{N_CLASSES - 1}")
    if not st.set_label(sample_id, body.label):
        raise HTTPException(404, f"sample {sample_id} not found")
    return {"id": sample_id, "label": LABEL_NAMES[body.label], "stats": st.stats()}


class BootstrapRequest(BaseModel):
    site: str = "driveway"
    n: int = 5000
    seed: int = 0


@router.post("/api/federated/samples/bootstrap", tags=["Federated"])
async def bootstrap_samples(body: BootstrapRequest, request: Request):
    """Seed the store from the simulator's generator.

    A short capture cannot produce enough labelled events to train on. Rows
    added here are tagged synthetic so they stay distinguishable from
    anything the camera actually saw.
    """
    st = _store(request)
    n = st.bootstrap(body.site, int(body.n), int(body.seed))
    return {"added": n, "synthetic": True, "stats": st.stats()}


# ============================================================================
# Head
# ============================================================================

@router.get("/api/federated/head", tags=["Federated"])
async def head_status(request: Request):
    h = _head(request)
    st = _store(request)
    return {
        "params": h.n_params,
        "features": N_FEATURES,
        "classes": list(LABEL_NAMES),
        "samples": st.stats(),
        "last_metrics": _state["last_metrics"],
        "model": model_state.summary(request.app),
    }


class TrainRequest(BaseModel):
    epochs: int = 8
    min_samples: int = 200
    tolerance: float = 0.01


def _train_local_candidate(app, epochs: int):
    from src.federated.head import FederatedHead
    from src.federated.store import holdout_split

    X, y = app.state.fl_store.labelled()
    Xt, yt, _, _ = holdout_split(X, y, seed=0)
    candidate = FederatedHead(seed=0)
    candidate.set_weights(app.state.fl_head.get_weights())
    t0 = time.perf_counter()
    candidate.fit(Xt, yt, epochs=epochs)
    path = Path(settings.DATA_DIR) / "fl_local_candidate.npz"
    candidate.save(path)
    return path, round(time.perf_counter() - t0, 2), int(len(Xt))


@router.post("/api/federated/head/train", tags=["Federated"])
async def train_head(body: TrainRequest, request: Request):
    """Train a copy locally, then gate it into the live node.

    Trains a copy rather than the live head, so a bad run never reaches
    inference, and runs off the event loop so the API keeps serving.
    """
    _head(request)
    st = _store(request)
    X, _ = st.labelled()
    if len(X) < body.min_samples:
        raise HTTPException(
            409,
            f"only {len(X)} labelled samples, need {body.min_samples}. "
            f"Label more, or POST /api/federated/samples/bootstrap.",
        )
    path, secs, n_train = await asyncio.to_thread(
        _train_local_candidate, request.app, body.epochs)
    decision = await asyncio.to_thread(
        model_state.accept_candidate, request.app, path, "local-train",
        body.tolerance, {"seconds": secs, "train_samples": n_train})
    _state["last_metrics"] = decision
    return decision


@router.post("/api/federated/predict", tags=["Federated"])
async def predict(request: Request):
    """Run the head on a raw 19-dim feature vector."""
    body = await request.json()
    x = np.asarray(body.get("features", []), dtype=np.float32)
    if x.size != N_FEATURES:
        raise HTTPException(422, f"expected {N_FEATURES} features, got {x.size}")
    idx = _head(request).infer_one(x)
    return {"label": LABEL_NAMES[idx], "index": idx}


# ============================================================================
# Federated round
# ============================================================================

class RoundRequest(BaseModel):
    server: Optional[str] = None
    epochs: int = 6
    min_samples: int = 200
    tolerance: float = 0.01


def _pump(proc, sink: List[str]):
    for line in iter(proc.stdout.readline, ""):
        sink.append(line.rstrip())
        del sink[:-400]
    proc.stdout.close()


def _finish_round(app, proc, run_dir: Path, tolerance: float):
    """Wait for the client, then gate the returned global model into the node."""
    code = proc.wait()
    candidate = run_dir / "global.npz"
    rounds = 0
    events = run_dir / "events.jsonl"
    if events.exists():
        for line in events.read_text(encoding="utf-8").splitlines():
            try:
                if json.loads(line).get("kind") == "global":
                    rounds += 1
            except Exception:
                pass

    if code != 0 or not candidate.exists():
        _state["round_result"] = {
            "accepted": False,
            "exit_code": code,
            "rounds": rounds,
            "reason": "round did not complete" if code != 0
                      else "server never sent a global model",
        }
        return

    try:
        decision = model_state.accept_candidate(
            app, candidate, "federated", tolerance,
            {"rounds": rounds, "run_dir": str(run_dir)})
    except Exception as exc:
        logger.exception("Validation gate failed")
        decision = {"accepted": False, "reason": f"gate failed: {exc}"}
    _state["round_result"] = decision


@router.post("/api/federated/round", tags=["Federated"])
async def start_round(body: RoundRequest, request: Request):
    """Join the aggregator, train on this node's samples, take the update back.

    The client runs as its own process, starting from the weights live now.
    When it exits, the final global model the server returned goes through
    the validation gate and, if it holds up, replaces the live head.
    """
    proc = _state["round_proc"]
    if proc is not None and proc.poll() is None:
        raise HTTPException(409, "a round is already running")

    st, head = _store(request), _head(request)
    X, _ = st.labelled()
    if len(X) < body.min_samples:
        raise HTTPException(
            409, f"only {len(X)} labelled samples, need {body.min_samples}")

    server = body.server or settings.FL_SERVER_URL
    run_dir = Path(settings.DATA_DIR) / "fl_rounds" / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    head.save(run_dir / "init.npz")

    # Without --node-name the client calls itself "node"; without
    # --central-api it never POSTs /api/nodes/register, and the registry's
    # update_training() silently no-ops on an id it has never seen. The
    # result is a server that trains correctly and reports zero nodes.
    # rounds.py passes both -- this path had drifted from it, which is the
    # exact divergence that module exists to prevent.
    cmd = [sys.executable, "-m", "src.federated.node_client",
           "--store", str(st.path), "--init", str(run_dir / "init.npz"),
           "--out", str(run_dir), "--server", server,
           "--epochs", str(body.epochs),
           "--node-name", settings.NODE_NAME]
    if getattr(settings, "FL_CENTRAL_API", ""):
        cmd += ["--central-api", settings.FL_CENTRAL_API]

    proc = subprocess.Popen(
        cmd,
        cwd=str(BASE_DIR), stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    _state.update(round_proc=proc, round_started=time.time(), round_log=[],
                  round_dir=str(run_dir), round_result=None)
    threading.Thread(target=_pump, args=(proc, _state["round_log"]),
                     daemon=True).start()
    threading.Thread(target=_finish_round,
                     args=(request.app, proc, run_dir, body.tolerance),
                     daemon=True).start()
    return {"started": True, "server": server,
            "labelled_samples": int(len(X)), "run_dir": str(run_dir)}


@router.get("/api/federated/round", tags=["Federated"])
async def round_status():
    proc = _state["round_proc"]
    running = proc is not None and proc.poll() is None
    return {
        "running": running,
        "exit_code": None if running or proc is None else proc.returncode,
        "started": _state["round_started"],
        "run_dir": _state["round_dir"],
        "result": _state["round_result"],
        "log": _state["round_log"][-60:],
    }
