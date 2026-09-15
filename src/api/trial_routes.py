"""
Endpoints for the live federated trial, plus the dashboard compatibility shim.

Two separate concerns in one module because they are wired into the app the
same way and are both scaffolding for the same demo.

1. Federated trial -- capture samples off the live camera, label them, train
   the head, run a Flower round, see the result.
The dashboard's own contract lives in src/api/dashboard_routes.py.
"""

from __future__ import annotations

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
    }


class TrainRequest(BaseModel):
    epochs: int = 8
    min_samples: int = 200


@router.post("/api/federated/head/train", tags=["Federated"])
async def train_head(body: TrainRequest, request: Request):
    """Train locally, without a Flower server. The fast inner loop."""
    h, st = _head(request), _store(request)
    X, y = st.labelled()
    if len(X) < body.min_samples:
        raise HTTPException(
            409,
            f"only {len(X)} labelled samples, need {body.min_samples}. "
            f"Label more, or POST /api/federated/samples/bootstrap.",
        )
    cut = int(len(X) * 0.8)
    t0 = time.perf_counter()
    h.fit(X[:cut], y[:cut], epochs=body.epochs)
    loss, acc = h.evaluate(X[cut:], y[cut:])
    bal = h.balanced_accuracy(X[cut:], y[cut:])
    _state["last_metrics"] = {
        "loss": round(loss, 4),
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal, 4),
        "train_samples": cut,
        "val_samples": len(X) - cut,
        "seconds": round(time.perf_counter() - t0, 2),
        "at": time.time(),
    }
    return _state["last_metrics"]


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


def _pump(proc, sink: List[str]):
    for line in iter(proc.stdout.readline, ""):
        sink.append(line.rstrip())
        del sink[:-400]
    proc.stdout.close()


@router.post("/api/federated/round", tags=["Federated"])
async def start_round(body: RoundRequest, request: Request):
    """Connect to the aggregator and run a round now.

    The scheduler fires at 02:00 on a 14-day interval, which is correct for
    deployment and useless for a demo with people watching. This is the
    manual trigger.
    """
    proc = _state["round_proc"]
    if proc is not None and proc.poll() is None:
        raise HTTPException(409, "a round is already running")

    st = _store(request)
    X, _ = st.labelled()
    if len(X) < body.min_samples:
        raise HTTPException(
            409, f"only {len(X)} labelled samples, need {body.min_samples}"
        )

    server = body.server or settings.FL_SERVER_URL
    _state["round_log"] = []
    _state["round_started"] = time.time()
    proc = subprocess.Popen(
        [sys.executable, str(BASE_DIR / "sim" / "fl.py"), "client",
         "--address", server, "--epochs", str(body.epochs)],
        cwd=str(BASE_DIR), stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    _state["round_proc"] = proc
    threading.Thread(target=_pump, args=(proc, _state["round_log"]),
                     daemon=True).start()
    return {"started": True, "server": server, "labelled_samples": int(len(X))}


@router.get("/api/federated/round", tags=["Federated"])
async def round_status():
    proc = _state["round_proc"]
    running = proc is not None and proc.poll() is None
    return {
        "running": running,
        "exit_code": None if running or proc is None else proc.returncode,
        "started": _state["round_started"],
        "log": _state["round_log"][-60:],
    }
