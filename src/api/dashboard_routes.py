"""
The adam9798/argus-dashboard backend contract.

That repo's README documents the shape it was built against, and says
"implement against this shape and no frontend changes should be needed".
This is that implementation, backed by the sample store the collector
fills from live detections.

    POST /api/login             -> { access_token }
    GET  /api/status            -> { fps, accelerator, model_version, nodes_online }
    GET  /api/events            -> DetectionEvent[]
    POST /api/events/{id}/label -> { action: "confirm" | "correct", label? }
    WS   /ws/events             -> DetectionEvent, pushed

Two of these collide with stubs already in routes.py -- /api/status
returned hardcoded "stopped" strings and /api/events returned an empty
list behind a TODO. Both are replaced there rather than shadowed here, so
there is one definition of each and no route-ordering trap.

Why the label endpoint matters more than it looks
-------------------------------------------------
It is the human-in-the-loop that makes the federated head honest. The
simulator derives labels from the same rule that generates its data;
on the node the label has to come from outside the model. Confirm and
Correct in the dashboard are how it gets there, and each click writes a
label into the store the next FL round trains on.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from src.config import settings
from src.federated.features import ANOMALY, CLASS_NAMES, LABEL_NAMES

logger = logging.getLogger(__name__)

router = APIRouter()

# DetectionEvent.cls is typed "person" | "vehicle" | "package" | "animal"
# in lib/api.ts. Our feature class order is person, vehicle, animal, package,
# so this is an explicit remap rather than an index cast -- getting it wrong
# silently renders every dog as a parcel.
_CLS_FOR_DASHBOARD = {0: "person", 1: "vehicle", 2: "animal", 3: "package"}

_PRETTY = {
    "person": "Person",
    "vehicle": "Vehicle",
    "animal": "Animal",
    "package": "Package delivery",
}

# Vocabularies for reading the dashboard's free-text corrections. Anomaly
# words are checked first, so "unrecognised person" is an anomaly, not a
# routine person.
_ANOMALY_WORDS = {
    "anomaly", "stranger", "unusual", "suspicious", "intruder", "threat",
    "unknown", "unrecognized", "unrecognised", "alert", "wrong", "trespasser",
}
_ROUTINE_WORDS = {
    "routine", "normal", "ok", "okay", "fine", "expected", "resident",
    "member", "household", "familiar", "benign", "neighbor", "neighbour",
}
# Keyed by head class index: 0 person, 1 vehicle, 2 animal, 3 package.
_CLASS_WORDS = {
    0: {"person", "people", "human", "man", "woman", "kid", "child",
        "pedestrian", "visitor", "guest", "driver", "courier"},
    1: {"vehicle", "car", "truck", "van", "bike", "bicycle", "motorcycle", "bus"},
    2: {"animal", "dog", "cat", "bird", "pet", "deer", "raccoon", "squirrel"},
    3: {"package", "parcel", "box", "delivery", "mail"},
}


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _event_from_row(row: dict, head) -> dict:
    """Render one stored sample as a dashboard DetectionEvent."""
    x = np.array(row["x"], dtype=np.float32)
    cls_idx = int(np.argmax(x[:4])) if x[:4].any() else 0
    cls = _CLS_FOR_DASHBOARD.get(cls_idx, "person")

    # If the operator has already labelled this sample, that is ground truth
    # and outranks the model. Otherwise fall back to the head's judgement.
    if row.get("y") is not None:
        verdict = int(row["y"])
        confirmed = 1
    else:
        verdict = head.infer_one(x) if head is not None else cls_idx
        confirmed = 0

    # status is member | stranger | na. With face recognition out of scope
    # there is no "member", so the field carries the head's anomaly call
    # instead: that is what the dashboard renders in red, and flagging the
    # unusual is exactly what the head is for.
    if verdict == ANOMALY:
        status, label = "stranger", f"Unusual for this site ({cls})"
    else:
        status, label = "na", _PRETTY.get(cls, cls.title())

    return {
        "id": int(row.get("n") or 0),
        "cls": cls,
        "zone": row.get("meta", {}).get("zone") or settings.CAMERA_ZONE,
        "node": row.get("meta", {}).get("node") or settings.NODE_NAME,
        "ts": _iso(row.get("t", time.time())),
        "status": status,
        "label": label,
        "confidence": int(round(float(x[18]) * 100)),
        "confirmed": confirmed,
    }


# ============================================================================
# Auth
# ============================================================================

class LoginBody(BaseModel):
    username: str
    password: str


@router.post("/api/login", tags=["Dashboard"])
async def dashboard_login(body: LoginBody, request: Request):
    """Alias for /api/auth/login in the shape the dashboard expects.

    app/api/auth/login/route.ts reads `access_token` off this response and
    sets the HttpOnly cookie itself, so the key name is load-bearing.
    """
    from src.api.routes import login as backend_login, LoginRequest

    result = await backend_login(LoginRequest(**body.model_dump()), request)
    if isinstance(result, dict):
        token = (result.get("access_token") or result.get("token")
                 or result.get("access"))
        if token:
            return {"access_token": token}
    raise HTTPException(401, "login failed")


# ============================================================================
# Status
# ============================================================================

@router.get("/api/dashboard/status", tags=["Dashboard"], include_in_schema=False)
async def _status_alias(request: Request):
    return build_status(request)


def build_status(request: Request) -> dict:
    """The /api/status body. Called from routes.py so there is one source."""
    fps = 0.0
    accelerator = "none"
    try:
        from src.detection import detection_service
        st = detection_service.status()
        fps = float(st.get("fps", 0.0))
        accelerator = {
            "hailo": "Hailo-10H",
            "tensorrt": "TensorRT",
            "cpu": "CPU",
            "none": "none",
        }.get(st.get("backend", "none"), st.get("backend", "none"))
    except Exception as exc:
        logger.debug("status: detection service unavailable: %s", exc)

    # The status strip should show what the node is really doing: the
    # worker's measured detection rate, not the stream's.
    worker = getattr(request.app.state, "detection_worker", None)
    if worker is not None and worker.running:
        fps = worker.detect_rate.value()

    store = getattr(request.app.state, "fl_store", None)
    from src.federated.model_state import model_version as _model_version
    model_version = _model_version(request.app)

    return {
        "fps": round(fps, 1),
        "accelerator": accelerator,
        "model_version": model_version,
        "nodes_online": int(getattr(request.app.state, "fl_nodes_online", 1)),
        # Beyond the contract, ignored by the dashboard, useful in curl.
        "node": settings.NODE_NAME,
        "samples": store.stats() if store is not None else None,
    }


# ============================================================================
# Events
# ============================================================================

def build_events(request: Request, limit: int = 50) -> List[dict]:
    """The /api/events body: newest first, as a bare array."""
    store = getattr(request.app.state, "fl_store", None)
    head = getattr(request.app.state, "fl_head", None)
    if store is None:
        return []
    rows = [r for r in store._rows if not r.get("meta", {}).get("synthetic")]
    out = [_event_from_row(r, head) for r in rows[-limit:]]
    out.reverse()
    return out


class LabelBody(BaseModel):
    action: str                  # "confirm" | "correct"
    label: Optional[str] = None  # class name or ARGUS label name


def _resolve_label(action: str, label: Optional[str], current: dict) -> int:
    """Map a Confirm/Correct click onto one of the five head classes."""
    if action == "confirm":
        # Confirming an anomaly flag means anomaly; confirming anything else
        # means that object class is routine here.
        if current["status"] == "stranger":
            return ANOMALY
        for i, name in _CLS_FOR_DASHBOARD.items():
            if name == current["cls"]:
                return i
        return 0

    if action != "correct":
        raise HTTPException(422, 'action must be "confirm" or "correct"')

    if not label:
        # Correct with no label is the inverse of what was shown: a flagged
        # event becomes routine, a routine one becomes an anomaly.
        if current["status"] == "stranger":
            for i, name in _CLS_FOR_DASHBOARD.items():
                if name == current["cls"]:
                    return i
            return 0
        return ANOMALY

    # The dashboard's Correct box is free text ("Delivery driver", "false
    # alarm"), not a picker, so an exact-match vocabulary would reject most
    # real corrections with a 422. Read intent from keywords instead, and
    # when nothing matches treat the correction as "the call was wrong".
    key = label.strip().lower()
    if key.replace(" ", "_") in LABEL_NAMES:
        return LABEL_NAMES.index(key.replace(" ", "_"))

    words = set(key.replace("_", " ").replace("-", " ").split())
    phrase = " ".join(key.replace("_", " ").replace("-", " ").split())

    if words & _ANOMALY_WORDS:
        return ANOMALY
    for i, synonyms in _CLASS_WORDS.items():
        if words & synonyms:
            return i
    if words & _ROUTINE_WORDS or any(p in phrase for p in ("false alarm", "false positive")):
        for i, name in _CLS_FOR_DASHBOARD.items():
            if name == current["cls"]:
                return i
        return 0
    return _resolve_label("correct", None, current)


@router.post("/api/events/{event_id}/label", tags=["Dashboard"])
async def label_event(event_id: int, body: LabelBody, request: Request):
    """Confirm or correct one event. This is what produces training labels."""
    store = getattr(request.app.state, "fl_store", None)
    head = getattr(request.app.state, "fl_head", None)
    if store is None:
        raise HTTPException(503, "Sample store not initialised")

    row = store.by_n(event_id)
    if row is None:
        raise HTTPException(404, f"event {event_id} not found")

    current = _event_from_row(row, head)
    y = _resolve_label(body.action, body.label, current)
    store.set_label_by_n(event_id, y, note=body.label)

    return {
        "ok": True,
        "id": event_id,
        "label": LABEL_NAMES[y],
        "labelled_total": store.stats()["labelled"],
    }


# ============================================================================
# Live feed
# ============================================================================

@router.websocket("/ws/events")
async def events_socket(websocket: WebSocket):
    """Push new events as they are collected.

    connectEventsSocket() in lib/api.ts expects the DetectionEvent shape,
    identical to the REST records. Polls the store rather than subscribing
    to it: at a few events a minute the difference is invisible, and it
    keeps the collector free of socket plumbing.
    """
    await websocket.accept()
    app = websocket.app
    store = getattr(app.state, "fl_store", None)
    head = getattr(app.state, "fl_head", None)
    if store is None:
        await websocket.close(code=1011)
        return

    seen = max((r.get("n", 0) for r in store._rows), default=0)
    try:
        while True:
            fresh = [
                r for r in store._rows
                if (r.get("n") or 0) > seen
                and not r.get("meta", {}).get("synthetic")
            ]
            for r in fresh:
                seen = max(seen, r.get("n") or 0)
                await websocket.send_text(json.dumps(_event_from_row(r, head)))
            await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.debug("events socket closed: %s", exc)
