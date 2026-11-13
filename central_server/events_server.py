"""
ARGUS Cloud Connect: central event-clip store.

Nodes PUT short H.264 event clips here (src/cloud/uploader.py); the console's
Events page lists, plays, saves and downloads them.

Storage (defaults; both inside Dropbox, so Dropbox is the cloud backup):
    events root   <ARGUS_EVENTS_DIR>/<node>/<YYYY-MM-DD>/<id>.mp4|.jpg|.json
    saved root    <ARGUS_SAVED_DIR>/<node>/<YYYY-MM-DD>/...   never auto-deleted
Every clip has a .json sidecar, so the folders describe themselves: the
SQLite index (kept outside Dropbox, so Dropbox never syncs a live database)
is rebuilt from the sidecars if it is missing.

Retention: unsaved events older than retention_days (default 30, set from the
console) are deleted by an hourly sweep and right after the setting changes.

Auth: if ARGUS_EVENTS_TOKEN (or ~/.argus/events_token) is set, uploads need header X-Argus-Token, and so
does every read that does not come from this machine (the console proxies
from localhost).

Run:  .venv\\Scripts\\python.exe central_server\\events_server.py   (port 8095)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import sqlite3
import threading
import time
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse

logger = logging.getLogger("argus.events")

HOME = Path.home()
ARGUS_FOLDER = HOME / "Dropbox" / "gio" / "Athena Files" / "03 ARGUS"
EVENTS_DIR = Path(os.getenv("ARGUS_EVENTS_DIR", ARGUS_FOLDER / "events"))
SAVED_DIR = Path(os.getenv("ARGUS_SAVED_DIR", ARGUS_FOLDER / "events-saved"))
STATE_DIR = Path(os.getenv("ARGUS_EVENTS_STATE", HOME / ".argus"))
DB_PATH = STATE_DIR / "events.db"
SETTINGS_PATH = STATE_DIR / "events_settings.json"
STAGING = STATE_DIR / "staging"

def _token() -> str:
    """ARGUS_EVENTS_TOKEN, else the contents of <state>/events_token."""
    if os.getenv("ARGUS_EVENTS_TOKEN"):
        return os.environ["ARGUS_EVENTS_TOKEN"]
    try:
        return (STATE_DIR / "events_token").read_text().strip()
    except OSError:
        return ""


TOKEN = _token()
PORT = int(os.getenv("ARGUS_EVENTS_PORT", "8095"))
DEFAULT_RETENTION_DAYS = int(os.getenv("ARGUS_RETENTION_DAYS", "30"))
MAX_UPLOAD = 200 * 1024 * 1024
SWEEP_EVERY_S = 3600

TYPE_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
# What a node uploads until someone changes it on the Events tab. Vehicles
# are off: a street-facing camera records dozens an hour.
DEFAULT_UPLOAD_TYPES = ["human", "dangerous_person", "face", "package", "anomaly"]
# Shown as toggles even before a node has reported in.
BASE_TYPES = ["human", "vehicle", "dangerous_person", "face", "package", "animal", "anomaly"]

ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,95}$")
LOCAL = {"127.0.0.1", "::1", "localhost"}

_db_lock = threading.Lock()


# -- settings ---------------------------------------------------------------

def load_settings() -> dict:
    try:
        data = json.loads(SETTINGS_PATH.read_text())
    except (OSError, ValueError):
        data = {}
    data.setdefault("retention_days", DEFAULT_RETENTION_DAYS)
    data.setdefault("upload_types", {})     # node -> [types]; missing = defaults
    data.setdefault("known_types", {})      # node -> [types it can record]
    data.setdefault("last_seen", {})        # node -> ISO time of last config fetch
    return data


def upload_types_for(data: dict, node: str) -> list:
    return list(data["upload_types"].get(node, DEFAULT_UPLOAD_TYPES))


def settings_view(data: dict) -> dict:
    with _db_lock, db() as conn:
        seen = [r[0] for r in conn.execute("SELECT DISTINCT node FROM events")]
    nodes = sorted(set(seen) | set(data["known_types"]) | set(data["upload_types"]))
    return {
        "retention_days": data["retention_days"],
        "defaults": DEFAULT_UPLOAD_TYPES,
        "nodes": [{
            "node": n,
            "known_types": data["known_types"].get(n, BASE_TYPES),
            "upload_types": upload_types_for(data, n),
            "last_seen": data["last_seen"].get(n),
        } for n in nodes],
    }


def save_settings(data: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = SETTINGS_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1))
    tmp.replace(SETTINGS_PATH)


# -- index ------------------------------------------------------------------

COLUMNS = ("id", "node", "started_at", "ended_at", "ts", "date", "event_type",
           "confidence", "duration_s", "saved", "dir", "has_thumb", "bytes",
           "sha256", "received_at")


@contextmanager
def db():
    """One connection per use: commit on success, always close (Windows
    holds the file otherwise)."""
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def init_db() -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    STAGING.mkdir(parents=True, exist_ok=True)
    EVENTS_DIR.mkdir(parents=True, exist_ok=True)
    SAVED_DIR.mkdir(parents=True, exist_ok=True)
    with _db_lock, db() as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS events (
            id TEXT PRIMARY KEY, node TEXT, started_at TEXT, ended_at TEXT,
            ts REAL, date TEXT, event_type TEXT, confidence REAL,
            duration_s REAL, saved INTEGER DEFAULT 0, dir TEXT,
            has_thumb INTEGER DEFAULT 0, bytes INTEGER, sha256 TEXT,
            received_at TEXT)""")
        conn.execute("CREATE INDEX IF NOT EXISTS ev_ts ON events(ts)")
        empty = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 0
    if empty:
        n = rebuild_index()
        if n:
            logger.info("index rebuilt from %d sidecar(s)", n)


def rebuild_index() -> int:
    """Re-read every .json sidecar under both roots into the index."""
    count = 0
    for root, saved in ((EVENTS_DIR, 0), (SAVED_DIR, 1)):
        for side in root.glob("*/*/*.json"):
            try:
                meta = json.loads(side.read_text())
                clip = side.with_suffix(".mp4")
                if not clip.exists():
                    continue
                meta["saved"] = saved
                upsert(meta, side.parent, clip.stat().st_size,
                       side.with_suffix(".jpg").exists())
                count += 1
            except Exception as exc:
                logger.warning("skipping sidecar %s: %s", side, exc)
    return count


def upsert(meta: dict, folder: Path, size: int, has_thumb: bool) -> None:
    started = meta["started_at"]
    row = {
        "id": meta["event_id"], "node": meta.get("node", "?"),
        "started_at": started, "ended_at": meta.get("ended_at"),
        "ts": datetime.fromisoformat(started).timestamp(),
        "date": started[:10], "event_type": meta.get("event_type", "event"),
        "confidence": float(meta.get("confidence") or 0),
        "duration_s": float(meta.get("duration_s") or 0),
        "saved": int(meta.get("saved", 0)), "dir": str(folder),
        "has_thumb": int(has_thumb), "bytes": size,
        "sha256": meta.get("sha256"), "received_at": meta.get("received_at"),
    }
    with _db_lock, db() as conn:
        conn.execute(
            f"INSERT OR REPLACE INTO events ({','.join(COLUMNS)}) "
            f"VALUES ({','.join('?' * len(COLUMNS))})",
            [row[c] for c in COLUMNS])


def get_row(event_id: str) -> dict:
    if not ID_RE.match(event_id):
        raise HTTPException(404, "No such event")
    with _db_lock, db() as conn:
        row = conn.execute("SELECT * FROM events WHERE id=?", (event_id,)).fetchone()
    if row is None:
        raise HTTPException(404, "No such event")
    return dict(row)


def public(row: dict) -> dict:
    out = {k: row[k] for k in COLUMNS if k not in ("dir", "sha256")}
    out["saved"] = bool(row["saved"])
    out["has_thumb"] = bool(row["has_thumb"])
    return out


# -- files ------------------------------------------------------------------

def node_folder(node: str) -> str:
    return re.sub(r"[^A-Za-z0-9 _-]+", "_", node).strip() or "unknown"


def files_of(row: dict) -> list:
    folder = Path(row["dir"])
    return [folder / f"{row['id']}{ext}" for ext in (".mp4", ".jpg", ".json")]


def inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def remove_empty_dirs(folder: Path, root: Path) -> None:
    while inside(folder, root) and folder.resolve() != root.resolve():
        try:
            folder.rmdir()
        except OSError:
            return
        folder = folder.parent


def move_event(row: dict, saved: bool) -> dict:
    src_root, dst_root = (EVENTS_DIR, SAVED_DIR) if saved else (SAVED_DIR, EVENTS_DIR)
    src = Path(row["dir"])
    dst = dst_root / node_folder(row["node"]) / row["date"]
    dst.mkdir(parents=True, exist_ok=True)
    for f in files_of(row):
        if f.exists():
            shutil.move(str(f), str(dst / f.name))
    side = dst / f"{row['id']}.json"
    try:
        meta = json.loads(side.read_text())
        meta["saved"] = int(saved)
        side.write_text(json.dumps(meta, indent=1))
    except (OSError, ValueError):
        pass
    remove_empty_dirs(src, src_root)
    with _db_lock, db() as conn:
        conn.execute("UPDATE events SET saved=?, dir=? WHERE id=?",
                     (int(saved), str(dst), row["id"]))
    row.update(saved=int(saved), dir=str(dst))
    return row


def delete_event(row: dict) -> None:
    root = SAVED_DIR if row["saved"] else EVENTS_DIR
    for f in files_of(row):
        if inside(f, root):
            f.unlink(missing_ok=True)
    remove_empty_dirs(Path(row["dir"]), root)
    with _db_lock, db() as conn:
        conn.execute("DELETE FROM events WHERE id=?", (row["id"],))


def sweep() -> int:
    """Delete unsaved events older than the retention period."""
    days = load_settings()["retention_days"]
    cutoff = time.time() - days * 86400
    with _db_lock, db() as conn:
        rows = [dict(r) for r in conn.execute(
            "SELECT * FROM events WHERE saved=0 AND ts < ?", (cutoff,))]
    for row in rows:
        try:
            delete_event(row)
        except Exception as exc:
            logger.warning("retention: could not delete %s: %s", row["id"], exc)
    if rows:
        logger.info("retention (%d days): deleted %d event(s)", days, len(rows))
    # Thumbnails whose clip never arrived.
    for f in STAGING.glob("*"):
        if time.time() - f.stat().st_mtime > 86400:
            f.unlink(missing_ok=True)
    return len(rows)


# -- app --------------------------------------------------------------------

def check_token(request: Request, always: bool = False) -> None:
    if not TOKEN:
        return
    host = request.client.host if request.client else ""
    if not always and host in LOCAL:
        return
    if request.headers.get("x-argus-token") != TOKEN:
        raise HTTPException(401, "Bad or missing X-Argus-Token")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    stop = threading.Event()

    def loop():
        while True:
            try:
                sweep()
            except Exception as exc:
                logger.warning("retention sweep failed: %s", exc)
            if stop.wait(SWEEP_EVERY_S):
                return
    threading.Thread(target=loop, name="retention", daemon=True).start()
    logger.info("events root %s, saved root %s, retention %d days, token %s",
                EVENTS_DIR, SAVED_DIR, load_settings()["retention_days"],
                "on" if TOKEN else "OFF (anyone on the LAN can read and upload)")
    yield
    stop.set()


app = FastAPI(title="ARGUS Cloud Connect", lifespan=lifespan)


async def read_body(request: Request) -> bytes:
    if int(request.headers.get("content-length") or 0) > MAX_UPLOAD:
        raise HTTPException(413, "Too large")
    body = await request.body()
    if not body:
        raise HTTPException(400, "Empty body")
    return body


@app.get("/api/health")
def health(request: Request):
    check_token(request)
    with _db_lock, db() as conn:
        n = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    return {"ok": True, "events": n, "events_dir": str(EVENTS_DIR),
            "saved_dir": str(SAVED_DIR), "retention_days": load_settings()["retention_days"]}


@app.put("/api/ingest/{event_id}/thumb")
async def ingest_thumb(event_id: str, request: Request):
    check_token(request, always=True)
    if not ID_RE.match(event_id):
        raise HTTPException(400, "Bad event id")
    body = await read_body(request)
    (STAGING / f"{event_id}.jpg").write_bytes(body)
    return {"ok": True}


@app.put("/api/ingest/{event_id}/clip")
async def ingest_clip(event_id: str, request: Request):
    check_token(request, always=True)
    if not ID_RE.match(event_id):
        raise HTTPException(400, "Bad event id")
    try:
        meta = json.loads(request.headers.get("x-argus-meta") or "")
        datetime.fromisoformat(meta["started_at"])
    except (ValueError, KeyError, TypeError):
        raise HTTPException(400, "Missing or bad X-Argus-Meta")
    if meta.get("event_id") != event_id:
        raise HTTPException(400, "event_id mismatch")
    body = await read_body(request)
    digest = hashlib.sha256(body).hexdigest()

    # Re-upload after a lost reply: same bytes, nothing to do.
    with _db_lock, db() as conn:
        old = conn.execute("SELECT sha256 FROM events WHERE id=?", (event_id,)).fetchone()
    if old is not None and old["sha256"] == digest:
        return {"ok": True, "id": event_id, "sha256": digest, "duplicate": True}

    folder = EVENTS_DIR / node_folder(meta.get("node", "unknown")) / meta["started_at"][:10]
    folder.mkdir(parents=True, exist_ok=True)
    clip = folder / f"{event_id}.mp4"
    tmp = folder / f"{event_id}.mp4.part"
    tmp.write_bytes(body)
    tmp.replace(clip)
    staged = STAGING / f"{event_id}.jpg"
    has_thumb = staged.exists()
    if has_thumb:
        shutil.move(str(staged), str(folder / f"{event_id}.jpg"))
    meta.update(sha256=digest, saved=0, bytes=len(body),
                received_at=datetime.now().isoformat(timespec="seconds"))
    (folder / f"{event_id}.json").write_text(json.dumps(meta, indent=1))
    upsert(meta, folder, len(body), has_thumb)
    logger.info("stored %s (%s, %.1f MB)", event_id, meta.get("event_type"), len(body) / 1e6)
    return {"ok": True, "id": event_id, "sha256": digest}


@app.get("/api/events")
def list_events(request: Request, node: Optional[str] = None,
                date_from: Optional[str] = None, date_to: Optional[str] = None,
                saved: Optional[bool] = None, event_type: Optional[str] = None,
                limit: int = 100, offset: int = 0):
    check_token(request)
    where, args = [], []
    if node:
        where.append("node=?"); args.append(node)
    if date_from:
        where.append("date>=?"); args.append(date_from[:10])
    if date_to:
        where.append("date<=?"); args.append(date_to[:10])
    if saved is not None:
        where.append("saved=?"); args.append(int(saved))
    if event_type:
        where.append("event_type=?"); args.append(event_type)
    clause = f"WHERE {' AND '.join(where)}" if where else ""
    limit = max(1, min(limit, 500))
    with _db_lock, db() as conn:
        total = conn.execute(f"SELECT COUNT(*) FROM events {clause}", args).fetchone()[0]
        rows = conn.execute(f"SELECT * FROM events {clause} ORDER BY ts DESC LIMIT ? OFFSET ?",
                            [*args, limit, max(0, offset)]).fetchall()
        nodes = [r[0] for r in conn.execute("SELECT DISTINCT node FROM events ORDER BY node")]
        types = [r[0] for r in conn.execute("SELECT DISTINCT event_type FROM events ORDER BY 1")]
    return {"events": [public(dict(r)) for r in rows], "total": total,
            "nodes": nodes, "event_types": types,
            "retention_days": load_settings()["retention_days"]}


@app.get("/api/events/{event_id}")
def event_detail(event_id: str, request: Request):
    check_token(request)
    return public(get_row(event_id))


def _file(event_id: str, ext: str) -> Path:
    row = get_row(event_id)
    path = Path(row["dir"]) / f"{event_id}{ext}"
    if not path.exists():
        raise HTTPException(404, "File missing")
    return path


@app.get("/api/events/{event_id}/clip")
def event_clip(event_id: str, request: Request):
    check_token(request)
    return FileResponse(_file(event_id, ".mp4"), media_type="video/mp4")


@app.get("/api/events/{event_id}/thumb")
def event_thumb(event_id: str, request: Request):
    check_token(request)
    return FileResponse(_file(event_id, ".jpg"), media_type="image/jpeg",
                        headers={"Cache-Control": "max-age=86400"})


@app.get("/api/events/{event_id}/download")
def event_download(event_id: str, request: Request):
    check_token(request)
    row = get_row(event_id)
    stamp = row["started_at"][:19].replace("T", " ").replace(":", "-")
    name = f"ARGUS {node_folder(row['node'])} {stamp} {row['event_type']}.mp4"
    return FileResponse(_file(event_id, ".mp4"), media_type="video/mp4", filename=name)


@app.post("/api/events/{event_id}/save")
def event_save(event_id: str, request: Request):
    check_token(request)
    row = get_row(event_id)
    if not row["saved"]:
        row = move_event(row, True)
    return public(row)


@app.post("/api/events/{event_id}/unsave")
def event_unsave(event_id: str, request: Request):
    check_token(request)
    row = get_row(event_id)
    if row["saved"]:
        row = move_event(row, False)
    return public(row)


@app.delete("/api/events/{event_id}")
def event_delete(event_id: str, request: Request):
    check_token(request)
    delete_event(get_row(event_id))
    return {"ok": True}


_settings_lock = threading.Lock()


@app.get("/api/node-config")
def node_config(request: Request, node: str, known: str = ""):
    """Polled by each node (about once a minute): which event types to
    upload. The node reports the types it can record, for the console."""
    check_token(request, always=True)
    node = node.strip()[:64]
    if not node:
        raise HTTPException(400, "node required")
    types = sorted({t for t in known.split(",") if TYPE_RE.match(t)})
    with _settings_lock:
        data = load_settings()
        changed = False
        if types and data["known_types"].get(node) != types:
            data["known_types"][node] = types
            changed = True
        now = datetime.now().isoformat(timespec="seconds")
        last = data["last_seen"].get(node)
        if not last or now[:16] != last[:16]:    # write at most once a minute
            data["last_seen"][node] = now
            changed = True
        if changed:
            save_settings(data)
    return {"node": node, "event_types": upload_types_for(data, node)}


@app.get("/api/settings")
def get_settings(request: Request):
    check_token(request)
    return settings_view(load_settings())


@app.put("/api/settings")
async def put_settings(request: Request):
    """Body may carry retention_days and/or upload_types {node: [types]}."""
    check_token(request)
    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(400, "JSON object expected")
    deleted = 0
    with _settings_lock:
        data = load_settings()
        if "retention_days" in body:
            try:
                days = int(body["retention_days"])
            except (TypeError, ValueError):
                raise HTTPException(400, "retention_days must be a whole number")
            if not 1 <= days <= 3650:
                raise HTTPException(400, "retention_days must be 1 to 3650")
            data["retention_days"] = days
        if "upload_types" in body:
            ut = body["upload_types"]
            if not isinstance(ut, dict):
                raise HTTPException(400, "upload_types must map node to a list")
            for node, types in ut.items():
                if not isinstance(types, list) or not all(
                        isinstance(t, str) and TYPE_RE.match(t) for t in types):
                    raise HTTPException(400, f"bad type list for {node}")
                data["upload_types"][str(node)[:64]] = sorted(set(types))
        save_settings(data)
    if "retention_days" in body:
        deleted = sweep()
    return {**settings_view(data), "deleted_now": deleted}


if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
