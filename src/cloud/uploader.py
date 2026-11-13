"""
Cloud Connect: ship every event clip to the central server.

The node's card cannot hold weeks of footage (Node B's died nearly full),
so history lives on the central server and the node keeps only a short
queue. Flow for one event:

1. The recorder closes a clip (it already has a 5 s pre-roll ring buffer).
   `enqueue` hard-links the raw file into the queue folder at once, so the
   clip keeper deleting its copy 30 s later does not lose it, and writes
   meta.json next to it.
2. The worker thread trims it to CLOUD_CLIP_MAX_S, re-encodes to H.264
   (plays in any browser), grabs a thumbnail, and drops the raw file.
3. It PUTs the thumbnail, then the clip with its metadata. Only when the
   server answers with the same SHA-256 is the local folder deleted.
   Any failure leaves the folder in place and retries with backoff.

The queue never fills the disk: it is capped at CLOUD_QUEUE_MAX_MB (oldest
events dropped first), and nothing is queued below the recorder's
free-space floor.

Plain urllib, no new dependencies.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

logger = logging.getLogger(__name__)

MAX_BACKOFF_S = 300.0
TRANSCODE_TRIES = 3


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower() or "node"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class CloudUploader:
    def __init__(self, server_url: str, node_name: str, queue_dir: Path,
                 token: str = "", clip_max_s: float = 20.0,
                 queue_max_bytes: int = 1_500_000_000,
                 min_free_bytes: int = 0, pre_roll_s: float = 5.0,
                 event_types=("human", "dangerous_person", "face", "package", "anomaly"),
                 known_types=(), config_every_s: float = 60.0):
        self.server = server_url.rstrip("/")
        self.node = node_name
        self.queue_dir = Path(queue_dir)
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        self.token = token
        self.clip_max_s = clip_max_s
        self.queue_max_bytes = queue_max_bytes
        self.min_free_bytes = min_free_bytes
        self.pre_roll_s = pre_roll_s
        # Which event types upload: recorder triggers, plus "anomaly" for
        # clips the keeper keeps on a P1 verdict. Set from the console
        # (Events tab) and fetched from the server every config_every_s, so
        # a change needs no restart. CLOUD_EVENT_TYPES is only the fallback
        # until the first fetch; the last fetched list is cached on disk.
        self.event_types = {t.strip() for t in event_types if t.strip()}
        self.known_types = sorted({t for t in known_types if t} | {"anomaly"})
        self.config_every_s = config_every_s
        self._config_at = float("-inf")
        self.config_source = "default"
        self._config_cache = self.queue_dir.parent / "cloud_config.json"
        try:
            cached = json.loads(self._config_cache.read_text())
            self.event_types = set(cached["event_types"])
            self.config_source = "cache"
        except (OSError, ValueError, KeyError, TypeError):
            pass
        self.ffmpeg = shutil.which("ffmpeg")
        if self.ffmpeg is None:
            logger.warning("cloud: ffmpeg not found; clips upload as mp4v "
                           "without thumbnails (sudo apt install ffmpeg)")

        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._retry_at: dict = {}      # event_id -> earliest next attempt
        self._fails: dict = {}         # event_id -> consecutive failures
        self._transcode_fails: dict = {}
        self.uploaded = self.dropped = 0
        self.last_error: Optional[str] = None
        self.last_ok_at: Optional[str] = None

    # -- enqueue (called on the detection thread; must be quick) ----------

    def wants(self, trigger: str) -> bool:
        return "all" in self.event_types or trigger in self.event_types

    def refresh_config(self, force: bool = False) -> None:
        """Fetch this node's upload types from the server (rate-limited)."""
        now = time.monotonic()
        if not force and now - self._config_at < self.config_every_s:
            return
        self._config_at = now
        q = urllib.parse.urlencode({"node": self.node, "known": ",".join(self.known_types)})
        req = urllib.request.Request(f"{self.server}/api/node-config?{q}")
        if self.token:
            req.add_header("X-Argus-Token", self.token)
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                types = json.loads(resp.read())["event_types"]
        except Exception as exc:
            logger.debug("cloud: config fetch failed: %s", exc)
            return
        new = {str(t) for t in types}
        if new != self.event_types or self.config_source != "server":
            if new != self.event_types:
                logger.info("cloud: upload types now %s", sorted(new))
            self.event_types = new
            self.config_source = "server"
            try:
                self._config_cache.write_text(json.dumps({"event_types": sorted(new)}))
            except OSError:
                pass

    def enqueue_anomaly(self, path: Path, start: float, end: float,
                        trigger: str, confidence: float) -> None:
        """A clip the keeper kept for a P1 (anomaly) event. Skipped when its
        trigger already uploaded it on close, or anomalies are switched off."""
        if self.wants(trigger) or not self.wants("anomaly"):
            return
        self.enqueue(SimpleNamespace(
            path=str(path), started_at=datetime.fromtimestamp(start).isoformat(),
            ended_at=datetime.fromtimestamp(end).isoformat(),
            duration_s=end - start, trigger=trigger, peak_confidence=confidence,
        ), event_type="anomaly")

    def enqueue(self, record, event_type: Optional[str] = None) -> None:
        if event_type is None and not self.wants(record.trigger):
            return
        src = Path(record.path)
        if not src.exists():
            return
        try:
            if shutil.disk_usage(self.queue_dir).free < self.min_free_bytes:
                logger.warning("cloud: disk below free floor, not queueing %s", src.name)
                self.dropped += 1
                return
        except OSError:
            pass

        started = datetime.fromisoformat(record.started_at)
        event_id = f"{_slug(self.node)}-{started.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        item = self.queue_dir / event_id
        try:
            item.mkdir(parents=True)
            try:
                os.link(src, item / "raw.mp4")
            except OSError:
                shutil.copy2(src, item / "raw.mp4")
            meta = {
                "event_id": event_id,
                "node": self.node,
                "started_at": record.started_at,
                "ended_at": record.ended_at,
                "duration_s": round(float(record.duration_s), 2),
                "event_type": event_type or record.trigger,
                "trigger": record.trigger,
                "confidence": round(float(record.peak_confidence), 3),
                "pre_roll_s": self.pre_roll_s,
            }
            (item / "meta.json").write_text(json.dumps(meta))
        except Exception as exc:
            logger.warning("cloud: could not queue %s: %s", src.name, exc)
            shutil.rmtree(item, ignore_errors=True)
            return
        logger.info("cloud: queued %s (%s %.2f)", event_id, record.trigger,
                    record.peak_confidence)
        self._enforce_cap()
        self._wake.set()

    # -- queue housekeeping ------------------------------------------------

    def _items(self):
        return sorted(p for p in self.queue_dir.iterdir()
                      if p.is_dir() and (p / "meta.json").exists())

    def _enforce_cap(self) -> None:
        items = self._items()
        sizes = {p: sum(f.stat().st_size for f in p.iterdir() if f.is_file()) for p in items}
        total = sum(sizes.values())
        while items and total > self.queue_max_bytes:
            oldest = items.pop(0)
            total -= sizes[oldest]
            shutil.rmtree(oldest, ignore_errors=True)
            self.dropped += 1
            logger.warning("cloud: queue over %d MB, dropped %s",
                           self.queue_max_bytes // 1_000_000, oldest.name)
        # Orphans: a folder whose meta.json never got written.
        for p in self.queue_dir.iterdir():
            if p.is_dir() and not (p / "meta.json").exists() and \
                    time.time() - p.stat().st_mtime > 600:
                shutil.rmtree(p, ignore_errors=True)

    # -- prepare: trim, H.264, thumbnail ----------------------------------

    def _prepare(self, item: Path) -> bool:
        raw, clip = item / "raw.mp4", item / "clip.mp4"
        if clip.exists() and not raw.exists():
            return True
        if not raw.exists():
            return False
        if self.ffmpeg is None or self._transcode_fails.get(item.name, 0) >= TRANSCODE_TRIES:
            raw.replace(clip)       # better an mp4v clip than none
            return True
        nice = ["nice", "-n", "19"] if shutil.which("nice") else []
        tmp = item / "clip.tmp.mp4"
        cmd = nice + [self.ffmpeg, "-y", "-loglevel", "error", "-i", str(raw),
                      "-t", str(self.clip_max_s),
                      "-vf", "scale='min(1280,iw)':-2", "-c:v", "libx264",
                      "-preset", "veryfast", "-crf", "28", "-pix_fmt", "yuv420p",
                      "-threads", "2", "-movflags", "+faststart", "-an", str(tmp)]
        try:
            subprocess.run(cmd, check=True, timeout=300, capture_output=True, text=True)
        except Exception as exc:
            n = self._transcode_fails.get(item.name, 0) + 1
            self._transcode_fails[item.name] = n
            err = getattr(exc, "stderr", "") or str(exc)
            logger.warning("cloud: transcode %d/%d failed for %s: %s",
                           n, TRANSCODE_TRIES, item.name, err.strip()[-300:])
            tmp.unlink(missing_ok=True)
            return False
        tmp.replace(clip)
        raw.unlink(missing_ok=True)
        # Thumbnail just after the trigger (the end of the pre-roll).
        thumb = item / "thumb.jpg"
        try:
            subprocess.run(nice + [self.ffmpeg, "-y", "-loglevel", "error",
                                   "-ss", str(self.pre_roll_s + 0.5), "-i", str(clip),
                                   "-frames:v", "1", "-vf", "scale=480:-2", str(thumb)],
                           check=True, timeout=60, capture_output=True)
            if not thumb.exists() or thumb.stat().st_size == 0:
                subprocess.run(nice + [self.ffmpeg, "-y", "-loglevel", "error",
                                       "-i", str(clip), "-frames:v", "1",
                                       "-vf", "scale=480:-2", str(thumb)],
                               check=True, timeout=60, capture_output=True)
        except Exception as exc:
            logger.warning("cloud: thumbnail failed for %s: %s", item.name, exc)
        return True

    # -- upload ------------------------------------------------------------

    def _put(self, path: str, body: bytes, headers: dict) -> dict:
        req = urllib.request.Request(self.server + path, data=body, method="PUT")
        req.add_header("Content-Type", "application/octet-stream")
        if self.token:
            req.add_header("X-Argus-Token", self.token)
        for k, v in headers.items():
            req.add_header(k, v)
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read() or b"{}")

    def _upload(self, item: Path) -> bool:
        meta = json.loads((item / "meta.json").read_text())
        eid = urllib.parse.quote(meta["event_id"])
        thumb, clip = item / "thumb.jpg", item / "clip.mp4"
        if thumb.exists():
            self._put(f"/api/ingest/{eid}/thumb", thumb.read_bytes(), {})
        digest = _sha256(clip)
        meta["sha256"] = digest
        reply = self._put(f"/api/ingest/{eid}/clip", clip.read_bytes(),
                          {"X-Argus-Meta": json.dumps(meta, ensure_ascii=True)})
        if reply.get("sha256") != digest:
            raise RuntimeError(f"server checksum mismatch: {reply}")
        return True

    # -- worker ------------------------------------------------------------

    def run_once(self) -> None:
        self.refresh_config()
        now = time.time()
        for item in self._items():
            if self._stop.is_set():
                return
            name = item.name
            if self._retry_at.get(name, 0) > now:
                continue
            try:
                if not self._prepare(item):
                    if not (item / "raw.mp4").exists() and not (item / "clip.mp4").exists():
                        shutil.rmtree(item, ignore_errors=True)
                    continue
                self._upload(item)
            except Exception as exc:
                n = self._fails.get(name, 0) + 1
                self._fails[name] = n
                delay = min(MAX_BACKOFF_S, 5.0 * 2 ** (n - 1))
                self._retry_at[name] = time.time() + delay
                self.last_error = f"{type(exc).__name__}: {exc}"
                logger.warning("cloud: upload of %s failed (try %d, retry in %.0fs): %s",
                               name, n, delay, self.last_error)
                # Server down: no point hammering the rest of the queue now.
                if isinstance(exc, (urllib.error.URLError, OSError)):
                    return
                continue
            shutil.rmtree(item, ignore_errors=True)
            for d in (self._retry_at, self._fails, self._transcode_fails):
                d.pop(name, None)
            self.uploaded += 1
            self.last_ok_at = datetime.now().isoformat(timespec="seconds")
            self.last_error = None
            logger.info("cloud: uploaded %s", name)

    def start(self, interval_s: float = 5.0) -> None:
        def loop():
            while not self._stop.is_set():
                try:
                    self.run_once()
                except Exception as exc:
                    logger.warning("cloud: worker error: %s", exc)
                self._wake.wait(interval_s)
                self._wake.clear()
        self._thread = threading.Thread(target=loop, name="cloud-uploader", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()

    def status(self) -> dict:
        items = self._items()
        return {
            "server": self.server,
            "node": self.node,
            "queued": len(items),
            "queue_bytes": sum(f.stat().st_size for p in items for f in p.iterdir() if f.is_file()),
            "uploaded": self.uploaded,
            "dropped": self.dropped,
            "last_ok_at": self.last_ok_at,
            "last_error": self.last_error,
            "transcode": self.ffmpeg is not None,
            "event_types": sorted(self.event_types),
            "config_source": self.config_source,
        }
