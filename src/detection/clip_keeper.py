"""
Which clips survive.

The recorder writes footage around every trigger; the card cannot hold
that (18 GB of it filled Node A on 2026-10-25). The keeper decides
afterwards: a finished clip is held for a short grace period, then kept
only if a priority-1 event overlaps it and deleted otherwise. Kept clips
are re-encoded to H.264 (the recorder's mp4v does not play in Chrome or
Edge) and the folder is held under a size cap, oldest first.

Priority
--------
1 = the federated head's anomaly call (or an operator's anomaly label):
the red "Unusual for this site" alerts. Priorities 2 and 3 come later;
until then everything else is unranked (None) and gets no footage.

An event's priority is frozen the first time it is decided, in
index.json next to the clips, so a later model round cannot silently
demote an event whose clip was kept because of it.

Why match against the store rather than hook the collector
----------------------------------------------------------
A track is only emitted once it has been gone for max_idle_s, and the
clip closes post_roll_s after the last detection, so either can finish
first. Waiting a grace period and then asking the store "which events
overlap this clip" has no ordering to get wrong.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from src.federated.features import ANOMALY

logger = logging.getLogger(__name__)

P1 = 1

# An event is emitted up to max_idle_s after its track was last seen, and
# carries its duration, so its span is [t - duration - slack, t].
EMIT_SLACK_S = 3.0


def priority_of(row: dict, head) -> Optional[int]:
    """1 for an anomaly (operator label first, then the head), else None."""
    if row.get("y") is not None:
        verdict = int(row["y"])
    elif head is not None:
        verdict = head.infer_one(np.array(row["x"], dtype=np.float32))
    else:
        return None
    return P1 if verdict == ANOMALY else None


def event_span(row: dict) -> tuple:
    t = float(row.get("t") or 0.0)
    dur = float((row.get("meta") or {}).get("duration_s") or 0.0)
    return t - dur - EMIT_SLACK_S, t


class ClipKeeper:
    def __init__(self, clip_dir: Path, store, head_source: Callable[[], object],
                 max_bytes: int, min_free_bytes: int, grace_s: float = 30.0,
                 transcode: bool = True):
        self.clip_dir = Path(clip_dir)
        self.clip_dir.mkdir(parents=True, exist_ok=True)
        self.store = store
        self._head = head_source
        self.max_bytes = max_bytes
        self.min_free_bytes = min_free_bytes
        self.grace_s = grace_s
        self.ffmpeg = shutil.which("ffmpeg") if transcode else None
        if transcode and self.ffmpeg is None:
            logger.warning("clips: ffmpeg not found; kept clips stay mp4v and "
                           "will not play in the browser (sudo apt install ffmpeg)")

        self._lock = threading.Lock()
        self._pending: List[dict] = []     # finished clips awaiting a verdict
        self._index_path = self.clip_dir / "index.json"
        self._index = self._load_index()   # {"events": {n: {...}}, "clips": {name: {...}}}
        self.kept = self.dropped = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # -- index -----------------------------------------------------------

    def _load_index(self) -> dict:
        try:
            data = json.loads(self._index_path.read_text())
            data.setdefault("events", {})
            data.setdefault("clips", {})
            return data
        except (OSError, ValueError):
            return {"events": {}, "clips": {}}

    def _save_index(self) -> None:
        tmp = self._index_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._index, indent=1))
        tmp.replace(self._index_path)

    def frozen_priority(self, n: int) -> Optional[int]:
        entry = self._index["events"].get(str(n))
        return entry.get("priority") if entry else None

    def clip_for(self, n: int) -> Optional[Path]:
        entry = self._index["events"].get(str(n))
        name = entry.get("clip") if entry else None
        if not name:
            return None
        path = (self.clip_dir / name).resolve()
        if path.parent != self.clip_dir.resolve() or not path.exists():
            return None
        return path

    def priority(self, row: dict, head) -> Optional[int]:
        """The frozen priority if there is one, else the live call."""
        return self.frozen_priority(int(row.get("n") or 0)) or priority_of(row, head)

    def freeze(self, n: int, priority: Optional[int], clip: Optional[str] = None) -> None:
        """Record an event's priority (and clip) once; later calls only add a clip."""
        key = str(n)
        entry = self._index["events"].setdefault(key, {"priority": priority})
        if clip and not entry.get("clip"):
            entry["clip"] = clip

    # -- recorder hooks --------------------------------------------------

    def can_record(self) -> bool:
        """False when the disk is too full to start another clip."""
        try:
            return shutil.disk_usage(self.clip_dir).free >= self.min_free_bytes
        except OSError:
            return True

    def on_clip(self, record) -> None:
        """EventRecorder's on_clip: queue the clip for a verdict."""
        start = datetime.fromisoformat(record.started_at).timestamp()
        end = datetime.fromisoformat(record.ended_at).timestamp()
        with self._lock:
            self._pending.append({"path": Path(record.path), "start": start,
                                  "end": end, "trigger": record.trigger})

    # -- the decision ----------------------------------------------------

    def p1_events_in(self, start: float, end: float) -> List[dict]:
        head = self._head()
        hits = []
        for row in list(self.store._rows):
            a, b = event_span(row)
            if b < start or a > end:
                continue
            if self.priority(row, head) == P1:
                hits.append(row)
        return hits

    def claim(self, path: Path, start: float, end: float) -> Optional[str]:
        """Keep the clip if a P1 event overlaps it; delete it otherwise.

        Returns the kept file's name, or None if it was deleted.
        """
        hits = self.p1_events_in(start, end)
        if not hits:
            path.unlink(missing_ok=True)
            self.dropped += 1
            return None

        kept = self._transcode(path)
        for row in hits:
            self.freeze(int(row["n"]), P1, kept.name)
        self._index["clips"][kept.name] = {
            "start": start, "end": end, "bytes": kept.stat().st_size,
            "events": [int(r["n"]) for r in hits],
        }
        self.kept += 1
        logger.info("clips: kept %s for P1 event(s) %s", kept.name,
                    [int(r["n"]) for r in hits])
        return kept.name

    def _transcode(self, path: Path) -> Path:
        """mp4v -> H.264, at most 1280 wide, faststart. Falls back to the original."""
        if self.ffmpeg is None:
            return path
        out = path.with_name(path.stem + "_h264.mp4")
        cmd = ["nice", "-n", "19"] if shutil.which("nice") else []
        cmd += [self.ffmpeg, "-y", "-loglevel", "error", "-i", str(path),
                "-vf", "scale='min(1280,iw)':-2", "-c:v", "libx264",
                "-preset", "veryfast", "-crf", "28", "-pix_fmt", "yuv420p",
                "-threads", "1", "-movflags", "+faststart", "-an", str(out)]
        try:
            subprocess.run(cmd, check=True, timeout=600)
            path.unlink(missing_ok=True)
            return out
        except Exception as exc:
            logger.warning("clips: transcode failed for %s: %s", path.name, exc)
            out.unlink(missing_ok=True)
            return path

    def enforce_cap(self) -> None:
        """Delete the oldest kept clips until the folder is under max_bytes."""
        clips = sorted(self._index["clips"].items(), key=lambda kv: kv[1]["start"])
        total = sum(c["bytes"] for _, c in clips)
        while clips and total > self.max_bytes:
            name, info = clips.pop(0)
            (self.clip_dir / name).unlink(missing_ok=True)
            total -= info["bytes"]
            del self._index["clips"][name]
            # The event keeps its priority; only the footage goes.
            for n in info.get("events", []):
                entry = self._index["events"].get(str(n))
                if entry and entry.get("clip") == name:
                    entry["clip"] = None
            logger.info("clips: cap reached, deleted %s", name)

    def sweep(self, now: Optional[float] = None) -> None:
        now = now if now is not None else time.time()
        with self._lock:
            due = [c for c in self._pending if now - c["end"] >= self.grace_s]
            self._pending = [c for c in self._pending if now - c["end"] < self.grace_s]
        if not due:
            return
        for clip in due:
            try:
                self.claim(clip["path"], clip["start"], clip["end"])
            except Exception as exc:
                logger.warning("clips: verdict failed for %s: %s", clip["path"], exc)
        self.enforce_cap()
        self._save_index()

    # -- lifecycle -------------------------------------------------------

    def start(self, interval_s: float = 10.0) -> None:
        def loop():
            while not self._stop.wait(interval_s):
                self.sweep()
        self._thread = threading.Thread(target=loop, name="clip-keeper", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Decide every pending clip now, so nothing is orphaned on shutdown."""
        self._stop.set()
        self.sweep(now=float("inf"))

    def status(self) -> dict:
        used = sum(c["bytes"] for c in self._index["clips"].values())
        return {
            "kept": self.kept, "dropped": self.dropped,
            "pending": len(self._pending),
            "clips_on_disk": len(self._index["clips"]),
            "bytes_used": used, "max_bytes": self.max_bytes,
            "p1_events": sum(1 for e in self._index["events"].values()
                             if e.get("priority") == P1),
            "transcode": self.ffmpeg is not None,
        }
