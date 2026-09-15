"""
Sample store for the federated head.

Every detection the pipeline resolves becomes a candidate training sample:
a 19-dim feature vector plus whatever context helps a human label it later.
Samples land unlabelled. A label arrives separately, from the operator via
the dashboard.

Why not auto-label
------------------
The simulator derives labels from a site rule it also generates the data
from, which is fine for measuring throughput and useless as evidence about
accuracy. On the live node the label has to come from outside the model or
the whole exercise is circular. So: capture unlabelled, label by hand, and
report how many are labelled so nobody quotes an accuracy figure computed
over eleven samples.

JSONL on disk, one sample per line -- inspectable with `tail`, survives a
restart, and needs no schema migration when a field is added.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.federated.features import LABEL_NAMES, N_CLASSES, N_FEATURES, describe

logger = logging.getLogger(__name__)


class SampleStore:
    def __init__(self, path: Path, max_samples: int = 200_000):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_samples = max_samples
        self._lock = threading.Lock()
        self._rows: List[dict] = []
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        bad = 0
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
                if len(r.get("x", [])) == N_FEATURES:
                    self._rows.append(r)
                else:
                    bad += 1
            except Exception:
                bad += 1
        if bad:
            # Loud rather than silent: a truncated last line is normal after a
            # hard power cut, but a large count means the schema moved.
            logger.warning("SampleStore: skipped %d unreadable rows in %s",
                           bad, self.path)
        logger.info("SampleStore: loaded %d samples (%d labelled)",
                    len(self._rows), self.n_labelled)

    # ------------------------------------------------------------------
    def _next_n(self) -> int:
        return max((r.get("n", 0) for r in self._rows), default=0) + 1

    def by_n(self, n: int) -> Optional[dict]:
        for r in self._rows:
            if r.get("n") == n:
                return r
        return None

    @property
    def n_labelled(self) -> int:
        return sum(1 for r in self._rows if r.get("y") is not None)

    def append(self, x: np.ndarray, label: Optional[int] = None,
               meta: Optional[dict] = None) -> str:
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        if x.size != N_FEATURES:
            raise ValueError(f"expected {N_FEATURES} features, got {x.size}")
        if not np.isfinite(x).all():
            raise ValueError("feature vector contains NaN or inf")

        sid = f"{int(time.time() * 1000):x}-{len(self._rows):05d}"
        # n is the dashboard-facing id. DetectionEvent.id is typed number in
        # lib/api.ts, and POST /api/events/{id}/label addresses it, so a
        # stable monotonic integer is part of the contract, not a nicety.
        row = {
            "id": sid,
            "n": self._next_n(),
            "t": time.time(),
            "x": [round(float(v), 6) for v in x],
            "y": int(label) if label is not None else None,
            "meta": meta or {},
        }
        with self._lock:
            self._rows.append(row)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            if len(self._rows) > self.max_samples:
                self._compact()
        return sid

    def set_label_by_n(self, n: int, label: int) -> bool:
        with self._lock:
            r = self.by_n(n)
            if r is None:
                return False
            r["y"] = int(label)
            self._rewrite()
            return True

    def set_label(self, sample_id: str, label: int) -> bool:
        if not 0 <= int(label) < N_CLASSES:
            raise ValueError(f"label must be 0..{N_CLASSES - 1}")
        with self._lock:
            for r in self._rows:
                if r["id"] == sample_id:
                    r["y"] = int(label)
                    self._rewrite()
                    return True
        return False

    def labelled(self) -> Tuple[np.ndarray, np.ndarray]:
        rows = [r for r in self._rows if r.get("y") is not None]
        if not rows:
            return (np.zeros((0, N_FEATURES), np.float32),
                    np.zeros((0,), np.int64))
        X = np.array([r["x"] for r in rows], dtype=np.float32)
        y = np.array([r["y"] for r in rows], dtype=np.int64)
        return X, y

    def recent(self, limit: int = 50) -> List[dict]:
        out = []
        for r in self._rows[-limit:][::-1]:
            out.append({
                "id": r["id"],
                "n": r.get("n"),
                "t": r["t"],
                "label": LABEL_NAMES[r["y"]] if r.get("y") is not None else None,
                "features": describe(np.array(r["x"], dtype=np.float32)),
                "meta": r.get("meta", {}),
            })
        return out

    def stats(self) -> dict:
        _, y = self.labelled()
        counts = np.bincount(y, minlength=N_CLASSES) if y.size else np.zeros(N_CLASSES, int)
        return {
            "total": len(self._rows),
            "labelled": int(y.size),
            "unlabelled": len(self._rows) - int(y.size),
            "per_label": {LABEL_NAMES[i]: int(counts[i]) for i in range(N_CLASSES)},
            "path": str(self.path),
        }

    def bootstrap(self, site_key: str, n: int, seed: int = 0) -> int:
        """Seed the store from the simulator's generator.

        A ten-minute capture does not produce enough labelled events to
        train on. This makes a demo possible while keeping the distinction
        visible: bootstrapped rows are tagged synthetic, so stats() can
        separate them from anything the camera actually saw.
        """
        import sys
        sim = Path(__file__).resolve().parents[2] / "sim"
        if str(sim) not in sys.path:
            sys.path.insert(0, str(sim))
        import workload  # noqa: E402

        X, y = workload.generate(workload.SITES[site_key], n, seed=seed)
        for xi, yi in zip(X, y):
            self.append(xi, int(yi), meta={"synthetic": True, "site": site_key})
        return int(len(X))

    # ------------------------------------------------------------------
    def _rewrite(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for r in self._rows:
                f.write(json.dumps(r) + "\n")
        tmp.replace(self.path)

    def _compact(self) -> None:
        """Drop oldest unlabelled rows first; labelled data is the asset."""
        keep_lab = [r for r in self._rows if r.get("y") is not None]
        keep_unl = [r for r in self._rows if r.get("y") is None]
        room = max(0, self.max_samples - len(keep_lab))
        self._rows = keep_lab + keep_unl[-room:]
        self._rows.sort(key=lambda r: r["t"])
        self._rewrite()
