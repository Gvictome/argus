"""
Sample store for the federated head.

Every detection the pipeline resolves becomes a candidate training sample:
a 19-dim feature vector plus whatever context helps a human label it later.
Samples land unlabelled. A label arrives separately, from the operator via
the dashboard.

Why provenance
--------------
The simulator derives labels from a site rule it also generates the data
from, which is fine for measuring throughput and useless as evidence about
accuracy. On the live node a label has to come from outside the model or
the whole exercise is circular. Machine suggestions are useful, but only
if nothing can mistake one for a person's decision, so:

  - A row's `prov` says who decided its label and whether a person looked.
  - A row with a label and no `prov` is machine and unreviewed. Rows written
    before this field existed were rule-derived.
  - Suggestions live in their own file (src/federated/suggest.py) and never
    touch `y`.

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

from src.federated.features import ANOMALY, LABEL_NAMES, N_CLASSES, N_FEATURES, describe

logger = logging.getLogger(__name__)


# Where a label came from when the writer did not say. Deliberately machine:
# an old console or a script can only ever understate what was reviewed.
LEGACY_SOURCE = "legacy:rule-derived"
UNATTRIBUTED_SOURCE = "api:unattributed"
IMPORT_SOURCE = "import"


class ReviewedLabelConflict(Exception):
    """An unattributed write tried to replace a person's decision."""


def machine_prov(source: str) -> dict:
    return {"origin": "machine", "source": source, "reviewed": False,
            "at": time.time()}


def human_prov(label: int, review: str, suggested_label: Optional[int] = None,
               suggested_source: Optional[str] = None,
               via: Optional[str] = None) -> dict:
    """Provenance for a decision a person made on this event.

    `review` is how they made it:
      blind    labelled with no suggestion on screen
      correct  a suggestion was shown and they changed it
      accept   a suggestion was shown and they kept it

    An accepted label stays origin=machine and names its source, because
    the idea was the machine's. The reviewed flag records that a person
    looked. Blind and corrected labels are the only ones with no machine
    input to the final answer.
    """
    if review not in ("blind", "correct", "accept"):
        raise ValueError(f"unknown review kind {review!r}")
    if review != "blind" and (suggested_label is None or not suggested_source):
        raise ValueError(f"review {review!r} needs the suggestion that was shown")
    if review == "accept" and int(suggested_label) != int(label):
        raise ValueError("an accepted suggestion must match the label")
    if review == "correct" and int(suggested_label) == int(label):
        raise ValueError("a correction must differ from the suggestion")

    prov = {
        "origin": "machine" if review == "accept" else "human",
        "source": suggested_source if review == "accept" else "operator",
        "reviewed": True,
        "review": review,
        "at": time.time(),
    }
    if review != "blind":
        prov["suggested_label"] = LABEL_NAMES[int(suggested_label)]
        prov["suggested_source"] = suggested_source
    if via:
        prov["via"] = str(via)[:40]
    return prov


def provenance(row: dict) -> Optional[dict]:
    """Who decided this row's label. None when it has no label."""
    if row.get("y") is None:
        return None
    prov = row.get("prov")
    if isinstance(prov, dict) and prov.get("origin") in ("human", "machine"):
        return prov
    return {"origin": "machine", "source": LEGACY_SOURCE, "reviewed": False}


def is_reviewed(row: dict) -> bool:
    prov = provenance(row)
    return bool(prov and prov.get("reviewed") is True)


def holdout_split(X, y, seed: int = 0, val_frac: float = 0.2):
    """Deterministic train/validation split.

    Shared by the FL client and the API's validation gate, so a candidate
    is always scored on samples it did not train on.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    cut = int(len(X) * (1.0 - val_frac))
    return X[idx[:cut]], y[idx[:cut]], X[idx[cut:]], y[idx[cut:]]


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
               meta: Optional[dict] = None,
               label_source: str = IMPORT_SOURCE) -> str:
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
        if label is not None:
            row["prov"] = machine_prov(label_source)
        with self._lock:
            self._rows.append(row)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            if len(self._rows) > self.max_samples:
                self._compact()
        return sid

    def append_many(self, items) -> int:
        """Append many (x, label, meta) samples with a single file write.

        Same validation and row format as append(); for bulk imports such as
        the MEVA converter, where one open() per row is the bottleneck on
        Windows. Rows are validated first, so a bad vector writes nothing.
        """
        rows = []
        base_n = self._next_n()
        for i, (x, label, meta) in enumerate(items):
            x = np.asarray(x, dtype=np.float32).reshape(-1)
            if x.size != N_FEATURES:
                raise ValueError(f"expected {N_FEATURES} features, got {x.size}")
            if not np.isfinite(x).all():
                raise ValueError("feature vector contains NaN or inf")
            row = {
                "id": f"{int(time.time() * 1000):x}-{len(self._rows) + i:05d}",
                "n": base_n + i,
                "t": time.time(),
                "x": [round(float(v), 6) for v in x],
                "y": int(label) if label is not None else None,
                "meta": meta or {},
            }
            if label is not None:
                row["prov"] = machine_prov(IMPORT_SOURCE)
            rows.append(row)
        with self._lock:
            self._rows.extend(rows)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write("".join(json.dumps(r) + "\n" for r in rows))
            if len(self._rows) > self.max_samples:
                self._compact()
        return len(rows)

    def _apply_label(self, r: dict, label: int, prov: Optional[dict]) -> None:
        if prov is None:
            if is_reviewed(r):
                raise ReviewedLabelConflict(
                    f"sample {r.get('n')} was reviewed by a person; an "
                    "unattributed write cannot replace that")
            prov = machine_prov(UNATTRIBUTED_SOURCE)
        r["y"] = int(label)
        r["prov"] = prov

    def set_label_by_n(self, n: int, label: int, note: Optional[str] = None,
                       prov: Optional[dict] = None) -> bool:
        """Write a label. With no `prov` it is recorded as an unattributed
        machine label and may not replace a reviewed one."""
        with self._lock:
            r = self.by_n(n)
            if r is None:
                return False
            self._apply_label(r, label, prov)
            if note:
                # Keep the operator's own words: the label is lossy, the
                # text says what they actually saw.
                r.setdefault("meta", {})["operator_label"] = note[:120]
            self._rewrite()
            return True

    def set_label(self, sample_id: str, label: int,
                  prov: Optional[dict] = None) -> bool:
        if not 0 <= int(label) < N_CLASSES:
            raise ValueError(f"label must be 0..{N_CLASSES - 1}")
        with self._lock:
            for r in self._rows:
                if r["id"] == sample_id:
                    self._apply_label(r, label, prov)
                    self._rewrite()
                    return True
        return False

    def labelled(self, human_only: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Features and labels for training.

        human_only keeps rows a person reviewed. A machine suggestion is
        never in here, whatever this flag says: suggestions are not labels.
        """
        rows = [r for r in self._rows if r.get("y") is not None
                and (not human_only or is_reviewed(r))]
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
                "provenance": provenance(r),
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
            "provenance": self.provenance_counts(),
            "path": str(self.path),
        }

    def provenance_counts(self) -> dict:
        """How much of the labelled data a person actually looked at."""
        out = {"reviewed": 0, "reviewed_blind": 0, "reviewed_corrected": 0,
               "reviewed_accepted": 0, "reviewed_anomalies": 0,
               "machine_unreviewed": 0}
        kinds = {"blind": "reviewed_blind", "correct": "reviewed_corrected",
                 "accept": "reviewed_accepted"}
        for r in self._rows:
            prov = provenance(r)
            if prov is None:
                continue
            if prov.get("reviewed") is not True:
                out["machine_unreviewed"] += 1
                continue
            out["reviewed"] += 1
            if prov.get("review") in kinds:
                out[kinds[prov["review"]]] += 1
            if r["y"] == ANOMALY:
                out["reviewed_anomalies"] += 1
        return out

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

        if site_key not in workload.SITES:
            raise KeyError(
                f"unknown site {site_key!r}; the simulator defines "
                f"{', '.join(sorted(workload.SITES))}")

        X, y = workload.generate(workload.SITES[site_key], n, seed=seed)
        for xi, yi in zip(X, y):
            self.append(xi, int(yi), meta={"synthetic": True, "site": site_key},
                        label_source="sim:bootstrap")
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
