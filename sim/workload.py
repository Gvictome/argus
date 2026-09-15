"""
Synthetic per-site event generator for the ARGUS federated head.

This replaces the CIFAR-10 stand-in. The head no longer classifies pixels;
it classifies a *detection in context* -- what YOLO found, how big it was,
where it was, how long it lingered, and what time it happened.

Why this shape
--------------
YOLO already emits person / vehicle / animal / package. A head that
reproduces that adds nothing. The job left over is the one YOLO cannot do:
deciding whether a given detection is routine *for this site*. A delivery
van at 14:00 is routine; the same van at 03:00 is not. That judgement is
site-specific, which is exactly what makes it worth federating.

Non-IID on purpose
------------------
flower_training_report_2026-04-24.md:36 partitioned CIFAR-10 with
`IidPartitioner` -- "each node gets a random, balanced sample". That is the
easiest possible split and it validates plumbing, not learning. Here each
site gets both a different *label* distribution and a different
*conditional* distribution (the feature -> anomaly rule itself differs).
That is the hard case, and the one a real deployment has.

Feature vector (19 dims)
------------------------
 0-3   class one-hot: person, vehicle, animal, package
 4-9   geometry: w, h, sqrt(area), aspect, cx, cy
10-13  dynamics: vx, vy, speed, dwell_frames (normalised)
14-17  temporal: hour sin/cos, weekday sin/cos
18     detector confidence

Labels (5) -- the class list from flower_training_report_2026-04-24.md:18
  0 routine_person  1 routine_vehicle  2 routine_animal
  3 routine_package 4 anomaly
"""

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

N_FEATURES = 19
N_CLASSES = 5
CLASS_NAMES = ("person", "vehicle", "animal", "package")
LABEL_NAMES = (
    "routine_person",
    "routine_vehicle",
    "routine_animal",
    "routine_package",
    "anomaly",
)


@dataclass(frozen=True)
class Site:
    """One physical camera location with its own idea of 'normal'."""

    key: str
    label: str
    class_prior: Tuple[float, float, float, float]
    active_hours: Tuple[int, int]      # inclusive start, exclusive end
    anomalous_classes: Tuple[str, ...]  # classes never routine here
    dwell_limit: float                  # frames; longer than this is loitering
    note: str


SITES: Dict[str, Site] = {
    "driveway": Site(
        key="driveway",
        label="Residential driveway, street-facing",
        class_prior=(0.40, 0.42, 0.10, 0.08),
        active_hours=(6, 22),
        anomalous_classes=(),
        dwell_limit=0.72,
        note="Vehicles and residents all day. Night presence is the signal.",
    ),
    "backdoor": Site(
        key="backdoor",
        label="Rear service door, pedestrian only",
        class_prior=(0.46, 0.04, 0.14, 0.36),
        active_hours=(8, 19),
        anomalous_classes=("vehicle",),
        dwell_limit=0.80,
        note="Deliveries and staff. A vehicle here is always wrong.",
    ),
    "yard": Site(
        key="yard",
        label="Side yard, fenced",
        class_prior=(0.22, 0.03, 0.56, 0.19),
        active_hours=(7, 20),
        anomalous_classes=(),
        dwell_limit=0.85,
        note="Mostly animals. Tolerant of loitering, intolerant of people at night.",
    ),
}


def _hour_weights(active: Tuple[int, int], night_rate: float) -> np.ndarray:
    """Event likelihood per hour: busy while active, sparse overnight."""
    w = np.full(24, night_rate, dtype=np.float64)
    start, end = active
    w[start:end] = 1.0
    return w / w.sum()


def generate(
    site: Site,
    n: int,
    seed: int = 0,
    night_rate: float = 0.12,
    label_noise: float = 0.02,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (X[n, 19] float32, y[n] int64) for one site."""
    rng = np.random.default_rng(seed)

    cls = rng.choice(4, size=n, p=np.asarray(site.class_prior))
    hour = rng.choice(24, size=n, p=_hour_weights(site.active_hours, night_rate))
    weekday = rng.integers(0, 7, size=n)

    # Geometry correlates with class: vehicles wide, people tall, animals small.
    base_w = np.array([0.12, 0.34, 0.14, 0.11])[cls]
    base_h = np.array([0.30, 0.20, 0.10, 0.09])[cls]
    w = np.clip(base_w * rng.lognormal(0.0, 0.28, n), 0.01, 1.0)
    h = np.clip(base_h * rng.lognormal(0.0, 0.28, n), 0.01, 1.0)
    cx = np.clip(rng.normal(0.5, 0.22, n), 0.0, 1.0)
    cy = np.clip(rng.normal(0.58, 0.16, n), 0.0, 1.0)
    aspect = w / np.maximum(h, 1e-3)

    # Dynamics: vehicles move fast and linger briefly, animals dart, packages
    # are static once dropped.
    speed_scale = np.array([0.18, 0.62, 0.34, 0.02])[cls]
    vx = rng.normal(0.0, 1.0, n) * speed_scale
    vy = rng.normal(0.0, 0.45, n) * speed_scale
    speed = np.sqrt(vx**2 + vy**2)
    dwell = np.clip(
        rng.beta(1.6, 4.0, n) + np.array([0.10, 0.0, 0.05, 0.55])[cls],
        0.0, 1.0,
    )

    conf = np.clip(rng.beta(7.0, 2.2, n), 0.0, 1.0)

    # ---- site-specific normalcy rule -------------------------------------
    start, end = site.active_hours
    off_hours = (hour < start) | (hour >= end)
    forbidden = np.zeros(n, dtype=bool)
    for name in site.anomalous_classes:
        forbidden |= cls == CLASS_NAMES.index(name)
    # Loitering applies to things that are supposed to move on. A package
    # sitting where it was dropped is the normal case, not an anomaly.
    loitering = (dwell > site.dwell_limit) & (cls != 3)

    is_anomaly = forbidden | (off_hours & (cls != 2)) | loitering

    y = np.where(is_anomaly, 4, cls).astype(np.int64)

    # A little label noise so nothing hits a suspicious 1.000.
    flip = rng.random(n) < label_noise
    y[flip] = rng.integers(0, N_CLASSES, size=int(flip.sum()))

    onehot = np.zeros((n, 4), dtype=np.float64)
    onehot[np.arange(n), cls] = 1.0
    ang_h = 2 * np.pi * hour / 24.0
    ang_d = 2 * np.pi * weekday / 7.0

    X = np.column_stack([
        onehot,
        w, h, np.sqrt(w * h), np.clip(aspect, 0, 6) / 6.0, cx, cy,
        np.tanh(vx), np.tanh(vy), np.tanh(speed), dwell,
        np.sin(ang_h), np.cos(ang_h), np.sin(ang_d), np.cos(ang_d),
        conf,
    ]).astype(np.float32)

    return X, y


def split(X: np.ndarray, y: np.ndarray, val_frac: float = 0.2, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    cut = int(len(X) * (1.0 - val_frac))
    tr, va = idx[:cut], idx[cut:]
    return X[tr], y[tr], X[va], y[va]


def describe(site: Site, y: np.ndarray) -> str:
    counts = np.bincount(y, minlength=N_CLASSES)
    parts = " ".join(
        f"{LABEL_NAMES[i].replace('routine_', '')[:4]}={counts[i]}"
        for i in range(N_CLASSES)
    )
    return f"{site.key:9} n={len(y):5}  {parts}  anomaly={counts[4] / len(y):.1%}"
