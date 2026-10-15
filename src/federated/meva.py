"""
MEVA tracks to ARGUS head events.

MEVA ships ground-truth actor and vehicle tracks (KPF), so detection can be
skipped for the federated benchmark. This module turns one camera's tracks
into the same 19-feature events the live node stores, via the node's own
``features.Track`` / ``features.extract``. Nothing about the features is
re-implemented here; if ``features.py`` changes, this follows.

KPF input (per five-minute clip, three YAML-ish files, one dict per line)
    <clip>.geom.yml        {'geom': {'g0': 'x1 y1 x2 y2', 'id1': track, 'ts0': frame}}
    <clip>.types.yml       {'types': {'cset3': {'person': 1.0}, 'id1': track}}
    <clip>.activities.yml  {'act': {'act2': {'name': 1.0}, 'actors': [{'id1': track}]}}
Two spellings exist in the download (Python-quoted and unquoted flow style);
the line parsers accept both. Files are read line by line with regexes
rather than yaml.load, because a geom file can be 100 MB.

How a track becomes an event (mirrors src/federated/collector.py)
    - boxes are normalised by the frame size (1920x1080; boxes that overhang
      by a few pixels are clipped to the frame);
    - Track.update() is called once per detection, so prev_cx / dwell mean
      what they mean on the node;
    - tracks with fewer than 3 updates are dropped (collector min_frames);
    - one event per track, features extracted at the track's last-seen time
      (the collector emits at close), confidence 1.0 (ground truth);
    - ``det_fps`` (default 10) thins the 30 fps annotation to a detector-like
      rate before update(), so the per-update velocity is comparable with the
      node, which does not run YOLO at 30 fps on a Pi. Set 30 for every frame.
    - Timestamp = clip start (from the clip name) + frame / 30 fps.

Class mapping (MEVA cset3 -> ARGUS cls index)
    person -> 0 (person)     vehicle -> 1 (vehicle)     bag -> 3 (package)
    DROPPED: bicycle, other, receptacle and anything unknown: no ARGUS class
    fits, and feeding them to the head as zeros is wrong. MEVA has no animals.
    Drops are counted and printed.

Labels (fixed in advance; no tuning against model results)
    The head's label space is routine_person / routine_vehicle /
    routine_animal / routine_package (0-3) or anomaly (4). MEVA has no
    "routine" label, so anomaly is defined per camera from that camera's own
    statistics. An event is ANOMALY (label 4) if ANY of:
      (a) rare_class:    its mapped class is < 2% of that camera's events;
      (a) rare_activity: it participates in an activity whose share of all
                         (event, activity) participations at that camera is
                         < 1%;
      (b) off_hours:     its clock hour is inactive for that camera, where an
                         hour is active if it holds >= 2% as many events as
                         that camera's busiest hour.
    Otherwise the label is its class index (0 person, 1 vehicle, 3 package).
    Statistics are computed over the whole camera (train and holdout clips
    together), because they define what "rare at this camera" means.
    The 2% / 1% / 2% thresholds were set by reading the class, activity and
    hour frequency tables, never model results. The hour threshold started at
    10%; the first G336 run showed that flagged hours 12-13, which are only
    thinly covered by the 95 sampled clips (a sampling artifact, not a quiet
    camera), so it was lowered to 2% before any model was trained. MEVA
    records only daytime, so rule (b) is nearly vacuous on G336; it matters
    for cameras that record around the clock.

Holdout: whole clips, chosen by a seeded permutation of the sorted clip
names. Neighbouring events in one clip are near-duplicates, so a random
event-level split would leak.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from src.federated.features import ANOMALY, Track, extract
from src.federated.store import SampleStore

logger = logging.getLogger(__name__)

FPS = 30.0
FRAME_SIZE = (1920, 1080)
DEFAULT_DET_FPS = 10.0
MIN_FRAMES = 3  # collector.EventCollector default

CLASS_RARE_SHARE = 0.02
ACTIVITY_RARE_SHARE = 0.01
ACTIVE_HOUR_MIN_FRACTION_OF_PEAK = 0.02

# MEVA cset3 name -> ARGUS cls index. Anything absent is dropped.
CLASS_MAP: Dict[str, int] = {"person": 0, "vehicle": 1, "bag": 3}

_NUM = r"-?\d+(?:\.\d+)?"
_GEOM_HEAD_RE = re.compile(r"^\s*-\s*\{\s*'?geom'?\s*:")
_TS0_RE = re.compile(r"ts0'?\s*:\s*(\d+)")
_G0_RE = re.compile(rf"g0'?\s*:\s*'?({_NUM})\s+({_NUM})\s+({_NUM})\s+({_NUM})")
_TYPES_RE = re.compile(
    r"^\s*-\s*\{\s*'?types'?\s*:.*?cset3'?\s*:\s*\{\s*'?(\w+)'?\s*:")
_ID1_RE = re.compile(r"id1'?\s*:\s*(\d+)")
_ACT_RE = re.compile(r"^\s*-\s*\{\s*'?act'?\s*:\s*\{\s*'?act2'?\s*:\s*\{\s*'?(\w+)'?\s*:")
_CLIP_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2})\.(\d{2}-\d{2}-\d{2})\.(\d{2}-\d{2}-\d{2})\.([^.]+)\.(G\d+)")


# ---------------------------------------------------------------------------
# Line parsers
# ---------------------------------------------------------------------------
def parse_geom_line(line: str) -> Optional[Tuple[int, int, Tuple[float, float, float, float]]]:
    """(track_id, frame, (x1, y1, x2, y2)) from one geom line, else None."""
    if not _GEOM_HEAD_RE.match(line):
        return None
    tid, ts, g0 = _ID1_RE.search(line), _TS0_RE.search(line), _G0_RE.search(line)
    if not (tid and ts and g0):
        return None
    return int(tid.group(1)), int(ts.group(1)), tuple(float(v) for v in g0.groups())


def parse_types_line(line: str) -> Optional[Tuple[int, str]]:
    """(track_id, class_name) from one types line, else None."""
    m = _TYPES_RE.match(line)
    i = _ID1_RE.search(line)
    if not m or not i:
        return None
    return int(i.group(1)), m.group(1)


def parse_activity_line(line: str) -> Optional[Tuple[str, Tuple[int, ...]]]:
    """(activity_name, actor_track_ids) from one activities line, else None."""
    m = _ACT_RE.match(line)
    if not m:
        return None
    _, _, tail = line.partition("actors")
    return m.group(1), tuple(int(v) for v in _ID1_RE.findall(tail))


def map_class(name: str) -> Optional[int]:
    """ARGUS class index for a MEVA class name, or None if it has no home."""
    return CLASS_MAP.get(name)


# ---------------------------------------------------------------------------
# Clip names
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ClipInfo:
    name: str
    start: datetime
    site: str
    camera: str


def parse_clip_name(name: str) -> ClipInfo:
    """Parse ``2018-03-05.13-10-00.13-15-00.school.G336[.geom.yml]``."""
    m = _CLIP_RE.match(Path(name).name)
    if not m:
        raise ValueError(f"not a MEVA clip name: {name!r}")
    date, t0, t1, site, cam = m.groups()
    start = datetime.strptime(f"{date} {t0}", "%Y-%m-%d %H-%M-%S")
    return ClipInfo(f"{date}.{t0}.{t1}.{site}.{cam}", start, site, cam)


def discover_clips(annotations: Path, camera: str) -> List[Path]:
    """geom.yml paths for one camera, sorted by clip name, one per clip name.

    The same clip can appear under more than one annotation team directory;
    the first by sorted path wins. A clip without a types file is skipped.
    """
    found: Dict[str, Path] = {}
    for p in sorted(Path(annotations).rglob("*.geom.yml")):
        try:
            info = parse_clip_name(p.name)
        except ValueError:
            continue
        if info.camera != camera or info.name in found:
            continue
        if not p.with_name(p.name.replace(".geom.yml", ".types.yml")).exists():
            logger.warning("meva: %s has no types file, skipped", info.name)
            continue
        found[info.name] = p
    return [found[k] for k in sorted(found)]


# ---------------------------------------------------------------------------
# Tracks to events
# ---------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class Event:
    clip: str
    camera: str
    track_id: int
    cls: int
    when: datetime
    x: np.ndarray
    activities: Tuple[str, ...]
    frames: int


@dataclass
class ConvertStats:
    tracks_seen: int = 0
    untyped: int = 0
    dropped_short: int = 0
    dropped_class: Dict[str, int] = field(default_factory=dict)


def _read_geom(path: Path) -> Dict[int, Dict[int, Tuple[float, float, float, float]]]:
    tracks: Dict[int, Dict[int, Tuple[float, float, float, float]]] = defaultdict(dict)
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            rec = parse_geom_line(line)
            if rec:
                tid, frame, box = rec
                tracks[tid][frame] = box
    return tracks


def _read_types(path: Path) -> Dict[int, str]:
    out: Dict[int, str] = {}
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            rec = parse_types_line(line)
            if rec:
                out[rec[0]] = rec[1]
    return out


def _read_activities(path: Path) -> Dict[int, List[str]]:
    out: Dict[int, List[str]] = defaultdict(list)
    if not path.exists():
        return out
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            rec = parse_activity_line(line)
            if rec:
                for tid in rec[1]:
                    if rec[0] not in out[tid]:
                        out[tid].append(rec[0])
    return out


def _thin(frames: Sequence[int], stride: int) -> List[int]:
    """Keep a frame once at least ``stride`` frames passed; always keep the last."""
    kept = [frames[0]]
    for fr in frames[1:]:
        if fr - kept[-1] >= stride:
            kept.append(fr)
    if kept[-1] != frames[-1]:
        kept.append(frames[-1])
    return kept


def _norm_box(box, frame_size) -> Tuple[float, float, float, float]:
    w, h = frame_size
    x1, y1, x2, y2 = box
    clip = lambda v: min(1.0, max(0.0, v))  # noqa: E731
    return clip(x1 / w), clip(y1 / h), clip(x2 / w), clip(y2 / h)


def clip_to_events(geom_path: Path, det_fps: float = DEFAULT_DET_FPS,
                   frame_size: Tuple[int, int] = FRAME_SIZE, fps: float = FPS,
                   min_frames: int = MIN_FRAMES,
                   stats: Optional[ConvertStats] = None) -> List[Event]:
    """One event per usable track in a clip, built like the node's collector."""
    geom_path = Path(geom_path)
    info = parse_clip_name(geom_path.name)
    base = geom_path.name[: -len(".geom.yml")]
    stats = stats if stats is not None else ConvertStats()
    tracks = _read_geom(geom_path)
    types = _read_types(geom_path.with_name(base + ".types.yml"))
    acts = _read_activities(geom_path.with_name(base + ".activities.yml"))
    stride = max(1, int(round(fps / det_fps)))

    events: List[Event] = []
    for tid in sorted(tracks):
        stats.tracks_seen += 1
        cname = types.get(tid)
        if cname is None:
            stats.untyped += 1
            continue
        cls = map_class(cname)
        if cls is None:
            stats.dropped_class[cname] = stats.dropped_class.get(cname, 0) + 1
            continue
        boxes = tracks[tid]
        frames = _thin(sorted(boxes), stride)
        t0 = frames[0] / fps
        track = Track(track_id=tid, cls=cls, first_seen=t0, last_seen=t0)
        for fr in frames:
            track.update(_norm_box(boxes[fr], frame_size), 1.0, fr / fps)
        if track.frames < min_frames:
            stats.dropped_short += 1
            continue
        when = info.start + timedelta(seconds=track.last_seen)
        events.append(Event(
            clip=info.name, camera=info.camera, track_id=tid, cls=cls, when=when,
            x=extract(track, when), activities=tuple(acts.get(tid, ())),
            frames=track.frames))
    return events


# ---------------------------------------------------------------------------
# Anomaly rule
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Profile:
    """What 'normal' means at one camera, derived from its own events."""

    class_share: Dict[int, float]
    activity_share: Dict[str, float]
    active_hours: FrozenSet[int]


def build_profile(events: Iterable[Event]) -> Profile:
    events = list(events)
    n = len(events)
    cls_counts = Counter(e.cls for e in events)
    act_counts = Counter(a for e in events for a in e.activities)
    n_act = sum(act_counts.values())
    hour_counts = Counter(e.when.hour for e in events)
    peak = max(hour_counts.values(), default=0)
    active = frozenset(h for h, c in hour_counts.items()
                       if c >= ACTIVE_HOUR_MIN_FRACTION_OF_PEAK * peak)
    return Profile(
        class_share={c: v / n for c, v in cls_counts.items()} if n else {},
        activity_share={a: v / n_act for a, v in act_counts.items()} if n_act else {},
        active_hours=active,
    )


def anomaly_reasons(event: Event, profile: Profile) -> List[str]:
    """Which parts of the fixed rule fire for this event (empty = routine)."""
    reasons: List[str] = []
    if profile.class_share.get(event.cls, 0.0) < CLASS_RARE_SHARE:
        reasons.append("rare_class")
    if any(profile.activity_share.get(a, 0.0) < ACTIVITY_RARE_SHARE
           for a in event.activities):
        reasons.append("rare_activity")
    if event.when.hour not in profile.active_hours:
        reasons.append("off_hours")
    return reasons


def label_for(event: Event, profile: Profile) -> int:
    """Head label: ANOMALY (4) if the rule fires, else the class index."""
    return ANOMALY if anomaly_reasons(event, profile) else event.cls


# ---------------------------------------------------------------------------
# Holdout
# ---------------------------------------------------------------------------
def split_clips(clips: Sequence[str], holdout_frac: float = 0.2,
                seed: int = 0) -> Tuple[List[str], List[str]]:
    """Whole-clip split. Returns (train, holdout), both sorted.

    The input order does not matter: names are sorted before the seeded
    permutation, so the same clip set and seed always give the same split.
    """
    names = sorted(set(clips))
    if len(names) < 2:
        return names, []
    n_hold = min(len(names) - 1, max(1, int(round(len(names) * holdout_frac))))
    perm = np.random.default_rng(seed).permutation(len(names))
    hold = sorted(names[i] for i in perm[:n_hold])
    held = set(hold)
    return [n for n in names if n not in held], hold


# ---------------------------------------------------------------------------
# One camera, end to end
# ---------------------------------------------------------------------------
@dataclass
class ConvertResult:
    camera: str
    clips: int
    tracks: int
    events: int
    class_mix: Dict[str, int]
    anomaly_count: int
    anomaly_rate: float
    reason_counts: Dict[str, int]
    train_clips: int
    holdout_clips: int
    train_events: int
    holdout_events: int
    active_hours: List[int]
    stats: ConvertStats
    out_dir: str


_CLS_NAMES = {0: "person", 1: "vehicle", 2: "animal", 3: "package"}


def convert_camera(annotations: Path, camera: str, out_dir: Path,
                   holdout_frac: float = 0.2, seed: int = 0,
                   limit_clips: Optional[int] = None,
                   det_fps: float = DEFAULT_DET_FPS) -> ConvertResult:
    """Convert every clip of ``camera`` into one SampleStore under ``out_dir``.

    Writes ``samples.jsonl`` (replacing any previous run) and ``holdout.json``.
    Each row carries meta.split ("train" or "holdout") and meta.clip.
    """
    paths = discover_clips(Path(annotations), camera)
    if limit_clips:
        paths = paths[:limit_clips]
    if not paths:
        raise FileNotFoundError(f"no clips for camera {camera} under {annotations}")

    stats = ConvertStats()
    events: List[Event] = []
    for p in paths:
        events.extend(clip_to_events(p, det_fps=det_fps, stats=stats))
    events.sort(key=lambda e: (e.when, e.track_id))

    names = [parse_clip_name(p.name).name for p in paths]
    train, hold = split_clips(names, holdout_frac, seed)
    hold_set = set(hold)
    profile = build_profile(events)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples = out_dir / "samples.jsonl"
    samples.unlink(missing_ok=True)
    store = SampleStore(samples, max_samples=max(200_000, len(events) + 1))

    mix: Counter = Counter()
    reasons: Counter = Counter()
    batch = []
    n_anom = n_hold_ev = 0
    for e in events:
        why = anomaly_reasons(e, profile)
        label = ANOMALY if why else e.cls
        split = "holdout" if e.clip in hold_set else "train"
        n_hold_ev += split == "holdout"
        n_anom += bool(why)
        mix[_CLS_NAMES[e.cls]] += 1
        reasons.update(why)
        batch.append((e.x, label, {
            "source": "meva", "synthetic": False, "camera": e.camera,
            "clip": e.clip, "split": split, "track_id": e.track_id,
            "frames": e.frames, "timestamp": e.when.isoformat(),
            "activities": list(e.activities), "anomaly_reasons": why,
        }))
    store.append_many(batch)

    (out_dir / "holdout.json").write_text(json.dumps({
        "camera": camera, "seed": seed, "holdout_frac": holdout_frac,
        "det_fps": det_fps, "train_clips": train, "holdout_clips": hold,
        "active_hours": sorted(profile.active_hours),
        "thresholds": {"class_rare_share": CLASS_RARE_SHARE,
                       "activity_rare_share": ACTIVITY_RARE_SHARE,
                       "active_hour_min_fraction_of_peak": ACTIVE_HOUR_MIN_FRACTION_OF_PEAK},
    }, indent=2), encoding="utf-8")

    n = len(events)
    return ConvertResult(
        camera=camera, clips=len(paths), tracks=stats.tracks_seen, events=n,
        class_mix=dict(mix), anomaly_count=n_anom,
        anomaly_rate=(n_anom / n) if n else 0.0, reason_counts=dict(reasons),
        train_clips=len(train), holdout_clips=len(hold),
        train_events=n - n_hold_ev, holdout_events=n_hold_ev,
        active_hours=sorted(profile.active_hours), stats=stats,
        out_dir=str(out_dir),
    )
