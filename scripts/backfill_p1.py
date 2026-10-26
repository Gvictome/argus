#!/usr/bin/env python3
"""
Give the events already on a node their priority, and keep the clips of
the priority-1 ones.

The live clip keeper (src/detection/clip_keeper.py) only sees clips
recorded after it was switched on. This applies the same rule to what is
already on the card:

  * every past event the head calls unusual, or that is labelled anomaly,
    is frozen as P1 in media/events/index.json, the same P1 as
    scripts/prep_round.py, so a later model round cannot demote it;
  * every clip already in media/events is kept (re-encoded to H.264 so the
    dashboard can play it) if a P1 event overlaps it, and deleted if not;
  * the kept clips are held under RECORD_MAX_GB, oldest first.

Dry run by default: prints what it would keep and delete, touches nothing.
Stop the node first, because the running service writes the same index:

    PI  sudo systemctl stop argus-node
    PI  python scripts/backfill_p1.py
    PI  python scripts/backfill_p1.py --apply
    PI  sudo systemctl start argus-node
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import settings  # noqa: E402
from src.detection.clip_keeper import P1, ClipKeeper, priority_of  # noqa: E402
from src.federated.head import FederatedHead  # noqa: E402
from src.federated.store import SampleStore  # noqa: E402

STAMP = re.compile(r"_(\d{8}_\d{6})(?:_h264)?\.mp4$")


def clip_span(path: Path) -> tuple | None:
    """(start, end) epoch seconds from the filename stamp and the frame count."""
    m = STAMP.search(path.name)
    if not m:
        return None
    start = datetime.strptime(m.group(1), "%Y%m%d_%H%M%S").timestamp()
    try:
        import cv2
        cap = cv2.VideoCapture(str(path))
        frames, fps = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS) or 15.0
        cap.release()
        length = frames / fps if frames > 0 else settings.RECORD_MAX_CLIP_S
    except Exception:
        length = settings.RECORD_MAX_CLIP_S
    return start, start + length


def node_running() -> bool:
    try:
        out = subprocess.run(["systemctl", "is-active", "argus-node"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() == "active"
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--apply", action="store_true", help="write the index and delete/re-encode clips")
    args = ap.parse_args()

    if args.apply and node_running():
        print("argus-node is running; stop it first: sudo systemctl stop argus-node")
        return 1

    store = SampleStore(Path(settings.FL_TRAINING_DIR) / "samples.jsonl")
    head = FederatedHead(seed=0)
    head_file = Path(settings.DATA_DIR) / "fl_head.npz"
    if head_file.exists():
        head.load(head_file)
    else:
        print("warning: no saved head; P1 = labelled anomalies only")
        head = None

    keeper = ClipKeeper(
        clip_dir=Path(settings.MEDIA_DIR) / "events", store=store,
        head_source=lambda: head,
        max_bytes=int(settings.RECORD_MAX_GB * 1e9),
        min_free_bytes=int(settings.RECORD_MIN_FREE_GB * 1e9),
        transcode=args.apply,
    )

    rows = [r for r in store._rows if not (r.get("meta") or {}).get("synthetic")]
    p1 = [r for r in rows if priority_of(r, head) == P1]
    print(f"events: {len(rows)}   P1: {len(p1)}   "
          f"(labelled anomaly {sum(1 for r in p1 if r.get('y') is not None)}, "
          f"head-flagged {sum(1 for r in p1 if r.get('y') is None)})")

    known = set(keeper._index["clips"])
    clips = sorted(p for p in keeper.clip_dir.glob("*.mp4") if p.name not in known)
    keep, drop, unreadable = [], [], []
    for path in clips:
        span = clip_span(path)
        if span is None:
            unreadable.append(path)
            continue
        hits = keeper.p1_events_in(*span)
        (keep if hits else drop).append((path, span, hits))

    gb = lambda items: sum(p.stat().st_size for p, *_ in items) / 1e9  # noqa: E731
    print(f"clips on card: {len(clips)}   keep {len(keep)} ({gb(keep):.2f} GB)   "
          f"delete {len(drop)} ({gb(drop):.2f} GB)   unrecognised name {len(unreadable)}")
    for path, _, hits in keep:
        print(f"  keep {path.name}  <- P1 events {[int(r['n']) for r in hits]}")

    if not args.apply:
        print("dry run; nothing changed. Re-run with --apply.")
        return 0

    for r in p1:
        keeper.freeze(int(r["n"]), P1)
    for path, (start, end), _ in keep + drop:
        keeper.claim(path, start, end)
    keeper.enforce_cap()
    keeper._save_index()
    print(f"done: {keeper.status()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
