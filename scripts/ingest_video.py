#!/usr/bin/env python3
"""
Turn a saved video into labelled training data.  Runs on the PI or LAPTOP.

Replays a recording through the real detection pipeline, so every event it
produces is the same shape as one the live camera would have made: the same
features, the same still, the same crop. Then labels them all at once,
because a clip is usually one kind of thing -- "this is the postman", "this
is someone who should not be here".

    python scripts/ingest_video.py media/events/primary_human_2026.mp4 --label routine_person
    python scripts/ingest_video.py doorbell.mp4 --label anomaly --every-frame
    python scripts/ingest_video.py clip.mp4            # capture, label later

Labels: routine_person, routine_vehicle, routine_animal, routine_package,
anomaly.

Timestamps come from the video's own frame rate, not the wall clock. The
features include speed and how long something lingered, and replaying a
two-minute clip in fifteen seconds would otherwise record everything as
moving four times too fast.

Stop the node first if it is running -- both write to the same store.
"""

import argparse
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--label", default="",
                    help="applied to every event from this clip")
    ap.add_argument("--data-dir", default="",
                    help="where to store samples; defaults to the node's own")
    ap.add_argument("--every-frame", action="store_true",
                    help="classify every frame, not only moving ones")
    ap.add_argument("--max-frames", type=int, default=0, help="0 = whole video")
    ap.add_argument("--zone", default="", help="recorded on each event")
    args = ap.parse_args()

    video = Path(args.video)
    if not video.exists():
        print(f"no such video: {video}")
        return 2
    if args.data_dir:
        os.environ["ARGUS_DATA_DIR"] = args.data_dir

    import cv2

    from src.config import settings
    from src.detection import DetectionConfig, DetectionService
    from src.federated.collector import EventCollector
    from src.federated.features import LABEL_NAMES
    from src.federated.store import SampleStore

    label_index = None
    if args.label:
        if args.label not in LABEL_NAMES:
            print(f"unknown label {args.label!r}. Use one of: {', '.join(LABEL_NAMES)}")
            return 2
        label_index = LABEL_NAMES.index(args.label)

    service = DetectionService(DetectionConfig(
        cpu_export="auto", motion_gate=not args.every_frame))
    service.initialize()

    store = SampleStore(Path(settings.FL_TRAINING_DIR) / "samples.jsonl")
    before = max((r.get("n", 0) for r in store._rows), default=0)

    collector = EventCollector(
        store, db=None,
        snapshot_dir=Path(settings.DATA_DIR) / "snapshots",
        source=video.name,
    )

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"could not open {video}")
        return 2
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    step = 1.0 / fps
    clock = time.time()

    print(f"backend: {service.backend}   model: {service.config.model_name}")
    print(f"video:   {video.name} at {fps:.0f} fps"
          f"{' (classifying every frame)' if args.every_frame else ''}")

    frames = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames += 1
        detections = service.process_frame(frame)
        # The clip's own clock, so dwell and speed come out right.
        collector.observe(detections, frame.shape[:2], when=clock, frame=frame)
        clock += step
        if args.max_frames and frames >= args.max_frames:
            break
        if frames % 200 == 0:
            print(f"  {frames} frames, {collector.emitted} events")
    cap.release()
    collector.flush()

    new_rows = [r for r in store._rows if (r.get("n") or 0) > before]
    for row in new_rows:
        meta = row.setdefault("meta", {})
        meta["ingested_from"] = video.name
        if args.zone:
            meta["zone"] = args.zone
        if label_index is not None:
            store.set_label_by_n(row["n"], label_index, note=f"clip:{video.name}")

    print()
    print(f"frames processed : {frames}")
    print(f"events captured  : {len(new_rows)}")
    print(f"stills saved     : {collector.snapshots_saved} "
          f"(+{collector.crops_saved} crops)")
    if label_index is not None:
        print(f"labelled as      : {args.label}")
    else:
        print("labelled as      : nothing yet -- label them in the dashboard")
    print(f"store            : {store.stats()['labelled']} labelled of "
          f"{store.stats()['total']} total")
    if len(new_rows) == 0 and frames:
        print("\nNo events. Either nothing it recognises is in the clip, or the"
              "\nsubject never moves -- try --every-frame.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
