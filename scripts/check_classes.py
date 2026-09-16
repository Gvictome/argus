#!/usr/bin/env python3
"""
What does the detector actually see?  Runs on the PI or the LAPTOP.

    python scripts/check_classes.py --camera          a frame from the running node
    python scripts/check_classes.py media/vtest.avi   a video
    python scripts/check_classes.py photo.jpg         any image

Prints the four ARGUS classes on the left and the raw COCO labels behind
them, which is what separates "the model missed it" from "the model saw it
and it maps somewhere else".

--camera pulls through the node's API rather than opening the camera: the
detection worker owns the device, and a second reader gets nothing.
"""

import argparse
import collections
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import cv2  # noqa: E402

from src.detection import DetectionConfig, DetectionService  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def frames_from(source: str, limit: int):
    path = Path(source)
    if path.suffix.lower() in IMAGE_SUFFIXES:
        img = cv2.imread(str(path))
        if img is None:
            raise SystemExit(f"could not read {path}")
        return [img]

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"could not open {path}")
    out = []
    while len(out) < limit:
        ok, frame = cap.read()
        if not ok:
            break
        out.append(frame)
    cap.release()
    if not out:
        raise SystemExit(f"no frames in {path}")
    return out


def grab_from_node(api: str) -> Path:
    dest = Path("/tmp/argus_frame.jpg") if sys.platform != "win32" else REPO / "argus_frame.jpg"
    url = f"{api}/api/camera/snapshot?annotated=0"
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            data = r.read()
    except Exception as exc:
        raise SystemExit(f"could not reach the node at {api}: {exc}")
    if len(data) < 1000:
        raise SystemExit(
            f"the node returned {len(data)} bytes, not an image. "
            "Is the camera up? Check /api/detection/status for camera_error.")
    dest.write_bytes(data)
    print(f"frame from the node: {dest} ({len(data)} bytes)")
    return dest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", nargs="?", help="image or video path")
    ap.add_argument("--camera", action="store_true", help="grab a frame from the running node")
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--frames", type=int, default=120, help="max frames for a video")
    ap.add_argument("--threshold", type=float, default=0.25)
    ap.add_argument("--save", default="", help="write an annotated copy here")
    args = ap.parse_args()

    if not args.source and not args.camera:
        ap.error("give a file, or --camera")

    source = str(grab_from_node(args.api)) if args.camera else args.source

    service = DetectionService(DetectionConfig(
        cpu_export="auto", detection_threshold=args.threshold))
    service.initialize()
    print(f"backend: {service.backend}   model: {service.config.model_name}   "
          f"imgsz: {service.config.object_imgsz}   threshold: {args.threshold}")

    frames = frames_from(source, args.frames)
    classes, labels = collections.Counter(), collections.Counter()
    last = []
    for frame in frames:
        last = service.detect_objects(frame)
        for d in last:
            classes[d.type.value] += 1
            labels[f"{d.label}({d.coco_id})"] += 1

    print(f"frames: {len(frames)}")
    print(f"ARGUS classes: {dict(classes) or 'none'}")
    print(f"COCO labels:   {dict(labels.most_common(10)) or 'none'}")
    if not classes:
        print("\nNothing detected. Things that cause this:")
        print("  - the subject is small in frame; it needs roughly a third of it")
        print("  - a cardboard box is not a COCO class; package means backpack/handbag/suitcase")
        print("  - glare or motion blur, if you are holding a screen up to the camera")

    if args.save:
        from src.detection.annotate import draw_detections
        cv2.imwrite(args.save, draw_detections(frames[-1], last))
        print(f"annotated copy: {args.save}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
