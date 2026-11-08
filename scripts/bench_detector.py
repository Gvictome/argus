"""Object-detector FPS on this node: Hailo-10H vs CPU.   >>> RUN ON THE PI <<<

    python scripts/bench_detector.py --backend hailo --video media/vtest.avi
    python scripts/bench_detector.py --backend cpu   --video media/vtest.avi

Two loads, both through DetectionService.detect_objects (what the node runs):
  idle      a static black frame (nothing to detect)
  tracking  frames from --video (people walking), cycled
Prints mean latency, FPS, detections per frame, and saves one annotated frame
so the boxes can be checked by eye.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.detection import DetectionConfig, DetectionService  # noqa: E402

WARMUP = 10


def run(svc, frames, n):
    for i in range(WARMUP):
        svc.detect_objects(frames[i % len(frames)])
    found, t0 = 0, time.perf_counter()
    for i in range(n):
        found += len(svc.detect_objects(frames[i % len(frames)]))
    dt = time.perf_counter() - t0
    return dt / n * 1000, n / dt, found / n


def load_video(path, limit=300):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while len(frames) < limit:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    if not frames:
        sys.exit(f"could not read frames from {path}")
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["hailo", "cpu"], required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--out", default="bench_annotated.jpg")
    a = ap.parse_args()

    svc = DetectionService(DetectionConfig(
        hailo=a.backend == "hailo",
        cpu_export=os.getenv("OBJECT_BACKEND", "auto"),
        motion_gate=False,
    ))
    svc.initialize()
    print(f"backend={svc.backend} model={type(svc.object_model).__name__} "
          f"imgsz={svc.config.object_imgsz}")
    if a.backend == "hailo" and svc.backend != "hailo":
        sys.exit("Hailo requested but not loaded; see the warning above.")

    video = load_video(a.video)
    h, w = video[0].shape[:2]
    idle = [np.zeros((h, w, 3), np.uint8)]
    for name, frames in (("idle", idle), ("tracking", video)):
        ms, fps, per = run(svc, frames, a.frames)
        print(f"{name:9s} {ms:7.1f} ms/frame  {fps:6.1f} FPS  {per:4.1f} det/frame  ({w}x{h})")

    frame = video[len(video) // 2].copy()
    for d in svc.detect_objects(frame):
        x, y, bw, bh = d.bbox
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        cv2.putText(frame, f"{d.label} {d.confidence:.2f}", (x, max(0, y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imwrite(a.out, frame)
    print(f"annotated frame -> {a.out}")


if __name__ == "__main__":
    main()
