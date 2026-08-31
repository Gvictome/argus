#!/usr/bin/env python3
"""
Measure per-class detection stability across the whole cascade.

Answers the question "is every class as trustworthy as face recognition?"
with numbers rather than impressions. Face recognition sets the bar: on
clean stills it scores ~0.99 genuine against ~0.07 for a different
person, a separation of ~0.91.

For each detection class this reports:

  n            how many detections were seen
  mean/median  typical confidence
  p05          5th percentile -- the weak tail is what fails on a demo,
               not the average
  stdev        spread; a wide spread means an unpredictable box
  <thresh      share of detections within 0.10 of the decision threshold,
               i.e. one bad frame from flipping

With a labeled dataset (--dataset) it additionally reports true
precision/recall for the threat stage against ground truth.

Usage:
    python scripts/benchmark_classes.py --images path/to/dir
    python scripts/benchmark_classes.py --dataset path/to/Dataset/test
    python scripts/benchmark_classes.py --dataset ... --limit 150
"""

import argparse
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from src.config import settings  # noqa: E402
from src.detection import DetectionService, DetectionType  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# Ground-truth class ids in the upstream YOLO label files.
GT_NORMAL = 0
GT_DANGEROUS = 1


def _images(root: Path, limit: int):
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    return files[:limit] if limit else files


def _summarize(name: str, confs: list, threshold: float) -> dict:
    if not confs:
        return {"class": name, "n": 0}
    near = sum(1 for c in confs if c < threshold + 0.10) / len(confs)
    return {
        "class": name,
        "n": len(confs),
        "mean": statistics.mean(confs),
        "median": statistics.median(confs),
        "p05": np.percentile(confs, 5),
        "min": min(confs),
        "stdev": statistics.stdev(confs) if len(confs) > 1 else 0.0,
        "near": near,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Per-class detection stability.")
    ap.add_argument("--images", type=Path, help="Directory of images (no ground truth).")
    ap.add_argument("--dataset", type=Path,
                    help="YOLO split dir containing images/ and labels/.")
    ap.add_argument("--limit", type=int, default=0, help="Cap image count (0 = all).")
    ap.add_argument("--imgsz", type=int, default=None, help="Override threat imgsz.")
    args = ap.parse_args()

    if not args.images and not args.dataset:
        ap.error("pass --images or --dataset")

    root = args.dataset / "images" if args.dataset else args.images
    if not root.is_dir():
        print(f"[FAIL] not a directory: {root}")
        return 2

    label_dir = args.dataset / "labels" if args.dataset else None
    files = _images(root, args.limit)
    if not files:
        print(f"[FAIL] no images under {root}")
        return 2

    service = DetectionService()
    service.initialize()

    threat_on = False
    if settings.THREAT_ENABLED:
        try:
            from src.detection.threat import ThreatClassifier
            service.attach_threat_classifier(ThreatClassifier(
                model_path=settings.THREAT_MODEL_PATH,
                confidence=settings.THREAT_CONFIDENCE,
                imgsz=args.imgsz or settings.THREAT_IMGSZ,
            ))
            threat_on = True
        except Exception as exc:
            print(f"[WARN] threat stage unavailable: {exc}")

    print(f"images            : {len(files)}")
    print(f"object detection  : {service.object_model is not None} (backend={service.backend})")
    print(f"threat detection  : {threat_on}")
    print(f"object threshold  : {service.config.detection_threshold}")
    if threat_on:
        print(f"threat threshold  : {settings.THREAT_CONFIDENCE}"
              f"  imgsz={args.imgsz or settings.THREAT_IMGSZ}")
    print()

    confs = defaultdict(list)
    per_image_latency = []
    # Ground-truth counters for the threat stage.
    tp = fp = fn = tn = 0
    gt_available = 0

    for i, path in enumerate(files, 1):
        frame = cv2.imread(str(path))
        if frame is None:
            continue

        # process_frame gates everything behind motion, which a still image
        # cannot produce. Call the stages directly so every image counts.
        t0 = time.perf_counter()
        objects = service.detect_objects(frame)
        threats = service.detect_threats(frame) if threat_on else []
        per_image_latency.append((time.perf_counter() - t0) * 1000)

        for d in objects + threats:
            confs[d.type.value].append(d.confidence)

        if label_dir is not None:
            lp = label_dir / (path.stem + ".txt")
            if lp.exists():
                gt_available += 1
                gt_classes = set()
                for line in lp.read_text().splitlines():
                    parts = line.split()
                    if parts:
                        gt_classes.add(int(parts[0]))
                gt_dangerous = GT_DANGEROUS in gt_classes
                pred_dangerous = bool(threats)
                if gt_dangerous and pred_dangerous:
                    tp += 1
                elif gt_dangerous and not pred_dangerous:
                    fn += 1
                elif not gt_dangerous and pred_dangerous:
                    fp += 1
                else:
                    tn += 1

        if i % 25 == 0:
            print(f"\r  {i}/{len(files)}", end="", flush=True)

    print(f"\r  {len(files)}/{len(files)} done\n")

    # --- per-class confidence stability ---------------------------------
    print("Per-class confidence")
    print(f"{'class':<18}{'n':>6}{'mean':>8}{'med':>8}{'p05':>8}"
          f"{'min':>8}{'stdev':>8}{'near':>8}")
    print("-" * 72)

    rows = []
    for cls in sorted(confs, key=lambda c: -len(confs[c])):
        thr = (settings.THREAT_CONFIDENCE if cls == DetectionType.DANGEROUS_PERSON.value
               else service.config.detection_threshold)
        rows.append(_summarize(cls, confs[cls], thr))

    for r in rows:
        if not r["n"]:
            continue
        print(f"{r['class']:<18}{r['n']:>6}{r['mean']:>8.3f}{r['median']:>8.3f}"
              f"{r['p05']:>8.3f}{r['min']:>8.3f}{r['stdev']:>8.3f}"
              f"{r['near']*100:>7.0f}%")

    print("\n  near = share of detections within 0.10 of the threshold;")
    print("         these are the ones that flicker between frames.")

    if per_image_latency:
        print(f"\nLatency: mean {statistics.mean(per_image_latency):.1f} ms/image"
              f"  median {statistics.median(per_image_latency):.1f} ms")

    # --- ground-truth accuracy for the threat stage ----------------------
    if gt_available:
        print(f"\nThreat stage vs ground truth  ({gt_available} labeled images)")
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        acc = (tp + tn) / gt_available
        print(f"  true positives  : {tp:>5}")
        print(f"  false positives : {fp:>5}   <- flags an unarmed person")
        print(f"  false negatives : {fn:>5}   <- misses an armed person")
        print(f"  true negatives  : {tn:>5}")
        print(f"\n  precision : {prec:.3f}")
        print(f"  recall    : {rec:.3f}")
        print(f"  F1        : {f1:.3f}")
        print(f"  accuracy  : {acc:.3f}")
        print("\n  Image-level: 'did this frame contain an armed person'.")
        print("  Not directly comparable to the upstream box-level mAP.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
