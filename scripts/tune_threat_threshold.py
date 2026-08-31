#!/usr/bin/env python3
"""
Sweep the threat confidence threshold against ground truth.

The default 0.35 is inherited from the upstream model card's
demonstration value. It was never tuned for ARGUS, and the right
threshold depends on which mistake costs more.

Two asymmetries pull in opposite directions here:

  A false NEGATIVE misses an armed person. In a security system that is
  the failure the whole project exists to prevent.

  A false POSITIVE flags an innocent bystander as dangerous. In front of
  a judge, at a booth, pointed at a visitor, that is a credibility
  problem and an ethical one -- and this model runs on public imagery
  where most people are unarmed.

This prints precision, recall, F1, and both error counts at each
threshold so the choice can be made on evidence and written down.

Usage:
    python scripts/tune_threat_threshold.py --dataset .../Dataset/test
    python scripts/tune_threat_threshold.py --dataset ... --negatives .../coco128
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402

from src.config import settings  # noqa: E402

GT_DANGEROUS = 1
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
STEPS = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]


def main() -> int:
    ap = argparse.ArgumentParser(description="Tune the threat threshold.")
    ap.add_argument("--dataset", type=Path, required=True,
                    help="YOLO split dir with images/ and labels/.")
    ap.add_argument("--negatives", type=Path,
                    help="Extra images assumed to contain NO armed person "
                         "(e.g. COCO), for a real-world false-positive rate.")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from src.detection.threat import ThreatClassifier

    # Load once at the lowest threshold, then filter in Python. Re-running
    # inference per threshold would be ~10x the work for identical boxes.
    clf = ThreatClassifier(
        model_path=settings.THREAT_MODEL_PATH,
        confidence=min(STEPS),
        imgsz=settings.THREAT_IMGSZ,
    )

    images = sorted(p for p in (args.dataset / "images").iterdir()
                    if p.suffix.lower() in IMAGE_EXTS)
    if args.limit:
        images = images[:args.limit]

    print(f"labeled images : {len(images)}")

    # best confidence seen per image, plus whether it truly has a threat
    scored = []
    for i, p in enumerate(images, 1):
        lp = args.dataset / "labels" / (p.stem + ".txt")
        if not lp.exists():
            continue
        gt = GT_DANGEROUS in {
            int(l.split()[0]) for l in lp.read_text().splitlines() if l.split()
        }
        frame = cv2.imread(str(p))
        best = max((r.confidence for r in clf.detect_threats(frame)), default=0.0)
        scored.append((gt, best))
        if i % 50 == 0:
            print(f"\r  {i}/{len(images)}", end="", flush=True)
    print(f"\r  {len(images)}/{len(images)} scored\n")

    neg_scores = []
    if args.negatives:
        negs = sorted(p for p in args.negatives.rglob("*")
                      if p.suffix.lower() in IMAGE_EXTS)
        if args.limit:
            negs = negs[:args.limit]
        print(f"negative images: {len(negs)} (assumed no armed person)")
        for i, p in enumerate(negs, 1):
            frame = cv2.imread(str(p))
            if frame is None:
                continue
            neg_scores.append(
                max((r.confidence for r in clf.detect_threats(frame)), default=0.0)
            )
            if i % 25 == 0:
                print(f"\r  {i}/{len(negs)}", end="", flush=True)
        print(f"\r  {len(negs)}/{len(negs)} scored\n")

    header = f"{'thresh':>7}{'prec':>8}{'recall':>8}{'F1':>8}{'FP':>6}{'FN':>6}"
    if neg_scores:
        header += f"{'FP/100 neg':>12}"
    print(header)
    print("-" * len(header))

    best_f1 = (0.0, None)
    for t in STEPS:
        tp = sum(1 for gt, s in scored if gt and s >= t)
        fp = sum(1 for gt, s in scored if not gt and s >= t)
        fn = sum(1 for gt, s in scored if gt and s < t)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        if f1 > best_f1[0]:
            best_f1 = (f1, t)

        line = f"{t:>7.2f}{prec:>8.3f}{rec:>8.3f}{f1:>8.3f}{fp:>6}{fn:>6}"
        if neg_scores:
            rate = sum(1 for s in neg_scores if s >= t) / len(neg_scores) * 100
            line += f"{rate:>11.1f}%"
        print(line)

    print(f"\nbest F1: {best_f1[0]:.3f} at threshold {best_f1[1]:.2f}")
    print(f"current setting: {settings.THREAT_CONFIDENCE:.2f}")
    print("\nF1 is a starting point, not the answer. Decide which error you")
    print("would rather explain to a judge, then pick the threshold that")
    print("makes that tradeoff and write down why.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
