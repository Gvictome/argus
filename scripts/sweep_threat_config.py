#!/usr/bin/env python3
"""
Find the threat configuration that maximises accuracy at a usable
sensitivity.

Input size was originally pinned at 416 to keep the Pi CPU viable. With a
Hailo accelerator running inference, that constraint is gone -- so the
question becomes what the model can actually do when speed is not the
binding limit, and where the accuracy/latency knee sits.

Sweeps input size and confidence threshold against ground truth and
reports accuracy, precision, recall, and the false-alarm rate on ordinary
scenes. Optionally sweeps a second weights file so model variants can be
compared on identical data.

Usage:
    python scripts/sweep_threat_config.py --dataset <split>
    python scripts/sweep_threat_config.py --dataset <split> --negatives <dir>
    python scripts/sweep_threat_config.py --dataset <split> \
        --weights models/threat-yolo11n.pt models/threat-yolo11s.pt
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from src.config import settings  # noqa: E402

GT_DANGEROUS = 1
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

SIZES = [416, 512, 640, 832]
THRESHOLDS = [0.30, 0.40, 0.50, 0.55, 0.60, 0.70]


def _images(root: Path, limit: int):
    files = sorted(p for p in root.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    return files[:limit] if limit else files


def score_dataset(clf, images, label_dir):
    """Best dangerous-confidence per image, paired with ground truth."""
    out = []
    for i, p in enumerate(images, 1):
        lp = label_dir / (p.stem + ".txt")
        if not lp.exists():
            continue
        gt = GT_DANGEROUS in {
            int(l.split()[0]) for l in lp.read_text().splitlines() if l.split()
        }
        frame = cv2.imread(str(p))
        if frame is None:
            continue
        best = max((r.confidence for r in clf.detect_threats(frame)), default=0.0)
        out.append((gt, best))
        if i % 50 == 0:
            print(f"\r    {i}/{len(images)}", end="", flush=True)
    print(f"\r    {len(images)}/{len(images)} scored")
    return out


def score_negatives(clf, images):
    scores = []
    for i, p in enumerate(images, 1):
        frame = cv2.imread(str(p))
        if frame is None:
            continue
        scores.append(max((r.confidence for r in clf.detect_threats(frame)), default=0.0))
        if i % 50 == 0:
            print(f"\r    {i}/{len(images)}", end="", flush=True)
    print(f"\r    {len(images)}/{len(images)} scored")
    return scores


def metrics(scored, thresh):
    tp = sum(1 for gt, s in scored if gt and s >= thresh)
    fp = sum(1 for gt, s in scored if not gt and s >= thresh)
    fn = sum(1 for gt, s in scored if gt and s < thresh)
    tn = sum(1 for gt, s in scored if not gt and s < thresh)
    total = tp + fp + fn + tn or 1
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {
        "acc": (tp + tn) / total, "prec": prec, "rec": rec, "f1": f1,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Sweep threat model configuration.")
    ap.add_argument("--dataset", type=Path, required=True,
                    help="YOLO split dir with images/ and labels/.")
    ap.add_argument("--negatives", type=Path,
                    help="Images with no armed person, for a false-alarm rate.")
    ap.add_argument("--weights", type=Path, nargs="*",
                    help="Weights to compare. Defaults to the configured model.")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--min-recall", type=float, default=0.75,
                    help="Configurations below this recall are not reported as best.")
    args = ap.parse_args()

    from src.detection.threat import ThreatClassifier

    weights = args.weights or [settings.THREAT_MODEL_PATH]
    images = _images(args.dataset / "images", args.limit)
    label_dir = args.dataset / "labels"
    negatives = _images(args.negatives, args.limit) if args.negatives else []

    print(f"labelled images : {len(images)}")
    print(f"negatives       : {len(negatives)}")
    print(f"weights         : {[Path(w).name for w in weights]}")
    print(f"min recall      : {args.min_recall}\n")

    best = None
    rows = []

    for w in weights:
        w = Path(w)
        if not w.exists():
            print(f"[skip] {w} not found")
            continue

        for size in SIZES:
            # Load once at the lowest threshold, then filter in Python --
            # re-running inference per threshold would repeat identical work.
            clf = ThreatClassifier(w, confidence=min(THRESHOLDS), imgsz=size)

            print(f"  {w.name} @ imgsz={size}")
            t0 = time.perf_counter()
            scored = score_dataset(clf, images, label_dir)
            ms = (time.perf_counter() - t0) / max(1, len(images)) * 1000

            neg = score_negatives(clf, negatives) if negatives else []

            for t in THRESHOLDS:
                m = metrics(scored, t)
                fa = (sum(1 for s in neg if s >= t) / len(neg) * 100) if neg else float("nan")
                row = {"model": w.name, "imgsz": size, "thresh": t,
                       "ms": ms, "fa": fa, **m}
                rows.append(row)
                if m["rec"] >= args.min_recall and (best is None or m["acc"] > best["acc"]):
                    best = row

    if not rows:
        print("no configurations evaluated")
        return 1

    hdr = (f"{'model':<22}{'imgsz':>6}{'thr':>6}{'acc':>8}{'prec':>8}"
           f"{'rec':>8}{'F1':>8}{'FN':>5}{'ms':>7}{'FA%':>7}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in rows:
        star = " *" if best is not None and r is best else ""
        fa = "  n/a" if np.isnan(r["fa"]) else f"{r['fa']:6.1f}"
        print(f"{r['model'][:21]:<22}{r['imgsz']:>6}{r['thresh']:>6.2f}"
              f"{r['acc']:>8.3f}{r['prec']:>8.3f}{r['rec']:>8.3f}{r['f1']:>8.3f}"
              f"{r['fn']:>5}{r['ms']:>7.0f}{fa}{star}")

    print("\n  FA% = share of ordinary scenes falsely flagged.")
    print(f"  ms  = per-image inference on this machine, not the Pi.")

    if best:
        print(f"\nBest accuracy at recall >= {args.min_recall}:")
        print(f"  {best['model']}  imgsz={best['imgsz']}  threshold={best['thresh']:.2f}")
        print(f"  accuracy {best['acc']:.3f}   precision {best['prec']:.3f}   "
              f"recall {best['rec']:.3f}")
        print(f"\n  THREAT_IMGSZ={best['imgsz']} THREAT_CONFIDENCE={best['thresh']}")
    else:
        print(f"\nNo configuration reached recall {args.min_recall}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
