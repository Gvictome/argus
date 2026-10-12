#!/usr/bin/env python3
"""
Build a larger test set for the threat stage from public weapon datasets.

The threat model answers "is this person dangerous?" for a whole scene,
so its existing test set (314 images from the upstream thesis split) is
small for choosing a threshold. This turns weapon datasets into the same
image-level layout `tune_threat_threshold.py` reads:

  positives  images that contain a labelled weapon AND a person the COCO
             detector can find. A close-up of a knife on a table is not an
             armed person and would only measure the wrong thing.
  negatives  COCO val2017 images with people and no knife, scissors or
             baseball bat, so ordinary scenes give the false-alarm rate.

Inputs are Supervisely-format datasets as `dataset-tools` downloads them
(any folder tree with img/ and ann/ siblings). Output:

  data/datasets/threat_eval/positives/{images,labels}   label "1 ..." per image
  data/datasets/threat_eval/negatives/images

Then:
  python scripts/tune_threat_threshold.py --dataset data/datasets/threat_eval/positives \
      --negatives data/datasets/threat_eval/negatives
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT = ROOT / "data" / "datasets" / "threat_eval"
COCO = ROOT / "data" / "datasets" / "coco"
# COCO classes that are, or look like, a weapon. Images holding any of
# them are not safe negatives.
WEAPONISH = {34, 43, 76}  # baseball bat, knife, scissors
PERSON = 0


def weapon_images(roots):
    """(image path, source name) for every image with at least one object."""
    for root in roots:
        for ann_dir in Path(root).rglob("ann"):
            img_dir = ann_dir.parent / "img"
            if not img_dir.is_dir():
                continue
            for ann in sorted(ann_dir.glob("*.json")):
                try:
                    objs = json.loads(ann.read_text(encoding="utf-8")).get("objects", [])
                except Exception:
                    continue
                if not objs:
                    continue
                img = img_dir / ann.name[:-5]  # "x.jpg.json" -> "x.jpg"
                if img.exists():
                    yield img, Path(root).name


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("roots", nargs="+", help="downloaded weapon dataset folders")
    ap.add_argument("--max-positives", type=int, default=1500)
    ap.add_argument("--negatives", type=int, default=1000)
    ap.add_argument("--person-conf", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from ultralytics import YOLO
    det = YOLO("yolov8n.pt")
    rng = random.Random(args.seed)

    pos_img = OUT / "positives" / "images"
    pos_lbl = OUT / "positives" / "labels"
    neg_img = OUT / "negatives" / "images"
    for d in (pos_img, pos_lbl, neg_img):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    candidates = list(weapon_images(args.roots))
    rng.shuffle(candidates)
    kept, seen, by_source = 0, 0, {}
    for img, source in candidates:
        if kept >= args.max_positives:
            break
        seen += 1
        r = det.predict(str(img), conf=args.person_conf, classes=[PERSON], verbose=False, device="cpu")[0]
        if len(r.boxes) == 0:
            continue
        name = f"{source[:24].replace(' ', '_')}_{img.stem}{img.suffix}"
        shutil.copy2(img, pos_img / name)
        # Image-level ground truth: the scene contains a dangerous person.
        (pos_lbl / (Path(name).stem + ".txt")).write_text("1 0.5 0.5 1.0 1.0\n", encoding="utf-8")
        kept += 1
        by_source[source] = by_source.get(source, 0) + 1

    labels = COCO / "labels" / "val2017"
    negs = []
    for lf in sorted(labels.glob("*.txt")):
        classes = {int(line.split()[0]) for line in lf.read_text().splitlines() if line.strip()}
        if PERSON in classes and not classes & WEAPONISH:
            negs.append(COCO / "images" / "val2017" / (lf.stem + ".jpg"))
    rng.shuffle(negs)
    for p in negs[: args.negatives]:
        shutil.copy2(p, neg_img / p.name)

    summary = {"positives": kept, "weapon_images_checked": seen, "by_source": by_source,
               "negatives": min(args.negatives, len(negs)), "person_conf": args.person_conf}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
