"""
Roboflow detection export (COCO JSON) -> classification crops per ARGUS slot.

    python training/crop/prepare_roboflow.py --task carrier \
        --src D:/argus-data/roboflow/usps-merge --out D:/argus-data/crops/carrier
    python training/crop/prepare_roboflow.py --task size \
        --src D:/argus-data/roboflow/mailers-and-boxes --out D:/argus-data/crops/size

Download each dataset from Roboflow Universe in "COCO" format and unzip it;
the export holds train/ valid/ test/ folders, each with
_annotations.coco.json. Several --src folders can be merged in one run.

Each box is padded 10% (the same padding the node applies to YOLO boxes)
and saved under the ARGUS slot its class maps to. Classes that map to
nothing (logos, people) are skipped and counted, never guessed.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image

from taxonomy import argus_carrier, argus_size, crop_box

SPLITS = {"train": "train", "valid": "val", "test": "test"}
MAPPERS = {"carrier": argus_carrier, "size": argus_size}


def crops_from_export(src: Path, task: str, out: Path, counts: Counter, skipped: Counter) -> None:
    mapper = MAPPERS[task]
    for rf_split, split in SPLITS.items():
        ann_path = src / rf_split / "_annotations.coco.json"
        if not ann_path.exists():
            continue
        coco = json.loads(ann_path.read_text(encoding="utf-8"))
        cats = {c["id"]: c["name"] for c in coco["categories"]}
        images = {im["id"]: im for im in coco["images"]}
        for k, ann in enumerate(coco["annotations"]):
            name = cats[ann["category_id"]]
            slot = mapper(name)
            if slot is None:
                skipped[name] += 1
                continue
            im = images[ann["image_id"]]
            x, y, w, h = ann["bbox"]
            box = crop_box((x, y, x + w, y + h), im["width"], im["height"])
            if box is None:
                skipped["(too small)"] += 1
                continue
            with Image.open(src / rf_split / im["file_name"]) as img:
                crop = img.convert("RGB").crop(box)
            d = out / split / slot
            d.mkdir(parents=True, exist_ok=True)
            crop.save(d / f"{src.name}_{rf_split}_{k:06d}.jpg", quality=92)
            counts[(split, slot)] += 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=sorted(MAPPERS), required=True)
    ap.add_argument("--src", type=Path, nargs="+", required=True,
                    help="one or more unzipped Roboflow COCO exports")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    counts, skipped = Counter(), Counter()
    for src in args.src:
        if not src.is_dir():
            raise SystemExit(f"{src}: not a directory")
        crops_from_export(src, args.task, args.out, counts, skipped)
    if not counts:
        raise SystemExit("no crops written: check --src points at an unzipped COCO export")

    slots = sorted({slot for _, slot in counts})
    for split in ("train", "val", "test"):
        row = "  ".join(f"{s}={counts[(split, s)]}" for s in slots)
        print(f"  {split:5}  {row}")
    if skipped:
        print(f"  skipped: {dict(skipped)}")
    (args.out / "classes.json").write_text(json.dumps({
        "classes": slots, "task": args.task,
        "sources": [str(s) for s in args.src],
        "counts": {f"{sp}/{sl}": n for (sp, sl), n in counts.items()},
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
