"""
Stanford Cars (Hugging Face parquet mirror) -> make-level crop folders.

    python training/crop/prepare_stanford_cars.py \
        --src D:/argus-data/stanford-cars/data --out D:/argus-data/crops/make

Source: https://huggingface.co/datasets/Donghyun99/Stanford-Cars
(8,144 train / 8,041 test, 196 make-model-year classes). Collapsed here
to 49 makes; 10% of train is held out as val, stratified by make. Images
are resized to a 256 px short side once, so training does not decode
multi-megapixel JPEGs every epoch.

Licensing: the mirror states no license. The original Stanford release is
distributed for research use; treat it as non-commercial until confirmed.
"""

import argparse
import io
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

from taxonomy import make_of

SHORT_SIDE = 256


def class_names(card: Path) -> list:
    text = card.read_text(encoding="utf-8")
    pairs = re.findall(r"'(\d+)': (.+)", text)
    names = [name.strip() for _, name in sorted(pairs, key=lambda p: int(p[0]))]
    if len(names) != 196:
        raise SystemExit(f"{card}: expected 196 class names, found {len(names)}")
    return names


def resize(img: Image.Image) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size
    s = SHORT_SIDE / min(w, h)
    if s < 1.0:
        img = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BICUBIC)
    return img


def rows(files):
    for f in files:
        table = pq.read_table(f, columns=["image", "label"])
        for img, label in zip(table.column("image").to_pylist(),
                              table.column("label").to_pylist()):
            yield img["bytes"], int(label)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path("D:/argus-data/stanford-cars/data"))
    ap.add_argument("--card", type=Path, default=Path("D:/argus-data/_card/README.md"))
    ap.add_argument("--out", type=Path, default=Path("D:/argus-data/crops/make"))
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    names = class_names(args.card)
    makes = sorted({make_of(n) for n in names})
    counts = defaultdict(int)

    for split, pattern in (("train", "train-*.parquet"), ("test", "test-*.parquet")):
        files = sorted(args.src.glob(pattern))
        if not files:
            raise SystemExit(f"no {pattern} under {args.src}")
        by_make = defaultdict(list)
        for blob, label in rows(files):
            by_make[make_of(names[label])].append(blob)

        rng = random.Random(args.seed)
        for make, blobs in by_make.items():
            rng.shuffle(blobs)
            n_val = int(round(len(blobs) * args.val_frac)) if split == "train" else 0
            for i, blob in enumerate(blobs):
                dest_split = "val" if i < n_val else split
                d = args.out / dest_split / make
                d.mkdir(parents=True, exist_ok=True)
                resize(Image.open(io.BytesIO(blob))).save(d / f"{i:05d}.jpg", quality=92)
                counts[dest_split] += 1
        print(f"  {split}: {sum(len(b) for b in by_make.values())} images, {len(by_make)} makes")

    (args.out / "classes.json").write_text(json.dumps({
        "classes": makes,
        "source": "https://huggingface.co/datasets/Donghyun99/Stanford-Cars",
        "counts": dict(counts),
    }, indent=2), encoding="utf-8")
    print(f"  wrote {dict(counts)} -> {args.out}")


if __name__ == "__main__":
    main()
