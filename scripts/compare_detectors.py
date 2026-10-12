#!/usr/bin/env python3
"""
Pick the base detector: YOLOv8n vs YOLO11n vs YOLO26n, on the numbers
that decide it for ARGUS.

    accuracy   mAP on COCO val2017 (never the train split: coco8 and coco128
               are training images, which every one of these models has seen),
               overall and over the 16 COCO classes ARGUS maps to
               person / vehicle / animal / package
    speed      median CPU ms per frame on this machine, at the input sizes
               the node actually uses
    our scene  detections per frame on a real clip, by ARGUS class. No ground
               truth, so this only shows whether a model sees our kind of
               scene, not whether it is right
    accelerator  whether Ultralytics' Hailo exporter takes the model
    size       parameters and GFLOPs

Run it once with --prepare to fetch COCO val2017 (about 0.8 GB of images
plus the label archive) into data/datasets/coco; later runs reuse it.

    python scripts/compare_detectors.py --prepare
    python scripts/compare_detectors.py                       # 500 val images
    python scripts/compare_detectors.py --subset 5000         # all of val2017
    python scripts/compare_detectors.py --models yolov8n yolo11n --clip media/vtest.avi

Writes data/benchmarks/detector_compare_<date>.{json,md}. Laptop numbers are
not Pi numbers: run the speed part again on the Pi before quoting FPS.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.request
import zipfile
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data" / "datasets" / "coco"
OUT_DIR = ROOT / "data" / "benchmarks"

VAL_IMAGES_URL = "http://images.cocodataset.org/zips/val2017.zip"
LABELS_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels.zip"

# The COCO ids ARGUS keeps (src/detection/__init__.py, _YOLO_CLASS_MAP).
ARGUS_GROUPS = {
    "person": [0],
    "vehicle": [1, 2, 3, 5, 6, 7],
    "animal": [14, 15, 16, 17, 18, 19],
    "package": [24, 26, 28],
}
ARGUS_IDS = sorted(i for ids in ARGUS_GROUPS.values() for i in ids)

# Ultralytics format="hailo" support for detection, per
# https://docs.ultralytics.com/integrations/hailo (checked 2026-10-11).
HAILO_EXPORT = {"yolov8": True, "yolo11": True, "yolo26": True}


def _download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"  have {dest.name}")
        return
    print(f"  downloading {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)


def prepare() -> None:
    """Fetch val2017 images and the YOLO-format labels, val split only."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    zips = DATA_DIR / "_zips"
    zips.mkdir(exist_ok=True)
    _download(LABELS_URL, zips / "coco2017labels.zip")
    _download(VAL_IMAGES_URL, zips / "val2017.zip")

    labels_dir = DATA_DIR / "labels" / "val2017"
    if not labels_dir.exists():
        print("  extracting val labels")
        with zipfile.ZipFile(zips / "coco2017labels.zip") as z:
            for name in z.namelist():
                # The archive is rooted at coco/; keep only val2017 labels.
                if "/labels/val2017/" in name and name.endswith(".txt"):
                    target = labels_dir / Path(name).name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(z.read(name))
    images_dir = DATA_DIR / "images" / "val2017"
    if not images_dir.exists():
        print("  extracting val images")
        with zipfile.ZipFile(zips / "val2017.zip") as z:
            z.extractall(DATA_DIR / "images")
    n = len(list(images_dir.glob("*.jpg")))
    print(f"  ready: {n} val images in {images_dir}")


def _subset_yaml(n: int) -> Path:
    """A data yaml whose val split is the first n val2017 images, sorted, so
    every model and every run scores the same images."""
    images = sorted((DATA_DIR / "images" / "val2017").glob("*.jpg"))
    if not images:
        sys.exit("No COCO val images. Run with --prepare first.")
    chosen = images[: min(n, len(images))]
    listing = DATA_DIR / f"val_subset_{len(chosen)}.txt"
    listing.write_text("\n".join(str(p) for p in chosen) + "\n", encoding="utf-8")

    import yaml as pyyaml

    names = _coco_names()
    cfg = {"path": str(DATA_DIR), "train": str(listing), "val": str(listing),
           "names": {i: n for i, n in enumerate(names)}}
    out = DATA_DIR / f"argus_val_{len(chosen)}.yaml"
    out.write_text(pyyaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return out


def _coco_names() -> list:
    from ultralytics import YOLO
    return list(YOLO("yolov8n.pt").names.values())


def _family(name: str) -> str:
    for fam in HAILO_EXPORT:
        if name.startswith(fam):
            return fam
    return "unknown"


def accuracy(model, data_yaml: Path, imgsz: int) -> dict:
    m = model.val(data=str(data_yaml), imgsz=imgsz, batch=16, device="cpu",
                  plots=False, verbose=False, workers=0)
    maps = m.box.maps  # per-class mAP50-95, indexed by class id
    present = set(int(c) for c in m.box.ap_class_index)
    groups = {}
    for g, ids in ARGUS_GROUPS.items():
        vals = [float(maps[i]) for i in ids if i in present]
        groups[g] = round(statistics.mean(vals), 4) if vals else None
    argus_vals = [float(maps[i]) for i in ARGUS_IDS if i in present]
    return {
        "map50_95": round(float(m.box.map), 4),
        "map50": round(float(m.box.map50), 4),
        "argus_map50_95": round(statistics.mean(argus_vals), 4) if argus_vals else None,
        "by_argus_class": groups,
    }


def _frames(clip: Path, n: int):
    import cv2
    cap = cv2.VideoCapture(str(clip))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or n
    step = max(1, total // n)
    out, i = [], 0
    while len(out) < n:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok:
            break
        out.append(frame)
        i += step
    cap.release()
    return out


def speed(model, frames, imgsz: int, warmup: int = 5) -> dict:
    for f in frames[:warmup]:
        model.predict(f, imgsz=imgsz, device="cpu", verbose=False, classes=ARGUS_IDS)
    times = []
    for f in frames:
        t = time.perf_counter()
        model.predict(f, imgsz=imgsz, device="cpu", verbose=False, classes=ARGUS_IDS)
        times.append((time.perf_counter() - t) * 1000)
    med = statistics.median(times)
    return {"median_ms": round(med, 1), "p90_ms": round(sorted(times)[int(0.9 * len(times)) - 1], 1),
            "fps": round(1000 / med, 1)}


def scene(model, frames, imgsz: int, conf: float) -> dict:
    counts = {g: 0 for g in ARGUS_GROUPS}
    confs = []
    for f in frames:
        r = model.predict(f, imgsz=imgsz, conf=conf, device="cpu", verbose=False,
                          classes=ARGUS_IDS)[0]
        for c, p in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist()):
            for g, ids in ARGUS_GROUPS.items():
                if int(c) in ids:
                    counts[g] += 1
            confs.append(p)
    n = len(frames) or 1
    return {"per_frame": {g: round(v / n, 2) for g, v in counts.items()},
            "mean_conf": round(statistics.mean(confs), 3) if confs else None}


def size(model) -> dict:
    try:
        layers, params, grads, gflops = model.info(verbose=False)
        return {"params_m": round(params / 1e6, 2), "gflops": round(gflops, 1)}
    except Exception:
        params = sum(p.numel() for p in model.model.parameters())
        return {"params_m": round(params / 1e6, 2), "gflops": None}


def to_markdown(results: list, meta: dict) -> str:
    lines = [
        f"# Detector comparison, {meta['date']}",
        "",
        f"COCO val2017, first {meta['val_images']} images (sorted), imgsz {meta['val_imgsz']}. "
        f"CPU: {meta['cpu']}. Clip: `{meta['clip']}` ({meta['clip_frames']} frames, conf {meta['conf']}).",
        "",
        "| Model | Params (M) | GFLOPs | mAP50-95 | mAP50 | ARGUS-class mAP50-95 | person | vehicle | animal | package | CPU ms @640 | CPU ms @416 | Hailo export |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        a, g = r["accuracy"], r["accuracy"]["by_argus_class"]
        lines.append(
            f"| {r['model']} | {r['size']['params_m']} | {r['size']['gflops']} | {a['map50_95']} | {a['map50']} | "
            f"**{a['argus_map50_95']}** | {g['person']} | {g['vehicle']} | {g['animal']} | {g['package']} | "
            f"{r['speed']['640']['median_ms']} | {r['speed']['416']['median_ms']} | {'yes' if r['hailo_export'] else 'no'} |")
    lines += ["", "Detections per frame on the clip (no ground truth; shows what each model picks up):", "",
              "| Model | person | vehicle | animal | package | mean conf |", "|---|---|---|---|---|---|"]
    for r in results:
        s = r["scene"]["per_frame"]
        lines.append(f"| {r['model']} | {s['person']} | {s['vehicle']} | {s['animal']} | {s['package']} | {r['scene']['mean_conf']} |")
    lines += ["", "Laptop CPU times are not Pi times. Re-run the speed columns on the Pi (`--skip-accuracy`) before quoting FPS."]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--prepare", action="store_true", help="download COCO val2017 and exit")
    ap.add_argument("--models", nargs="+", default=["yolov8n", "yolo11n", "yolo26n"])
    ap.add_argument("--subset", type=int, default=500, help="val2017 images to score (max 5000)")
    ap.add_argument("--imgsz", type=int, default=640, help="input size for the accuracy run")
    ap.add_argument("--clip", default="media/vtest.avi", help="a real clip for speed and the scene check")
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--conf", type=float, default=0.5, help="the node's DETECTION_THRESHOLD")
    ap.add_argument("--skip-accuracy", action="store_true", help="speed and scene only (use on the Pi)")
    args = ap.parse_args()

    if args.prepare:
        prepare()
        return 0

    import platform

    from ultralytics import YOLO

    clip = ROOT / args.clip
    if not clip.exists():
        sys.exit(f"clip not found: {clip}")
    frames = _frames(clip, args.frames)
    data_yaml = None if args.skip_accuracy else _subset_yaml(args.subset)

    results = []
    for name in args.models:
        print(f"\n== {name}")
        model = YOLO(f"{name}.pt")
        r = {"model": name, "size": size(model), "hailo_export": HAILO_EXPORT.get(_family(name), False)}
        if data_yaml:
            print("   accuracy on COCO val...")
            r["accuracy"] = accuracy(model, data_yaml, args.imgsz)
        else:
            r["accuracy"] = {"map50_95": None, "map50": None, "argus_map50_95": None,
                             "by_argus_class": {g: None for g in ARGUS_GROUPS}}
        print("   speed...")
        r["speed"] = {str(s): speed(model, frames, s) for s in (640, 416)}
        r["scene"] = scene(model, frames, 640, args.conf)
        print(f"   {json.dumps(r)}")
        results.append(r)

    meta = {"date": date.today().isoformat(), "val_images": args.subset if data_yaml else 0,
            "val_imgsz": args.imgsz, "cpu": platform.processor() or platform.machine(),
            "clip": args.clip, "clip_frames": len(frames), "conf": args.conf}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUT_DIR / f"detector_compare_{meta['date']}"
    stem.with_suffix(".json").write_text(json.dumps({"meta": meta, "results": results}, indent=2), encoding="utf-8")
    md = to_markdown(results, meta)
    stem.with_suffix(".md").write_text(md, encoding="utf-8")
    print("\n" + md)
    print(f"Wrote {stem}.json and .md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
