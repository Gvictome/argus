#!/usr/bin/env python3
"""
Export YOLOv8n to an optimised CPU format.

    ncnn       the Ultralytics-recommended CPU path on a Raspberry Pi
    openvino   the fast CPU path on an x86 dev machine

Unlike the Hailo compile, both exports run on the target machine itself,
including the Pi. This is the speedup available today while no Hailo HEF
has been compiled for the AI HAT+ 2.

The export lands in models/, where DetectionService picks it up when
OBJECT_BACKEND is auto (the default). Delete the folder to go back to the
plain PyTorch weights.

    python scripts/export_cpu_models.py --format ncnn        # on the Pi
    python scripts/export_cpu_models.py --format openvino    # on x86
"""

import argparse
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.config import settings  # noqa: E402
from src.detection import _CPU_EXPORTS, cpu_export_dir  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--format", choices=sorted(_CPU_EXPORTS), required=True)
    ap.add_argument("--model", default=settings.OBJECT_MODEL,
                    help="yolov8n (default), yolov8s, yolov8m")
    ap.add_argument("--weights", default="",
                    help="weights file; defaults to <model>.pt, downloaded if absent")
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()
    weights = args.weights or str(REPO / f"{args.model}.pt")

    from ultralytics import YOLO

    target = REPO / "models" / cpu_export_dir(args.format, args.model)
    t0 = time.time()
    exported = Path(YOLO(weights).export(format=args.format, imgsz=args.imgsz))

    # Ultralytics writes the export beside the weights; move it to models/.
    if exported.resolve() != target.resolve():
        if target.exists():
            shutil.rmtree(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(exported), str(target))

    print(f"{args.format} export ready at {target} ({time.time() - t0:.0f}s)")
    print(f"Run the node with OBJECT_MODEL={args.model} to use it "
          f"(OBJECT_BACKEND=auto).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
