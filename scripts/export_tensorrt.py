#!/usr/bin/env python3
"""
Export ARGUS's models to TensorRT engines for the Jetson Orin.

TensorRT compiles a model for the exact GPU, driver, and TensorRT version
it is built on. **An engine is not portable** -- you cannot build it on a
laptop and copy it to the Orin, and an engine built before a JetPack
upgrade will refuse to load after one. So this runs on the Orin.

The build is slow (minutes per model, longer with --half on first run).
Do it once during setup, not at the booth.

Usage:
    python scripts/export_tensorrt.py                 # both models, FP16
    python scripts/export_tensorrt.py --model object  # just YOLOv8n
    python scripts/export_tensorrt.py --no-half       # FP32
    python scripts/export_tensorrt.py --check         # report only

Exit codes:
    0  engines present (or built)
    1  export failed
    2  prerequisite missing
"""

import argparse
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import BASE_DIR, settings  # noqa: E402

MODELS_DIR = BASE_DIR / "models"

TARGETS = {
    "object": {
        "source": BASE_DIR / "yolov8n.pt",
        "engine": MODELS_DIR / "yolov8n.engine",
        "imgsz": 640,
        "why": "COCO object detection (person/vehicle/animal)",
    },
    "threat": {
        "source": settings.THREAT_MODEL_PATH,
        "engine": MODELS_DIR / "threat-yolo11n.engine",
        "imgsz": settings.THREAT_IMGSZ,
        "why": "dangerous-person classification",
    },
}


def _say(tag: str, msg: str) -> None:
    print(f"[{tag}] {msg}")


def check_platform() -> bool:
    """Report what we can see, and whether this looks like a Jetson."""
    from src.camera.platform_detect import BOARD, Board, board_description

    _say("info", f"board: {board_description()}")

    ok = True
    try:
        import torch
        _say("info", f"torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            _say("info", f"device: {torch.cuda.get_device_name(0)}")
        else:
            _say("FAIL", "CUDA is not available — TensorRT export needs the GPU")
            ok = False
    except ImportError:
        _say("FAIL", "torch is not installed")
        ok = False

    if BOARD is not Board.JETSON:
        _say("warn", "this does not look like a Jetson; engines built here "
                     "will not load on one")

    if shutil.which("trtexec") is None:
        _say("info", "trtexec not on PATH (fine — ultralytics uses the Python API)")

    return ok


def export_one(key: str, half: bool, workspace: int) -> bool:
    spec = TARGETS[key]
    source, engine = Path(spec["source"]), Path(spec["engine"])

    print()
    _say("info", f"{key}: {spec['why']}")

    if not source.exists():
        _say("FAIL", f"{key}: source model missing at {source}")
        if key == "threat":
            _say("info", "fetch it first: python scripts/fetch_models.py")
        return False

    if engine.exists():
        _say("ok", f"{key}: engine already exists at {engine}")
        print("       delete it to rebuild")
        return True

    try:
        from ultralytics import YOLO
    except ImportError:
        _say("FAIL", "ultralytics is not installed")
        return False

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    _say("info", f"{key}: building at imgsz={spec['imgsz']}, "
                 f"{'FP16' if half else 'FP32'} — this takes minutes")

    started = time.time()
    try:
        model = YOLO(str(source))
        produced = model.export(
            format="engine",
            imgsz=spec["imgsz"],
            half=half,
            workspace=workspace,
            device=0,
            verbose=False,
        )
    except Exception as exc:
        _say("FAIL", f"{key}: export failed: {exc}")
        return False

    elapsed = time.time() - started

    # ultralytics writes the engine next to the source; move it to models/
    # so DetectionService's probe finds it at a predictable path.
    produced = Path(produced) if produced else source.with_suffix(".engine")
    if produced.exists() and produced.resolve() != engine.resolve():
        engine.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(produced), str(engine))

    if not engine.exists():
        _say("FAIL", f"{key}: export reported success but {engine} is missing")
        return False

    size_mb = engine.stat().st_size / (1024 * 1024)
    _say("ok", f"{key}: wrote {engine} ({size_mb:.1f} MB) in {elapsed:.0f}s")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Export models to TensorRT engines.")
    ap.add_argument("--model", choices=list(TARGETS) + ["all"], default="all")
    ap.add_argument("--no-half", action="store_true",
                    help="FP32 instead of FP16. Slower, marginally more accurate.")
    ap.add_argument("--workspace", type=int, default=4, help="Build workspace, GB.")
    ap.add_argument("--check", action="store_true", help="Report status and exit.")
    args = ap.parse_args()

    print("TensorRT export")
    print("---------------")
    platform_ok = check_platform()

    if args.check:
        print()
        for key, spec in TARGETS.items():
            engine = Path(spec["engine"])
            state = "present" if engine.exists() else "MISSING"
            print(f"  {key:8s} {state:8s} {engine}")
        print()
        print("DetectionService selects: tensorrt > hailo > cpu")
        return 0

    if not platform_ok:
        _say("FAIL", "prerequisites missing — see above")
        return 2

    keys = list(TARGETS) if args.model == "all" else [args.model]
    results = {k: export_one(k, not args.no_half, args.workspace) for k in keys}

    print()
    if all(results.values()):
        _say("ok", "all engines ready")
        print()
        print("Restart ARGUS; /api/detection/status should report")
        print('  "backend": "tensorrt"')
        return 0

    failed = [k for k, v in results.items() if not v]
    _say("FAIL", f"failed: {', '.join(failed)}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
