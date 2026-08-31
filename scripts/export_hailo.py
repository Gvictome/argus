#!/usr/bin/env python3
"""
Compile ARGUS's models for the Raspberry Pi AI HAT+ / AI HAT+ 2.

    THIS DOES NOT RUN ON THE PI.

The Hailo Dataflow Compiler is x86_64 Linux only. You compile on a
workstation and copy the result to the Pi, which needs only the HailoRT
runtime. This is the opposite of the TensorRT flow, where the engine must
be built on the target device -- easy to get backwards, so the script
checks and refuses rather than failing halfway through a long build.

Ultralytics produces an export *directory*, not a bare .hef: the HEF
alone lacks the metadata describing input size, classes, and NMS. Copy
the whole directory.

Usage (on an x86_64 Linux workstation, or WSL2):
    python scripts/export_hailo.py --hw-arch hailo8         # AI HAT+
    python scripts/export_hailo.py --hw-arch hailo10h       # AI HAT+ 2
    python scripts/export_hailo.py --check                  # report only

Then on the Pi:
    sudo apt install hailo-all        # Hailo-8 / 8L  (AI HAT+)
    sudo apt install hailo-h10-all    # Hailo-10H      (AI HAT+ 2)
    hailortcli fw-control identify    # confirm the board is seen
    scp -r models/yolov8n_hailo_model pi@<host>:~/argus/models/

Exit codes:
    0  exports present (or built)
    1  export failed
    2  wrong platform, or a prerequisite is missing
"""

import argparse
import platform
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import BASE_DIR, settings  # noqa: E402

MODELS_DIR = BASE_DIR / "models"

# Which accelerator is on which board. The AI HAT+ 2 is a different chip
# from the AI HAT+, not a faster revision of it, and a HEF compiled for
# one will not load on the other.
HW_ARCHS = {
    "hailo8":   "AI HAT+ (26 TOPS, Hailo-8)",
    "hailo8l":  "AI Kit / AI HAT+ 13 TOPS (Hailo-8L)",
    "hailo10h": "AI HAT+ 2 (40 TOPS INT8, 8 GB onboard, Hailo-10H)",
}

TARGETS = {
    "object": {
        "source": BASE_DIR / "yolov8n.pt",
        "out": MODELS_DIR / "yolov8n_hailo_model",
        "imgsz": 640,
        "why": "COCO object detection (person/vehicle/animal)",
    },
    "threat": {
        "source": settings.THREAT_MODEL_PATH,
        "out": MODELS_DIR / "threat_hailo_model",
        "imgsz": settings.THREAT_IMGSZ,
        "why": "dangerous-person classification",
    },
}


def _say(tag: str, msg: str) -> None:
    print(f"[{tag}] {msg}")


def check_platform() -> bool:
    """The DFC needs x86_64 Linux. Say so plainly rather than failing late."""
    system, machine = platform.system(), platform.machine()
    _say("info", f"host: {system} {machine}")

    ok = True
    if system != "Linux" or machine not in ("x86_64", "AMD64"):
        _say("FAIL", "the Hailo Dataflow Compiler requires x86_64 Linux.")
        print("       You cannot compile on the Pi, on macOS, or on Windows")
        print("       directly. Use a Linux workstation, WSL2, or Ultralytics")
        print("       Platform, then copy the export directory to the Pi.")
        ok = False

    try:
        import hailo_sdk_client  # noqa: F401
        _say("ok", "Hailo Dataflow Compiler found")
    except ImportError:
        _say("FAIL", "hailo_sdk_client not installed — install the Hailo DFC")
        print("       (developer.hailo.ai, requires a free account)")
        ok = False

    try:
        import ultralytics
        _say("ok", f"ultralytics {ultralytics.__version__}")
    except ImportError:
        _say("FAIL", "ultralytics is not installed")
        ok = False

    return ok


def export_one(key: str, hw_arch: str) -> bool:
    spec = TARGETS[key]
    source, out = Path(spec["source"]), Path(spec["out"])

    print()
    _say("info", f"{key}: {spec['why']}")

    if not source.exists():
        _say("FAIL", f"{key}: source model missing at {source}")
        if key == "threat":
            _say("info", "fetch it first: python scripts/fetch_models.py")
        return False

    if out.exists():
        _say("ok", f"{key}: export already present at {out}")
        print("       delete it to rebuild")
        return True

    from ultralytics import YOLO

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    _say("info", f"{key}: compiling for {hw_arch} at imgsz={spec['imgsz']}")
    print("       This takes a long time — tens of minutes is normal.")

    started = time.time()
    try:
        produced = YOLO(str(source)).export(
            format="hailo",
            imgsz=spec["imgsz"],
            hw_arch=hw_arch,
        )
    except Exception as exc:
        _say("FAIL", f"{key}: export failed: {exc}")
        return False

    elapsed = time.time() - started

    produced = Path(produced) if produced else None
    if produced and produced.exists() and produced.resolve() != out.resolve():
        if out.exists():
            shutil.rmtree(out, ignore_errors=True)
        shutil.move(str(produced), str(out))

    if not out.exists():
        _say("FAIL", f"{key}: export reported success but {out} is missing")
        return False

    _say("ok", f"{key}: wrote {out} in {elapsed/60:.1f} min")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Compile models for Hailo.")
    ap.add_argument("--model", choices=list(TARGETS) + ["all"], default="all")
    ap.add_argument("--hw-arch", choices=list(HW_ARCHS), default="hailo8",
                    help="Which accelerator. AI HAT+ 2 is hailo10h.")
    ap.add_argument("--check", action="store_true", help="Report and exit.")
    args = ap.parse_args()

    print("Hailo export")
    print("------------")

    if args.check:
        print()
        for key, spec in TARGETS.items():
            out = Path(spec["out"])
            print(f"  {key:8s} {'present' if out.exists() else 'MISSING':8s} {out}")
        print()
        print("  Accelerators:")
        for arch, desc in HW_ARCHS.items():
            print(f"    {arch:9s} {desc}")
        print()
        print("  DetectionService selects: tensorrt > hailo > cpu")
        return 0

    _say("info", f"target: {HW_ARCHS[args.hw_arch]}")
    if not check_platform():
        return 2

    keys = list(TARGETS) if args.model == "all" else [args.model]
    results = {k: export_one(k, args.hw_arch) for k in keys}

    print()
    if not all(results.values()):
        _say("FAIL", f"failed: {', '.join(k for k, v in results.items() if not v)}")
        return 1

    _say("ok", "all exports ready")
    print()
    print("Copy to the Pi, then restart ARGUS:")
    for k in keys:
        print(f"  scp -r {TARGETS[k]['out']} pi@<host>:~/argus/models/")
    print()
    print('/api/detection/status should then report  "backend": "hailo"')
    return 0


if __name__ == "__main__":
    sys.exit(main())
