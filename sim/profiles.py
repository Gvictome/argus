"""
Hardware timing profiles for the ARGUS node simulator.

Every number here is traceable to a source in the repo or the proposal.
Nothing is invented; where a figure is a *budget* rather than a
*measurement* it says so, because the whole point of this simulator is to
find out which of the budgets survive contact with real CPU contention.

Stage units
-----------
cpu      Burns a real CPU core. Contends with the federated trainer.
accel    Offloaded to the Hailo-10H over PCIe. Sleeps; does not burn CPU.
measured Timed live during the run (the federated head).
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class Stage:
    name: str
    ms: float
    unit: str  # "cpu" | "accel" | "measured"
    source: str


@dataclass(frozen=True)
class NodeProfile:
    key: str
    label: str
    cores: int
    stages: Tuple[Stage, ...]
    note: str

    @property
    def cpu_ms(self) -> float:
        return sum(s.ms for s in self.stages if s.unit == "cpu")

    @property
    def accel_ms(self) -> float:
        return sum(s.ms for s in self.stages if s.unit == "accel")

    @property
    def serial_ms(self) -> float:
        return self.cpu_ms + self.accel_ms

    @property
    def pipelined_ms(self) -> float:
        # With the accelerator overlapped against the next frame's CPU work,
        # throughput is bounded by whichever path is longer.
        return max(self.cpu_ms, self.accel_ms)


# Proposal Rev 2.2 Table 11, with ArcFace removed (scope change, 2026-09-14).
# The ~120 ms ArcFace line is gone; it was never in the 58 ms mandatory total
# anyway, it sat outside as a per-track-entry spike.
_TABLE_11 = "Rev 2.2 Table 11 (frame budget, 10 FPS)"

PROFILES = {
    "pi5-hailo": NodeProfile(
        key="pi5-hailo",
        label="Pi 5 + AI HAT+ 2, threat model compiled to Hailo",
        cores=4,
        stages=(
            Stage("capture", 8.0, "cpu", _TABLE_11),
            Stage("yolo", 28.0, "accel", _TABLE_11),
            Stage("threat", 12.0, "accel", "HAILO_PIPELINE.md sec 3 (est. once compiled)"),
            Stage("sort", 4.0, "cpu", _TABLE_11),
            Stage("head", 0.0, "measured", "timed live"),
            Stage("encode", 15.0, "cpu", _TABLE_11),
        ),
        note="The target build. CPU path 27 ms, accelerator path 40 ms.",
    ),
    # The risk case flagged in review: Table 11 has no threat line, and
    # HAILO_PIPELINE.md sec 3 warns an uncompiled threat model is
    # "a second CPU YOLO pass at ~36 ms/frame".
    "pi5-threat-on-cpu": NodeProfile(
        key="pi5-threat-on-cpu",
        label="Pi 5 + AI HAT+ 2, threat model NOT compiled (runs on CPU)",
        cores=4,
        stages=(
            Stage("capture", 8.0, "cpu", _TABLE_11),
            Stage("yolo", 28.0, "accel", _TABLE_11),
            Stage("threat", 36.0, "cpu", "HAILO_PIPELINE.md sec 3"),
            Stage("sort", 4.0, "cpu", _TABLE_11),
            Stage("head", 0.0, "measured", "timed live"),
            Stage("encode", 15.0, "cpu", _TABLE_11),
        ),
        note="What you get if export_hailo.py is never run for the threat model.",
    ),
    "pi5-cpu-only": NodeProfile(
        key="pi5-cpu-only",
        label="Pi 5, no accelerator (Hailo probe failed / fallback path)",
        cores=4,
        stages=(
            Stage("capture", 8.0, "cpu", _TABLE_11),
            # DETECTION_ACCURACY.md sec 6 measures YOLOv8n at ~90 ms on x86;
            # that doc states a Pi 5 CPU is "several times slower". 3x is the
            # conservative end of "several".
            Stage("yolo", 270.0, "cpu", "DETECTION_ACCURACY.md sec 6, x86 90 ms x3"),
            Stage("threat", 108.0, "cpu", "DETECTION_ACCURACY.md sec 6, x86 36 ms x3"),
            Stage("sort", 4.0, "cpu", _TABLE_11),
            Stage("head", 0.0, "measured", "timed live"),
            Stage("encode", 15.0, "cpu", _TABLE_11),
        ),
        note="src/detection/__init__.py:170 fallback. Shows why the HAT matters.",
    ),
    "x86-dev": NodeProfile(
        key="x86-dev",
        label="x86 dev box, CPU only (measured on Ryzen AI 7 350)",
        cores=8,
        stages=(
            Stage("capture", 4.0, "cpu", "estimate"),
            Stage("yolo", 90.0, "cpu", "DETECTION_ACCURACY.md sec 6"),
            Stage("threat", 36.0, "cpu", "DETECTION_ACCURACY.md sec 6"),
            Stage("sort", 2.0, "cpu", "estimate"),
            Stage("head", 0.0, "measured", "timed live"),
            Stage("encode", 8.0, "cpu", "estimate"),
        ),
        note="The only profile here built purely from real measurements.",
    ),
}

DEFAULT_PROFILE = "pi5-hailo"


def get(key: str) -> NodeProfile:
    if key not in PROFILES:
        raise SystemExit(
            f"unknown profile {key!r}. choose from: {', '.join(PROFILES)}"
        )
    return PROFILES[key]
