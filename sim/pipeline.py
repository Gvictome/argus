"""
The per-frame detection loop.

The one thing that makes this simulator worth running rather than reading
off a spreadsheet: CPU stages burn a real core, accelerator stages do not.

  cpu    -> a calibrated numpy matmul loop. Real cycles, real memory
            traffic, releases the GIL. Contends with the federated trainer
            exactly the way the real pipeline would.
  accel  -> time.sleep(). The Hailo-10H works over PCIe and the CPU is free
            while it does, so sleeping is the honest model.

If CPU stages were also sleeps, this would show no FPS impact from a
concurrent training round, which is the entire question being asked.
"""

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

_UNIT_A: Optional[np.ndarray] = None
_UNIT_B: Optional[np.ndarray] = None
_UNIT_MS: float = 0.0


def calibrate() -> float:
    """Size a matmul so one iteration is a known, small slice of a ms."""
    global _UNIT_A, _UNIT_B, _UNIT_MS
    rng = np.random.default_rng(0)
    n = 48
    dt = 0.0
    A = B = None
    while n < 2048:
        A = rng.random((n, n), dtype=np.float32)
        B = rng.random((n, n), dtype=np.float32)
        A @ B  # warm
        t = time.perf_counter()
        for _ in range(3):
            A @ B
        dt = (time.perf_counter() - t) / 3
        if dt >= 3e-4:
            break
        n = int(n * 1.35)
    _UNIT_A, _UNIT_B, _UNIT_MS = A, B, dt * 1000.0
    return _UNIT_MS


def burn(ms: float) -> None:
    """Consume `ms` of real CPU time."""
    if ms <= 0:
        return
    end = time.perf_counter() + ms / 1000.0
    while time.perf_counter() < end:
        _UNIT_A @ _UNIT_B


@dataclass
class FrameStats:
    frames: int = 0
    behind: int = 0
    latencies: List[float] = field(default_factory=list)
    stage_ms: Dict[str, float] = field(default_factory=dict)
    detections: Dict[int, int] = field(default_factory=dict)

    def fps_over(self, window: int = 30) -> float:
        if len(self.latencies) < 2:
            return 0.0
        recent = self.latencies[-window:]
        return 1000.0 / (sum(recent) / len(recent))


class Pipeline:
    """One node's frame loop."""

    def __init__(self, profile, head, feed, target_fps, pipelined=False):
        self.profile = profile
        self.head = head
        self.feed = feed  # callable -> (feature_vector, true_label)
        self.target_fps = target_fps
        self.pipelined = pipelined
        self.stats = FrameStats()
        self._stop = False
        for s in profile.stages:
            self.stats.stage_ms[s.name] = 0.0

    def stop(self):
        self._stop = True

    def _run_frame(self) -> float:
        t0 = time.perf_counter()
        accel_total = sum(s.ms for s in self.profile.stages if s.unit == "accel")

        # Pipelined: the accelerator overlaps the CPU path, so throughput is
        # bounded by the longer of the two rather than their sum.
        accel_done = None
        if self.pipelined and accel_total > 0:
            accel_done = time.perf_counter() + accel_total / 1000.0

        n = self.stats.frames + 1
        for s in self.profile.stages:
            st = time.perf_counter()
            if s.unit == "cpu":
                burn(s.ms)
            elif s.unit == "accel":
                if not self.pipelined:
                    time.sleep(s.ms / 1000.0)
            else:
                # The federated head, timed for real rather than budgeted.
                x, _truth = self.feed()
                pred = self.head.infer_one(x)
                self.stats.detections[pred] = self.stats.detections.get(pred, 0) + 1
            el = (time.perf_counter() - st) * 1000.0
            self.stats.stage_ms[s.name] += (el - self.stats.stage_ms[s.name]) / n

        if accel_done is not None:
            rem = accel_done - time.perf_counter()
            if rem > 0:
                time.sleep(rem)

        return (time.perf_counter() - t0) * 1000.0

    def run(self, seconds: float, on_tick: Optional[Callable] = None) -> FrameStats:
        period = 1.0 / self.target_fps
        start = time.perf_counter()
        deadline = start
        last_tick = start

        while not self._stop and (time.perf_counter() - start) < seconds:
            lat = self._run_frame()
            self.stats.frames += 1
            self.stats.latencies.append(lat)

            deadline += period
            now = time.perf_counter()
            if now < deadline:
                time.sleep(deadline - now)
            else:
                self.stats.behind += 1
                deadline = now  # do not accumulate scheduling debt

            if on_tick and (time.perf_counter() - last_tick) >= 1.0:
                last_tick = time.perf_counter()
                on_tick(self.stats)

        return self.stats
