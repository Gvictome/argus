"""
Thermal monitor for the node.

The enclosure has to keep the Pi 5 below its throttle point while it
detects all day, and the only way to show that is to measure it. This
samples the SoC temperature, the firmware's throttle flags, the fan and
the CPU clock every THERMAL_SAMPLE_S, next to the detection rate, and:

- keeps the last 24 h in memory for /api/system/thermal/history, and
- appends every sample to data/thermal/thermal-YYYY-MM-DD.csv, labelled
  with THERMAL_ENCLOSURE, so a soak test leaves a record per enclosure.

The Pi 5 starts lowering its clock at 80 C and throttles hard at 85 C,
so peak temperature and any throttle flag during a run are the numbers
that matter. Off a Pi (the laptop, CI) every reading is None and the
monitor still runs, so the API shape never changes.
"""

from __future__ import annotations

import csv
import glob
import logging
import subprocess
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Deque, Dict, List, Optional

logger = logging.getLogger(__name__)

SOC_TEMP_PATH = "/sys/class/thermal/thermal_zone0/temp"
CPU_FREQ_PATH = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq"
FAN_GLOB = "/sys/devices/platform/cooling_fan/hwmon/hwmon*/fan1_input"

# Pi 5 limits, for the dashboard and the soak pass/fail.
SOFT_LIMIT_C = 80.0
HARD_LIMIT_C = 85.0

# vcgencmd get_throttled bits. Low bits are "now", bits 16+ are "has
# happened since boot".
THROTTLE_BITS = {
    0: "under_voltage",
    1: "freq_capped",
    2: "throttled",
    3: "soft_temp_limit",
}

CSV_FIELDS = [
    "ts", "node", "enclosure", "soc_temp_c", "cpu_mhz", "fan_rpm",
    "throttled_hex", "throttled_now", "detect_fps",
]


def _read_number(path: str) -> Optional[float]:
    try:
        return float(Path(path).read_text().strip())
    except (OSError, ValueError):
        return None


def read_soc_temp_c() -> Optional[float]:
    milli = _read_number(SOC_TEMP_PATH)
    return round(milli / 1000.0, 1) if milli is not None else None


def read_cpu_mhz() -> Optional[int]:
    khz = _read_number(CPU_FREQ_PATH)
    return int(khz / 1000) if khz is not None else None


def read_fan_rpm() -> Optional[int]:
    for path in glob.glob(FAN_GLOB):
        rpm = _read_number(path)
        if rpm is not None:
            return int(rpm)
    return None


def read_throttled_raw() -> Optional[int]:
    """The firmware's throttle word, or None when vcgencmd is not there."""
    try:
        out = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True,
                             text=True, timeout=2).stdout
        return int(out.strip().split("=")[1], 16)
    except (OSError, IndexError, ValueError, subprocess.SubprocessError):
        return None


def decode_throttled(raw: Optional[int]) -> Dict[str, object]:
    """Split the throttle word into what is happening now and since boot."""
    if raw is None:
        return {"hex": None, "now": [], "since_boot": []}
    return {
        "hex": f"0x{raw:x}",
        "now": [name for bit, name in THROTTLE_BITS.items() if raw & (1 << bit)],
        "since_boot": [name for bit, name in THROTTLE_BITS.items() if raw & (1 << (bit + 16))],
    }


@dataclass
class ThermalSample:
    ts: float
    soc_temp_c: Optional[float]
    cpu_mhz: Optional[int]
    fan_rpm: Optional[int]
    throttled_hex: Optional[str]
    throttled_now: List[str]
    detect_fps: Optional[float]


class ThermalMonitor:
    """Samples on its own thread; readers are injectable for tests."""

    def __init__(
        self,
        node: str,
        enclosure: str,
        interval_s: float = 30.0,
        log_dir: Optional[Path] = None,
        fps_source: Optional[Callable[[], Optional[float]]] = None,
        readers: Optional[Dict[str, Callable]] = None,
        history_hours: float = 24.0,
    ) -> None:
        self.node = node
        self.enclosure = enclosure
        self.interval_s = max(1.0, float(interval_s))
        self.log_dir = Path(log_dir) if log_dir else None
        self.fps_source = fps_source
        r = readers or {}
        self._temp = r.get("temp", read_soc_temp_c)
        self._mhz = r.get("mhz", read_cpu_mhz)
        self._fan = r.get("fan", read_fan_rpm)
        self._throttle = r.get("throttle", read_throttled_raw)
        self.samples: Deque[ThermalSample] = deque(
            maxlen=max(1, int(history_hours * 3600 / self.interval_s)))
        self.started = time.time()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_raw: Optional[int] = None

    # ---- sampling -------------------------------------------------------

    def sample(self) -> ThermalSample:
        raw = self._throttle()
        self._last_raw = raw
        fps = None
        if self.fps_source is not None:
            try:
                fps = self.fps_source()
            except Exception:  # a dead worker must not stop the thermal log
                fps = None
        s = ThermalSample(
            ts=time.time(),
            soc_temp_c=self._temp(),
            cpu_mhz=self._mhz(),
            fan_rpm=self._fan(),
            throttled_hex=decode_throttled(raw)["hex"],
            throttled_now=decode_throttled(raw)["now"],
            detect_fps=round(fps, 1) if fps is not None else None,
        )
        with self._lock:
            self.samples.append(s)
        self._write_csv(s)
        return s

    def _write_csv(self, s: ThermalSample) -> None:
        if self.log_dir is None:
            return
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            path = self.log_dir / f"thermal-{datetime.fromtimestamp(s.ts):%Y-%m-%d}.csv"
            new = not path.exists()
            with path.open("a", newline="") as fh:
                w = csv.writer(fh)
                if new:
                    w.writerow(CSV_FIELDS)
                w.writerow([
                    datetime.fromtimestamp(s.ts).isoformat(timespec="seconds"), self.node,
                    self.enclosure, s.soc_temp_c, s.cpu_mhz, s.fan_rpm, s.throttled_hex,
                    "|".join(s.throttled_now), s.detect_fps,
                ])
        except OSError as exc:
            logger.warning("thermal: could not write %s: %s", self.log_dir, exc)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.sample()
            except Exception as exc:
                logger.warning("thermal: sample failed: %s", exc)
            self._stop.wait(self.interval_s)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="thermal", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    # ---- reporting ------------------------------------------------------

    def history(self, minutes: float = 120.0) -> List[dict]:
        cutoff = time.time() - minutes * 60
        with self._lock:
            return [asdict(s) for s in self.samples if s.ts >= cutoff]

    def summary(self, minutes: float = 120.0) -> dict:
        rows = self.history(minutes)
        temps = [r["soc_temp_c"] for r in rows if r["soc_temp_c"] is not None]
        throttled = [r for r in rows if r["throttled_now"]]
        return {
            "window_min": minutes,
            "samples": len(rows),
            "temp_min_c": min(temps) if temps else None,
            "temp_max_c": max(temps) if temps else None,
            "temp_avg_c": round(sum(temps) / len(temps), 1) if temps else None,
            "throttled_samples": len(throttled),
            # Pass = never reached the soft limit and never throttled.
            "pass": (bool(temps) and max(temps) < SOFT_LIMIT_C and not throttled),
        }

    def status(self) -> dict:
        with self._lock:
            last = self.samples[-1] if self.samples else None
        return {
            "node": self.node,
            "enclosure": self.enclosure,
            "interval_s": self.interval_s,
            "running_since": datetime.fromtimestamp(self.started).isoformat(timespec="seconds"),
            "limits_c": {"soft": SOFT_LIMIT_C, "hard": HARD_LIMIT_C},
            "current": asdict(last) if last else None,
            "throttled": decode_throttled(self._last_raw),
            "summary": self.summary(),
        }

    def csv_text(self, minutes: float = 120.0) -> str:
        import io
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(CSV_FIELDS)
        for r in self.history(minutes):
            w.writerow([
                datetime.fromtimestamp(r["ts"]).isoformat(timespec="seconds"), self.node,
                self.enclosure, r["soc_temp_c"], r["cpu_mhz"], r["fan_rpm"],
                r["throttled_hex"], "|".join(r["throttled_now"]), r["detect_fps"],
            ])
        return buf.getvalue()
