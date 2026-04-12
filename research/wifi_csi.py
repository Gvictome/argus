"""
Argus WiFi CSI Motion Detection Module
========================================
Detects human presence, motion, and estimates skeleton pose using
WiFi Channel State Information (CSI) signal analysis.

CSI captures how WiFi signals propagate between transmitter and receiver.
Human bodies reflect/scatter WiFi signals, causing measurable changes in
CSI amplitude and phase across OFDM subcarriers. By analyzing these
changes we can detect:
  - Presence (someone is in the room)
  - Motion (someone is moving)
  - Activity (walking, sitting, falling)
  - Skeleton pose (experimental — requires ML model)

Backends:
  - ESP32 CSI: ESP32 dev boards as sensor nodes, send CSI over UDP to Pi
  - Nexmon CSI: Extract CSI from Pi's Broadcom WiFi chip (requires patched firmware)
  - Mock: Simulated CSI data for development/testing

References:
  - WiFi sensing overview: spatialintelligence.ai/p/your-wifi-can-see-you-heres-how
  - ESP32 CSI: github.com/StevenMHernandez/ESP32-CSI-Tool
  - Nexmon CSI: github.com/seemoo-lab/nexmon_csi
  - DensePose from WiFi: arxiv.org/abs/2301.00250
"""

import threading
import time
import logging
import json
import os
import struct
import socket
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict

import numpy as np

try:
    from scipy.signal import butter, filtfilt
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

logger = logging.getLogger("argus.wifi_csi")


# ─── Data Structures ────────────────────────────────────────────────

@dataclass
class CSIFrame:
    """Single CSI measurement frame."""
    timestamp: float
    mac_address: str
    rssi: int
    subcarrier_count: int
    amplitude: np.ndarray   # shape: (subcarrier_count,)
    phase: np.ndarray        # shape: (subcarrier_count,)
    bandwidth: int = 20      # MHz (20, 40, 80)
    channel: int = 6

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "mac": self.mac_address,
            "rssi": self.rssi,
            "subcarriers": self.subcarrier_count,
            "mean_amplitude": float(np.mean(self.amplitude)),
            "amplitude_var": float(np.var(self.amplitude)),
            "bandwidth": self.bandwidth,
            "channel": self.channel,
        }


@dataclass
class MotionEvent:
    """Detected motion event from CSI analysis."""
    timestamp: float
    confidence: float        # 0.0 - 1.0
    motion_level: float      # magnitude of detected motion
    zone: str = "unknown"    # estimated zone/area
    activity: str = "motion" # motion, walking, stationary, breathing
    person_count: int = 1
    skeleton: Optional[dict] = None  # skeleton keypoints if available

    def to_dict(self) -> dict:
        d = {
            "timestamp": self.timestamp,
            "confidence": round(self.confidence, 3),
            "motion_level": round(self.motion_level, 3),
            "zone": self.zone,
            "activity": self.activity,
            "person_count": self.person_count,
            "source": "wifi_csi",
        }
        if self.skeleton:
            d["skeleton"] = self.skeleton
        return d


# ─── CSI Backend Interface ──────────────────────────────────────────

class CSIBackend(ABC):
    """Abstract CSI data source."""

    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def stop(self):
        pass

    @abstractmethod
    def get_frame(self) -> Optional[CSIFrame]:
        """Get next CSI frame (non-blocking, returns None if no data)."""
        pass

    @abstractmethod
    def is_running(self) -> bool:
        pass


class ESP32Backend(CSIBackend):
    """Receive CSI data from ESP32 nodes over UDP.

    ESP32 CSI Tool format: each UDP packet contains:
      - MAC address (6 bytes)
      - RSSI (1 byte signed)
      - Channel (1 byte)
      - Subcarrier count (2 bytes)
      - CSI data (subcarrier_count * 2 bytes, interleaved I/Q)
    """

    def __init__(self, listen_port: int = 5500, listen_addr: str = "0.0.0.0"):
        self.listen_port = listen_port
        self.listen_addr = listen_addr
        self._sock: Optional[socket.socket] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._buffer: deque = deque(maxlen=500)
        self._lock = threading.Lock()

    def start(self):
        if self._running:
            return
        self._running = True
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.settimeout(1.0)
        self._sock.bind((self.listen_addr, self.listen_port))
        self._thread = threading.Thread(target=self._receive_loop, daemon=True)
        self._thread.start()
        logger.info(f"ESP32 CSI backend listening on {self.listen_addr}:{self.listen_port}")

    def _receive_loop(self):
        while self._running:
            try:
                data, addr = self._sock.recvfrom(4096)
                frame = self._parse_esp32_packet(data, addr)
                if frame:
                    with self._lock:
                        self._buffer.append(frame)
            except socket.timeout:
                continue
            except Exception as e:
                logger.error(f"ESP32 receive error: {e}")

    def _parse_esp32_packet(self, data: bytes, addr: tuple) -> Optional[CSIFrame]:
        """Parse ESP32 CSI Tool UDP packet."""
        try:
            if len(data) < 10:
                return None

            # Parse header
            mac = ":".join(f"{b:02x}" for b in data[0:6])
            rssi = struct.unpack("b", data[6:7])[0]
            channel = data[7]
            sc_count = struct.unpack("<H", data[8:10])[0]

            if len(data) < 10 + sc_count * 2:
                return None

            # Parse I/Q data into amplitude and phase
            iq_data = data[10:10 + sc_count * 2]
            amplitudes = []
            phases = []
            for i in range(0, len(iq_data), 2):
                real = struct.unpack("b", iq_data[i:i+1])[0]
                imag = struct.unpack("b", iq_data[i+1:i+2])[0]
                amp = np.sqrt(real**2 + imag**2)
                phase = np.arctan2(imag, real)
                amplitudes.append(amp)
                phases.append(phase)

            return CSIFrame(
                timestamp=time.time(),
                mac_address=mac,
                rssi=rssi,
                subcarrier_count=sc_count,
                amplitude=np.array(amplitudes, dtype=np.float32),
                phase=np.array(phases, dtype=np.float32),
                channel=channel,
            )
        except Exception as e:
            logger.debug(f"Parse error: {e}")
            return None

    def get_frame(self) -> Optional[CSIFrame]:
        with self._lock:
            if self._buffer:
                return self._buffer.popleft()
        return None

    def is_running(self) -> bool:
        return self._running

    def stop(self):
        self._running = False
        if self._sock:
            self._sock.close()
        if self._thread:
            self._thread.join(timeout=3)


class MockBackend(CSIBackend):
    """Simulated CSI data for development and testing.

    Generates realistic-looking CSI amplitude patterns:
    - Baseline: stable with small noise
    - Motion events: periodic amplitude fluctuations
    - Breathing: subtle low-frequency modulation
    """

    def __init__(self, subcarrier_count: int = 52, frame_rate: float = 100.0):
        self.sc_count = subcarrier_count
        self.frame_interval = 1.0 / frame_rate
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._buffer: deque = deque(maxlen=500)
        self._lock = threading.Lock()
        self._start_time = 0.0
        # Simulation state
        self._motion_active = False
        self._motion_start = 0.0
        self._motion_duration = 5.0
        self._motion_interval = 15.0  # seconds between motion events

    def start(self):
        if self._running:
            return
        self._running = True
        self._start_time = time.time()
        self._thread = threading.Thread(target=self._generate_loop, daemon=True)
        self._thread.start()
        logger.info("Mock CSI backend started")

    def _generate_loop(self):
        while self._running:
            t = time.time() - self._start_time

            # Determine if motion is happening (periodic bursts)
            cycle_pos = t % self._motion_interval
            self._motion_active = cycle_pos < self._motion_duration

            # Base amplitude: stable per-subcarrier profile
            base = 15.0 + 5.0 * np.sin(np.linspace(0, 2 * np.pi, self.sc_count))

            # Add noise
            noise = np.random.randn(self.sc_count) * 0.5

            # Add motion signature: correlated fluctuations across subcarriers
            if self._motion_active:
                motion_freq = 2.0  # Hz (walking)
                motion_sig = 3.0 * np.sin(
                    2 * np.pi * motion_freq * t +
                    np.linspace(0, np.pi, self.sc_count)
                )
                # Add breathing undertone
                breathing = 0.5 * np.sin(2 * np.pi * 0.3 * t) * np.ones(self.sc_count)
                amplitude = base + motion_sig + breathing + noise
            else:
                amplitude = base + noise

            phase = np.random.uniform(-np.pi, np.pi, self.sc_count).astype(np.float32)
            if self._motion_active:
                # Phase shifts correlate during motion
                phase_shift = 0.5 * np.sin(2 * np.pi * 2.0 * t + np.linspace(0, 2*np.pi, self.sc_count))
                phase = (phase + phase_shift).astype(np.float32)

            frame = CSIFrame(
                timestamp=time.time(),
                mac_address="aa:bb:cc:dd:ee:ff",
                rssi=-45 + int(np.random.randn() * 3),
                subcarrier_count=self.sc_count,
                amplitude=np.abs(amplitude).astype(np.float32),
                phase=phase,
                channel=6,
            )

            with self._lock:
                self._buffer.append(frame)

            time.sleep(self.frame_interval)

    def get_frame(self) -> Optional[CSIFrame]:
        with self._lock:
            if self._buffer:
                return self._buffer.popleft()
        return None

    def is_running(self) -> bool:
        return self._running

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)


# ─── Signal Processing Pipeline ────────────────────────────────────

class CSIProcessor:
    """Process raw CSI frames into motion detection signals.

    Pipeline:
      1. Collect sliding window of CSI amplitude frames
      2. Apply bandpass filter to isolate human motion frequencies
      3. Compute variance across subcarriers (motion indicator)
      4. Apply Hampel filter for outlier removal
      5. Threshold to detect motion events
    """

    def __init__(self, window_size: int = 100, motion_threshold: float = 2.0):
        self.window_size = window_size
        self.motion_threshold = motion_threshold
        self._amplitude_window: deque = deque(maxlen=window_size)
        self._motion_scores: deque = deque(maxlen=500)
        self._baseline: Optional[np.ndarray] = None
        self._baseline_var: Optional[np.ndarray] = None
        self._calibration_frames = 50
        self._frame_count = 0

        # Bandpass filter coefficients for human motion (1-10 Hz at ~100 Hz sample rate)
        if HAS_SCIPY:
            self._b_motion, self._a_motion = butter(4, [1.0, 10.0], btype='band', fs=100.0)
            self._b_breath, self._a_breath = butter(4, [0.1, 0.5], btype='band', fs=100.0)
        else:
            self._b_motion = self._a_motion = None
            self._b_breath = self._a_breath = None

    def process_frame(self, frame: CSIFrame) -> Optional[float]:
        """Process a single CSI frame. Returns motion score or None during calibration."""
        self._frame_count += 1
        self._amplitude_window.append(frame.amplitude)

        # Calibration phase: establish baseline
        if self._frame_count <= self._calibration_frames:
            if self._frame_count == self._calibration_frames:
                window = np.array(list(self._amplitude_window))
                self._baseline = np.mean(window, axis=0)
                self._baseline_var = np.var(window, axis=0) + 1e-6
                logger.info(f"CSI calibration complete ({self._calibration_frames} frames)")
            return None

        if self._baseline is None:
            return None

        # Compute deviation from baseline (normalized)
        deviation = (frame.amplitude - self._baseline) / np.sqrt(self._baseline_var)

        # Motion score: mean absolute deviation across subcarriers
        motion_score = float(np.mean(np.abs(deviation)))

        # Apply temporal smoothing if we have enough history
        if len(self._amplitude_window) >= 20 and HAS_SCIPY:
            try:
                window = np.array(list(self._amplitude_window))
                # Take mean subcarrier amplitude over time
                mean_amp = np.mean(window, axis=1)
                # Bandpass filter for motion frequencies
                if len(mean_amp) >= 15:  # need enough samples for filter
                    filtered = filtfilt(self._b_motion, self._a_motion, mean_amp,
                                        padlen=min(12, len(mean_amp)-1))
                    motion_score = float(np.std(filtered[-20:]))
            except Exception:
                pass  # fall back to simple score

        self._motion_scores.append(motion_score)
        return motion_score

    def detect_motion(self, score: float) -> Tuple[bool, float]:
        """Determine if motion score indicates real motion.

        Returns (is_motion, confidence).
        """
        if score > self.motion_threshold:
            # Scale confidence: threshold=0.5, 2x threshold=0.9, 3x=0.95
            confidence = min(0.95, 0.5 + 0.4 * (score / self.motion_threshold - 1))
            return True, confidence
        return False, 0.0

    def estimate_activity(self, window_scores: List[float]) -> str:
        """Classify activity from recent motion scores."""
        if len(window_scores) < 10:
            return "unknown"

        recent = np.array(window_scores[-30:])
        mean_score = np.mean(recent)
        std_score = np.std(recent)

        if mean_score < 0.5:
            return "stationary"
        elif mean_score < 1.0 and std_score < 0.3:
            return "breathing"
        elif mean_score < 2.0:
            return "subtle_motion"
        elif std_score > 1.0:
            return "walking"
        else:
            return "motion"

    def estimate_person_count(self) -> int:
        """Rough person count from CSI pattern complexity.

        This is a very rough heuristic — proper implementation would
        use a trained ML model. We estimate based on the number of
        distinct frequency components in the motion signal.
        """
        if len(self._amplitude_window) < 50:
            return 0

        window = np.array(list(self._amplitude_window))
        mean_amp = np.mean(window, axis=1)

        # FFT to find frequency components
        fft = np.abs(np.fft.rfft(mean_amp - np.mean(mean_amp)))
        freqs = np.fft.rfftfreq(len(mean_amp), d=0.01)  # 100 Hz sample rate

        # Count significant peaks in human motion range (0.5-5 Hz)
        motion_mask = (freqs >= 0.5) & (freqs <= 5.0)
        motion_fft = fft[motion_mask]

        if len(motion_fft) == 0:
            return 0

        threshold = np.mean(motion_fft) + 2 * np.std(motion_fft)
        peaks = np.sum(motion_fft > threshold)

        # Rough: 1-2 peaks = 1 person, 3-5 = 2, 6+ = 3+
        if peaks == 0:
            return 0
        elif peaks <= 2:
            return 1
        elif peaks <= 5:
            return 2
        else:
            return min(peaks // 3, 5)

    def get_recent_scores(self, n: int = 50) -> List[float]:
        return list(self._motion_scores)[-n:]


# ─── Skeleton Estimation (Stub) ─────────────────────────────────────

class SkeletonEstimator:
    """WiFi-based skeleton pose estimation.

    This is a placeholder for ML-based approaches like:
    - DensePose from WiFi (Meta, 2023): Uses 3 WiFi transmitters to
      estimate UV coordinates of human body surface
    - WiPose: WiFi-based 3D pose estimation using CSI
    - Person-in-WiFi: Real-time pose estimation from WiFi signals

    For production, this would load a pre-trained model and run inference
    on CSI features. For now, it returns a simplified skeleton based on
    motion detection heuristics.
    """

    KEYPOINTS = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle",
    ]

    def __init__(self):
        self._model_loaded = False
        self._model_path = None

    def load_model(self, path: str) -> bool:
        """Load skeleton estimation model. Returns True if successful."""
        if os.path.exists(path):
            self._model_path = path
            self._model_loaded = True
            logger.info(f"Skeleton model loaded from {path}")
            return True
        logger.warning(f"Skeleton model not found at {path} — using heuristic mode")
        return False

    def estimate(self, csi_features: np.ndarray, motion_detected: bool) -> Optional[dict]:
        """Estimate skeleton from CSI features.

        Returns skeleton dict with keypoints or None if no person detected.
        """
        if not motion_detected:
            return None

        if self._model_loaded:
            return self._model_inference(csi_features)

        # Heuristic mode: return a generic standing pose with noise
        # This signals "a person is here" without precise keypoints
        return self._heuristic_skeleton()

    def _model_inference(self, features: np.ndarray) -> Optional[dict]:
        """Run actual model inference. Placeholder for real implementation."""
        # TODO: Load ONNX/TFLite model and run inference
        return self._heuristic_skeleton()

    def _heuristic_skeleton(self) -> dict:
        """Generate a rough skeleton estimate based on detection."""
        # Normalized coordinates (0-1) for a standing person at center
        base_skeleton = {
            "nose": [0.50, 0.10],
            "left_shoulder": [0.42, 0.22],
            "right_shoulder": [0.58, 0.22],
            "left_elbow": [0.35, 0.38],
            "right_elbow": [0.65, 0.38],
            "left_wrist": [0.33, 0.50],
            "right_wrist": [0.67, 0.50],
            "left_hip": [0.44, 0.55],
            "right_hip": [0.56, 0.55],
            "left_knee": [0.43, 0.72],
            "right_knee": [0.57, 0.72],
            "left_ankle": [0.42, 0.90],
            "right_ankle": [0.58, 0.90],
        }

        # Add small random perturbation (WiFi resolution isn't perfect)
        noisy = {}
        for joint, coords in base_skeleton.items():
            noisy[joint] = [
                round(coords[0] + np.random.randn() * 0.03, 3),
                round(coords[1] + np.random.randn() * 0.03, 3),
            ]

        return {
            "keypoints": noisy,
            "confidence": 0.3,  # low confidence for heuristic
            "model": "heuristic",
        }


# ─── Main CSI Engine ────────────────────────────────────────────────

class WiFiCSIEngine:
    """Main WiFi CSI processing engine.

    Ties together the backend, signal processor, and skeleton estimator.
    Runs in a background thread, produces MotionEvents.
    """

    def __init__(self, backend: str = "mock", config: Optional[dict] = None):
        config = config or {}

        # Select backend
        if backend == "esp32":
            self._backend = ESP32Backend(
                listen_port=config.get("esp32_port", 5500),
                listen_addr=config.get("esp32_addr", "0.0.0.0"),
            )
        elif backend == "mock":
            self._backend = MockBackend(
                subcarrier_count=config.get("subcarriers", 52),
                frame_rate=config.get("frame_rate", 100.0),
            )
        else:
            raise ValueError(f"Unknown CSI backend: {backend}")

        self._processor = CSIProcessor(
            window_size=config.get("window_size", 100),
            motion_threshold=config.get("motion_threshold", 2.0),
        )
        self._skeleton = SkeletonEstimator()
        self._events: deque = deque(maxlen=200)
        self._events_lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._prometheus_metrics = None

        # State
        self.motion_detected = False
        self.current_activity = "unknown"
        self.person_count = 0
        self.mean_amplitude = 0.0
        self.last_motion_time = 0.0
        self._frames_processed = 0

        # Load skeleton model if available
        model_path = config.get("skeleton_model", "")
        if model_path:
            self._skeleton.load_model(model_path)

    def set_prometheus_metrics(self, motion_gauge, amplitude_gauge, persons_gauge):
        """Accept Prometheus gauge references."""
        self._prometheus_metrics = {
            "motion": motion_gauge,
            "amplitude": amplitude_gauge,
            "persons": persons_gauge,
        }

    def start(self):
        if self._running:
            return
        self._running = True
        self._backend.start()
        self._thread = threading.Thread(target=self._processing_loop, daemon=True)
        self._thread.start()
        logger.info("WiFi CSI engine started")

    def _processing_loop(self):
        while self._running:
            frame = self._backend.get_frame()
            if frame is None:
                time.sleep(0.005)  # 5ms idle wait
                continue

            self._frames_processed += 1

            # Process CSI frame
            score = self._processor.process_frame(frame)
            if score is None:
                continue  # Still calibrating

            # Update state
            self.mean_amplitude = float(np.mean(frame.amplitude))

            # Detect motion
            is_motion, confidence = self._processor.detect_motion(score)
            self.motion_detected = is_motion

            if is_motion:
                self.last_motion_time = time.time()

            # Classify activity
            scores = self._processor.get_recent_scores(30)
            self.current_activity = self._processor.estimate_activity(scores)

            # Estimate person count
            self.person_count = self._processor.estimate_person_count()

            # Update Prometheus metrics (every 10th frame to reduce overhead)
            if self._prometheus_metrics and self._frames_processed % 10 == 0:
                self._prometheus_metrics["motion"].set(1 if is_motion else 0)
                self._prometheus_metrics["amplitude"].set(self.mean_amplitude)
                self._prometheus_metrics["persons"].set(self.person_count)

            # Generate motion event (throttled to avoid flooding)
            if is_motion and self._frames_processed % 50 == 0:
                # Get CSI features for skeleton estimation
                csi_features = frame.amplitude
                skeleton = self._skeleton.estimate(csi_features, is_motion)

                event = MotionEvent(
                    timestamp=time.time(),
                    confidence=confidence,
                    motion_level=score,
                    activity=self.current_activity,
                    person_count=self.person_count,
                    skeleton=skeleton,
                )

                with self._events_lock:
                    self._events.append(event)

                logger.debug(f"Motion event: {self.current_activity} "
                             f"(conf={confidence:.2f}, persons={self.person_count})")

    def stop(self):
        self._running = False
        self._backend.stop()
        if self._thread:
            self._thread.join(timeout=5)

    def get_recent_events(self, limit: int = 20) -> List[dict]:
        with self._events_lock:
            events = list(self._events)[-limit:]
        return [e.to_dict() for e in events]


# ─── Module-level API (called from server.py) ───────────────────────

_engine: Optional[WiFiCSIEngine] = None
_CSI_CONFIG_FILE = os.path.join(os.path.dirname(__file__), "csi_config.json")

DEFAULT_CSI_CONFIG = {
    "backend": "mock",  # "mock", "esp32", or "nexmon"
    "esp32_port": 5500,
    "subcarriers": 52,
    "frame_rate": 100.0,
    "window_size": 100,
    "motion_threshold": 2.0,
    "skeleton_model": "",
}


def _load_csi_config() -> dict:
    if os.path.exists(_CSI_CONFIG_FILE):
        with open(_CSI_CONFIG_FILE, "r") as f:
            return {**DEFAULT_CSI_CONFIG, **json.load(f)}
    return DEFAULT_CSI_CONFIG.copy()


def init_wifi_csi(prometheus_metrics: Optional[dict] = None) -> WiFiCSIEngine:
    """Initialize and start the WiFi CSI engine."""
    global _engine

    config = _load_csi_config()
    _engine = WiFiCSIEngine(backend=config["backend"], config=config)

    if prometheus_metrics:
        _engine.set_prometheus_metrics(
            prometheus_metrics["motion"],
            prometheus_metrics["amplitude"],
            prometheus_metrics["persons"],
        )

    _engine.start()
    logger.info(f"WiFi CSI initialized (backend={config['backend']})")
    return _engine


def get_csi_status() -> dict:
    """Get current WiFi CSI status for API."""
    if _engine is None:
        return {"wifi_csi_enabled": False}
    return {
        "wifi_csi_enabled": True,
        "motion_detected": _engine.motion_detected,
        "current_activity": _engine.current_activity,
        "person_count": _engine.person_count,
        "mean_amplitude": round(_engine.mean_amplitude, 2),
        "frames_processed": _engine._frames_processed,
        "backend": _load_csi_config().get("backend", "unknown"),
    }


def get_recent_detections(limit: int = 20) -> List[dict]:
    """Get recent motion detection events."""
    if _engine is None:
        return []
    return _engine.get_recent_events(limit)


def is_motion_detected() -> bool:
    """Quick check if WiFi CSI currently detects motion."""
    if _engine is None:
        return False
    return _engine.motion_detected
