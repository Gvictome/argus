"""
Argus Sensor Fusion Module
===========================
Combines WiFi CSI motion detection with camera-based YOLO detections
for multi-modal presence and activity awareness.

Fusion strategy:
  1. WiFi CSI provides always-on, room-wide presence detection
  2. Camera provides high-accuracy visual identification when active
  3. Bayesian confidence merging when both sensors agree
  4. WiFi can trigger camera wake-up for power saving
  5. Camera can validate WiFi false positives

Detection states:
  - IDLE:      No sensors detect anything
  - WIFI_ONLY: WiFi CSI detects motion, camera inactive/no detection
  - CAM_ONLY:  Camera detects, WiFi not detecting (close to camera, out of WiFi range)
  - CONFIRMED: Both sensors detect presence → highest confidence
  - CONFLICT:  One detects, other contradicts → investigate
"""

import time
import threading
import logging
from dataclasses import dataclass
from typing import Optional, List, Dict
from collections import deque

logger = logging.getLogger("argus.fusion")


@dataclass
class FusedDetection:
    """Combined detection from multiple sensor modalities."""
    timestamp: float
    state: str           # IDLE, WIFI_ONLY, CAM_ONLY, CONFIRMED, CONFLICT
    confidence: float    # fused confidence 0-1
    wifi_motion: bool
    wifi_confidence: float
    wifi_activity: str
    wifi_persons: int
    camera_detected: bool
    camera_label: str
    camera_confidence: float
    person_count_estimate: int
    action: str          # "none", "alert", "record", "wake_camera", "investigate"

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "state": self.state,
            "confidence": round(self.confidence, 3),
            "wifi": {
                "motion": self.wifi_motion,
                "confidence": round(self.wifi_confidence, 3),
                "activity": self.wifi_activity,
                "persons": self.wifi_persons,
            },
            "camera": {
                "detected": self.camera_detected,
                "label": self.camera_label,
                "confidence": round(self.camera_confidence, 3),
            },
            "person_count": self.person_count_estimate,
            "action": self.action,
        }


class SensorFusion:
    """Fuses WiFi CSI and camera detection signals.

    Uses a simple Bayesian approach:
      P(person | wifi, camera) ∝ P(wifi | person) * P(camera | person) * P(person)

    With temporal smoothing to avoid rapid state changes.
    """

    # Prior probability of a person being present
    PRIOR_PERSON = 0.1

    # Likelihood of sensor detecting given person present
    P_WIFI_GIVEN_PERSON = 0.85
    P_CAM_GIVEN_PERSON = 0.95
    # False positive rates
    P_WIFI_GIVEN_NO_PERSON = 0.05
    P_CAM_GIVEN_NO_PERSON = 0.01

    # Minimum confidence thresholds
    WIFI_CONFIDENCE_THRESHOLD = 0.4
    CAMERA_CONFIDENCE_THRESHOLD = 0.5
    ALERT_CONFIDENCE_THRESHOLD = 0.7

    # Temporal smoothing
    STATE_HOLD_TIME = 5.0  # seconds to hold a state before transitioning to IDLE

    def __init__(self):
        self._state = "IDLE"
        self._last_wifi_motion = False
        self._last_wifi_time = 0.0
        self._last_wifi_confidence = 0.0
        self._last_wifi_activity = "unknown"
        self._last_wifi_persons = 0
        self._last_cam_detected = False
        self._last_cam_time = 0.0
        self._last_cam_confidence = 0.0
        self._last_cam_label = ""
        self._history: deque = deque(maxlen=100)
        self._lock = threading.Lock()

    def update_wifi(self, motion: bool, confidence: float = 0.0,
                    activity: str = "unknown", persons: int = 0):
        """Update WiFi CSI state."""
        with self._lock:
            self._last_wifi_motion = motion
            self._last_wifi_time = time.time()
            self._last_wifi_confidence = confidence
            self._last_wifi_activity = activity
            self._last_wifi_persons = persons

    def update_camera(self, detected: bool, label: str = "",
                      confidence: float = 0.0):
        """Update camera detection state."""
        with self._lock:
            self._last_cam_detected = detected
            self._last_cam_time = time.time()
            self._last_cam_confidence = confidence
            self._last_cam_label = label

    def fuse(self) -> FusedDetection:
        """Compute fused detection state."""
        with self._lock:
            now = time.time()

            # Check if sensor data is stale
            wifi_stale = (now - self._last_wifi_time) > self.STATE_HOLD_TIME
            cam_stale = (now - self._last_cam_time) > self.STATE_HOLD_TIME

            wifi_active = self._last_wifi_motion and not wifi_stale
            cam_active = self._last_cam_detected and not cam_stale

            wifi_conf = self._last_wifi_confidence if wifi_active else 0.0
            cam_conf = self._last_cam_confidence if cam_active else 0.0

            # Determine state
            if wifi_active and cam_active:
                state = "CONFIRMED"
            elif wifi_active and not cam_active:
                state = "WIFI_ONLY"
            elif cam_active and not wifi_active:
                state = "CAM_ONLY"
            else:
                state = "IDLE"

            # Bayesian fusion
            fused_confidence = self._bayesian_fuse(wifi_active, wifi_conf,
                                                    cam_active, cam_conf)

            # Person count: take max of both estimates
            person_count = max(
                self._last_wifi_persons if wifi_active else 0,
                1 if cam_active else 0,
            )

            # Determine action
            action = self._decide_action(state, fused_confidence)

            self._state = state

            detection = FusedDetection(
                timestamp=now,
                state=state,
                confidence=fused_confidence,
                wifi_motion=wifi_active,
                wifi_confidence=wifi_conf,
                wifi_activity=self._last_wifi_activity if wifi_active else "idle",
                wifi_persons=self._last_wifi_persons if wifi_active else 0,
                camera_detected=cam_active,
                camera_label=self._last_cam_label if cam_active else "",
                camera_confidence=cam_conf,
                person_count_estimate=person_count,
                action=action,
            )

            self._history.append(detection)
            return detection

    def _bayesian_fuse(self, wifi: bool, wifi_conf: float,
                       cam: bool, cam_conf: float) -> float:
        """Bayesian confidence fusion."""
        # Weighted likelihoods based on sensor confidence
        if wifi:
            p_wifi = self.P_WIFI_GIVEN_PERSON * wifi_conf
            p_wifi_neg = self.P_WIFI_GIVEN_NO_PERSON
        else:
            p_wifi = 1 - self.P_WIFI_GIVEN_PERSON
            p_wifi_neg = 1 - self.P_WIFI_GIVEN_NO_PERSON

        if cam:
            p_cam = self.P_CAM_GIVEN_PERSON * cam_conf
            p_cam_neg = self.P_CAM_GIVEN_NO_PERSON
        else:
            p_cam = 1 - self.P_CAM_GIVEN_PERSON
            p_cam_neg = 1 - self.P_CAM_GIVEN_NO_PERSON

        # Bayes' rule
        p_person = p_wifi * p_cam * self.PRIOR_PERSON
        p_no_person = p_wifi_neg * p_cam_neg * (1 - self.PRIOR_PERSON)

        total = p_person + p_no_person
        if total == 0:
            return 0.0

        return min(p_person / total, 1.0)

    def _decide_action(self, state: str, confidence: float) -> str:
        """Decide what action to take based on fused state."""
        if state == "IDLE":
            return "none"
        elif state == "WIFI_ONLY":
            if confidence > self.WIFI_CONFIDENCE_THRESHOLD:
                return "wake_camera"  # WiFi detects → activate camera to confirm
            return "none"
        elif state == "CAM_ONLY":
            if confidence > self.ALERT_CONFIDENCE_THRESHOLD:
                return "alert"
            return "none"
        elif state == "CONFIRMED":
            if confidence > self.ALERT_CONFIDENCE_THRESHOLD:
                return "record"  # Both confirm → record
            return "alert"
        return "investigate"

    def get_current_state(self) -> dict:
        """Get current fusion state for API."""
        detection = self.fuse()
        return detection.to_dict()

    def get_history(self, limit: int = 20) -> List[dict]:
        """Get recent fused detection history."""
        with self._lock:
            events = list(self._history)[-limit:]
        return [e.to_dict() for e in events]


# ─── Module-level singleton ─────────────────────────────────────────

_fusion: Optional[SensorFusion] = None


def get_fusion() -> SensorFusion:
    """Get or create the global SensorFusion instance."""
    global _fusion
    if _fusion is None:
        _fusion = SensorFusion()
    return _fusion


def update_wifi_state(motion: bool, confidence: float = 0.0,
                      activity: str = "unknown", persons: int = 0):
    """Update WiFi CSI state in the fusion engine."""
    get_fusion().update_wifi(motion, confidence, activity, persons)


def update_camera_state(detected: bool, label: str = "", confidence: float = 0.0):
    """Update camera state in the fusion engine."""
    get_fusion().update_camera(detected, label, confidence)


def get_fused_status() -> dict:
    """Get current fused detection state."""
    return get_fusion().get_current_state()


def get_fusion_history(limit: int = 20) -> List[dict]:
    """Get fused detection history."""
    return get_fusion().get_history(limit)
