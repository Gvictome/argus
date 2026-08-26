"""
Detection service module

Handles:
- Motion detection via frame differencing
- Object classification via YOLOv8n
- Face recognition via OpenCV CascadeClassifier
- AI model management
"""

import logging
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Tuple

import cv2
import numpy as np

from src.config import BASE_DIR

logger = logging.getLogger(__name__)

# Optional ultralytics import — not required on all deployments
try:
    from ultralytics import YOLO as _YOLO
    _ULTRALYTICS_AVAILABLE = True
except ImportError:
    _YOLO = None
    _ULTRALYTICS_AVAILABLE = False
    logger.warning("ultralytics not installed — YOLOv8n object detection disabled")


class DetectionType(Enum):
    MOTION = "motion"
    HUMAN = "human"
    FACE = "face"
    ANIMAL = "animal"
    VEHICLE = "vehicle"
    UNKNOWN = "unknown"


# YOLO class-id → DetectionType mapping (COCO dataset classes)
_YOLO_CLASS_MAP: dict[int, DetectionType] = {
    0: DetectionType.HUMAN,     # person
    1: DetectionType.ANIMAL,    # bicycle (treat as unknown via fallback)
    2: DetectionType.VEHICLE,   # car
    3: DetectionType.VEHICLE,   # motorcycle
    5: DetectionType.VEHICLE,   # bus
    7: DetectionType.VEHICLE,   # truck
    15: DetectionType.ANIMAL,   # cat
    16: DetectionType.ANIMAL,   # dog
    17: DetectionType.ANIMAL,   # horse
    18: DetectionType.ANIMAL,   # sheep
    19: DetectionType.ANIMAL,   # cow
}


@dataclass
class Detection:
    """Detection result"""
    type: DetectionType
    confidence: float
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    label: Optional[str] = None
    face_id: Optional[str] = None


# Frames retained for the rolling FPS average. Short enough that the number
# tracks what the demo is doing right now rather than its lifetime mean.
FPS_WINDOW = 30


@dataclass
class DetectionConfig:
    """Detection configuration"""
    motion_threshold: int = 25
    detection_threshold: float = 0.5
    face_recognition_threshold: float = 0.6
    min_detection_size: Tuple[int, int] = (30, 30)
    # Frame differencing runs on a 1/N copy. The diff only locates *where*
    # something moved, which survives downscaling, and the blur and dilate
    # steps dominate the cost at full resolution. 1 disables scaling.
    motion_detection_scale: int = 4


class DetectionService:
    """
    Detection service for motion, objects, and faces.

    Uses OpenCV for motion detection and face detection.
    Uses YOLOv8n (ultralytics) for object classification when available.
    Optional Hailo AI HAT+ acceleration via ultralytics export.
    """

    def __init__(self, config: Optional[DetectionConfig] = None):
        self.config = config or DetectionConfig()
        self.object_model = None      # YOLOv8n YOLO instance
        self.face_cascade = None      # cv2.CascadeClassifier
        self.face_recognizer = None   # FaceRecognitionService (set via attach_face_recognizer)
        self.previous_frame = None    # for frame differencing
        self.backend = "none"         # "hailo" | "cpu" | "none"
        self._initialized = False
        self._frame_times: deque = deque(maxlen=FPS_WINDOW)

    def initialize(self) -> bool:
        """
        Initialize detection models.

        Checks for optimized Hailo (.hef) models for Raspberry Pi AI HAT+.
        Falls back to standard YOLOv8n (ultralytics).
        """
        success = True

        # Object detection — YOLOv8.1 (8.4.x)
        if _ULTRALYTICS_AVAILABLE:
            try:
                # Check for Hailo optimized model first
                hailo_model = BASE_DIR / "yolov8n_hailo_model.hef"
                if hailo_model.exists():
                    logger.info("Optimized Hailo model found! Using AI HAT+ acceleration.")
                    self.object_model = _YOLO(str(hailo_model))
                    self.backend = "hailo"
                else:
                    logger.info("No Hailo model found. Using standard YOLOv8n (CPU).")
                    self.object_model = _YOLO("yolov8n.pt")
                    self.backend = "cpu"
                
                logger.info("YOLO model loaded successfully")
            except Exception as exc:
                # Loud on purpose. A silent fallback here disables object
                # detection, and the cascade only reaches face recognition
                # after YOLO reports a human -- so this failure reads on the
                # demo as "face recognition is broken".
                logger.error("Failed to load YOLO model: %s", exc, exc_info=True)
                self.object_model = None
                self.backend = "none"
        else:
            logger.info("Skipping YOLOv8.1 — ultralytics not available")

        # Face detection — OpenCV Haar cascade (bundled with cv2)
        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            if self.face_cascade.empty():
                logger.warning("Failed to load Haar cascade for face detection")
                self.face_cascade = None
            else:
                logger.info("OpenCV face cascade loaded")
        except Exception as exc:
            logger.warning("Face cascade init failed: %s", exc)
            self.face_cascade = None

        self._initialized = True

        logger.info(
            "DetectionService initialized — objects=%s faces=%s",
            self.object_model is not None,
            self.face_cascade is not None,
        )
        return success

    def detect_motion(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect motion by comparing the current frame with the previous frame.

        Pipeline: grayscale → GaussianBlur → absdiff → threshold →
                  findContours → filter by min_detection_size

        Args:
            frame: Current frame as numpy array (BGR).

        Returns:
            List of motion detections; empty on first call (no prior frame).
        """
        detections: List[Detection] = []

        if self.previous_frame is None:
            self.previous_frame = frame.copy()
            return detections

        min_w, min_h = self.config.min_detection_size

        # Work on a shrunken copy; boxes are scaled back to full-frame
        # coordinates below so callers never see the reduced space.
        scale = max(1, int(self.config.motion_detection_scale))
        current, previous = frame, self.previous_frame
        if scale > 1:
            h, w = frame.shape[:2]
            small = (max(1, w // scale), max(1, h // scale))
            current = cv2.resize(frame, small, interpolation=cv2.INTER_NEAREST)
            previous = cv2.resize(self.previous_frame, small, interpolation=cv2.INTER_NEAREST)
            # The size filter is expressed in full-frame pixels.
            min_w = max(1, min_w // scale)
            min_h = max(1, min_h // scale)

        # Convert both frames to grayscale
        gray_current = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
        gray_previous = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)

        # Reduce noise before differencing
        gray_current = cv2.GaussianBlur(gray_current, (21, 21), 0)
        gray_previous = cv2.GaussianBlur(gray_previous, (21, 21), 0)

        # Frame difference
        diff = cv2.absdiff(gray_previous, gray_current)

        # Binary threshold
        _, thresh = cv2.threshold(
            diff, self.config.motion_threshold, 255, cv2.THRESH_BINARY
        )

        # Dilate to fill gaps between contours
        thresh = cv2.dilate(thresh, None, iterations=2)

        # Find contours of changed regions
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w < min_w or h < min_h:
                continue

            # Confidence proxy: ratio of contour area to bounding-box area
            contour_area = cv2.contourArea(contour)
            bbox_area = float(w * h) if w * h > 0 else 1.0
            confidence = min(contour_area / bbox_area, 1.0)

            detections.append(
                Detection(
                    type=DetectionType.MOTION,
                    confidence=float(confidence),
                    bbox=(x * scale, y * scale, w * scale, h * scale),
                    label="motion",
                )
            )

        self.previous_frame = frame.copy()
        return detections

    def detect_objects(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect and classify objects in a frame using YOLOv8n.

        Args:
            frame: Current frame as numpy array (BGR).

        Returns:
            List of object detections; empty if model not loaded.
        """
        detections: List[Detection] = []

        if self.object_model is None:
            return detections

        try:
            results = self.object_model(frame, verbose=False)
        except Exception as exc:
            logger.error("YOLOv8n inference error: %s", exc)
            return detections

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                confidence = float(box.conf[0])
                if confidence < self.config.detection_threshold:
                    continue

                cls_id = int(box.cls[0])
                label = result.names.get(cls_id, "unknown")
                det_type = _YOLO_CLASS_MAP.get(cls_id, DetectionType.UNKNOWN)

                # box.xywh returns [x_center, y_center, w, h] — convert to top-left
                xc, yc, w, h = box.xywh[0].tolist()
                x = int(xc - w / 2)
                y = int(yc - h / 2)

                detections.append(
                    Detection(
                        type=det_type,
                        confidence=confidence,
                        bbox=(x, y, int(w), int(h)),
                        label=label,
                    )
                )

        return detections

    def recognize_faces(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect and optionally identify faces in a frame.

        If a FaceRecognitionService is attached, uses ArcFace 512-d
        embeddings for identity matching.  Otherwise falls back to Haar
        cascade detection only (no identity).

        Args:
            frame: Current frame as numpy array (BGR).

        Returns:
            List of face detections with face_id populated when recognized.
        """
        detections: List[Detection] = []

        # Prefer FaceRecognitionService if available
        if self.face_recognizer is not None:
            matches = self.face_recognizer.recognize(frame)
            for m in matches:
                detections.append(
                    Detection(
                        type=DetectionType.FACE,
                        confidence=m.confidence if m.face_id else self.config.face_recognition_threshold,
                        bbox=m.bbox,
                        label=m.name or "unknown",
                        face_id=m.face_id,
                    )
                )
            return detections

        # Fallback: Haar cascade (detection only, no identity)
        if self.face_cascade is None:
            return detections

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=self.config.min_detection_size,
        )

        for x, y, w, h in faces:
            detections.append(
                Detection(
                    type=DetectionType.FACE,
                    confidence=self.config.face_recognition_threshold,
                    bbox=(int(x), int(y), int(w), int(h)),
                    label="face",
                    face_id=None,
                )
            )

        return detections

    def attach_face_recognizer(self, recognizer) -> None:
        """
        Attach a FaceRecognitionService for identity-aware face detection.

        Args:
            recognizer: FaceRecognitionService instance.
        """
        self.face_recognizer = recognizer
        logger.info("FaceRecognitionService attached to DetectionService")

    def process_frame(self, frame: np.ndarray) -> List[Detection]:
        """
        Full detection pipeline for a single frame.

        Runs motion detection first (cheap); only invokes the object model
        if motion is found; only invokes face detection if humans are detected.

        Args:
            frame: Current frame as numpy array (BGR).

        Returns:
            All detections from all active models.
        """
        all_detections: List[Detection] = []
        self._frame_times.append(time.monotonic())

        # Motion detection (fast, always runs)
        motion = self.detect_motion(frame)
        all_detections.extend(motion)

        # Only run expensive models if motion is detected
        if motion:
            objects = self.detect_objects(frame)
            all_detections.extend(objects)

            # Faces are the expensive stage, so normally they run only once
            # YOLO has placed a human in the frame. When the object model is
            # unavailable there are no human detections to gate on, and that
            # gate would disable face recognition entirely -- so motion alone
            # becomes sufficient.
            humans = [d for d in objects if d.type == DetectionType.HUMAN]
            if humans or self.object_model is None:
                faces = self.recognize_faces(frame)
                all_detections.extend(faces)

        return all_detections

    def current_fps(self) -> float:
        """
        Processed frames per second over the last FPS_WINDOW frames.

        Measured across the window's span rather than per-frame, so one slow
        frame does not swing the reading. Returns 0.0 until two frames have
        been processed, since a rate needs an interval.
        """
        if len(self._frame_times) < 2:
            return 0.0

        elapsed = self._frame_times[-1] - self._frame_times[0]
        if elapsed <= 0:
            return 0.0

        return (len(self._frame_times) - 1) / elapsed

    def status(self) -> dict:
        """
        Runtime state for GET /api/detection/status.

        Reports what is actually loaded rather than what was configured: on
        the Pi the useful question is whether the accelerator and the face
        recognizer came up, not whether they were requested.
        """
        known_faces = 0
        if self.face_recognizer is not None:
            try:
                known_faces = len(self.face_recognizer.list_known_faces())
            except Exception as exc:
                logger.warning("Could not count known faces: %s", exc)

        return {
            "status": "running" if self._initialized else "stopped",
            "fps": round(self.current_fps(), 2),
            "backend": self.backend,
            "motion_detection": self._initialized,
            "object_detection": self.object_model is not None,
            "face_recognition": self.face_recognizer is not None,
            "known_faces": known_faces,
        }

    def shutdown(self) -> None:
        """Release model references and reset state."""
        self.object_model = None
        self.face_cascade = None
        self.face_recognizer = None
        self.previous_frame = None
        self.backend = "none"
        self._initialized = False
        self._frame_times.clear()
        logger.info("DetectionService shut down")


# Global detection service instance (mirrors `camera_service` in src.camera).
# Initialized at application startup; see src/api/app.py.
detection_service = DetectionService()
