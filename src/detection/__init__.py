"""
Detection service module

Handles:
- Motion detection via frame differencing
- Object classification via YOLOv8n
- Face recognition via OpenCV CascadeClassifier
- AI model management
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Tuple

import cv2
import numpy as np

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


@dataclass
class DetectionConfig:
    """Detection configuration"""
    motion_threshold: int = 25
    detection_threshold: float = 0.5
    face_recognition_threshold: float = 0.6
    min_detection_size: Tuple[int, int] = (30, 30)


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
                else:
                    logger.info("No Hailo model found. Using standard YOLOv8n (CPU).")
                    self.object_model = _YOLO("yolov8n.pt")
                
                logger.info("YOLO model loaded successfully")
            except Exception as exc:
                logger.warning("Failed to load YOLO model: %s", exc)
                self.object_model = None
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

        # Convert both frames to grayscale
        gray_current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_previous = cv2.cvtColor(self.previous_frame, cv2.COLOR_BGR2GRAY)

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
                    bbox=(x, y, w, h),
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

        If a FaceRecognitionService is attached, uses dlib-based 128-d
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

        # Motion detection (fast, always runs)
        motion = self.detect_motion(frame)
        all_detections.extend(motion)

        # Only run expensive models if motion is detected
        if motion:
            objects = self.detect_objects(frame)
            all_detections.extend(objects)

            # Only run face detection if a human was detected
            humans = [d for d in objects if d.type == DetectionType.HUMAN]
            if humans:
                faces = self.recognize_faces(frame)
                all_detections.extend(faces)

        return all_detections

    def shutdown(self) -> None:
        """Release model references and reset state."""
        self.object_model = None
        self.face_cascade = None
        self.previous_frame = None
        logger.info("DetectionService shut down")
