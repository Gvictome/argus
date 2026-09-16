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
from pathlib import Path
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
    # Produced only by the threat stage, never by the COCO object model.
    DANGEROUS_PERSON = "dangerous_person"


# YOLO class-id → DetectionType mapping (COCO dataset classes).
#
# COCO ONLY. The threat model is a separate two-class checkpoint whose
# class 1 means "dangerous person", not "bicycle" -- routing its output
# through this map would render every threat as a green ANIMAL box. It
# has its own mapping in src/detection/threat.py.
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


# Where a compiled Hailo model may live, in preference order.
#
# Ultralytics loads the export *directory* -- `YOLO("yolo11n_hailo_model")`
# -- because the HEF alone is not enough: the directory also carries the
# metadata describing input size, classes, and NMS. An earlier version of
# this probe looked only for a bare `yolov8n_hailo_model.hef` file, which
# is not what `format="hailo"` produces, so the accelerator would never
# have been found on a correctly-exported install.
#
# The bare .hef is still accepted last, so an existing hand-placed file
# keeps working.
_HAILO_CANDIDATES = (
    ("models", "yolov8n_hailo_model"),
    ("models", "yolo11n_hailo_model"),
    ("yolov8n_hailo_model",),
    ("yolov8n_hailo_model.hef",),
)


# Everything ARGUS maps to a class, plus the COCO ids that stand in for
# "package". Passing these to the model drops the other COCO classes
# before NMS instead of after.
_RELEVANT_COCO_IDS = sorted(set(_YOLO_CLASS_MAP) | {24, 26, 28})

# Optimised CPU exports. NCNN is the Ultralytics-recommended CPU path on a
# Raspberry Pi; OpenVINO is the fast path on x86. Both are what to run
# while no Hailo HEF has been compiled for the board.
_CPU_EXPORTS = {
    "openvino": "yolov8n_openvino_model",
    "ncnn": "yolov8n_ncnn_model",
}


def _find_cpu_export(preference: str):
    """(format, path) of an optimised CPU export to load, or None."""
    import platform

    preference = (preference or "none").lower()
    if preference in ("none", "pt"):
        return None
    if preference == "auto":
        arm = platform.machine().lower() in ("aarch64", "arm64")
        order = ["ncnn", "openvino"] if arm else ["openvino", "ncnn"]
    elif preference in _CPU_EXPORTS:
        order = [preference]
    else:
        return None
    for fmt in order:
        path = BASE_DIR / "models" / _CPU_EXPORTS[fmt]
        if path.exists():
            return fmt, path
    return None


def _export_imgsz(path) -> Optional[int]:
    """Input size an exported model was compiled for, from its metadata.yaml.

    NCNN, OpenVINO, and Hailo exports have a fixed input shape. Asking one
    for a different size does not resize -- it returns no detections at all,
    quickly, which reads as a large speedup with detection silently off.
    Measured: OpenVINO exported at 640 and run at 416 gave 211 FPS and zero
    boxes. So the export's own size wins over OBJECT_IMGSZ.
    """
    if path is None:
        return None
    p = Path(path)
    meta = (p if p.is_dir() else p.parent) / "metadata.yaml"
    if not meta.exists():
        return None
    try:
        import yaml

        imgsz = (yaml.safe_load(meta.read_text(encoding="utf-8")) or {}).get("imgsz")
    except Exception:
        return None
    if isinstance(imgsz, (list, tuple)) and imgsz:
        imgsz = imgsz[0]
    try:
        return int(imgsz)
    except (TypeError, ValueError):
        return None


def _find_hailo_model():
    """The first Hailo export present, or None."""
    for parts in _HAILO_CANDIDATES:
        candidate = BASE_DIR.joinpath(*parts)
        if candidate.exists():
            return candidate
    return None


@dataclass
class Detection:
    """Detection result"""
    type: DetectionType
    confidence: float
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    label: Optional[str] = None
    face_id: Optional[str] = None
    # Raw COCO class id. "package" has no DetectionType, so the federated
    # head recognises it by id.
    coco_id: Optional[int] = None


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
    # DEPRECATED 2026-09-14. Face recognition cost more than YOLO on CPU and
    # identity is out of scope. The stage stays behind this flag so it can
    # be re-enabled; by default it never runs and nothing loads for it.
    faces_enabled: bool = False
    # YOLO input size.
    object_imgsz: int = 640
    # Motion detection differences a copy downscaled to this width.
    motion_width: int = 320
    # none | auto | pt | openvino | ncnn. Defaults to none so a bare
    # DetectionService() is unaffected by whatever sits in models/; the app
    # passes OBJECT_BACKEND through.
    cpu_export: str = "none"


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
        self.collector = None         # EventCollector (set via attach_collector)
        self.threat_classifier = None  # ThreatClassifier (set via attach_threat_classifier)
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
                # Accelerators first, in descending order of speed, then
                # plain CPU. Each is a file probe rather than a hardware
                # query: an engine only exists if someone exported it for
                # this machine, and a TensorRT engine is not portable
                # between devices or JetPack versions anyway.
                tensorrt_model = BASE_DIR / "models" / "yolov8n.engine"
                hailo_model = _find_hailo_model()
                cpu_export = _find_cpu_export(self.config.cpu_export)

                if tensorrt_model.exists():
                    logger.info("TensorRT engine found — using Jetson GPU acceleration.")
                    self.object_model = _YOLO(str(tensorrt_model))
                    self.backend = "tensorrt"
                elif hailo_model is not None:
                    logger.info("Hailo model found at %s — using AI HAT+ acceleration.",
                                hailo_model)
                    self.object_model = _YOLO(str(hailo_model))
                    self.backend = "hailo"
                elif cpu_export is not None:
                    fmt, path = cpu_export
                    logger.info("%s export found at %s, using optimised CPU inference.", fmt, path)
                    self.object_model = _YOLO(str(path), task="detect")
                    self.backend = f"cpu-{fmt}"
                else:
                    logger.info("No accelerator model found. Using standard YOLOv8n (CPU).")
                    self.object_model = _YOLO("yolov8n.pt")
                    self.backend = "cpu"

                if self.backend == "hailo" or self.backend.startswith("cpu-"):
                    export_path = hailo_model if self.backend == "hailo" else cpu_export[1]
                    fixed = _export_imgsz(export_path)
                    if fixed and fixed != self.config.object_imgsz:
                        logger.warning(
                            "OBJECT_IMGSZ=%d ignored: %s export is fixed at %d. "
                            "Re-export at the size you want instead.",
                            self.config.object_imgsz, self.backend, fixed,
                        )
                        self.config.object_imgsz = fixed

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

        # Face detection is deprecated; the cascade loads only when enabled.
        if self.config.faces_enabled:
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
        else:
            logger.info("Face stage disabled (deprecated)")

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

        # Differencing does not need full resolution to see a person move,
        # and blurring two full frames per call was the most expensive
        # "cheap" stage. Work on a small copy, keep this call's blurred
        # result for the next call instead of re-blurring the previous
        # frame, and map boxes back to full-resolution coordinates.
        h, w = frame.shape[:2]
        scale = min(1.0, max(16, int(self.config.motion_width or w)) / float(w))
        small = frame if scale >= 1.0 else cv2.resize(
            frame, (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        k = max(3, int(round(21 * scale)) | 1)
        gray = cv2.GaussianBlur(gray, (k, k), 0)

        previous = self.previous_frame
        self.previous_frame = gray
        if previous is None or previous.shape != gray.shape:
            return detections

        diff = cv2.absdiff(previous, gray)
        _, thresh = cv2.threshold(
            diff, self.config.motion_threshold, 255, cv2.THRESH_BINARY
        )
        thresh = cv2.dilate(thresh, None, iterations=2 if scale >= 1.0 else 1)
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        inv = 1.0 / scale
        min_w, min_h = self.config.min_detection_size
        for contour in contours:
            x, y, bw, bh = cv2.boundingRect(contour)
            fx, fy, fw, fh = int(x * inv), int(y * inv), int(bw * inv), int(bh * inv)
            if fw < min_w or fh < min_h:
                continue

            # Confidence proxy: ratio of contour area to bounding-box area
            contour_area = cv2.contourArea(contour)
            bbox_area = float(bw * bh) if bw * bh > 0 else 1.0
            confidence = min(contour_area / bbox_area, 1.0)

            detections.append(
                Detection(
                    type=DetectionType.MOTION,
                    confidence=float(confidence),
                    bbox=(fx, fy, fw, fh),
                    label="motion",
                )
            )

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
            results = self.object_model(
                frame,
                verbose=False,
                imgsz=self.config.object_imgsz,
                conf=self.config.detection_threshold,
                classes=_RELEVANT_COCO_IDS,
            )
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
                        coco_id=cls_id,
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

    def attach_collector(self, collector) -> None:
        """
        Attach an EventCollector so resolved detections become training data.

        Without this nothing in the pipeline is ever persisted: db.log_event
        exists and was never called, so /api/events returned an empty list
        and the federated head had nothing to learn from.
        """
        self.collector = collector
        logger.info("EventCollector attached to DetectionService")

    def attach_threat_classifier(self, classifier) -> None:
        """
        Attach a ThreatClassifier for the dangerous-person cascade stage.

        Args:
            classifier: ThreatClassifier instance.
        """
        self.threat_classifier = classifier
        logger.info("ThreatClassifier attached to DetectionService")

    def detect_threats(self, frame: np.ndarray) -> List[Detection]:
        """
        Find people carrying weapons, using the threat model's own boxes.

        One inference over the whole frame. The threat model is itself a
        person detector, so it locates and classifies in a single pass --
        it does not need the COCO model to find people first.

        Deliberately *not* gated on a COCO human detection. Measured over
        the 314-image upstream test set, gating that way lost 20% of true
        positives outright: the COCO model simply never found those
        people, so the threat stage never ran on them. Recall went from
        0.800 to 0.612.

        Args:
            frame: Current frame as numpy array (BGR).

        Returns:
            One DANGEROUS_PERSON detection per flagged person.
        """
        detections: List[Detection] = []

        if self.threat_classifier is None:
            return detections

        try:
            results = self.threat_classifier.detect_threats(frame)
        except Exception as exc:
            # A failure in the newest stage must not take down object
            # detection or face recognition -- those carry the demo.
            logger.warning("Threat detection failed: %s", exc)
            return detections

        for r in results:
            detections.append(
                Detection(
                    type=DetectionType.DANGEROUS_PERSON,
                    confidence=r.confidence,
                    bbox=r.bbox,
                    label="dangerous",
                )
            )

        return detections

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
            if self.config.faces_enabled:
                humans = [d for d in objects if d.type == DetectionType.HUMAN]
                if humans or self.object_model is None:
                    faces = self.recognize_faces(frame)
                    all_detections.extend(faces)

            # Threat detection locates people itself, so it runs on motion
            # alone rather than behind a COCO human box. Gating it on one
            # cost 20% of true positives on the upstream test set.
            all_detections.extend(self.detect_threats(frame))

        # Feed the collector so resolved tracks become federated training
        # samples. Wrapped because a collector fault must never take down
        # the detection loop -- losing training data is recoverable, losing
        # the camera during a demo is not.
        collector = getattr(self, "collector", None)
        if collector is not None:
            try:
                collector.observe(all_detections, frame.shape[:2], frame=frame)
            except Exception as exc:
                logger.warning("EventCollector.observe failed: %s", exc)

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
            "faces_enabled": self.config.faces_enabled,
            "object_imgsz": self.config.object_imgsz,
            "threat_detection": self.threat_classifier is not None,
            "known_faces": known_faces,
        }

    def shutdown(self) -> None:
        """Release model references and reset state."""
        self.object_model = None
        self.face_cascade = None
        self.face_recognizer = None
        self.threat_classifier = None
        self.previous_frame = None
        self.backend = "none"
        self._initialized = False
        self._frame_times.clear()
        logger.info("DetectionService shut down")


# Global detection service instance (mirrors `camera_service` in src.camera).
# Initialized at application startup; see src/api/app.py.
detection_service = DetectionService()
