"""
Threat classification for the ARGUS detection cascade.

Runs a two-class YOLO11 model over a person crop and reports whether that
person appears to be carrying something dangerous (a firearm or knife in
the source frame).

This is a *fourth* cascade stage, layered on top of the existing COCO
object detection rather than replacing it. The COCO model still answers
"is there a person / vehicle / animal here?"; this model answers only
"is this particular person carrying something dangerous?".

Model provenance:
    Trained by Tran Si Nam as a graduation thesis artifact.
    https://github.com/Nambekai/dangerous-person-detection-yolo11
    Weights are AGPL-3.0; see NOTICE.md.

Dependencies:
    pip install ultralytics
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    from ultralytics import YOLO as _YOLO
    _ULTRALYTICS_AVAILABLE = True
except ImportError:
    _YOLO = None
    _ULTRALYTICS_AVAILABLE = False
    logger.warning(
        "ultralytics not installed — ThreatClassifier unavailable. "
        "Install with: pip install ultralytics"
    )


# Class indices in the trained checkpoint. These are pinned by INDEX and
# never by name on purpose: the upstream config/data.yaml documents the
# classes as "normal_person" / "potentially_dangerous_person", but the
# checkpoint actually ships Vietnamese names
# ({0: 'nguoi_binh_thuong', 1: 'doi_tuong_nguy_hiem'}). Matching on the
# documented English strings silently detects nothing, which on a live
# demo is indistinguishable from a broken camera.
CLASS_NORMAL = 0
CLASS_DANGEROUS = 1

# The class names the v1.0.0 checkpoint actually ships. Recorded so that a
# re-export or a swapped model that renumbers the classes is reported
# loudly instead of silently inverting every verdict.
#
# Checked rather than trusted: main.py runs uvicorn at log_level="warning"
# unless DEBUG is set, so an informational line here would never reach the
# operator on the demo unit. A mismatch has to warn to be seen at all.
KNOWN_CLASS_NAMES = {
    CLASS_NORMAL: "nguoi_binh_thuong",
    CLASS_DANGEROUS: "doi_tuong_nguy_hiem",
}

# Fraction of the person box added on every side before classifying.
#
# This is not a tuning nicety -- it is load-bearing. The model was trained
# on people in full scenes, so a crop tight to the COCO person box cuts off
# the arm and the weapon and the model returns *nothing*. Measured on the
# upstream handgun sample, for the person actually holding the gun:
#
#     tight crop  -> no detection at all
#     15% padding -> dangerous, 0.699
#     30% padding -> dangerous, 0.533
#     50% padding -> dangerous, 0.436
#
# 15% also beats classifying the whole frame (0.436 at imgsz 416), because
# the person fills more of the input. Padding wider dilutes that again.
#
# The tradeoff: padding can pull a neighbour into the crop, so a person
# standing beside an armed one can inherit the flag. 15% keeps that narrow
# while still restoring the context the model needs.
CROP_PADDING = 0.15


@dataclass
class ThreatResult:
    """Outcome of classifying one person crop."""
    is_dangerous: bool
    confidence: float
    label: Optional[str] = None  # the model's own class name, for logging


class ThreatClassifier:
    """
    Two-class YOLO11 classifier answering "is this person dangerous?".

    Deliberately mirrors FaceRecognitionService: constructed once at
    startup, attached to DetectionService, and raising a clear RuntimeError
    rather than degrading silently when its dependency is missing.
    """

    def __init__(
        self,
        model_path,
        confidence: float = 0.35,
        imgsz: int = 416,
        crop_padding: float = CROP_PADDING,
    ):
        """
        Args:
            model_path: Path to the trained .pt weights.
            confidence: Minimum detection confidence. Defaults to 0.35, the
                        value the upstream model card names for demonstration.
            imgsz: Inference input size. The model was trained at 832, but
                   832 is far too slow on a Pi CPU (~110ms/frame measured on
                   x86); 416 roughly halves that at a small accuracy cost.
            crop_padding: Fraction of the person box to add around it before
                          classifying. See CROP_PADDING -- zero here means
                          the model detects nothing at all.
        """
        if not _ULTRALYTICS_AVAILABLE:
            raise RuntimeError(
                "ultralytics is required for threat classification. "
                "Install with: pip install ultralytics"
            )

        path = Path(model_path)
        if not path.exists():
            raise RuntimeError(
                f"Threat model not found at {path}. "
                "Fetch it with: python scripts/fetch_models.py"
            )

        self.confidence = confidence
        self.imgsz = imgsz
        self.crop_padding = crop_padding
        self._model = _YOLO(str(path))

        names = getattr(self._model, "names", None) or {}
        logger.info("Threat model loaded from %s — classes=%s", path, names)

        # Verdicts are decided by class index, so a checkpoint whose
        # indices mean something different silently inverts every result.
        # Warn, because info-level never reaches the demo unit's logs.
        if {int(k): v for k, v in names.items()} != KNOWN_CLASS_NAMES:
            logger.warning(
                "Threat model classes differ from the recorded checkpoint. "
                "expected=%s actual=%s — index %d is treated as dangerous, "
                "so verify that still holds.",
                KNOWN_CLASS_NAMES, names, CLASS_DANGEROUS,
            )

        self.class_names = names

    def classify_region(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> Optional[ThreatResult]:
        """
        Classify the person inside `bbox`.

        The region is padded by `crop_padding` first; a tight crop loses the
        context the model was trained on and detects nothing.

        Args:
            frame: Full BGR frame.
            bbox: (x, y, w, h) person region from the object detector.

        Returns:
            The result for the crop, or None when the crop is empty or
            nothing cleared the confidence threshold.
        """
        x, y, w, h = bbox
        fh, fw = frame.shape[:2]

        pad = int(self.crop_padding * max(w, h))
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(fw, x + w + pad), min(fh, y + h + pad)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        return self._classify(crop)

    def classify_frame(self, frame: np.ndarray) -> Optional[ThreatResult]:
        """Classify a whole frame. Used by the offline verification script."""
        if frame is None or frame.size == 0:
            return None
        return self._classify(frame)

    def _classify(self, image: np.ndarray) -> Optional[ThreatResult]:
        """
        Run inference and reduce the boxes to one answer.

        Any dangerous detection wins over every normal one, regardless of
        confidence. Taking the global maximum instead would let a confidently
        -detected bystander mask an armed person standing next to them --
        on the upstream sample frame the normal person scores 0.86 and the
        armed one 0.62, so a plain argmax reports "nothing to see".

        Confidence therefore reads as "how sure we are about the thing we
        are reporting", not "how sure we are about the most obvious person".
        """
        try:
            results = self._model(
                image,
                imgsz=self.imgsz,
                conf=self.confidence,
                verbose=False,
            )
        except Exception as exc:
            logger.error("Threat inference error: %s", exc)
            return None

        best_dangerous: Optional[ThreatResult] = None
        best_normal: Optional[ThreatResult] = None

        for result in results:
            boxes = getattr(result, "boxes", None)
            if not boxes:
                continue
            for box in boxes:
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                candidate = ThreatResult(
                    is_dangerous=(cls_id == CLASS_DANGEROUS),
                    confidence=conf,
                    label=self.class_names.get(cls_id) if self.class_names else None,
                )

                if candidate.is_dangerous:
                    if best_dangerous is None or conf > best_dangerous.confidence:
                        best_dangerous = candidate
                elif best_normal is None or conf > best_normal.confidence:
                    best_normal = candidate

        return best_dangerous or best_normal
