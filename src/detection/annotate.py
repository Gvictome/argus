"""
Detection overlay drawing for the live demo stream (showcase P0-2).

Boxes and labels are burned into the frame server-side, before JPEG
encoding, so a label can never drift out of sync with the frame it
describes.  Drawing them in the browser would require a parallel metadata
channel and would desynchronize under load.
"""

from typing import Iterable, List

import cv2
import numpy as np

from src.detection import Detection, DetectionType

# BGR, matching OpenCV's channel order.
BOX_KNOWN = (0, 255, 0)     # green: recognized person
BOX_UNKNOWN = (0, 0, 255)   # red: a face with no identity match
BOX_THREAT = (0, 165, 255)  # orange: person flagged as carrying a weapon

# Motion boxes cover most of the frame and would bury the useful ones.
_SKIP_TYPES = frozenset({DetectionType.MOTION})

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.6
_THICKNESS = 2


def _is_known(detection: Detection) -> bool:
    """A face is identified only when the recognizer resolved a face_id."""
    return detection.type is DetectionType.FACE and detection.face_id is not None


def build_label(detection: Detection) -> str:
    """
    Text drawn above a detection's box.

    An unrecognized face reports no confidence: the number describes the
    detector's certainty that a face is present, not any claim about who
    it is, and showing it next to UNKNOWN invites exactly that misreading.
    """
    if detection.type is DetectionType.FACE and detection.face_id is None:
        return "UNKNOWN"
    name = detection.label or detection.type.value
    return f"{name} {detection.confidence:.2f}"


def box_color(detection: Detection) -> tuple:
    """
    Orange for a flagged threat, red for an unidentified face, green
    otherwise.

    Threat is checked first: it is the most consequential thing on the
    frame, and without its own color it would draw green and read as an
    ordinary person.
    """
    if detection.type is DetectionType.DANGEROUS_PERSON:
        return BOX_THREAT
    if detection.type is DetectionType.FACE and detection.face_id is None:
        return BOX_UNKNOWN
    return BOX_KNOWN


def draw_detections(frame: np.ndarray, detections: Iterable[Detection]) -> np.ndarray:
    """
    Draw boxes and labels onto a copy of `frame`.

    Args:
        frame: BGR frame as a numpy array.
        detections: Detections from DetectionService.process_frame().

    Returns:
        A new annotated frame. The input is never modified, because the
        caller may still need the clean frame for recording or enrollment.
    """
    annotated = frame.copy()
    height, width = annotated.shape[:2]

    detections = list(detections)

    # A flagged person yields two detections sharing one bbox: the HUMAN
    # box from the object model and the DANGEROUS_PERSON box from the
    # threat stage. Drawing both stacks two labels on the same pixel and
    # renders as unreadable overlapping text -- and implies two subjects
    # where there is one. The threat box supersedes; the detection list
    # itself is untouched, so events and the API still see both.
    threat_boxes = {
        d.bbox for d in detections if d.type is DetectionType.DANGEROUS_PERSON
    }

    for det in detections:
        if det.type in _SKIP_TYPES:
            continue

        if det.type is DetectionType.HUMAN and det.bbox in threat_boxes:
            continue

        x, y, w, h = (int(v) for v in det.bbox)
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(width - 1, x + w), min(height - 1, y + h)
        if x2 <= x1 or y2 <= y1:
            continue

        color = box_color(det)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, _THICKNESS)

        # Keep the label on-screen when the box is flush with the top edge.
        label_y = y1 - 6 if y1 - 6 > 10 else min(y2 + 18, height - 4)
        cv2.putText(
            annotated,
            build_label(det),
            (x1, label_y),
            _FONT,
            _FONT_SCALE,
            color,
            _THICKNESS,
        )

    return annotated
