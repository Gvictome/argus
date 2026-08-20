"""
Tests for detection overlay drawing (showcase P0-2 / G-2).

The demo depends on a judge seeing a red box labeled UNKNOWN turn into a
green box labeled with their name.  These tests pin that behavior.
"""

import numpy as np
import pytest

from src.detection import Detection, DetectionType
from src.detection.annotate import BOX_KNOWN, BOX_UNKNOWN, draw_detections


def blank(h=240, w=320):
    """Black frame; any drawn pixel is therefore non-zero."""
    return np.zeros((h, w, 3), dtype=np.uint8)


def face(face_id=None, name="unknown", conf=0.5, bbox=(40, 30, 60, 60)):
    return Detection(
        type=DetectionType.FACE,
        confidence=conf,
        bbox=bbox,
        label=name,
        face_id=face_id,
    )


def pixels_of_color(frame, bgr):
    """Count pixels exactly matching a BGR color."""
    return int(np.all(frame == np.array(bgr, dtype=np.uint8), axis=-1).sum())


class TestDrawDetections:
    def test_returns_new_frame_leaving_input_unmodified(self):
        original = blank()
        result = draw_detections(original, [face()])
        assert result is not original
        assert original.sum() == 0, "input frame was mutated"

    def test_no_detections_leaves_frame_blank(self):
        result = draw_detections(blank(), [])
        assert result.sum() == 0

    def test_known_face_drawn_in_green(self):
        result = draw_detections(blank(), [face(face_id="f1", name="Giovanny", conf=0.83)])
        assert pixels_of_color(result, BOX_KNOWN) > 0
        assert pixels_of_color(result, BOX_UNKNOWN) == 0

    def test_unknown_face_drawn_in_red(self):
        result = draw_detections(blank(), [face(face_id=None, name="unknown")])
        assert pixels_of_color(result, BOX_UNKNOWN) > 0
        assert pixels_of_color(result, BOX_KNOWN) == 0

    def test_box_is_drawn_at_the_detection_bbox(self):
        result = draw_detections(blank(), [face(face_id="f1", bbox=(100, 50, 40, 40))])
        # Top edge of the box lies on row 50, between columns 100 and 140.
        assert result[50, 100:140].any(), "no box edge at the bbox top"
        # A far corner of the frame stays untouched.
        assert result[200, 300].sum() == 0

    def test_motion_detections_are_not_drawn(self):
        """Motion boxes are large and noisy; they would clutter the demo."""
        motion = Detection(
            type=DetectionType.MOTION, confidence=0.9, bbox=(0, 0, 320, 240), label="motion"
        )
        assert draw_detections(blank(), [motion]).sum() == 0

    def test_human_detection_is_drawn(self):
        human = Detection(
            type=DetectionType.HUMAN, confidence=0.91, bbox=(10, 10, 80, 200), label="person"
        )
        assert draw_detections(blank(), [human]).sum() > 0

    def test_bbox_clamped_to_frame_bounds(self):
        """A bbox running off-frame must not raise."""
        out = draw_detections(blank(), [face(face_id="f1", bbox=(300, 220, 200, 200))])
        assert out.shape == (240, 320, 3)


class TestLabelText:
    def test_known_face_label_includes_name_and_confidence(self):
        assert build_label(face(face_id="f1", name="Giovanny", conf=0.83)) == "Giovanny 0.83"

    def test_unknown_face_label_is_uppercase_unknown(self):
        assert build_label(face(face_id=None, name="unknown")) == "UNKNOWN"

    def test_unknown_face_label_omits_confidence(self):
        """Confidence on an unrecognized face is meaningless to a judge."""
        assert "0." not in build_label(face(face_id=None, name="unknown", conf=0.31))


# imported here so the class above reads cleanly
from src.detection.annotate import build_label  # noqa: E402
