"""
ThreatClassifier tests: class mapping, the cascade stage, and degradation.

ultralytics is faked at the module boundary, matching
tests/test_detection_service.py. The point of these tests is the wiring
and the failure behavior, neither of which needs real weights.
"""

import numpy as np
import pytest

from src.detection import (
    DetectionService,
    DetectionType,
)
import src.detection as detection_module
import src.detection.threat as threat_module
from src.detection.threat import (
    CLASS_DANGEROUS,
    CLASS_NORMAL,
    ThreatClassifier,
    ThreatResult,
)
from src.detection.annotate import (
    BOX_KNOWN,
    BOX_THREAT,
    BOX_UNKNOWN,
    box_color,
    draw_detections,
)
from src.detection import Detection


def _frame(value: int = 0) -> np.ndarray:
    return np.full((480, 640, 3), value, dtype=np.uint8)


def _moving_frame() -> np.ndarray:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[100:300, 100:300] = 255
    return frame


class _Box:
    def __init__(self, cls_id: int, conf: float):
        self.cls = [cls_id]
        self.conf = [conf]


class _Result:
    """Mimics an ultralytics Result. Names are Vietnamese, as shipped."""

    names = {0: "nguoi_binh_thuong", 1: "doi_tuong_nguy_hiem"}

    def __init__(self, boxes):
        self.boxes = boxes


class _FakeThreatYOLO:
    """Returns whatever boxes the test configured."""

    boxes_to_return = []
    names = {0: "nguoi_binh_thuong", 1: "doi_tuong_nguy_hiem"}
    last_kwargs = None

    def __init__(self, path):
        self.path = str(path)
        self.names = _FakeThreatYOLO.names

    def __call__(self, image, **kwargs):
        _FakeThreatYOLO.last_kwargs = kwargs
        return [_Result(list(_FakeThreatYOLO.boxes_to_return))]


@pytest.fixture
def weights(tmp_path):
    """A stub weights file, so the existence check passes."""
    path = tmp_path / "threat-yolo11n.pt"
    path.write_bytes(b"stub")
    return path


@pytest.fixture
def fake_threat_yolo(monkeypatch):
    _FakeThreatYOLO.boxes_to_return = []
    _FakeThreatYOLO.names = {0: "nguoi_binh_thuong", 1: "doi_tuong_nguy_hiem"}
    _FakeThreatYOLO.last_kwargs = None
    monkeypatch.setattr(threat_module, "_ULTRALYTICS_AVAILABLE", True)
    monkeypatch.setattr(threat_module, "_YOLO", _FakeThreatYOLO)
    return _FakeThreatYOLO


class TestClassMapping:
    """
    The upstream checkpoint documents English class names but ships
    Vietnamese ones. Matching on names detects nothing and looks exactly
    like a broken camera, so the mapping is pinned by index.
    """

    def test_class_index_one_is_dangerous(self, weights, fake_threat_yolo):
        fake_threat_yolo.boxes_to_return = [_Box(CLASS_DANGEROUS, 0.62)]
        clf = ThreatClassifier(weights)

        result = clf.classify_region(_frame(), (0, 0, 100, 200))

        assert result is not None
        assert result.is_dangerous is True
        assert result.confidence == pytest.approx(0.62)

    def test_class_index_zero_is_not_dangerous(self, weights, fake_threat_yolo):
        fake_threat_yolo.boxes_to_return = [_Box(CLASS_NORMAL, 0.86)]
        clf = ThreatClassifier(weights)

        result = clf.classify_region(_frame(), (0, 0, 100, 200))

        assert result is not None
        assert result.is_dangerous is False

    def test_mapping_survives_renamed_classes(self, weights, fake_threat_yolo):
        """
        A re-export with different display names must not change behavior:
        index 1 stays dangerous even if the label is English, Vietnamese,
        or absent entirely.
        """
        fake_threat_yolo.names = {0: "normal_person", 1: "potentially_dangerous_person"}
        fake_threat_yolo.boxes_to_return = [_Box(CLASS_DANGEROUS, 0.55)]
        clf = ThreatClassifier(weights)

        result = clf.classify_region(_frame(), (0, 0, 100, 200))

        assert result.is_dangerous is True

    def test_unexpected_class_names_warn(self, weights, fake_threat_yolo, caplog):
        """
        Verdicts hinge on class index, so a checkpoint whose indices mean
        something else inverts every result. It must warn: main.py runs
        uvicorn at log_level="warning" unless DEBUG, so an info line would
        never reach the operator on the demo unit.
        """
        fake_threat_yolo.names = {0: "cat", 1: "dog"}

        with caplog.at_level("WARNING"):
            ThreatClassifier(weights)

        assert any("classes differ" in r.message.lower() or "classes differ" in r.getMessage().lower()
                   for r in caplog.records), "drifted class names did not warn"

    def test_known_class_names_do_not_warn(self, weights, fake_threat_yolo, caplog):
        """The shipped checkpoint is the normal case and must stay quiet."""
        with caplog.at_level("WARNING"):
            ThreatClassifier(weights)

        assert not [r for r in caplog.records if "classes differ" in r.getMessage().lower()]

    def test_highest_confidence_dangerous_wins_among_dangerous(self, weights, fake_threat_yolo):
        fake_threat_yolo.boxes_to_return = [
            _Box(CLASS_DANGEROUS, 0.41),
            _Box(CLASS_DANGEROUS, 0.91),
        ]
        clf = ThreatClassifier(weights)

        result = clf.classify_region(_frame(), (0, 0, 100, 200))

        assert result.is_dangerous is True
        assert result.confidence == pytest.approx(0.91)

    def test_dangerous_beats_a_more_confident_normal(self, weights, fake_threat_yolo):
        """
        Regression, found running the real weights: on the upstream handgun
        sample the bystander scores 0.86 and the armed person 0.62. A plain
        argmax over all boxes reports the bystander and the threat vanishes.
        Any dangerous detection must outrank every normal one.
        """
        fake_threat_yolo.boxes_to_return = [
            _Box(CLASS_NORMAL, 0.86),
            _Box(CLASS_DANGEROUS, 0.62),
        ]
        clf = ThreatClassifier(weights)

        result = clf.classify_region(_frame(), (0, 0, 100, 200))

        assert result.is_dangerous is True, "a confident bystander masked an armed person"
        assert result.confidence == pytest.approx(0.62), (
            "confidence must describe the threat being reported, not the bystander"
        )

    def test_normal_reported_when_nothing_dangerous(self, weights, fake_threat_yolo):
        fake_threat_yolo.boxes_to_return = [
            _Box(CLASS_NORMAL, 0.51),
            _Box(CLASS_NORMAL, 0.77),
        ]
        clf = ThreatClassifier(weights)

        result = clf.classify_region(_frame(), (0, 0, 100, 200))

        assert result.is_dangerous is False
        assert result.confidence == pytest.approx(0.77)

    def test_no_detections_returns_none(self, weights, fake_threat_yolo):
        fake_threat_yolo.boxes_to_return = []
        clf = ThreatClassifier(weights)

        assert clf.classify_region(_frame(), (0, 0, 100, 200)) is None


class TestConstruction:
    """Failures at construction must name their own cause."""

    def test_missing_weights_raises(self, tmp_path, fake_threat_yolo):
        with pytest.raises(RuntimeError, match="not found"):
            ThreatClassifier(tmp_path / "absent.pt")

    def test_missing_ultralytics_raises(self, weights, monkeypatch):
        monkeypatch.setattr(threat_module, "_ULTRALYTICS_AVAILABLE", False)

        with pytest.raises(RuntimeError, match="ultralytics"):
            ThreatClassifier(weights)

    def test_inference_uses_configured_imgsz_and_confidence(self, weights, fake_threat_yolo):
        """832 is the trained size but too slow for a Pi; the override must reach the model."""
        clf = ThreatClassifier(weights, confidence=0.5, imgsz=320)
        clf.classify_region(_frame(), (0, 0, 100, 200))

        assert fake_threat_yolo.last_kwargs["imgsz"] == 320
        assert fake_threat_yolo.last_kwargs["conf"] == 0.5


class _ShapeRecordingYOLO(_FakeThreatYOLO):
    """Records the shape of every crop handed to the model."""

    shapes = []

    def __call__(self, image, **kwargs):
        _ShapeRecordingYOLO.shapes.append(image.shape[:2])
        return super().__call__(image, **kwargs)


class TestCropping:
    def test_out_of_bounds_bbox_is_clamped(self, weights, fake_threat_yolo):
        """A person box running off-frame must not produce an empty crop."""
        fake_threat_yolo.boxes_to_return = [_Box(CLASS_DANGEROUS, 0.7)]
        clf = ThreatClassifier(weights)

        result = clf.classify_region(_frame(), (-50, -50, 200, 200))

        assert result is not None

    def test_fully_outside_bbox_returns_none(self, weights, fake_threat_yolo):
        clf = ThreatClassifier(weights)

        assert clf.classify_region(_frame(), (5000, 5000, 10, 10)) is None

    def test_crop_is_padded_beyond_the_person_box(self, weights, monkeypatch):
        """
        Regression, found running the real weights: the model was trained on
        people in full scenes, so a crop tight to the person box cuts off the
        weapon and returns *nothing*. Padding by 15% recovers it at 0.699.
        A zero-padding regression would silently detect no threats at all.
        """
        _ShapeRecordingYOLO.shapes = []
        _ShapeRecordingYOLO.boxes_to_return = []
        monkeypatch.setattr(threat_module, "_ULTRALYTICS_AVAILABLE", True)
        monkeypatch.setattr(threat_module, "_YOLO", _ShapeRecordingYOLO)

        clf = ThreatClassifier(weights, crop_padding=0.15)
        clf.classify_region(_frame(), (200, 100, 100, 200))

        crop_h, crop_w = _ShapeRecordingYOLO.shapes[0]
        # pad = 0.15 * max(100, 200) = 30 on every side
        assert crop_w == 100 + 60, "crop was not padded horizontally"
        assert crop_h == 200 + 60, "crop was not padded vertically"

    def test_padding_is_clamped_at_frame_edges(self, weights, monkeypatch):
        """Padding must never index outside the frame."""
        _ShapeRecordingYOLO.shapes = []
        _ShapeRecordingYOLO.boxes_to_return = []
        monkeypatch.setattr(threat_module, "_ULTRALYTICS_AVAILABLE", True)
        monkeypatch.setattr(threat_module, "_YOLO", _ShapeRecordingYOLO)

        clf = ThreatClassifier(weights, crop_padding=0.5)
        result = clf.classify_region(_frame(), (0, 0, 640, 480))

        crop_h, crop_w = _ShapeRecordingYOLO.shapes[0]
        assert (crop_h, crop_w) == (480, 640)
        assert result is None  # no boxes configured, but it must not raise


class _StubClassifier:
    """Stands in for ThreatClassifier inside DetectionService."""

    def __init__(self, result=None, raises=False):
        self.result = result
        self.raises = raises
        self.calls = 0

    def classify_region(self, frame, bbox):
        self.calls += 1
        if self.raises:
            raise RuntimeError("inference exploded")
        return self.result


class _HumanYOLO:
    """A YOLO stand-in reporting one person."""

    last_path = None

    def __init__(self, path):
        _HumanYOLO.last_path = str(path)

    def __call__(self, frame, verbose=False):
        class _B:
            conf = [0.9]
            cls = [0]
            xywh = [np.array([200.0, 200.0, 100.0, 200.0])]

        class _R:
            names = {0: "person"}
            boxes = [_B()]

        return [_R()]


class _EmptyYOLO(_HumanYOLO):
    def __call__(self, frame, verbose=False):
        return []


@pytest.fixture
def human_yolo(monkeypatch):
    monkeypatch.setattr(detection_module, "_ULTRALYTICS_AVAILABLE", True)
    monkeypatch.setattr(detection_module, "_YOLO", _HumanYOLO)


class TestCascadeStage:
    """Where the threat stage sits in process_frame()."""

    def test_dangerous_person_detection_emitted(self, human_yolo):
        service = DetectionService()
        service.initialize()
        stub = _StubClassifier(ThreatResult(is_dangerous=True, confidence=0.62))
        service.attach_threat_classifier(stub)

        service.process_frame(_frame(0))
        detections = service.process_frame(_moving_frame())

        threats = [d for d in detections if d.type is DetectionType.DANGEROUS_PERSON]
        assert len(threats) == 1
        assert threats[0].confidence == pytest.approx(0.62)

    def test_normal_person_emits_no_threat(self, human_yolo):
        service = DetectionService()
        service.initialize()
        service.attach_threat_classifier(
            _StubClassifier(ThreatResult(is_dangerous=False, confidence=0.86))
        )

        service.process_frame(_frame(0))
        detections = service.process_frame(_moving_frame())

        assert not [d for d in detections if d.type is DetectionType.DANGEROUS_PERSON]

    def test_stage_skipped_when_no_humans(self, monkeypatch):
        """Threat classification is per-person; no person means no work."""
        monkeypatch.setattr(detection_module, "_ULTRALYTICS_AVAILABLE", True)
        monkeypatch.setattr(detection_module, "_YOLO", _EmptyYOLO)

        service = DetectionService()
        service.initialize()
        stub = _StubClassifier(ThreatResult(is_dangerous=True, confidence=0.9))
        service.attach_threat_classifier(stub)

        service.process_frame(_frame(0))
        service.process_frame(_moving_frame())

        assert stub.calls == 0

    def test_threat_failure_does_not_break_the_frame(self, human_yolo):
        """
        A crash in the newest, least-proven stage must not take down object
        detection -- that stage carries the demo on its own.
        """
        service = DetectionService()
        service.initialize()
        service.attach_threat_classifier(_StubClassifier(raises=True))

        service.process_frame(_frame(0))
        detections = service.process_frame(_moving_frame())

        humans = [d for d in detections if d.type is DetectionType.HUMAN]
        assert len(humans) == 1, "object detection lost when threat stage raised"

    def test_no_classifier_attached_is_inert(self, human_yolo):
        """The default path, with weights never fetched."""
        service = DetectionService()
        service.initialize()

        service.process_frame(_frame(0))
        detections = service.process_frame(_moving_frame())

        assert not [d for d in detections if d.type is DetectionType.DANGEROUS_PERSON]

    def test_threat_bbox_matches_the_person_bbox(self, human_yolo):
        service = DetectionService()
        service.initialize()
        service.attach_threat_classifier(
            _StubClassifier(ThreatResult(is_dangerous=True, confidence=0.7))
        )

        service.process_frame(_frame(0))
        detections = service.process_frame(_moving_frame())

        human = next(d for d in detections if d.type is DetectionType.HUMAN)
        threat = next(d for d in detections if d.type is DetectionType.DANGEROUS_PERSON)
        assert threat.bbox == human.bbox


class TestStatus:
    def test_status_reports_threat_detection_off_by_default(self):
        assert DetectionService().status()["threat_detection"] is False

    def test_status_reports_threat_detection_when_attached(self):
        service = DetectionService()
        service.attach_threat_classifier(_StubClassifier())

        assert service.status()["threat_detection"] is True

    def test_shutdown_releases_the_classifier(self):
        service = DetectionService()
        service.attach_threat_classifier(_StubClassifier())
        service.shutdown()

        assert service.threat_classifier is None


class TestAnnotationColor:
    """
    Without its own color a threat box draws green and reads as an
    ordinary person -- the one thing the operator must not miss.
    """

    def _det(self, det_type, face_id=None):
        return Detection(type=det_type, confidence=0.8, bbox=(0, 0, 10, 10), face_id=face_id)

    def test_threat_is_orange(self):
        assert box_color(self._det(DetectionType.DANGEROUS_PERSON)) == BOX_THREAT

    def test_threat_color_differs_from_person_and_unknown_face(self):
        assert BOX_THREAT != BOX_KNOWN
        assert BOX_THREAT != BOX_UNKNOWN

    def test_unknown_face_still_red(self):
        assert box_color(self._det(DetectionType.FACE)) == BOX_UNKNOWN

    def test_human_still_green(self):
        assert box_color(self._det(DetectionType.HUMAN)) == BOX_KNOWN


class TestThreatOverlay:
    """
    A flagged person produces two detections on one bbox. Drawing both
    stacks two labels on the same pixel -- unreadable, and it implies two
    subjects where there is one.
    """

    def _frame_and_dets(self):
        bbox = (100, 100, 80, 200)
        return (
            np.zeros((480, 640, 3), dtype=np.uint8),
            [
                Detection(type=DetectionType.HUMAN, confidence=0.86, bbox=bbox),
                Detection(type=DetectionType.DANGEROUS_PERSON, confidence=0.70, bbox=bbox),
            ],
        )

    def test_threat_box_supersedes_the_human_box(self):
        frame, dets = self._frame_and_dets()

        out = draw_detections(frame, dets)

        painted = {tuple(int(c) for c in px) for px in out.reshape(-1, 3) if px.any()}
        assert BOX_THREAT in painted, "threat box was not drawn"
        assert BOX_KNOWN not in painted, "redundant green human box drawn under the threat box"

    def test_unflagged_human_still_drawn(self):
        """Suppression must apply only to the person actually flagged."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = [
            Detection(type=DetectionType.HUMAN, confidence=0.86, bbox=(100, 100, 80, 200)),
            Detection(type=DetectionType.DANGEROUS_PERSON, confidence=0.70, bbox=(300, 100, 80, 200)),
        ]

        out = draw_detections(frame, dets)

        painted = {tuple(int(c) for c in px) for px in out.reshape(-1, 3) if px.any()}
        assert BOX_THREAT in painted
        assert BOX_KNOWN in painted, "a bystander's box was suppressed"

    def test_input_frame_is_not_modified(self):
        frame, dets = self._frame_and_dets()
        before = frame.copy()

        draw_detections(frame, dets)

        assert np.array_equal(frame, before)
