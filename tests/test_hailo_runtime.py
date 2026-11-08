"""HailoRT backend: decoding, HEF lookup, and fallback to CPU. No chip needed."""

import numpy as np
import pytest

import src.detection as det
from src.detection import DetectionConfig, DetectionService, DetectionType
from src.detection.hailo_runtime import decode, find_zoo_hef, letterbox

INPUT_HW = (640, 640)


def _by_class(n_classes=80, **boxes):
    """List-of-arrays NMS output; boxes given as cls<id>=[[ymin,xmin,ymax,xmax,score]]."""
    out = [np.zeros((0, 5), np.float32) for _ in range(n_classes)]
    for key, rows in boxes.items():
        out[int(key[3:])] = np.array(rows, np.float32)
    return out


def test_letterbox_pads_wide_frame_vertically():
    frame = np.zeros((480, 640, 3), np.uint8)
    img, scale, (px, py) = letterbox(frame, INPUT_HW)
    assert img.shape == (640, 640, 3)
    assert scale == pytest.approx(1.0)
    assert (px, py) == (0, 80)


def test_decode_maps_box_back_to_frame_pixels():
    # 1280x720 frame -> scale 0.5, pad y = (640 - 360) / 2 = 140
    frame_hw = (720, 1280)
    _, scale, pad = letterbox(np.zeros((*frame_hw, 3), np.uint8), INPUT_HW)
    # person spanning x 100..300, y 200..400 in frame pixels
    ymin, ymax = (200 * scale + pad[1]) / 640, (400 * scale + pad[1]) / 640
    xmin, xmax = (100 * scale) / 640, (300 * scale) / 640
    buf = _by_class(cls0=[[ymin, xmin, ymax, xmax, 0.9]])

    boxes = decode(buf, 0.5, None, INPUT_HW, scale, pad, frame_hw)

    assert len(boxes) == 1
    cls_id, score, x, y, w, h = boxes[0]
    assert (cls_id, round(score, 2)) == (0, 0.9)
    assert (x, y) == pytest.approx((100, 200), abs=1)
    assert (w, h) == pytest.approx((200, 200), abs=1)


def test_decode_filters_low_scores_and_unwanted_classes():
    buf = _by_class(cls0=[[0.1, 0.1, 0.5, 0.5, 0.3]], cls56=[[0.1, 0.1, 0.5, 0.5, 0.9]])
    assert decode(buf, 0.5, {0}, INPUT_HW, 1.0, (0, 0), INPUT_HW) == []


def test_decode_accepts_flat_count_prefixed_layout():
    # class 0: one box, class 1: none, class 2: one box
    flat = np.array([1, 0.1, 0.1, 0.5, 0.5, 0.8, 0, 1, 0.2, 0.2, 0.6, 0.6, 0.7], np.float32)
    boxes = decode(flat, 0.5, None, INPUT_HW, 1.0, (0, 0), INPUT_HW)
    assert [b[0] for b in boxes] == [0, 2]


def test_find_zoo_hef_picks_newest_version(tmp_path):
    zoo = tmp_path / "models" / "hailo10h"
    zoo.mkdir(parents=True)
    for name in ("yolov8n_v5.1.0.hef", "yolov8n_v5.2.0.hef", "yolov11n_v5.2.0.hef"):
        (zoo / name).write_bytes(b"x")
    assert find_zoo_hef(tmp_path, "yolov8n").name == "yolov8n_v5.2.0.hef"
    assert find_zoo_hef(tmp_path, "yolo11n").name == "yolov11n_v5.2.0.hef"
    assert find_zoo_hef(tmp_path, "yolov8s") is None


class _FakeHailo:
    input_hw = INPUT_HW

    def __init__(self, hef):
        self.hef = hef

    def infer(self, frame, conf, classes):
        return [(0, 0.88, 10, 20, 30, 40), (2, 0.7, 1, 2, 3, 4)]


@pytest.fixture
def zoo_dir(tmp_path, monkeypatch):
    zoo = tmp_path / "models" / "hailo10h"
    zoo.mkdir(parents=True)
    (zoo / "yolov8n_v5.2.0.hef").write_bytes(b"x")
    monkeypatch.setattr(det, "BASE_DIR", tmp_path)
    return tmp_path


def test_service_uses_hailo_when_runtime_loads(zoo_dir, monkeypatch):
    monkeypatch.setattr(det, "HailoYolo", _FakeHailo)
    svc = DetectionService(DetectionConfig(hailo=True))
    svc.initialize()

    assert svc.backend == "hailo"
    found = svc.detect_objects(np.zeros((480, 640, 3), np.uint8))
    assert [(d.type, d.label, d.bbox) for d in found] == [
        (DetectionType.HUMAN, "person", (10, 20, 30, 40)),
        (DetectionType.VEHICLE, "car", (1, 2, 3, 4)),
    ]


def test_service_falls_back_to_cpu_when_hailort_fails(zoo_dir, monkeypatch):
    def broken(_hef):
        raise ImportError("No module named 'hailo_platform'")

    monkeypatch.setattr(det, "HailoYolo", broken)
    monkeypatch.setattr(det, "_YOLO", lambda *a, **k: object())
    monkeypatch.setattr(det, "_ULTRALYTICS_AVAILABLE", True)
    svc = DetectionService(DetectionConfig(hailo=True))
    svc.initialize()

    assert svc.backend == "cpu"


def test_hailo_off_by_default_even_with_hef_present(zoo_dir, monkeypatch):
    monkeypatch.setattr(det, "HailoYolo", _FakeHailo)
    monkeypatch.setattr(det, "_YOLO", lambda *a, **k: object())
    monkeypatch.setattr(det, "_ULTRALYTICS_AVAILABLE", True)
    svc = DetectionService()
    svc.initialize()

    assert svc.backend == "cpu"
