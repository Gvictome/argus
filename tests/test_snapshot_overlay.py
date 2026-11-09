"""
Event photo overlay tests.

The photos are how an operator decides what to label, so a box on the
wrong subject is worse than no box. These lock in: new events store the
subject's box, an old event's box is recovered from its crop and cached,
the endpoint draws by default and returns the clean file with ?overlay=0.
"""

import json

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.config import settings
from src.detection import Detection, DetectionType
from src.detection.snapshot_overlay import CROP_PAD, draw_overlay, event_box, locate_crop


def _scene(w=1920, h=1080, seed=0):
    rng = np.random.default_rng(seed)
    img = cv2.resize(rng.integers(0, 255, (h // 20, w // 20, 3), dtype=np.uint8), (w, h),
                     interpolation=cv2.INTER_CUBIC)
    return img


def _still_and_crop(box_px):
    """A 1920x1080 frame, its 640x360 still and the padded full-res crop."""
    frame = _scene()
    x1, y1, x2, y2 = box_px
    bw, bh = x2 - x1, y2 - y1
    cx1, cy1 = int(x1 - bw * CROP_PAD), int(y1 - bh * CROP_PAD)
    cx2, cy2 = int(x2 + bw * CROP_PAD), int(y2 + bh * CROP_PAD)
    crop = frame[cy1:cy2, cx1:cx2]
    still = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
    return still, crop


class TestLocate:
    def test_recovers_box_from_crop(self):
        truth = (600, 300, 1000, 700)
        still, crop = _still_and_crop(truth)
        box = locate_crop(still, crop)
        assert box is not None
        expected = (600 / 1920, 300 / 1080, 1000 / 1920, 700 / 1080)
        assert all(abs(a - b) < 0.01 for a, b in zip(box, expected))

    def test_unrelated_crop_is_rejected(self):
        still, _ = _still_and_crop((600, 300, 1000, 700))
        other = _scene(seed=7)[100:400, 100:400]
        assert locate_crop(still, other) is None

    def test_result_is_cached_beside_the_still(self, tmp_path):
        still, crop = _still_and_crop((600, 300, 1000, 700))
        cv2.imwrite(str(tmp_path / "ab.jpg"), still)
        cv2.imwrite(str(tmp_path / "ab_crop.jpg"), crop)
        meta = {"snapshot": "ab.jpg", "crop": "ab_crop.jpg"}
        first = event_box(meta, tmp_path)
        assert first is not None
        cached = json.loads((tmp_path / "ab.box.json").read_text())["box"]
        assert cached == [round(v, 4) for v in first]

    def test_stored_box_wins(self, tmp_path):
        assert event_box({"box": [0.1, 0.2, 0.3, 0.4]}, tmp_path) == (0.1, 0.2, 0.3, 0.4)


class TestDraw:
    def test_draws_a_box(self):
        img = np.zeros((360, 640, 3), np.uint8)
        ok, buf = cv2.imencode(".jpg", img)
        out = cv2.imdecode(np.frombuffer(draw_overlay(buf.tobytes(), (0.25, 0.25, 0.75, 0.75),
                                                      "Vehicle 84%", False), np.uint8), cv2.IMREAD_COLOR)
        assert out.shape == img.shape
        assert out[90, 320].sum() > 100  # top edge of the box is drawn


class TestCollectorStoresBox:
    def test_new_event_has_its_box(self, tmp_path):
        from src.federated.collector import EventCollector
        from src.federated.store import SampleStore

        store = SampleStore(tmp_path / "samples.jsonl")
        collector = EventCollector(store, snapshot_dir=tmp_path / "snapshots")
        for i in range(6):
            frame = np.full((480, 640, 3), 40, np.uint8)
            frame[100:300, 200:360] = 220
            det = Detection(type=DetectionType.HUMAN, confidence=0.9, bbox=(200, 100, 160, 200))
            collector.observe([det], (480, 640), when=1000.0 + i * 0.1, frame=frame)
        collector.flush()

        box = store.recent(1)[0]["meta"]["box"]
        assert box == [round(200 / 640, 4), round(100 / 480, 4), round(360 / 640, 4), round(300 / 480, 4)]


class TestEndpoint:
    @pytest.fixture
    def client(self, monkeypatch, tmp_path):
        monkeypatch.setattr(settings, "AUTH_REQUIRED", False)
        monkeypatch.setattr(settings, "DB_PATH", tmp_path / "o.db")
        monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
        with TestClient(create_app()) as c:
            yield c

    def test_overlay_by_default_and_clean_on_request(self, client, tmp_path):
        from src.federated.store import SampleStore

        snaps = tmp_path / "snaps"
        snaps.mkdir()
        img = np.full((360, 640, 3), 30, np.uint8)
        cv2.imwrite(str(snaps / "s1.jpg"), img)
        store = SampleStore(tmp_path / "samples.jsonl")
        x = [0.0] * 19
        x[1] = 1.0
        x[18] = 0.84
        n = store.append(x, label=None, meta={"snapshot": "s1.jpg", "box": [0.2, 0.2, 0.6, 0.8]})

        client.app.state.fl_store = store
        client.app.state.fl_collector = type("C", (), {"snapshot_dir": snaps, "flush": lambda self: 0})()
        row_n = store.recent(1)[0]["n"]

        clean = client.get(f"/api/events/{row_n}/snapshot?overlay=0")
        drawn = client.get(f"/api/events/{row_n}/snapshot")
        assert clean.status_code == drawn.status_code == 200
        assert clean.content == (snaps / "s1.jpg").read_bytes()
        assert drawn.content != clean.content
        assert drawn.headers["content-type"] == "image/jpeg"
