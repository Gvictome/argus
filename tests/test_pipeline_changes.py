"""
The 2026-09-14 pipeline changes: face stage deprecated, downscaled motion,
continuous background detection, file camera source, and the validation
gate that decides whether a federated update replaces the live model.
"""

import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from src.detection import Detection, DetectionService, DetectionType
from src.detection.worker import DetectionWorker


def _frame(value, shape=(1080, 1920, 3)):
    return np.full(shape, value, dtype=np.uint8)


class TestFaceStageDeprecated:
    def test_faces_never_run_by_default(self):
        """Even with a recognizer attached and motion present, the stage is off."""

        class _Recognizer:
            calls = 0

            def recognize(self, frame):
                _Recognizer.calls += 1
                return []

        service = DetectionService()
        service.object_model = None
        service.attach_face_recognizer(_Recognizer())

        service.process_frame(_frame(0))
        moved = _frame(0)
        moved[200:700, 300:900] = 255
        service.process_frame(moved)

        assert _Recognizer.calls == 0
        assert service.status()["faces_enabled"] is False


class TestDownscaledMotion:
    def test_first_frame_only_primes(self):
        assert DetectionService().detect_motion(_frame(0)) == []

    def test_boxes_come_back_in_full_resolution_coordinates(self):
        """Motion runs on a 320px copy; boxes must still land on the object."""
        service = DetectionService()
        service.detect_motion(_frame(0))
        moved = _frame(0)
        moved[400:700, 900:1200] = 255  # a 300x300 block

        detections = service.detect_motion(moved)

        assert detections, "motion was missed"
        x, y, w, h = max(detections, key=lambda d: d.bbox[2] * d.bbox[3]).bbox
        # Blur and dilation at the small scale widen the box a little; it
        # must still sit on the block, in full-frame units.
        assert 820 <= x <= 920 and 320 <= y <= 420
        assert 280 <= w <= 460 and 280 <= h <= 460

    def test_static_scene_reports_no_motion(self):
        service = DetectionService()
        service.detect_motion(_frame(40))
        assert service.detect_motion(_frame(40)) == []


class _FakeCamera:
    def __init__(self, frames):
        self.frames = frames
        self.i = 0
        self.is_initialized = False

    def initialize(self):
        self.is_initialized = True
        return True

    def get_frame_array(self):
        time.sleep(0.004)
        frame = self.frames[self.i % len(self.frames)]
        self.i += 1
        return frame


class _SlowDetector:
    def __init__(self):
        self.seen = 0

    def process_frame(self, frame):
        time.sleep(0.03)
        self.seen += 1
        return [Detection(type=DetectionType.HUMAN, confidence=0.9, bbox=(1, 1, 10, 10))]


class TestDetectionWorker:
    def test_detects_with_no_viewer_and_skips_stale_frames(self):
        camera = _FakeCamera([_frame(i, (60, 80, 3)) for i in range(5)])
        detector = _SlowDetector()
        worker = DetectionWorker(camera, detector)

        worker.start()
        deadline = time.time() + 3
        while time.time() < deadline and detector.seen < 5:
            time.sleep(0.02)
        _, detections, _ = worker.latest()
        status = worker.status()
        worker.stop()

        assert detector.seen >= 5, "detection did not run without a stream"
        assert detections and detections[0].type is DetectionType.HUMAN
        # Capture outpaces the detector, which works on the newest frame
        # instead of queueing every one.
        assert status["frames_captured"] > status["frames_detected"]

    def test_camera_failure_is_reported_not_raised(self):
        class _DeadCamera:
            is_initialized = False

            def initialize(self):
                return False

        worker = DetectionWorker(_DeadCamera(), _SlowDetector())
        worker.start()
        time.sleep(0.2)

        assert worker.status()["camera_error"]
        assert worker.running is False
        worker.stop()


class TestFileCameraSource:
    def test_file_source_loops_at_end_of_stream(self, tmp_path):
        from src.camera.backends import OpenCVBackend

        path = tmp_path / "clip.avi"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (64, 48))
        if not writer.isOpened():
            pytest.skip("this OpenCV build cannot write MJPG")
        for i in range(3):
            writer.write(_frame(i * 80, (48, 64, 3)))
        writer.release()

        backend = OpenCVBackend(index=str(path), realtime=False)
        assert backend.open()
        frames = [backend.read() for _ in range(8)]
        backend.close()

        assert all(f is not None for f in frames), "file source did not loop"


class TestModelGate:
    @pytest.fixture
    def store(self, tmp_path):
        from src.federated.store import SampleStore

        s = SampleStore(tmp_path / "samples.jsonl")
        s.bootstrap("driveway", 1500, seed=3)
        return s

    @pytest.fixture
    def gate(self, tmp_path, monkeypatch, store):
        from src.federated import model_state
        from src.federated.head import FederatedHead

        monkeypatch.setattr(model_state.settings, "DATA_DIR", tmp_path)
        app = SimpleNamespace(state=SimpleNamespace(
            fl_head=FederatedHead(seed=0), fl_store=store, fl_model_state=None))
        return app, model_state

    def _trained(self, store, tmp_path, name):
        from src.federated.head import FederatedHead
        from src.federated.store import holdout_split

        X, y = store.labelled()
        Xt, yt, _, _ = holdout_split(X, y, seed=0)
        head = FederatedHead(seed=0)
        head.fit(Xt, yt, epochs=6)
        path = tmp_path / name
        head.save(path)
        return head, path

    def test_first_model_is_accepted_swapped_in_and_persisted(self, gate, store, tmp_path):
        app, model_state = gate
        trained, path = self._trained(store, tmp_path, "c1.npz")

        decision = model_state.accept_candidate(app, path, "federated")

        assert decision["accepted"] and decision["version"] == 1
        assert model_state.head_path().exists()
        assert model_state.model_version(app) == "fl-r1"
        assert all(np.array_equal(a, b) for a, b in
                   zip(app.state.fl_head.get_weights(), trained.get_weights()))

    def test_regressing_candidate_is_rejected_and_live_model_kept(self, gate, store, tmp_path):
        from src.federated.head import FederatedHead

        app, model_state = gate
        _, good = self._trained(store, tmp_path, "good.npz")
        model_state.accept_candidate(app, good, "federated")
        before = [w.copy() for w in app.state.fl_head.get_weights()]

        bad = tmp_path / "bad.npz"
        FederatedHead(seed=9).save(bad)  # untrained
        decision = model_state.accept_candidate(app, bad, "federated")

        assert decision["accepted"] is False
        assert "regressed" in decision["reason"]
        assert all(np.array_equal(a, b) for a, b in
                   zip(app.state.fl_head.get_weights(), before))
        assert model_state.model_version(app) == "fl-r1"

    def test_saved_head_is_restored_at_startup(self, gate, store, tmp_path):
        from src.federated.head import FederatedHead

        app, model_state = gate
        trained, path = self._trained(store, tmp_path, "c.npz")
        model_state.accept_candidate(app, path, "local-train")

        fresh = SimpleNamespace(state=SimpleNamespace(fl_head=FederatedHead(seed=4)))
        model_state.load_model_state(fresh)

        assert all(np.array_equal(a, b) for a, b in
                   zip(fresh.state.fl_head.get_weights(), trained.get_weights()))
        assert model_state.model_version(fresh) == "local-r1"


class TestDashboardCorrections:
    """The dashboard's Correct box is free text; every phrase must resolve."""

    ROUTINE_PERSON = {"cls": "person", "status": "na"}
    FLAGGED_VEHICLE = {"cls": "vehicle", "status": "stranger"}

    @pytest.mark.parametrize("text, current, expected", [
        ("Suspicious person", ROUTINE_PERSON, 4),
        ("Unrecognized person", ROUTINE_PERSON, 4),
        ("Delivery driver", ROUTINE_PERSON, 0),
        ("it's a parcel", ROUTINE_PERSON, 3),
        ("dog", ROUTINE_PERSON, 2),
        ("false alarm", FLAGGED_VEHICLE, 1),
        ("resident", FLAGGED_VEHICLE, 1),
        ("routine_animal", ROUTINE_PERSON, 2),
    ])
    def test_free_text_resolves_to_intent(self, text, current, expected):
        from src.api.dashboard_routes import _resolve_label

        assert _resolve_label("correct", text, current) == expected

    def test_unmatched_text_means_the_call_was_wrong(self):
        """Gibberish never 422s; it inverts what was shown."""
        from src.api.dashboard_routes import _resolve_label

        assert _resolve_label("correct", "asdf qwerty", self.ROUTINE_PERSON) == 4
        assert _resolve_label("correct", "asdf qwerty", self.FLAGGED_VEHICLE) == 1

    def test_confirm_keeps_what_was_shown(self):
        from src.api.dashboard_routes import _resolve_label

        assert _resolve_label("confirm", None, self.FLAGGED_VEHICLE) == 4
        assert _resolve_label("confirm", None, self.ROUTINE_PERSON) == 0


class TestMotionSegments:
    class _Store:
        def append(self, *args, **kwargs):
            return "unused"

    class _DB:
        def __init__(self):
            self.events = []

        def log_event(self, **kwargs):
            self.events.append(kwargs)

    def _motion(self):
        return [Detection(type=DetectionType.MOTION, confidence=1.0, bbox=(0, 0, 400, 400))]

    def test_unbroken_motion_is_logged_in_chunks(self):
        """A scene that never goes still must still produce motion events."""
        from src.federated.collector import EventCollector

        db = self._DB()
        collector = EventCollector(self._Store(), db=db, motion_max_s=10.0)
        for i in range(260):  # 26 s of continuous motion at 10 fps
            collector.observe(self._motion(), (720, 1280), when=1000.0 + i * 0.1)

        motion = [e for e in db.events if e["event_type"] == "motion"]
        assert len(motion) >= 2

    def test_brief_flicker_is_not_logged(self):
        from src.federated.collector import EventCollector

        db = self._DB()
        collector = EventCollector(self._Store(), db=db)
        collector.observe(self._motion(), (720, 1280), when=1000.0)
        collector.observe([], (720, 1280), when=1000.1)
        collector.observe([], (720, 1280), when=1005.0)
        collector.flush()

        assert not [e for e in db.events if e["event_type"] == "motion"]


class TestHeadWeightIsolation:
    def test_training_a_copy_never_changes_the_source_head(self):
        """Regression: set_weights aliased the arrays, so training a "copy"
        trained the live model in place and the validation gate compared
        a model against itself."""
        from src.federated.head import FederatedHead
        from src.federated.store import SampleStore, holdout_split

        live = FederatedHead(seed=0)
        before = [w.copy() for w in live.get_weights()]

        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as d:
            store = SampleStore(Path(d) / "s.jsonl")
            store.bootstrap("driveway", 600, seed=2)
            X, y = store.labelled()
        Xt, yt, _, _ = holdout_split(X, y)

        candidate = FederatedHead(seed=0)
        candidate.set_weights(live.get_weights())
        candidate.fit(Xt, yt, epochs=2)

        assert all(np.array_equal(a, b) for a, b in zip(live.get_weights(), before))
        assert not all(np.array_equal(a, b) for a, b in
                       zip(candidate.get_weights(), before))


class TestCameraColourOrder:
    """Red and blue were exchanged on the Pi path, so people looked blue."""

    class _Cam:
        def __init__(self, arr):
            self.arr = arr

        def capture_array(self):
            return self.arr

    def _frame(self):
        arr = np.zeros((2, 2, 3), dtype=np.uint8)
        arr[..., 0] = 10    # first channel
        arr[..., 2] = 200   # third channel
        return arr

    def test_frames_pass_through_untouched_by_default(self):
        """picamera2's "RGB888" is already BGR; reversing it is the bug."""
        from src.camera.backends import Picamera2Backend

        backend = Picamera2Backend()
        backend._cam = self._Cam(self._frame())

        out = backend.read()

        assert out[0, 0, 0] == 10 and out[0, 0, 2] == 200

    def test_swap_rb_exchanges_channels_and_keeps_it_contiguous(self):
        """A negative-stride view breaks OpenCV calls, including JPEG encode."""
        from src.camera.backends import Picamera2Backend

        backend = Picamera2Backend(swap_rb=True)
        backend._cam = self._Cam(self._frame())

        out = backend.read()

        assert out[0, 0, 0] == 200 and out[0, 0, 2] == 10
        assert out.flags["C_CONTIGUOUS"]

    def test_setting_reaches_the_backend(self):
        from src.camera import CameraConfig
        from src.camera.backends import build_backend
        from src.camera.platform_detect import Board

        backend = build_backend(Board.RASPBERRY_PI, 0, CameraConfig(swap_rb=True))

        assert backend.swap_rb is True


class TestArgusClassTaxonomy:
    """person / vehicle / animal / package, and nothing landing in the wrong one."""

    def test_a_bicycle_is_a_vehicle(self):
        """Regression: COCO id 1 was mapped to ANIMAL."""
        from src.detection import _YOLO_CLASS_MAP

        assert _YOLO_CLASS_MAP[1] is DetectionType.VEHICLE

    def test_carried_items_are_packages(self):
        from src.detection import _YOLO_CLASS_MAP

        for coco_id in (24, 26, 28):  # backpack, handbag, suitcase
            assert _YOLO_CLASS_MAP[coco_id] is DetectionType.PACKAGE

    def test_every_mapped_class_is_one_of_the_four(self):
        from src.detection import _YOLO_CLASS_MAP

        assert set(_YOLO_CLASS_MAP.values()) == {
            DetectionType.HUMAN, DetectionType.VEHICLE,
            DetectionType.ANIMAL, DetectionType.PACKAGE,
        }

    def test_the_class_filter_matches_the_map(self):
        from src.detection import _RELEVANT_COCO_IDS, _YOLO_CLASS_MAP

        assert _RELEVANT_COCO_IDS == sorted(_YOLO_CLASS_MAP)

    def test_packages_reach_the_federated_head(self):
        from src.federated.features import class_index

        assert class_index("package") == 3
        assert class_index("vehicle") == 1

    def test_a_delivery_is_worth_recording(self):
        from src.detection.event_recorder import DEFAULT_TRIGGERS

        assert DetectionType.PACKAGE in DEFAULT_TRIGGERS


class TestModelSelection:
    def test_export_folder_follows_the_model_name(self):
        from src.detection import cpu_export_dir

        assert cpu_export_dir("ncnn", "yolov8s") == "yolov8s_ncnn_model"
        assert cpu_export_dir("openvino", "yolov8n") == "yolov8n_openvino_model"

    def test_a_bigger_model_loads_its_own_export(self, tmp_path, monkeypatch):
        import src.detection as detection_module

        export = tmp_path / "models" / "yolov8s_ncnn_model"
        export.mkdir(parents=True)
        (export / "metadata.yaml").write_text("task: detect\nimgsz:\n- 640\n- 640\n")

        loaded = {}

        class _YOLO:
            def __init__(self, path, **kwargs):
                loaded["path"] = str(path)

        monkeypatch.setattr(detection_module, "BASE_DIR", tmp_path)
        monkeypatch.setattr(detection_module, "_ULTRALYTICS_AVAILABLE", True)
        monkeypatch.setattr(detection_module, "_YOLO", _YOLO)

        service = detection_module.DetectionService(
            detection_module.DetectionConfig(model_name="yolov8s", cpu_export="auto"))
        service.initialize()

        assert service.backend == "cpu-ncnn"
        assert "yolov8s_ncnn_model" in loaded["path"]
        assert service.status()["model"] == "yolov8s"

    def test_without_an_export_it_loads_the_named_weights(self, tmp_path, monkeypatch):
        import src.detection as detection_module

        loaded = {}

        class _YOLO:
            def __init__(self, path, **kwargs):
                loaded["path"] = str(path)

        monkeypatch.setattr(detection_module, "BASE_DIR", tmp_path)
        monkeypatch.setattr(detection_module, "_ULTRALYTICS_AVAILABLE", True)
        monkeypatch.setattr(detection_module, "_YOLO", _YOLO)

        service = detection_module.DetectionService(
            detection_module.DetectionConfig(model_name="yolov8m", cpu_export="auto"))
        service.initialize()

        assert service.backend == "cpu"
        assert loaded["path"] == "yolov8m.pt"


class TestSnapshotsAndClips:
    """Every stored event keeps a still; every clip lands in the events table."""

    def _collector(self, tmp_path):
        from src.federated.collector import EventCollector
        from src.federated.store import SampleStore

        class _DB:
            def __init__(self):
                self.events = []

            def log_event(self, **kwargs):
                self.events.append(kwargs)

        db = _DB()
        store = SampleStore(tmp_path / "samples.jsonl")
        collector = EventCollector(store, db=db, snapshot_dir=tmp_path / "snapshots")
        return collector, store, db

    def test_event_keeps_a_snapshot_and_records_its_path(self, tmp_path):
        collector, store, db = self._collector(tmp_path)

        for i in range(6):
            frame = _frame(40, (480, 640, 3))
            frame[100:300, 200 + i * 5:360 + i * 5] = 220
            det = Detection(type=DetectionType.HUMAN, confidence=0.9,
                            bbox=(200 + i * 5, 100, 160, 200))
            collector.observe([det], (480, 640), when=1000.0 + i * 0.1, frame=frame)
        collector.flush()

        files = list((tmp_path / "snapshots").glob("*.jpg"))
        assert len(files) == 1, "no still saved for the event"
        assert files[0].stat().st_size > 0

        row = store.recent(1)[0]
        assert row["meta"]["snapshot"] == files[0].name
        detection_events = [e for e in db.events if e["event_type"] == "detection"]
        assert detection_events[0]["media_path"].endswith(files[0].name)

    def test_no_frame_means_no_snapshot_but_still_an_event(self, tmp_path):
        """Callers without a frame keep working."""
        collector, store, db = self._collector(tmp_path)

        for i in range(6):
            det = Detection(type=DetectionType.HUMAN, confidence=0.9, bbox=(10, 10, 60, 90))
            collector.observe([det], (480, 640), when=2000.0 + i * 0.1)
        collector.flush()

        assert store.stats()["total"] == 1
        assert not list((tmp_path / "snapshots").glob("*.jpg"))

    def test_discarded_flicker_leaves_no_snapshot_behind(self, tmp_path):
        collector, store, db = self._collector(tmp_path)

        frame = _frame(40, (480, 640, 3))
        det = Detection(type=DetectionType.HUMAN, confidence=0.9, bbox=(10, 10, 60, 90))
        collector.observe([det], (480, 640), when=3000.0, frame=frame)
        collector.flush()

        assert store.stats()["total"] == 0
        assert not list((tmp_path / "snapshots").glob("*.jpg"))
        assert collector._snaps == {}

    def test_finished_clip_is_logged_to_the_events_table(self, tmp_path):
        from src.detection.event_recorder import EventRecorder, RecorderConfig

        logged = []
        recorder = EventRecorder(
            RecorderConfig(pre_roll_s=0.0, post_roll_s=0.1, fps=5,
                           output_dir=tmp_path / "clips"),
            camera_name="primary",
            on_clip=lambda record: logged.append(record),
        )
        person = Detection(type=DetectionType.HUMAN, confidence=0.9, bbox=(10, 10, 40, 60))
        frame = _frame(50, (120, 160, 3))

        recorder.process(frame, [person], now=100.0)
        recorder.process(frame, [person], now=100.1)
        recorder.process(frame, [], now=100.5)  # past post-roll, clip closes
        recorder.close()

        assert logged, "clip finished without notifying anyone"
        assert logged[0].path.endswith(".mp4")
        assert logged[0].frames >= 1


class TestFixedShapeExports:
    """An export compiled at 640 returns nothing when asked for 416."""

    def test_configured_size_yields_to_the_export_size(self, tmp_path, monkeypatch):
        import src.detection as detection_module

        export = tmp_path / "models" / "yolov8n_openvino_model"
        export.mkdir(parents=True)
        (export / "metadata.yaml").write_text("task: detect\nimgsz:\n- 640\n- 640\n")

        class _ExportedYOLO:
            def __init__(self, path, **kwargs):
                self.path = path

            def __call__(self, frame, **kwargs):
                return []

        monkeypatch.setattr(detection_module, "BASE_DIR", tmp_path)
        monkeypatch.setattr(detection_module, "_ULTRALYTICS_AVAILABLE", True)
        monkeypatch.setattr(detection_module, "_YOLO", _ExportedYOLO)

        service = detection_module.DetectionService(
            detection_module.DetectionConfig(object_imgsz=416, cpu_export="openvino"))
        service.initialize()

        assert service.backend == "cpu-openvino"
        assert service.config.object_imgsz == 640

    def test_plain_weights_keep_the_configured_size(self, tmp_path, monkeypatch):
        import src.detection as detection_module

        class _Weights:
            def __init__(self, path, **kwargs):
                self.path = path

        monkeypatch.setattr(detection_module, "BASE_DIR", tmp_path)
        monkeypatch.setattr(detection_module, "_ULTRALYTICS_AVAILABLE", True)
        monkeypatch.setattr(detection_module, "_YOLO", _Weights)

        service = detection_module.DetectionService(
            detection_module.DetectionConfig(object_imgsz=416, cpu_export="auto"))
        service.initialize()

        assert service.backend == "cpu"
        assert service.config.object_imgsz == 416
