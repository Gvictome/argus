"""
FaceRecognitionService tests.

insightface is faked at the module boundary so the enrollment and matching
logic is testable without ONNX runtimes or a 300MB model download -- the
buffalo_l pack lives on the Pi, not on every machine that runs the suite.
The database is the real sqlite Database against a tmp file, so the faces
schema and the pickle round-trip are exercised for real.
"""

import numpy as np
import pytest

from src.database import Database
import src.detection.face_recognition as fr
from src.detection.face_recognition import FaceRecognitionService

THRESHOLD = 0.4


def unit_vector(seed: int, dim: int = 512) -> np.ndarray:
    """A deterministic normalized 512-d vector, standing in for an embedding."""
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim).astype(np.float32)
    return vec / np.linalg.norm(vec)


def blend(a: np.ndarray, b: np.ndarray, weight: float) -> np.ndarray:
    """Normalized mix of two embeddings, for tuning cosine similarity."""
    mixed = weight * a + (1.0 - weight) * b
    return mixed / np.linalg.norm(mixed)


class FakeFace:
    """Mirrors the attributes DetectionService reads off an insightface face."""

    def __init__(self, embedding, bbox=(100, 120, 180, 220)):
        self.bbox = np.array(bbox, dtype=np.float32)  # x1, y1, x2, y2
        self.normed_embedding = embedding


@pytest.fixture
def detected_faces():
    """Mutable list the fake detector returns; tests append FakeFace to it."""
    return []


@pytest.fixture
def service(monkeypatch, tmp_path, detected_faces):
    """A FaceRecognitionService backed by a real sqlite DB and a fake detector."""

    class _FakeFaceAnalysis:
        def __init__(self, name=None, providers=None):
            self.name = name

        def prepare(self, ctx_id=0, det_size=(640, 640)):
            pass

        def get(self, frame):
            return list(detected_faces)

    monkeypatch.setattr(fr, "_INSIGHTFACE_AVAILABLE", True)
    monkeypatch.setattr(fr, "_FaceAnalysis", _FakeFaceAnalysis)

    db = Database(tmp_path / "test.db")
    db.initialize()
    svc = FaceRecognitionService(db=db, similarity_threshold=THRESHOLD)
    yield svc
    db.shutdown()


def frame() -> np.ndarray:
    return np.zeros((480, 640, 3), dtype=np.uint8)


class TestReset:
    """
    Showcase P1-6 / G-4: the known-faces database must be clearable in one
    call between judges. An unpruned database enrolled over a multi-hour
    showcase degrades match quality until it produces a false positive,
    which is worse in front of a judge than a miss.
    """

    def test_reset_removes_every_enrolled_face(self, service, detected_faces):
        detected_faces.append(FakeFace(unit_vector(1)))
        service.enroll_face(frame(), "Giovanny")
        detected_faces[:] = [FakeFace(unit_vector(2))]
        service.enroll_face(frame(), "Adam")

        removed = service.reset()

        assert removed == 2
        assert service.list_known_faces() == []
        assert service.db.list_faces() == []

    def test_reset_on_empty_database_removes_nothing(self, service):
        assert service.reset() == 0

    def test_reset_makes_a_previously_known_face_unrecognized(self, service, detected_faces):
        """The point of the reset: identity must actually stop resolving."""
        embedding = unit_vector(3)
        detected_faces.append(FakeFace(embedding))
        service.enroll_face(frame(), "Giovanny")

        service.reset()
        matches = service.recognize(frame())

        assert len(matches) == 1
        assert matches[0].face_id is None
        assert matches[0].name is None


class TestEnrollment:
    def test_enroll_persists_the_embedding_and_caches_it(self, service, detected_faces):
        embedding = unit_vector(1)
        detected_faces.append(FakeFace(embedding))

        face_id = service.enroll_face(frame(), "Giovanny")

        assert face_id is not None
        stored = service.db.get_face(face_id)
        assert stored["name"] == "Giovanny"
        assert service.list_known_faces() == [{"face_id": face_id, "name": "Giovanny"}]

    def test_enrolled_embedding_survives_the_database_round_trip(self, service, detected_faces):
        """The embedding is pickled into a BLOB; a lossy round-trip would
        silently degrade every later match."""
        embedding = unit_vector(1)
        detected_faces.append(FakeFace(embedding))
        face_id = service.enroll_face(frame(), "Giovanny")

        service.reload()

        cached = service._known_faces[face_id][1]
        np.testing.assert_allclose(cached, embedding, rtol=1e-6)

    def test_enroll_returns_none_when_no_face_is_detected(self, service):
        assert service.enroll_face(frame(), "Nobody") is None

    def test_enroll_picks_the_largest_face_in_frame(self, service, detected_faces):
        """A bystander in the background must not be enrolled as the subject."""
        small = unit_vector(1)
        large = unit_vector(2)
        detected_faces.append(FakeFace(small, bbox=(0, 0, 20, 20)))
        detected_faces.append(FakeFace(large, bbox=(100, 100, 400, 400)))

        face_id = service.enroll_face(frame(), "Giovanny")

        np.testing.assert_allclose(service._known_faces[face_id][1], large, rtol=1e-6)


class TestRecognition:
    def test_recognize_identifies_an_enrolled_face(self, service, detected_faces):
        embedding = unit_vector(1)
        detected_faces.append(FakeFace(embedding))
        face_id = service.enroll_face(frame(), "Giovanny")

        matches = service.recognize(frame())

        assert len(matches) == 1
        assert matches[0].face_id == face_id
        assert matches[0].name == "Giovanny"
        assert matches[0].confidence == pytest.approx(1.0, abs=1e-5)

    def test_recognize_reports_unknown_below_the_similarity_threshold(self, service, detected_faces):
        """A stranger must resolve to unknown. Tuning this threshold is G-6;
        this test pins the behavior the threshold governs."""
        enrolled = unit_vector(1)
        detected_faces.append(FakeFace(enrolled))
        service.enroll_face(frame(), "Giovanny")

        stranger = blend(enrolled, unit_vector(99), 0.2)  # cosine ~0.24, under 0.4
        detected_faces[:] = [FakeFace(stranger)]
        matches = service.recognize(frame())

        assert len(matches) == 1
        assert matches[0].face_id is None
        assert matches[0].name is None

    def test_recognize_matches_a_near_but_not_identical_embedding(self, service, detected_faces):
        """The same person never produces a byte-identical embedding twice."""
        enrolled = unit_vector(1)
        detected_faces.append(FakeFace(enrolled))
        service.enroll_face(frame(), "Giovanny")

        same_person = blend(enrolled, unit_vector(99), 0.9)  # cosine ~0.99
        detected_faces[:] = [FakeFace(same_person)]

        assert service.recognize(frame())[0].name == "Giovanny"

    def test_recognize_converts_bbox_to_top_left_width_height(self, service, detected_faces):
        """insightface reports x1,y1,x2,y2; the Detection contract is x,y,w,h."""
        detected_faces.append(FakeFace(unit_vector(1), bbox=(100, 120, 180, 220)))

        assert service.recognize(frame())[0].bbox == (100, 120, 80, 100)

    def test_recognize_records_last_seen_for_a_match(self, service, detected_faces):
        detected_faces.append(FakeFace(unit_vector(1)))
        face_id = service.enroll_face(frame(), "Giovanny")
        assert service.db.get_face(face_id)["last_seen"] is None

        service.recognize(frame())

        assert service.db.get_face(face_id)["last_seen"] is not None

    def test_recognize_returns_empty_when_no_faces_present(self, service):
        assert service.recognize(frame()) == []

    def test_recognize_in_region_offsets_bbox_back_to_full_frame(self, service, detected_faces):
        """Coordinates come back in crop space; unadjusted, every box drawn
        from a YOLO person-crop lands in the wrong place on the stream."""
        detected_faces.append(FakeFace(unit_vector(1), bbox=(10, 20, 60, 90)))

        matches = service.recognize_in_region(frame(), bbox=(100, 100, 300, 300))

        assert matches[0].bbox == (110, 120, 50, 70)

    def test_recognize_in_region_returns_empty_for_an_offscreen_region(self, service, detected_faces):
        detected_faces.append(FakeFace(unit_vector(1)))

        assert service.recognize_in_region(frame(), bbox=(9999, 9999, 10, 10)) == []


class TestRemoval:
    def test_remove_face_clears_database_and_cache(self, service, detected_faces):
        detected_faces.append(FakeFace(unit_vector(1)))
        face_id = service.enroll_face(frame(), "Giovanny")

        assert service.remove_face(face_id) is True
        assert service.db.get_face(face_id) is None
        assert service.list_known_faces() == []

    def test_remove_face_returns_false_for_an_unknown_id(self, service):
        assert service.remove_face("no-such-face") is False


class TestAvailability:
    def test_constructing_without_insightface_raises_a_clear_error(self, monkeypatch, tmp_path):
        """Startup catches this to keep the API up with identity disabled;
        the message has to say what to install."""
        monkeypatch.setattr(fr, "_INSIGHTFACE_AVAILABLE", False)
        db = Database(tmp_path / "x.db")
        db.initialize()

        with pytest.raises(RuntimeError, match="insightface"):
            FaceRecognitionService(db=db)


class TestLastSeenPersistence:
    def test_recording_last_seen_uses_no_deprecated_datetime_adapter(self, service, detected_faces):
        """
        sqlite3's implicit datetime adapter is deprecated in Python 3.12 and
        slated for removal. update_face_seen() runs on every successful
        match, so this fires continuously during a demo and will become a
        hard error on a future interpreter.
        """
        import warnings

        detected_faces.append(FakeFace(unit_vector(1)))
        service.enroll_face(frame(), "Giovanny")

        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            service.recognize(frame())

    def test_last_seen_is_stored_as_an_iso_timestamp(self, service, detected_faces):
        from datetime import datetime

        detected_faces.append(FakeFace(unit_vector(1)))
        face_id = service.enroll_face(frame(), "Giovanny")

        service.recognize(frame())

        last_seen = service.db.get_face(face_id)["last_seen"]
        assert isinstance(last_seen, str)
        datetime.fromisoformat(last_seen)  # raises if not a parseable timestamp


class TestDetectorSize:
    """
    RetinaFace ran at 640x640 regardless of frame size. The stream now feeds
    it 640x360 frames, so a 640x640 detection canvas buys nothing and costs
    roughly 4x the pixels of 320x320.
    """

    def _service_recording_prepare(self, monkeypatch, tmp_path, **kwargs):
        recorded = {}

        class _Recording:
            def __init__(self, name=None, providers=None):
                pass

            def prepare(self, ctx_id=0, det_size=(640, 640)):
                recorded["det_size"] = det_size

            def get(self, frame):
                return []

        monkeypatch.setattr(fr, "_INSIGHTFACE_AVAILABLE", True)
        monkeypatch.setattr(fr, "_FaceAnalysis", _Recording)

        db = Database(tmp_path / "t.db")
        db.initialize()
        FaceRecognitionService(db=db, **kwargs)
        db.shutdown()
        return recorded

    def test_detector_defaults_to_320(self, monkeypatch, tmp_path):
        recorded = self._service_recording_prepare(monkeypatch, tmp_path)

        assert recorded["det_size"] == (320, 320)

    def test_detector_size_is_still_overridable(self, monkeypatch, tmp_path):
        recorded = self._service_recording_prepare(
            monkeypatch, tmp_path, det_size=(640, 640)
        )

        assert recorded["det_size"] == (640, 640)
