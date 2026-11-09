"""
Parked-object suppression tests.

One night on Node A produced 2,140 events, 1,754 of them the same few
parked cars re-detected whenever the palm fronds opened the motion gate.
These lock in: a still vehicle at a known spot is recorded once per hour,
not every time; a vehicle that moves is always an event; people are never
suppressed; and an object that leaves and comes back is new again.
"""

from src.detection import Detection, DetectionType
from src.federated.collector import EventCollector
from src.federated.store import SampleStore


def _collector(tmp_path, **kw):
    store = SampleStore(tmp_path / "samples.jsonl")
    return EventCollector(store, **kw), store


def _sighting(collector, t0, kind=DetectionType.VEHICLE, x=100, dx=0, frames=5):
    """One track: `frames` detections 0.1 s apart, then the track closes."""
    for i in range(frames):
        det = Detection(type=kind, confidence=0.9, bbox=(x + i * dx, 200, 160, 100))
        collector.observe([det], (480, 640), when=t0 + i * 0.1)
    collector.flush()


class TestParked:
    def test_same_parked_car_is_one_event_per_hour(self, tmp_path):
        collector, store = _collector(tmp_path)
        for k in range(6):                      # six re-detections, 5 minutes apart
            _sighting(collector, 1000.0 + k * 300)
        assert store.stats()["total"] == 1
        assert collector.status()["suppressed_stationary"] == 5
        assert store.recent(1)[0]["meta"]["stationary"] is True

    def test_kept_again_after_keep_window(self, tmp_path):
        collector, store = _collector(tmp_path, stationary_keep_s=3600, stationary_forget_s=7200)
        _sighting(collector, 1000.0)
        _sighting(collector, 1000.0 + 1800)     # suppressed
        _sighting(collector, 1000.0 + 3700)     # an hour on: kept as the hourly record
        assert store.stats()["total"] == 2

    def test_moving_vehicle_is_always_an_event(self, tmp_path):
        collector, store = _collector(tmp_path)
        for k in range(3):
            _sighting(collector, 1000.0 + k * 60, dx=40)  # drives across the frame
        assert store.stats()["total"] == 3
        assert all(r["meta"]["stationary"] is False for r in store.recent(3))

    def test_people_are_never_suppressed(self, tmp_path):
        collector, store = _collector(tmp_path)
        for k in range(4):
            _sighting(collector, 1000.0 + k * 60, kind=DetectionType.HUMAN)
        assert store.stats()["total"] == 4

    def test_car_that_left_and_came_back_is_new(self, tmp_path):
        collector, store = _collector(tmp_path, stationary_forget_s=600)
        _sighting(collector, 1000.0)
        _sighting(collector, 1000.0 + 1200)     # unseen for 20 min: forgotten, new object
        assert store.stats()["total"] == 2

    def test_two_parked_cars_are_two_objects(self, tmp_path):
        collector, store = _collector(tmp_path)
        for k in range(3):
            _sighting(collector, 1000.0 + k * 300, x=50)
            _sighting(collector, 1000.0 + k * 300 + 5, x=400)
        assert store.stats()["total"] == 2
        assert collector.status()["parked_objects"] == 2

    def test_off_when_no_classes(self, tmp_path):
        collector, store = _collector(tmp_path, stationary_classes=())
        for k in range(3):
            _sighting(collector, 1000.0 + k * 300)
        assert store.stats()["total"] == 3
