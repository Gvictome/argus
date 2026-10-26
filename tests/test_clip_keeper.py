"""Clip keeper: P1 clips stay, the rest go, the cap holds, priority is frozen."""

from datetime import datetime
from types import SimpleNamespace

from src.detection.clip_keeper import P1, ClipKeeper, priority_of
from src.federated.features import ANOMALY, N_FEATURES


class FakeStore:
    def __init__(self, rows):
        self._rows = rows

    def by_n(self, n):
        return next((r for r in self._rows if r["n"] == n), None)


class FakeHead:
    def __init__(self, verdict):
        self.verdict = verdict

    def infer_one(self, x):
        return self.verdict


# Windows cannot convert pre-1970 local times, so every time is offset.
T0 = 1_790_000_000.0


def row(n, t, y=None, duration=2.0):
    return {"n": n, "t": T0 + t, "x": [0.0] * N_FEATURES, "y": y,
            "meta": {"duration_s": duration}}


def clip(tmp_path, name, start, end, size=1000):
    path = tmp_path / name
    path.write_bytes(b"\0" * size)
    return SimpleNamespace(path=str(path),
                           started_at=datetime.fromtimestamp(T0 + start).isoformat(),
                           ended_at=datetime.fromtimestamp(T0 + end).isoformat(),
                           trigger="human")


def keeper(tmp_path, rows, head=None, max_bytes=10**9):
    return ClipKeeper(tmp_path, FakeStore(rows), lambda: head, max_bytes=max_bytes,
                      min_free_bytes=0, grace_s=30, transcode=False)


def test_priority_label_outranks_head():
    assert priority_of(row(1, 0, y=ANOMALY), FakeHead(0)) == P1
    assert priority_of(row(1, 0, y=0), FakeHead(ANOMALY)) is None
    assert priority_of(row(1, 0), FakeHead(ANOMALY)) == P1
    assert priority_of(row(1, 0), None) is None


def test_clip_overlapping_p1_event_is_kept_and_indexed(tmp_path):
    k = keeper(tmp_path, [row(7, 1010.0, y=ANOMALY)])
    k.on_clip(clip(tmp_path, "a.mp4", 1000.0, 1020.0))
    k.sweep(now=T0 + 1100.0)
    assert (tmp_path / "a.mp4").exists()
    assert k.clip_for(7) == (tmp_path / "a.mp4").resolve()
    assert k.frozen_priority(7) == P1


def test_clip_without_p1_event_is_deleted(tmp_path):
    k = keeper(tmp_path, [row(7, 1010.0, y=0), row(8, 5000.0, y=ANOMALY)])
    k.on_clip(clip(tmp_path, "a.mp4", 1000.0, 1020.0))
    k.sweep(now=T0 + 1100.0)
    assert not (tmp_path / "a.mp4").exists()
    assert k.dropped == 1


def test_clip_waits_out_the_grace_period(tmp_path):
    k = keeper(tmp_path, [])
    k.on_clip(clip(tmp_path, "a.mp4", 1000.0, 1020.0))
    k.sweep(now=T0 + 1030.0)  # 10 s after close: an event may still be on its way
    assert (tmp_path / "a.mp4").exists()
    assert k.status()["pending"] == 1


def test_cap_deletes_oldest_clip_but_keeps_priority(tmp_path):
    rows = [row(1, 110.0, y=ANOMALY), row(2, 210.0, y=ANOMALY)]
    k = keeper(tmp_path, rows, max_bytes=1500)
    k.on_clip(clip(tmp_path, "old.mp4", 100.0, 120.0))
    k.on_clip(clip(tmp_path, "new.mp4", 200.0, 220.0))
    k.sweep(now=T0 + 1000.0)
    assert not (tmp_path / "old.mp4").exists()
    assert (tmp_path / "new.mp4").exists()
    assert k.clip_for(1) is None and k.frozen_priority(1) == P1


def test_frozen_priority_survives_a_model_change(tmp_path):
    k = keeper(tmp_path, [], head=FakeHead(0))
    k.freeze(5, P1)
    k._save_index()
    reloaded = keeper(tmp_path, [], head=FakeHead(0))
    assert reloaded.priority(row(5, 0), FakeHead(0)) == P1


def test_clip_for_rejects_paths_outside_the_folder(tmp_path):
    k = keeper(tmp_path, [])
    k._index["events"]["3"] = {"priority": P1, "clip": "../outside.mp4"}
    (tmp_path.parent / "outside.mp4").write_bytes(b"x")
    assert k.clip_for(3) is None


class TestEndpoint:
    def test_clip_is_served_with_range_and_listed_on_the_event(self, monkeypatch, tmp_path):
        from fastapi.testclient import TestClient
        from src.api.app import create_app
        from src.config import settings

        monkeypatch.setattr(settings, "AUTH_REQUIRED", False)
        monkeypatch.setattr(settings, "DB_PATH", tmp_path / "o.db")
        monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
        clips = tmp_path / "clips"
        clips.mkdir()
        (clips / "a.mp4").write_bytes(bytes(range(256)) * 4)
        r = row(9, 1010.0, y=ANOMALY)
        k = keeper(clips, [r])
        k.freeze(9, P1, "a.mp4")

        with TestClient(create_app()) as client:
            client.app.state.clip_keeper = k
            client.app.state.fl_store = FakeStore([r])
            full = client.get("/api/events/9/clip")
            part = client.get("/api/events/9/clip", headers={"Range": "bytes=0-99"})
            missing = client.get("/api/events/10/clip")
            event = client.get("/api/events/9").json()

        assert full.status_code == 200 and full.headers["content-type"] == "video/mp4"
        assert part.status_code == 206 and len(part.content) == 100
        assert missing.status_code == 404
        assert event["priority"] == P1 and event["clip"] == "/api/events/9/clip"


def test_reencode_renames_clip_in_index(tmp_path, monkeypatch):
    k = keeper(tmp_path, [row(4, 1010.0, y=ANOMALY)])
    k.on_clip(clip(tmp_path, "raw.mp4", 1000.0, 1020.0))
    k.sweep(now=T0 + 1100.0)

    def fake_transcode(path):
        out = path.with_name(path.stem + "_h264.mp4")
        out.write_bytes(b"h264")
        path.unlink()
        return out
    monkeypatch.setattr(k, "_transcode", fake_transcode)

    assert k.reencode_kept() == 1
    assert k.clip_for(4) == (tmp_path / "raw_h264.mp4").resolve()
    assert k.reencode_kept() == 0
