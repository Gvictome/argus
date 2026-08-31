"""
Event-triggered recording.

Cameras run 24/7, disk does not. These lock in that footage is kept only
around real detections, and that the clip contains the part worth seeing.
"""

import numpy as np
import pytest

from src.detection import Detection, DetectionType
from src.detection.event_recorder import EventRecorder, RecorderConfig


def _frame(v=0):
    return np.full((48, 64, 3), v, dtype=np.uint8)


def _det(t=DetectionType.HUMAN, conf=0.9):
    return Detection(type=t, confidence=conf, bbox=(1, 1, 10, 10))


@pytest.fixture
def recorder(tmp_path):
    return EventRecorder(
        RecorderConfig(
            pre_roll_s=1.0, post_roll_s=2.0, fps=10,
            output_dir=tmp_path, min_confidence=0.5,
        ),
        camera_name="front",
    )


class TestTriggering:
    def test_idle_frames_are_not_recorded(self, recorder, tmp_path):
        """The whole point: 24/7 cameras must not mean 24/7 disk."""
        for i in range(50):
            recorder.process(_frame(i), [], now=i * 0.1)

        assert not recorder.is_recording
        assert list(tmp_path.glob("*.mp4")) == []

    def test_a_person_starts_a_clip(self, recorder):
        recorder.process(_frame(), [], now=0.0)
        recorder.process(_frame(), [_det()], now=0.1)

        assert recorder.is_recording

    def test_motion_alone_does_not_record(self, recorder):
        """
        Motion fires on a cloud or a branch. Recording on it is 24/7
        recording with extra steps.
        """
        for i in range(20):
            recorder.process(_frame(i), [_det(DetectionType.MOTION, 0.99)], now=i * 0.1)

        assert not recorder.is_recording

    def test_low_confidence_does_not_record(self, recorder):
        recorder.process(_frame(), [_det(conf=0.2)], now=0.1)

        assert not recorder.is_recording

    def test_a_threat_starts_a_clip(self, recorder):
        recorder.process(_frame(), [_det(DetectionType.DANGEROUS_PERSON, 0.8)], now=0.1)

        assert recorder.is_recording

    def test_clip_is_labelled_by_the_most_confident_trigger(self, recorder, tmp_path):
        recorder.process(_frame(), [
            _det(DetectionType.VEHICLE, 0.6),
            _det(DetectionType.DANGEROUS_PERSON, 0.95),
        ], now=0.1)
        recorder.close()

        assert any("dangerous_person" in p.name for p in tmp_path.glob("*.mp4"))


class TestPreRoll:
    def test_clip_includes_frames_from_before_the_trigger(self, recorder):
        """
        A clip starting at the trigger has already missed the approach --
        usually the most useful part of the footage.
        """
        for i in range(10):
            recorder.process(_frame(i), [], now=i * 0.1)
        recorder.process(_frame(99), [_det()], now=1.0)
        recorder.close()

        clip = recorder.recent_clips()[0]
        assert clip["frames"] > 1, "clip began at the trigger; pre-roll was lost"

    def test_pre_roll_is_bounded(self, recorder):
        """The buffer must not grow without limit while idle."""
        for i in range(500):
            recorder.process(_frame(i % 255), [], now=i * 0.1)

        assert len(recorder._pre_roll) <= recorder._pre_roll.maxlen


class TestPostRollAndCooldown:
    def test_recording_continues_briefly_after_the_last_detection(self, recorder):
        recorder.process(_frame(), [_det()], now=0.0)
        recorder.process(_frame(), [], now=1.0)          # inside post-roll

        assert recorder.is_recording

    def test_recording_stops_after_post_roll(self, recorder):
        recorder.process(_frame(), [_det()], now=0.0)
        recorder.process(_frame(), [], now=3.0)          # past 2s post-roll

        assert not recorder.is_recording

    def test_continued_presence_extends_one_clip(self, recorder):
        """
        Someone loitering must not produce hundreds of near-identical
        clips.
        """
        for i in range(30):
            recorder.process(_frame(i), [_det()], now=i * 0.5)
        recorder.close()

        assert len(recorder.recent_clips()) == 1

    def test_separate_events_make_separate_clips(self, recorder):
        recorder.process(_frame(), [_det()], now=0.0)
        recorder.process(_frame(), [], now=5.0)      # clip 1 closes
        recorder.process(_frame(), [_det()], now=6.0)
        recorder.close()

        assert len(recorder.recent_clips()) == 2

    def test_max_clip_length_is_enforced(self, tmp_path):
        """A stuck trigger must not record until the disk fills."""
        rec = EventRecorder(
            RecorderConfig(pre_roll_s=0.1, post_roll_s=10, max_clip_s=2.0,
                           fps=10, output_dir=tmp_path),
            camera_name="front",
        )
        for i in range(40):
            rec.process(_frame(i), [_det()], now=i * 0.2)

        assert len(rec.recent_clips()) >= 1, "max_clip_s never closed the clip"


class TestOutput:
    def test_clip_is_written_to_disk(self, recorder, tmp_path):
        for i in range(5):
            recorder.process(_frame(i), [_det()], now=i * 0.1)
        recorder.close()

        clips = list(tmp_path.glob("*.mp4"))
        assert len(clips) == 1
        assert clips[0].stat().st_size > 0

    def test_clip_metadata_is_recorded(self, recorder):
        recorder.process(_frame(), [_det(conf=0.87)], now=0.0)
        recorder.process(_frame(), [], now=5.0)

        clip = recorder.recent_clips()[0]
        assert clip["trigger"] == "human"
        assert clip["peak_confidence"] == pytest.approx(0.87)
        assert clip["duration_s"] > 0

    def test_close_finalizes_an_open_clip(self, recorder, tmp_path):
        """Shutdown mid-event must not leave a truncated file."""
        recorder.process(_frame(), [_det()], now=0.0)
        assert recorder.is_recording

        recorder.close()

        assert not recorder.is_recording
        assert len(list(tmp_path.glob("*.mp4"))) == 1

    def test_status_is_serializable(self, recorder):
        import json
        json.loads(json.dumps(recorder.status()))

    def test_unwritable_output_dir_does_not_raise(self, tmp_path):
        """A full or read-only disk must not take down the detection loop."""
        blocker = tmp_path / "events"
        blocker.write_text("not a directory")
        rec = EventRecorder(RecorderConfig(output_dir=blocker), camera_name="front")

        rec.process(_frame(), [_det()], now=0.0)   # must not raise

        assert not rec.is_recording
