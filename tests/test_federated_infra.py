"""
The federated infrastructure: what a round does when it cannot run, and
turning saved video into labelled training data.

The happy path needs a live aggregator and is exercised end to end by
hand; these cover the parts that must behave without one, because those
are what run unattended at 2am.
"""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]


def _app(tmp_path, monkeypatch, labelled=0):
    from src.federated import model_state, rounds
    from src.federated.head import FederatedHead
    from src.federated.store import SampleStore

    monkeypatch.setattr(model_state.settings, "DATA_DIR", tmp_path)
    monkeypatch.setattr(rounds.settings, "DATA_DIR", tmp_path)
    monkeypatch.setattr(rounds.settings, "NODE_NAME", "Test Node")

    store = SampleStore(tmp_path / "samples.jsonl")
    if labelled:
        store.bootstrap("driveway", labelled, seed=1)
    app = SimpleNamespace(state=SimpleNamespace(
        fl_store=store, fl_head=FederatedHead(seed=0), fl_model_state=None))
    return app, rounds


class TestRoundGuards:
    def test_too_few_labelled_samples_is_refused_before_connecting(
            self, tmp_path, monkeypatch):
        """No point waking the aggregator for a handful of events."""
        app, rounds = _app(tmp_path, monkeypatch, labelled=0)

        decision = rounds.run_round_blocking(
            app, server="127.0.0.1:9", min_samples=200)

        assert decision["accepted"] is False
        assert "labelled samples" in decision["reason"]
        assert not (tmp_path / "fl_rounds").exists(), "should not have started a run"

    def test_missing_head_is_reported_not_raised(self, tmp_path, monkeypatch):
        from src.federated import rounds

        monkeypatch.setattr(rounds.settings, "DATA_DIR", tmp_path)
        app = SimpleNamespace(state=SimpleNamespace(fl_store=None, fl_head=None))

        decision = rounds.run_round_blocking(app, server="127.0.0.1:9")

        assert decision["accepted"] is False
        assert "not initialised" in decision["reason"]

    @pytest.mark.slow
    def test_an_unreachable_aggregator_fails_cleanly(self, tmp_path, monkeypatch):
        """The cron must survive a server that is off, and say why.

        Port 9 is discard: nothing accepts a Flower connection there.
        """
        app, rounds = _app(tmp_path, monkeypatch, labelled=600)

        decision = rounds.run_round_blocking(
            app, server="127.0.0.1:9", epochs=1, min_samples=200, timeout_s=25)

        assert decision["accepted"] is False
        assert decision.get("run_dir"), "a failed round should still leave its evidence"
        # The live model must be untouched by a round that never happened.
        assert app.state.fl_model_state in (None, {}) or \
            not (app.state.fl_model_state or {}).get("version")


class TestVideoIngest:
    """Saved clips become labelled events, with the clip's own clock."""

    @pytest.mark.slow
    def test_a_clip_becomes_labelled_events(self, tmp_path):
        video = REPO / "media" / "vtest.avi"
        if not video.exists():
            pytest.skip("test video not present")

        result = subprocess.run(
            [sys.executable, str(REPO / "scripts" / "ingest_video.py"), str(video),
             "--label", "routine_person", "--data-dir", str(tmp_path),
             "--max-frames", "60", "--zone", "Test zone"],
            cwd=str(REPO), capture_output=True, text=True, timeout=600,
        )
        assert result.returncode == 0, result.stdout + result.stderr

        rows = [json.loads(line) for line in
                (tmp_path / "training" / "samples.jsonl").read_text(
                    encoding="utf-8").splitlines()]
        assert rows, "no events captured from the clip"

        # Every row is labelled, tagged with its source, and real.
        assert all(r["y"] == 0 for r in rows), "not labelled routine_person"
        assert all(r["meta"]["ingested_from"] == video.name for r in rows)
        assert all(r["meta"]["zone"] == "Test zone" for r in rows)
        assert all(r["meta"]["synthetic"] is False for r in rows)

        # Timestamps come from the video's frame rate, not the wall clock:
        # 60 frames at 10fps is about six seconds of footage, however long
        # the machine took to process it.
        span = max(r["t"] for r in rows) - min(r["t"] for r in rows)
        assert span < 30, f"events span {span:.1f}s of clip time, expected a few"

    def test_an_unknown_label_is_rejected(self, tmp_path):
        result = subprocess.run(
            [sys.executable, str(REPO / "scripts" / "ingest_video.py"),
             str(REPO / "media" / "vtest.avi"), "--label", "nonsense",
             "--data-dir", str(tmp_path)],
            cwd=str(REPO), capture_output=True, text=True, timeout=120,
        )

        assert result.returncode == 2
        assert "unknown label" in result.stdout
