"""
FL scheduling on a two-week interval.

The interval is long enough that its failure modes are invisible in
testing by wall-clock: a scheduler that silently never fires looks
identical to one waiting patiently. These assert the arithmetic and the
persistence directly.
"""

from datetime import datetime, timedelta

import pytest

from src.federated.scheduler import FLScheduler


class _Config:
    FL_ENABLED = True
    FL_SERVER_URL = "localhost:8080"
    FL_ROUND_HOUR = 2
    FL_ROUND_INTERVAL_DAYS = 14


class _Client:
    def __init__(self, config):
        self.config = config


@pytest.fixture
def scheduler(tmp_path):
    config = _Config()
    return FLScheduler(_Client(config), config, state_path=tmp_path / "fl_state.json")


class TestInterval:
    def test_interval_is_two_weeks_by_default(self, scheduler):
        assert scheduler.interval == timedelta(days=14)

    def test_interval_is_configurable(self, tmp_path):
        config = _Config()
        config.FL_ROUND_INTERVAL_DAYS = 7
        s = FLScheduler(_Client(config), config, state_path=tmp_path / "s.json")

        assert s.interval == timedelta(days=7)

    def test_interval_cannot_be_zero(self, tmp_path):
        """A zero interval would spin FL rounds continuously."""
        config = _Config()
        config.FL_ROUND_INTERVAL_DAYS = 0
        s = FLScheduler(_Client(config), config, state_path=tmp_path / "s.json")

        assert s.interval >= timedelta(days=1)


class TestScheduleArithmetic:
    def test_first_round_is_soon_not_in_a_fortnight(self, scheduler):
        """A fresh install should participate at the next round hour."""
        now = datetime(2026, 9, 1, 10, 0)

        due = scheduler.next_due(now)

        assert due - now < timedelta(days=1)
        assert due.hour == 2

    def test_next_round_is_an_interval_after_the_last(self, scheduler):
        scheduler.mark_round_complete(datetime(2026, 9, 1, 2, 0))

        due = scheduler.next_due(datetime(2026, 9, 2, 12, 0))

        assert due.date() == datetime(2026, 9, 15).date()
        assert due.hour == 2

    def test_not_due_inside_the_window(self, scheduler):
        scheduler.mark_round_complete(datetime(2026, 9, 1, 2, 0))

        assert not scheduler.is_due(datetime(2026, 9, 8, 2, 0))

    def test_due_after_the_window(self, scheduler):
        scheduler.mark_round_complete(datetime(2026, 9, 1, 2, 0))

        assert scheduler.is_due(datetime(2026, 9, 20, 2, 0))

    def test_an_overdue_round_runs_promptly(self, scheduler):
        """
        Powered off across the due date: the round must not be deferred
        another full interval, which would compound every missed round.
        """
        scheduler.mark_round_complete(datetime(2026, 9, 1, 2, 0))
        now = datetime(2026, 10, 30, 14, 0)   # long overdue

        assert scheduler.next_due(now) - now < timedelta(hours=1)

    def test_overdue_round_waits_out_the_startup_grace(self, scheduler):
        """
        A grace period keeps an overdue round out of the boot storm right
        after a power cut. It applies from process start, so it only
        exists once the scheduler has been started.
        """
        scheduler.mark_round_complete(datetime(2026, 9, 1, 2, 0))
        now = datetime(2026, 10, 30, 14, 0)
        scheduler._started_at = now          # as if the process just came up

        assert scheduler.next_due(now) > now
        assert not scheduler.is_due(now)
        assert scheduler.is_due(now + timedelta(minutes=10))

    def test_overdue_round_runs_immediately_once_grace_has_passed(self, scheduler):
        scheduler.mark_round_complete(datetime(2026, 9, 1, 2, 0))
        now = datetime(2026, 10, 30, 14, 0)
        scheduler._started_at = now - timedelta(hours=3)   # up for a while

        assert scheduler.is_due(now)


class TestPersistence:
    """
    The reason this exists: with an in-memory timer and a 14-day
    interval, a host rebooted weekly restarts the countdown every time
    and runs FL never, while appearing healthy.
    """

    def test_last_round_survives_a_restart(self, tmp_path):
        config = _Config()
        path = tmp_path / "fl_state.json"
        when = datetime(2026, 9, 1, 2, 0)

        FLScheduler(_Client(config), config, state_path=path).mark_round_complete(when)
        revived = FLScheduler(_Client(config), config, state_path=path)

        assert revived._last_round_at == when

    def test_schedule_survives_a_restart(self, tmp_path):
        config = _Config()
        path = tmp_path / "fl_state.json"
        FLScheduler(_Client(config), config, state_path=path).mark_round_complete(
            datetime(2026, 9, 1, 2, 0)
        )

        revived = FLScheduler(_Client(config), config, state_path=path)

        assert not revived.is_due(datetime(2026, 9, 8, 2, 0)), (
            "a restart reset the interval — FL would never run on a "
            "regularly-rebooted host"
        )

    def test_missing_state_file_means_never_ran(self, tmp_path):
        config = _Config()
        s = FLScheduler(_Client(config), config, state_path=tmp_path / "absent.json")

        assert s._last_round_at is None

    def test_corrupt_state_file_does_not_wedge_the_scheduler(self, tmp_path):
        """
        A truncated write (power cut mid-save) must not stop FL from ever
        running again. Treat it as "never ran".
        """
        path = tmp_path / "fl_state.json"
        path.write_text("{not valid json")
        config = _Config()

        s = FLScheduler(_Client(config), config, state_path=path)

        assert s._last_round_at is None
        assert s.next_due() is not None

    def test_unwritable_state_path_does_not_raise(self, tmp_path):
        config = _Config()
        s = FLScheduler(_Client(config), config, state_path=tmp_path)  # a directory

        s.mark_round_complete(datetime(2026, 9, 1, 2, 0))   # must not raise

        assert s._last_round_at is not None


class TestIdleGate:
    def test_idle_when_no_motion(self, scheduler):
        assert scheduler.is_idle()

    def test_not_idle_right_after_motion(self, scheduler):
        scheduler.notify_motion()

        assert not scheduler.is_idle()


class TestStatus:
    def test_status_reports_the_schedule(self, scheduler):
        scheduler.mark_round_complete(datetime(2026, 9, 1, 2, 0))

        status = scheduler.status()

        assert status["interval_days"] == 14
        assert status["last_round_at"].startswith("2026-09-01")
        assert status["next_due_at"].startswith("2026-09-15")
        assert status["seconds_until_next"] >= 0

    def test_status_before_any_round(self, scheduler):
        status = scheduler.status()

        assert status["last_round_at"] is None
        assert status["next_due_at"] is not None


class TestRoundOutcome:
    @pytest.mark.asyncio
    async def test_a_failed_round_does_not_consume_the_interval(self, scheduler, monkeypatch):
        """
        If the server was unreachable, retry on the next wake rather than
        deferring another two weeks.
        """
        def boom():
            raise RuntimeError("connection refused")
        monkeypatch.setattr(scheduler, "_blocking_fl_round", boom)

        ok = await scheduler.run_round()

        assert ok is False
        assert scheduler._last_round_at is None, "a failed round advanced the schedule"

    @pytest.mark.asyncio
    async def test_a_successful_round_advances_the_schedule(self, scheduler, monkeypatch):
        monkeypatch.setattr(scheduler, "_blocking_fl_round", lambda: None)

        ok = await scheduler.run_round()

        assert ok is True
        assert scheduler._last_round_at is not None

    @pytest.mark.asyncio
    async def test_a_round_is_skipped_while_motion_is_live(self, scheduler, monkeypatch):
        monkeypatch.setattr(scheduler, "_blocking_fl_round", lambda: None)
        scheduler.notify_motion()

        assert await scheduler.run_round() is False
        assert scheduler._last_round_at is None

    @pytest.mark.asyncio
    async def test_disabled_fl_does_not_run(self, scheduler, monkeypatch):
        scheduler.client.config.FL_ENABLED = False
        monkeypatch.setattr(scheduler, "_blocking_fl_round", lambda: None)

        assert await scheduler.run_round() is False
