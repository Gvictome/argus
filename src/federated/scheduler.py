"""
Federated Learning round scheduler for ARGUS.

Triggers FL participation on a fixed interval (default: every 14 days) at
a configured hour, and only while the system is idle so a round never
competes with live surveillance.

Two things this has to get right that a naive timer does not:

**The clock must survive a restart.** With a 14-day interval and an
in-memory timer, any reboot inside the window restarts the countdown. A
device rebooted weekly would run FL *never*, and would look fine doing
it. The last-round timestamp is therefore persisted to disk.

**An overdue round must still run.** If the host was powered off across
the due date, waiting another full interval compounds the miss. On
startup an overdue round is scheduled promptly instead.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# The scheduler wakes at least this often rather than sleeping for the
# whole interval in one await. A single 14-day sleep is fragile: it does
# not survive suspend/resume cleanly, it ignores wall-clock corrections
# (an NTP step after a cold boot with no RTC is common on an SBC), and it
# makes stop() take up to 14 days to notice it was asked to stop.
_WAKE_INTERVAL_S = 3600.0

# How long after a missed due date to wait before running, so a round
# does not fire during the boot storm right after a power cut.
_OVERDUE_GRACE_S = 300.0


class FLScheduler:
    """
    Schedules Federated Learning rounds on an interval, with persistence.

    Attributes:
        client: ArgusFlowerClient instance.
        config: Settings instance.
        state_path: Where the last-round timestamp is persisted.
    """

    def __init__(self, client, config, state_path: Optional[Path] = None) -> None:
        self.client = client
        self.config = config

        self.state_path = Path(
            state_path
            or getattr(config, "FL_STATE_PATH", None)
            or Path(getattr(config, "DATA_DIR", ".")) / "fl_state.json"
        )

        self._task: Optional[asyncio.Task] = None
        self._running: bool = False
        self._last_motion_at: Optional[datetime] = None
        self._started_at: Optional[datetime] = None
        self._last_round_at: Optional[datetime] = self._load_last_round()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @property
    def interval(self) -> timedelta:
        return timedelta(days=max(1, int(getattr(self.config, "FL_ROUND_INTERVAL_DAYS", 14))))

    def _load_last_round(self) -> Optional[datetime]:
        """Read the last-round timestamp, tolerating a missing or bad file."""
        try:
            if not self.state_path.exists():
                return None
            data = json.loads(self.state_path.read_text())
            raw = data.get("last_round_at")
            return datetime.fromisoformat(raw) if raw else None
        except Exception as exc:
            # A corrupt state file must not stop FL from ever running
            # again -- treat it as "never ran".
            logger.warning("Could not read FL state at %s: %s", self.state_path, exc)
            return None

    def _save_last_round(self, when: datetime) -> None:
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(json.dumps({
                "last_round_at": when.isoformat(),
                "interval_days": self.interval.days,
            }, indent=2))
        except Exception as exc:
            # Non-fatal, but it means the interval restarts on reboot.
            logger.error("Could not persist FL state to %s: %s", self.state_path, exc)

    # ------------------------------------------------------------------
    # Schedule arithmetic
    # ------------------------------------------------------------------

    def scheduled_target(self, now: Optional[datetime] = None) -> datetime:
        """
        The round's nominal due time, ignoring startup grace.

        Never run before: the next occurrence of the configured hour, so
        a fresh install participates soon rather than in a fortnight.
        """
        now = now or datetime.now()
        hour = int(getattr(self.config, "FL_ROUND_HOUR", 2))

        if self._last_round_at is None:
            target = now.replace(hour=hour, minute=0, second=0, microsecond=0)
            if target <= now:
                target += timedelta(days=1)
            return target

        return (self._last_round_at + self.interval).replace(
            hour=hour, minute=0, second=0, microsecond=0
        )

    def _grace_until(self) -> Optional[datetime]:
        """
        The earliest a round may start after this process came up.

        Keeps an overdue round out of the boot storm following a power
        cut. Applies only once per process, at startup.
        """
        if self._started_at is None:
            return None
        return self._started_at + timedelta(seconds=_OVERDUE_GRACE_S)

    def next_due(self, now: Optional[datetime] = None) -> datetime:
        """
        When the next round will actually run: the target, held back by
        startup grace if the target has already passed.
        """
        now = now or datetime.now()
        target = self.scheduled_target(now)

        if target > now:
            return target

        # Overdue. Run shortly -- never another full interval away.
        grace_until = self._grace_until()
        if grace_until is not None and grace_until > now:
            return grace_until
        return now

    def seconds_until_next(self, now: Optional[datetime] = None) -> float:
        now = now or datetime.now()
        return max(0.0, (self.next_due(now) - now).total_seconds())

    def is_due(self, now: Optional[datetime] = None) -> bool:
        """
        Whether a round should run right now.

        Compares against `scheduled_target`, not `next_due`: next_due
        returns a future instant while startup grace is in effect, so
        testing against it would mean an overdue round is never due --
        the scheduler would wait forever.
        """
        now = now or datetime.now()
        if self.scheduled_target(now) > now:
            return False
        grace_until = self._grace_until()
        return grace_until is None or now >= grace_until

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """
        Begin the scheduling loop. Must be called from a running loop.
        """
        if self._running:
            logger.warning("FLScheduler already running — ignoring start()")
            return

        self._running = True
        self._started_at = datetime.now()
        self._task = asyncio.ensure_future(self._loop())
        logger.info(
            "FLScheduler started — every %d days at %02d:00; last round %s; next %s",
            self.interval.days,
            int(getattr(self.config, "FL_ROUND_HOUR", 2)),
            self._last_round_at.isoformat() if self._last_round_at else "never",
            self.next_due().isoformat(),
        )

    def stop(self) -> None:
        """Cancel the scheduled task. Safe if start() was never called."""
        self._running = False
        if self._task is not None and not self._task.done():
            self._task.cancel()
            logger.info("FLScheduler stopped")
        self._task = None

    # ------------------------------------------------------------------
    # FL round execution
    # ------------------------------------------------------------------

    async def run_round(self) -> bool:
        """
        Participate in one FL round.

        Returns True if the round completed without error.
        """
        if not self.client.config.FL_ENABLED:
            logger.info("FL is disabled (FL_ENABLED=False) — skipping round")
            return False

        if not self.is_idle():
            logger.info("System not idle — deferring FL round")
            return False

        logger.info("Starting FL round — server=%s", self.config.FL_SERVER_URL)

        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._blocking_fl_round)
        except Exception as exc:
            # The timestamp is deliberately NOT written on failure, so a
            # server that was unreachable is retried on the next wake
            # rather than deferred for another two weeks.
            logger.error("FL round failed: %s", exc)
            return False

        self.mark_round_complete()
        logger.info("FL round completed; next due %s", self.next_due().isoformat())
        return True

    def mark_round_complete(self, when: Optional[datetime] = None) -> None:
        """Record a completed round and persist the new schedule."""
        when = when or datetime.now()
        self._last_round_at = when
        self._save_last_round(when)

    def _blocking_fl_round(self) -> None:
        """Synchronous FL round — runs in a thread executor."""
        from src.federated.client import start_client
        start_client(self.config.FL_SERVER_URL, self.client)

    # ------------------------------------------------------------------
    # Idle check
    # ------------------------------------------------------------------

    def is_idle(self) -> bool:
        """True if no motion has been seen in the last 10 minutes."""
        if self._last_motion_at is None:
            return True
        return datetime.now() - self._last_motion_at > timedelta(minutes=10)

    def notify_motion(self) -> None:
        """Record motion, so a round does not start over live activity."""
        self._last_motion_at = datetime.now()

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self) -> dict:
        """Schedule state, for /api/federated/status."""
        return {
            "enabled": bool(getattr(self.config, "FL_ENABLED", False)),
            "running": self._running,
            "interval_days": self.interval.days,
            "round_hour": int(getattr(self.config, "FL_ROUND_HOUR", 2)),
            "last_round_at": self._last_round_at.isoformat() if self._last_round_at else None,
            "next_due_at": self.next_due().isoformat(),
            "seconds_until_next": round(self.seconds_until_next()),
            "idle": self.is_idle(),
            "server": getattr(self.config, "FL_SERVER_URL", None),
        }

    # ------------------------------------------------------------------
    # Internal loop
    # ------------------------------------------------------------------

    async def _loop(self) -> None:
        """
        Wake periodically, run a round when one is due.

        Polling on a bounded interval rather than sleeping the full two
        weeks: see the note at the top of this module.
        """
        while self._running:
            wait = min(self.seconds_until_next(), _WAKE_INTERVAL_S)

            try:
                await asyncio.sleep(max(1.0, wait))
            except asyncio.CancelledError:
                logger.info("FL scheduler task cancelled")
                return

            if not self._running:
                return

            if self.is_due():
                await self.run_round()
