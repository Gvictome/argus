"""
Federated Learning round scheduler for ARGUS.

FLScheduler triggers FL participation once per night (default 2 AM) using
asyncio tasks.  It checks whether the system is idle before connecting to
the server so that FL rounds don't interfere with active surveillance.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)


class FLScheduler:
    """
    Schedules nightly Federated Learning participation using asyncio.

    The scheduler wakes up at the configured hour, checks for system
    idleness, then calls the FL client to participate in a round.

    Attributes:
        client: ArgusFlowerClient instance.
        config: Settings instance.
        _task: Background asyncio Task, or None if not running.
        _last_motion_at: Timestamp of the most recent motion detection.
    """

    def __init__(self, client, config) -> None:
        """
        Args:
            client: ArgusFlowerClient instance (src.federated.client).
            config: Settings instance (src.config.Settings).
        """
        self.client = client
        self.config = config

        self._task: Optional[asyncio.Task] = None
        self._running: bool = False
        self._last_motion_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """
        Schedule the nightly FL round.

        Creates an asyncio background task that sleeps until the configured
        hour (FL_ROUND_HOUR) and then participates in a FL round.  The task
        reschedules itself after each round.

        Must be called from within a running asyncio event loop.
        """
        if self._running:
            logger.warning("FLScheduler already running — ignoring start()")
            return

        self._running = True
        self._task = asyncio.ensure_future(self._loop())
        logger.info(
            "FLScheduler started — FL rounds scheduled for %02d:00",
            self.config.FL_ROUND_HOUR,
        )

    def stop(self) -> None:
        """
        Cancel the scheduled FL task.

        Safe to call even if start() was never called.
        """
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

        Connects to the FL server, runs fit and evaluate, then disconnects.
        The actual blocking I/O is offloaded to an executor thread so the
        event loop stays responsive.

        Returns:
            True if the round completed without error; False otherwise.
        """
        if not self.client.config.FL_ENABLED:
            logger.info("FL is disabled (FL_ENABLED=False) — skipping round")
            return False

        if not self.is_idle():
            logger.info("System not idle — skipping FL round")
            return False

        logger.info(
            "Starting FL round — server=%s", self.config.FL_SERVER_URL
        )

        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self._blocking_fl_round,
            )
            logger.info("FL round completed successfully")
            return True
        except Exception as exc:
            logger.error("FL round failed: %s", exc)
            return False

    def _blocking_fl_round(self) -> None:
        """
        Synchronous FL round — runs in a thread executor.

        Imports are deferred to avoid hard dependency at module load time.
        """
        from src.federated.client import start_client
        start_client(self.config.FL_SERVER_URL, self.client)

    # ------------------------------------------------------------------
    # Idle check
    # ------------------------------------------------------------------

    def is_idle(self) -> bool:
        """
        Return True if no motion has been detected in the last 10 minutes.

        The DetectionService (or any caller) should update _last_motion_at
        via notify_motion() whenever motion is detected.
        """
        if self._last_motion_at is None:
            return True
        idle_threshold = timedelta(minutes=10)
        return datetime.now() - self._last_motion_at > idle_threshold

    def notify_motion(self) -> None:
        """
        Record the current time as the most recent motion event.

        Call this from the camera processing loop whenever detect_motion()
        returns non-empty results.
        """
        self._last_motion_at = datetime.now()

    # ------------------------------------------------------------------
    # Internal scheduling loop
    # ------------------------------------------------------------------

    async def _loop(self) -> None:
        """
        Internal asyncio loop: sleep until the target hour, run a round, repeat.
        """
        while self._running:
            sleep_seconds = self._seconds_until_round_hour()
            logger.debug(
                "FL scheduler sleeping for %.0f seconds (until %02d:00)",
                sleep_seconds,
                self.config.FL_ROUND_HOUR,
            )

            try:
                await asyncio.sleep(sleep_seconds)
            except asyncio.CancelledError:
                logger.info("FL scheduler task cancelled")
                return

            if not self._running:
                return

            await self.run_round()

    def _seconds_until_round_hour(self) -> float:
        """
        Compute seconds until the next occurrence of FL_ROUND_HOUR.

        If the current time is past the target hour, schedules for the
        same hour tomorrow.
        """
        now = datetime.now()
        target = now.replace(
            hour=self.config.FL_ROUND_HOUR,
            minute=0,
            second=0,
            microsecond=0,
        )
        if target <= now:
            target += timedelta(days=1)
        return (target - now).total_seconds()
