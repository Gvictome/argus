"""
In-memory login rate limiting.

Counts FAILED logins in a sliding window, per client IP and per username
(case-insensitive). Once either key reaches the limit, further attempts get
HTTP 429 with Retry-After until the oldest failure ages out of the window.
A successful login clears the username counter (not the IP one).

Limits come from settings (LOGIN_MAX_ATTEMPTS, LOGIN_WINDOW_SECONDS) and are
read on every call. State is per-process and lost on restart, which is fine
for a single-node demo unit. Behind a tunnel every request may share the
proxy's IP, so the per-username limit is the one that matters there.
"""

import threading
import time
from collections import defaultdict, deque
from typing import Callable, Deque, Dict, Tuple


class LoginRateLimiter:
    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self._lock = threading.Lock()
        self._failures: Dict[Tuple[str, str], Deque[float]] = defaultdict(deque)

    @staticmethod
    def _keys(ip: str, username: str):
        return (("ip", ip or "unknown"), ("user", (username or "").strip().lower()))

    def _prune(self, key, window: float) -> Deque[float]:
        q = self._failures[key]
        cutoff = self._clock() - window
        while q and q[0] <= cutoff:
            q.popleft()
        if not q:
            self._failures.pop(key, None)
            return deque()
        return q

    def check(self, ip: str, username: str, max_attempts: int, window: float) -> int:
        """Seconds until a retry is allowed, or 0 if the attempt may proceed."""
        with self._lock:
            wait = 0.0
            for key in self._keys(ip, username):
                q = self._prune(key, window)
                if len(q) >= max_attempts:
                    wait = max(wait, q[0] + window - self._clock())
            return int(wait) + 1 if wait > 0 else 0

    def record_failure(self, ip: str, username: str) -> None:
        with self._lock:
            now = self._clock()
            for key in self._keys(ip, username):
                self._failures[key].append(now)

    def record_success(self, ip: str, username: str) -> None:
        with self._lock:
            self._failures.pop(("user", (username or "").strip().lower()), None)

    def reset(self) -> None:
        with self._lock:
            self._failures.clear()


login_limiter = LoginRateLimiter()


def reset_login_limiter() -> None:
    """Clear all counters (used by tests)."""
    login_limiter.reset()
