"""Retry with exponential backoff, distinguishing transient from permanent failures."""
from __future__ import annotations

import time
from typing import Callable, Sequence, TypeVar

T = TypeVar("T")


class PipelineError(Exception):
    """Base class for errors with a human-readable reason."""


class TransientError(PipelineError):
    """Worth retrying: timeouts, rate limits, 5xx, flaky downloads."""


class PermanentError(PipelineError):
    """Retrying will not help: bad credentials, invalid input, policy rejection."""


class RateLimitError(TransientError):
    """HTTP 429 / quota. Per-minute limits reset within a minute, so wait at least that long.
    If it persists through every retry, the quota is exhausted (or billing is missing)."""

    min_delay = 60.0


def backoff_delay(attempt: int, schedule: Sequence[float]) -> float:
    """Delay after failed attempt number `attempt` (1-based). Extends the schedule geometrically."""
    if not schedule:
        return 0.0
    if attempt <= len(schedule):
        return float(schedule[attempt - 1])
    return float(schedule[-1]) * (3 ** (attempt - len(schedule)))


def with_retry(
    fn: Callable[[], T],
    *,
    max_attempts: int,
    schedule: Sequence[float],
    on_retry: Callable[[int, Exception, float], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Call fn until it succeeds. PermanentError is raised immediately; anything else is retried."""
    attempt = 0
    while True:
        attempt += 1
        try:
            return fn()
        except PermanentError:
            raise
        except Exception as exc:  # noqa: BLE001 - unknown errors are treated as transient
            if attempt >= max_attempts:
                raise
            delay = max(backoff_delay(attempt, schedule), getattr(exc, "min_delay", 0.0))
            if on_retry:
                on_retry(attempt, exc, delay)
            sleep(delay)
