"""Exponential backoff for free-tier LLM calls.

Gemini's free tier enforces both a per-minute and a per-day quota, and both
surface as HTTP 429. They need opposite responses:

  * per-minute  -> wait and retry, you will get through
  * per-day     -> stop; no amount of waiting helps until the quota resets

so this module tells them apart instead of hammering a dead quota.

It deliberately prints nothing. Callers pass an `on_retry` callback, which
keeps this layer usable from a web handler later, where printing to stdout
would be useless.
"""

from __future__ import annotations

import random
import re
import time
from typing import Callable, TypeVar

T = TypeVar("T")

# Google returns e.g. 'retryDelay': '31s' in the error payload. Honouring the
# server's own number beats guessing.
_RETRY_DELAY_RE = re.compile(r"retryDelay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)s")


class QuotaExhausted(RuntimeError):
    """The daily free-tier quota is gone. Retrying will not help today."""


class RetryableError(RuntimeError):
    """A failure the caller has already judged worth retrying.

    Used to re-raise errors that a library caught and turned into a silent
    no-op, so they re-enter the backoff path instead of vanishing.
    """


def _text(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def is_rate_limit(exc: BaseException) -> bool:
    blob = _text(exc).lower()
    # google-genai raises ClientError with a .code attribute; other stacks only
    # put the status in the message, so check both.
    if getattr(exc, "code", None) == 429 or getattr(exc, "status_code", None) == 429:
        return True
    return "429" in blob or "resource_exhausted" in blob or "rate limit" in blob


def is_daily_quota(exc: BaseException) -> bool:
    """Distinguish 'slow down' from 'come back tomorrow'."""
    blob = _text(exc).lower()
    return is_rate_limit(exc) and (
        "perday" in blob.replace("_", "") or "per day" in blob or "daily" in blob
    )


def is_transient(exc: BaseException) -> bool:
    """Network blips and Google's 500/503s are worth another attempt."""
    if isinstance(exc, RetryableError):
        return True
    blob = _text(exc).lower()
    transient_markers = (
        "500",
        "502",
        "503",
        "504",
        "internal error",
        "unavailable",
        "deadline",
        "timeout",
        "timed out",
        "connection reset",
        "connection aborted",
        "temporarily",
    )
    return any(marker in blob for marker in transient_markers)


def _server_requested_delay(exc: BaseException) -> float | None:
    match = _RETRY_DELAY_RE.search(_text(exc))
    return float(match.group(1)) if match else None


def with_retry(
    fn: Callable[[], T],
    *,
    what: str,
    max_retries: int = 5,
    base_delay: float = 2.0,
    on_retry: Callable[[str, float, int, int], None] | None = None,
) -> T:
    """Run `fn`, retrying 429s and transient errors with exponential backoff.

    Args:
        fn: zero-arg callable to run.
        what: human label for messages, e.g. "Gemini reply".
        on_retry: called as (what, delay_seconds, attempt, max_retries) before
            each sleep, so the caller decides how to tell the user.

    Raises:
        QuotaExhausted: daily free-tier quota is spent.
        Exception: the original error, if it is not retryable or we ran out of
            attempts.
    """
    last: BaseException | None = None

    for attempt in range(1, max_retries + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - classify, then re-raise
            last = exc

            if is_daily_quota(exc):
                raise QuotaExhausted(
                    f"{what}: daily free-tier quota exhausted. It resets on "
                    f"Google's clock (midnight Pacific). Switch to BACKEND=ollama "
                    f"in your .env to keep working offline in the meantime."
                ) from exc

            if not (is_rate_limit(exc) or is_transient(exc)):
                raise  # a real bug - surface it immediately, do not mask it

            if attempt == max_retries:
                break

            delay = _server_requested_delay(exc) or base_delay * (2 ** (attempt - 1))
            # Jitter stops mem0's two extraction calls from retrying in lockstep
            # and colliding on the same per-minute window again.
            delay += random.uniform(0, 0.5)
            if on_retry:
                on_retry(what, delay, attempt, max_retries)
            time.sleep(delay)

    assert last is not None
    raise last
