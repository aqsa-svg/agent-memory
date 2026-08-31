"""Safety rails for a publicly reachable deployment.

None of this matters when you run the app locally for yourself. It matters a
great deal the moment the URL is in a post, because the deployment spends *one
person's* free-tier quota on behalf of everyone who visits.

Three separate concerns, deliberately kept apart:

  isolation   every visitor gets their own memory scope, so strangers cannot
              read each other's facts out of the sidebar
  rate limit  per-visitor, stops one person monopolising the demo
  budget      global daily ceiling, so the quota cannot be fully drained and
              the link degrades to a polite message instead of errors

Enabled with DEMO_MODE=true. Off by default: running this locally should not
rate-limit you in your own app.
"""

from __future__ import annotations

import os
import secrets
import threading
import time
from dataclasses import dataclass, field


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "").strip() or default)
    except ValueError:
        return default


DEMO_MODE = os.getenv("DEMO_MODE", "").strip().lower() in {"1", "true", "yes", "on"}

# Enough to demonstrate memory across a few turns, not enough to sit and chat.
PER_VISITOR_MESSAGES = _env_int("DEMO_VISITOR_LIMIT", 12)
PER_VISITOR_WINDOW = _env_int("DEMO_VISITOR_WINDOW", 3600)  # seconds

# Global ceiling. Sized well under the model's daily quota so that ordinary
# use of the same key elsewhere is not starved by the demo.
DAILY_MESSAGE_BUDGET = _env_int("DEMO_DAILY_BUDGET", 300)


@dataclass
class _Bucket:
    hits: list[float] = field(default_factory=list)


class DemoGuard:
    """In-process limiter. Single instance, guarded by a lock."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._visitors: dict[str, _Bucket] = {}
        self._day = time.gmtime().tm_yday
        self._spent_today = 0

    def _roll_day(self) -> None:
        today = time.gmtime().tm_yday
        if today != self._day:
            self._day = today
            self._spent_today = 0

    def check(self, visitor: str) -> tuple[bool, str]:
        """Return (allowed, message). Message is only meaningful when blocked."""
        if not DEMO_MODE:
            return True, ""

        now = time.time()
        with self._lock:
            self._roll_day()

            if self._spent_today >= DAILY_MESSAGE_BUDGET:
                return False, (
                    "This demo has used up today's shared free-tier budget. "
                    "It resets tomorrow. Clone the repo to run it with your own "
                    "key - it is free."
                )

            bucket = self._visitors.setdefault(visitor, _Bucket())
            bucket.hits = [h for h in bucket.hits if now - h < PER_VISITOR_WINDOW]

            if len(bucket.hits) >= PER_VISITOR_MESSAGES:
                minutes = max(1, int(PER_VISITOR_WINDOW / 60))
                return False, (
                    f"You have reached this demo's limit of "
                    f"{PER_VISITOR_MESSAGES} messages per {minutes} minutes. "
                    f"Clone the repo to run it without limits."
                )

            bucket.hits.append(now)
            self._spent_today += 1
            return True, ""

    def refund(self, visitor: str) -> None:
        """Give back an allowance when the turn failed through no fault of the visitor."""
        if not DEMO_MODE:
            return
        with self._lock:
            bucket = self._visitors.get(visitor)
            if bucket and bucket.hits:
                bucket.hits.pop()
            self._spent_today = max(0, self._spent_today - 1)

    def stats(self) -> dict[str, int | bool]:
        with self._lock:
            self._roll_day()
            return {
                "demo_mode": DEMO_MODE,
                "spent_today": self._spent_today,
                "daily_budget": DAILY_MESSAGE_BUDGET,
                "per_visitor": PER_VISITOR_MESSAGES,
            }


guard = DemoGuard()


def new_visitor_id() -> str:
    """An unguessable per-browser memory scope.

    In demo mode the user_id must NOT come from the client as free text. If it
    did, one visitor could type another's name - or simply leave the default -
    and read their memories straight out of the sidebar. A random token issued
    server-side and stored in the browser keeps every visitor's memories to
    themselves while still surviving a page reload, which is what makes the
    "close the tab and come back" demonstration work.
    """
    return "guest_" + secrets.token_hex(8)


def is_valid_visitor_id(value: str) -> bool:
    if not value.startswith("guest_"):
        return False
    token = value[len("guest_"):]
    return len(token) == 16 and all(c in "0123456789abcdef" for c in token)
