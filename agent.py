"""One turn of conversation, front-end agnostic.

This is the four-step cycle the whole project is about, extracted so that the
terminal, a Streamlit app and an HTTP handler all run *the same* logic rather
than three drifting copies of it:

    1. embed the message and semantically search stored memories
    2. inject the top matches into the system prompt
    3. call the LLM for a reply
    4. write the exchange back, letting mem0 decide ADD/UPDATE/DELETE/NONE

Like memory_store, this module never prints. It returns a Turn describing what
happened, and the front end decides how to render it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import llm
import memory_store
from config import Settings, get_settings
from memory_store import MemoryOp, Recalled

Message = dict[str, str]

# How many recent turns to carry in the prompt. Deliberately small: the long
# term is what memory is for, and stuffing the transcript back in would defeat
# the entire design.
HISTORY_WINDOW = 8


@dataclass(frozen=True)
class Turn:
    """Everything one exchange produced, including the internals."""

    user_message: str
    reply: str
    recalled: list[Recalled] = field(default_factory=list)
    ops: list[MemoryOp] = field(default_factory=list)

    @property
    def update_count(self) -> int:
        return sum(1 for op in self.ops if op.event == "UPDATE")

    @property
    def is_pending(self) -> bool:
        """True when a cloud write was queued but had not landed yet."""
        return any(op.event == "PENDING" for op in self.ops)

    def summary(self) -> str:
        """A short description of the turn, for a collapsed debug header."""
        parts = [f"{len(self.recalled)} recalled"]
        counts: dict[str, int] = {}
        for op in self.ops:
            if op.event != "PENDING":
                counts[op.event] = counts.get(op.event, 0) + 1
        for event in ("UPDATE", "DELETE", "ADD"):
            if counts.get(event):
                parts.append(f"{counts[event]} {event.lower()}")
        if self.is_pending:
            parts.append("still saving")
        if len(parts) == 1:
            parts.append("no changes")
        return " · ".join(parts)


def run_turn(
    user_message: str,
    user_id: str,
    history: list[Message],
    settings: Settings | None = None,
    on_recall: Callable[[list[Recalled]], None] | None = None,
) -> Turn:
    """Run one full exchange and return what happened.

    `history` is mutated in place: the user message and the reply are appended
    and the window is trimmed. Callers that want to keep their own transcript
    should read it from the returned Turn instead.

    `on_recall` fires after step 1 and before the LLM call, so a front end that
    streams output can show what was recalled while the model is still
    thinking. The terminal uses this; a web UI that renders once at the end
    does not need it.
    """
    settings = settings or get_settings()

    # 1. semantic search over what we already know. No LLM call, so this is
    #    free and uses no quota.
    memories = memory_store.recall(user_message, user_id=user_id)
    if on_recall:
        on_recall(memories)

    # 2. inject those memories into the system prompt.
    system_prompt = llm.build_system_prompt(memories)

    # 3. ask the LLM. `history` holds user/assistant turns only; the system
    #    prompt is passed out-of-band and never joins it.
    history.append({"role": "user", "content": user_message})
    reply = llm.generate_reply(system_prompt, history, settings)
    history.append({"role": "assistant", "content": reply})
    del history[:-HISTORY_WINDOW]

    # 4. write the exchange back. ONLY the two message strings - passing the
    #    system prompt here would feed the memories we just recalled straight
    #    back into extraction, storing duplicates of them every single turn.
    ops = memory_store.remember(user_message, reply, user_id=user_id)

    return Turn(
        user_message=user_message, reply=reply, recalled=memories, ops=ops
    )
