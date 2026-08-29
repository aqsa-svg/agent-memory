"""The LLM that writes your replies.

Separate from the memory layer on purpose: mem0 runs its own extraction model
(or, in cloud mode, mem0's servers do). This module is only about answering
the human.

Both backends are free:
    gemini  - free tier, no credit card, needs a key
    ollama  - runs on your machine, needs no key at all
"""

from __future__ import annotations

import json
from typing import Any, Callable, Iterable

from config import Settings, get_settings
from retry import with_retry

# One turn of conversation, as passed around by the chat loop.
Message = dict[str, str]

_Notifier = Callable[[str], None]
_notifier: _Notifier | None = None


def set_notifier(fn: _Notifier | None) -> None:
    """Install a callback for retry/backoff messages. See memory_store."""
    global _notifier
    _notifier = fn


def _on_retry(what: str, delay: float, attempt: int, total: int) -> None:
    if _notifier:
        _notifier(f"{what}: rate limited, retrying in {delay:.0f}s ({attempt}/{total})")


_client: Any = None


def _gemini_client(settings: Settings) -> Any:
    global _client
    if _client is None:
        from google import genai

        _client = genai.Client(api_key=settings.gemini_api_key)
    return _client


def _gemini_reply(
    system_prompt: str, history: Iterable[Message], settings: Settings
) -> str:
    from google.genai import types

    client = _gemini_client(settings)

    # Gemini takes the system prompt out-of-band rather than as a message, and
    # calls the assistant role "model".
    contents = [
        types.Content(
            role="model" if m["role"] == "assistant" else "user",
            parts=[types.Part(text=m["content"])],
        )
        for m in history
    ]

    response = client.models.generate_content(
        model=settings.gemini_model,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.7,
            # Generous on purpose. The 2.5+ models spend tokens on internal
            # reasoning before emitting any text, and those count against this
            # budget - set it too low and you get an empty reply with a
            # finish_reason of MAX_TOKENS, which reads like a broken key.
            max_output_tokens=2048,
        ),
    )

    text = (getattr(response, "text", "") or "").strip()
    if text:
        return text

    # Empty means blocked or truncated. Say which, rather than returning "".
    reason = ""
    try:
        candidate = (response.candidates or [None])[0]
        reason = str(getattr(candidate, "finish_reason", "") or "")
    except Exception:  # noqa: BLE001
        pass
    if "SAFETY" in reason.upper():
        return "(The model declined to answer that one - safety filter.)"
    if "MAX_TOKENS" in reason.upper():
        return "(The reply hit the token limit before producing text.)"
    return "(The model returned an empty reply.)"


def _ollama_reply(
    system_prompt: str, history: Iterable[Message], settings: Settings
) -> str:
    import requests

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(dict(m) for m in history)

    response = requests.post(
        f"{settings.ollama_base_url}/api/chat",
        json={"model": settings.ollama_model, "messages": messages, "stream": False},
        # Local models on CPU are slow; a short timeout would fail healthy runs.
        timeout=300,
    )
    response.raise_for_status()
    payload = response.json()
    return str(payload.get("message", {}).get("content", "")).strip() or (
        "(Ollama returned an empty reply.)"
    )


def generate_reply(
    system_prompt: str,
    history: list[Message],
    settings: Settings | None = None,
) -> str:
    """Ask the configured LLM for one reply.

    `history` is the running conversation (user/assistant turns only). The
    system prompt is passed separately and is NEVER appended to history - it
    holds the recalled memories, and letting it leak into what gets written
    back to mem0 is precisely the duplication bug this project avoids.
    """
    settings = settings or get_settings()

    def _call() -> str:
        if settings.backend == "gemini":
            return _gemini_reply(system_prompt, history, settings)
        return _ollama_reply(system_prompt, history, settings)

    return with_retry(
        _call,
        what=f"{settings.backend} reply",
        max_retries=settings.max_retries,
        base_delay=settings.retry_base_delay,
        on_retry=_on_retry,
    )


SYSTEM_PROMPT_TEMPLATE = """You are a helpful assistant with long-term memory of this user.

What you remember about them:
{memories}

Use these memories naturally when they are relevant - do not recite them back
or announce that you are remembering. If a memory contradicts what the user
just said, trust what they said now. If nothing above is relevant, simply
answer the question."""

NO_MEMORIES = "(nothing yet - this is a new user)"


def build_system_prompt(memories: Iterable[Any]) -> str:
    """Render recalled memories into the system prompt for this turn."""
    lines = [f"- {m}" for m in memories]
    return SYSTEM_PROMPT_TEMPLATE.format(memories="\n".join(lines) or NO_MEMORIES)
