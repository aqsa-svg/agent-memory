"""The only module in this project that imports mem0.

Everything above it (the chat loop today, an HTTP handler tomorrow) talks to
these four functions and the two dataclasses they return:

    recall(query, user_id)                  -> list[Recalled]
    remember(user_message, assistant_message, user_id) -> list[MemoryOp]
    all_memories(user_id)                   -> list[Recalled]
    forget_all(user_id)                     -> int

Two deliberate design rules:

1. This module never prints. It returns data. Rendering belongs to whatever is
   driving it, otherwise a web handler inherits a pile of stdout noise. If you
   want progress messages during retries, install a callback via set_notifier().

2. remember() takes the user and assistant strings as separate parameters
   rather than a message list. See the note on that function - the signature is
   load-bearing, not cosmetic.
"""

from __future__ import annotations

import atexit
import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Iterator

from config import Settings, ConfigError, get_settings, mem0_config, validate
from retry import QuotaExhausted, RetryableError, with_retry

# mem0, sentence-transformers and google-genai are all chatty at import and call
# time (model download bars, an "automatic function calling" advisory on every
# Gemini request, posthog telemetry retries). None of it is actionable, and it
# would bury the memory debug output that is the point of this project.
for noisy in (
    "mem0",
    "mem0.vector_stores",
    "sentence_transformers",
    "transformers",
    "httpx",
    "google_genai",
    "google_genai.models",
    "backoff",
):
    logging.getLogger(noisy).setLevel(logging.ERROR)

# posthog still builds a (disabled) client and warns about it on stdout.
for silent in ("posthog", "urllib3"):
    logging.getLogger(silent).setLevel(logging.CRITICAL + 1)

# mem0 phones home with anonymous usage stats unless told not to. Off by
# default here: this is a local-first, zero-cost project and silent outbound
# requests are a surprise nobody asked for.
os.environ.setdefault("MEM0_TELEMETRY", "False")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

# Windows without Developer Mode cannot make symlinks, so huggingface_hub
# prints a paragraph about it on every cold start. Caching still works.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
# A progress bar for a 90MB download is useful; one for loading 103 local
# weight tensors in 0.05s is just noise on top of the debug output.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")


@dataclass(frozen=True)
class Recalled:
    """One memory retrieved from the store."""

    id: str
    text: str
    score: float | None = None  # None for get_all(), which does not rank
    created_at: str | None = None

    def __str__(self) -> str:
        return self.text


@dataclass(frozen=True)
class MemoryOp:
    """One decision mem0 made about the store after an exchange."""

    event: str  # ADD | UPDATE | DELETE | NONE
    text: str
    previous: str | None = None  # populated on UPDATE: what it replaced
    id: str | None = None


# --------------------------------------------------------------------------
# Notifications (optional, injected by the caller)
# --------------------------------------------------------------------------

_Notifier = Callable[[str], None]
_notifier: _Notifier | None = None


def set_notifier(fn: _Notifier | None) -> None:
    """Install a callback for slow-path messages, e.g. rate-limit backoff.

    The chat loop passes a dim-grey printer. A web handler would pass a logger.
    """
    global _notifier
    _notifier = fn


def _notify(message: str) -> None:
    if _notifier:
        _notifier(message)


def _on_retry(what: str, delay: float, attempt: int, total: int) -> None:
    _notify(f"{what}: rate limited, retrying in {delay:.0f}s ({attempt}/{total})")


# --------------------------------------------------------------------------
# Un-swallowing mem0's internal errors
# --------------------------------------------------------------------------

# mem0 wraps its reconciliation LLM call like this:
#
#     except Exception as e:
#         logger.error(f"Error in new memory actions response: {e}")
#         response = ""
#
# so a 429 on that call never reaches us. add() just returns zero events and
# the turn renders as a calm "NONE" - indistinguishable from "nothing worth
# remembering". That is the worst possible failure for this project: you
# contradict yourself, the UPDATE is quietly dropped, and the store keeps the
# stale fact with no visible sign anything went wrong.
#
# So we watch mem0's logger for the specific messages it emits when it eats an
# error, and re-raise. with_retry then backs off and runs add() again.
_SWALLOWED_MARKERS = (
    "error in new memory actions response",  # reconcile call failed (incl. 429)
    "error in new_retrieved_facts",  # extraction returned unparseable JSON
    "invalid json response",  # reconcile returned unparseable JSON
)


class _SwallowedErrorWatcher(logging.Handler):
    """Captures ERROR records mem0 logs instead of raising."""

    def __init__(self) -> None:
        super().__init__(level=logging.ERROR)
        self.captured: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = record.getMessage()
        except Exception:  # noqa: BLE001 - a broken log record must not break us
            return
        lowered = message.lower()
        if any(marker in lowered for marker in _SWALLOWED_MARKERS):
            self.captured.append(message)


@contextmanager
def _watch_swallowed_errors() -> Iterator[_SwallowedErrorWatcher]:
    watcher = _SwallowedErrorWatcher()
    logger = logging.getLogger("mem0")  # parent of mem0.memory.main
    logger.addHandler(watcher)
    try:
        yield watcher
    finally:
        logger.removeHandler(watcher)


# --------------------------------------------------------------------------
# Response normalisation
# --------------------------------------------------------------------------


def _rows(response: Any) -> list[dict[str, Any]]:
    """Coerce any mem0 response into a plain list of dicts.

    The two backends disagree, and each has changed shape across versions:

        OSS Memory   -> {"results": [...]}          (v1.1)
        OSS Memory   -> [...]                       (older v1.0 paths)
        MemoryClient -> [...]                       (v1 endpoints)
        MemoryClient -> {"results": [...]}          (v2 endpoints / output_format)
        MemoryClient -> {"memories": [...]}         (some list endpoints)

    Rather than assume, accept all of them. Callers above this line always get
    a list, so nothing upstream needs to know which backend answered.
    """
    if response is None:
        return []
    if isinstance(response, list):
        return [row for row in response if isinstance(row, dict)]
    if isinstance(response, dict):
        for key in ("results", "memories", "data"):
            value = response.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
        # A single memory object returned bare, e.g. from an update endpoint.
        if "memory" in response or "id" in response:
            return [response]
    return []


# --------------------------------------------------------------------------
# Backends
# --------------------------------------------------------------------------


class MemoryStoreError(RuntimeError):
    """Setup failure with an actionable message attached."""


class _Backend:
    """Interface both backends implement. The chat loop never sees these."""

    label: str = "memory"
    # True when add() returns before the memory actually exists, so the caller
    # has to wait and diff rather than read events straight off the response.
    async_writes: bool = False

    def search(self, query: str, user_id: str, limit: int, threshold: float | None) -> Any:
        raise NotImplementedError

    def add(self, messages: list[dict[str, str]], user_id: str) -> Any:
        raise NotImplementedError

    def get_all(self, user_id: str, limit: int) -> Any:
        raise NotImplementedError

    def delete_all(self, user_id: str) -> None:
        raise NotImplementedError

    def close(self) -> None:
        """Release resources. No-op unless the backend holds a handle."""


class _OssBackend(_Backend):
    """Fully local: sentence-transformers on CPU + Qdrant on disk."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        # Imported here, not at module scope, so cloud mode never pays for
        # torch. `from mem0 import Memory` itself is cheap - the heavy
        # sentence-transformers import happens inside from_config, when the
        # embedder factory actually builds the huggingface embedder.
        from mem0 import Memory

        try:
            self.memory = Memory.from_config(mem0_config(settings))
        except Exception as exc:  # noqa: BLE001
            raise MemoryStoreError(_diagnose(exc, settings)) from exc

        where = (
            f"local Qdrant at {settings.qdrant_path.name}/"
            if settings.local_qdrant
            else f"Qdrant server {settings.qdrant_host}:{settings.qdrant_port}"
        )
        self.label = (
            f"oss (on this machine) - {where}, "
            f"embeddings {settings.embedding_model.split('/')[-1]} "
            f"[{settings.embedding_dims}d], extraction via {settings.llm_model}"
        )

    def search(self, query: str, user_id: str, limit: int, threshold: float | None) -> Any:
        kwargs: dict[str, Any] = {"user_id": user_id, "limit": limit}
        if threshold is not None:
            kwargs["threshold"] = threshold
        return self.memory.search(query, **kwargs)

    def add(self, messages: list[dict[str, str]], user_id: str) -> Any:
        # infer=True is what turns raw text into extracted facts. With
        # infer=False mem0 stores messages verbatim, which is a transcript.
        return self.memory.add(messages, user_id=user_id, infer=True)

    def get_all(self, user_id: str, limit: int) -> Any:
        return self.memory.get_all(user_id=user_id, limit=limit)

    def delete_all(self, user_id: str) -> None:
        self.memory.delete_all(user_id=user_id)

    def close(self) -> None:
        try:
            self.memory.vector_store.client.close()
        except Exception:  # noqa: BLE001 - shutdown is best-effort
            pass


class _CloudBackend(_Backend):
    """mem0's hosted API. No local model, no local vector store.

    Writes are asynchronous. add() returns

        {"results": [{"status": "PENDING", "event_id": "...", ...}]}

    and the memory only materialises once a background worker has run, so
    get_all() straight afterwards can still come back empty. That is why
    async_writes is True: remember() has to snapshot, wait, and diff to work
    out what actually happened, instead of reading events off the response.
    """

    async_writes = True

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        from mem0 import MemoryClient

        try:
            self.client = MemoryClient(api_key=settings.mem0_api_key)
        except Exception as exc:  # noqa: BLE001
            raise MemoryStoreError(_diagnose_cloud(exc)) from exc

        self.label = "cloud (mem0 hosted) - extraction and storage on mem0's servers"
        # Set once we learn whether this deployment accepts output_format.
        self._output_format: str | None = "v1.1"

    def search(self, query: str, user_id: str, limit: int, threshold: float | None) -> Any:
        # The hosted API spells the result cap top_k, unlike the OSS limit=.
        return self.client.search(query, user_id=user_id, top_k=limit)

    def add(self, messages: list[dict[str, str]], user_id: str) -> Any:
        # output_format="v1.1" is what makes the API report per-memory events
        # (ADD/UPDATE/DELETE) rather than a bare acknowledgement. Older
        # deployments reject the field, so fall back once and remember.
        if self._output_format:
            try:
                return self.client.add(
                    messages, user_id=user_id, output_format=self._output_format
                )
            except Exception as exc:  # noqa: BLE001
                if not _looks_like_bad_param(exc):
                    raise
                self._output_format = None
        return self.client.add(messages, user_id=user_id)

    def get_all(self, user_id: str, limit: int) -> Any:
        return self.client.get_all(user_id=user_id)

    def delete_all(self, user_id: str) -> None:
        self.client.delete_all(user_id=user_id)


def _is_pending(rows: list[dict[str, Any]]) -> bool:
    """Did the hosted API just queue the write instead of performing it?"""
    return any(
        str(row.get("status", "")).upper() == "PENDING" or "event_id" in row
        for row in rows
    )


def _snapshot(user_id: str) -> dict[str, str]:
    """Map memory id -> text, for diffing across an asynchronous write."""
    return {m.id: m.text for m in all_memories(user_id)}


def _diff_to_ops(
    before: dict[str, str], after: dict[str, str]
) -> list[MemoryOp]:
    """Reconstruct ADD / UPDATE / DELETE by comparing two snapshots.

    The hosted API tells us nothing about what it decided, so we work it out
    from the outcome. This keeps the debug output - the whole point of the
    project - identical between the two backends.
    """
    ops: list[MemoryOp] = []
    for mem_id, text in after.items():
        if mem_id not in before:
            ops.append(MemoryOp(event="ADD", text=text, id=mem_id))
        elif before[mem_id] != text:
            ops.append(
                MemoryOp(
                    event="UPDATE", text=text, previous=before[mem_id], id=mem_id
                )
            )
    for mem_id, text in before.items():
        if mem_id not in after:
            ops.append(MemoryOp(event="DELETE", text=text, id=mem_id))
    return ops


def _await_write(
    user_id: str, before: dict[str, str], settings: Settings
) -> list[MemoryOp]:
    """Poll until the queued write lands, then report what changed.

    Bounded: a free-tier worker can be slow or the exchange may genuinely
    contain nothing worth storing, and either way we must not hang the chat
    loop. On timeout we say PENDING rather than pretending it was NONE - the
    memory may still show up a moment later.
    """
    deadline = time.monotonic() + settings.cloud_write_timeout
    delay = 0.75
    while time.monotonic() < deadline:
        time.sleep(delay)
        after = _snapshot(user_id)
        if after != before:
            return _diff_to_ops(before, after)
        delay = min(delay * 1.5, 3.0)  # back off; do not hammer the API

    return [
        MemoryOp(
            event="PENDING",
            text=(
                "mem0 Cloud queued this write and it had not landed after "
                f"{settings.cloud_write_timeout:.0f}s. It may still appear - "
                "check /memories in a moment."
            ),
        )
    ]


def _looks_like_bad_param(exc: BaseException) -> bool:
    """Did the API reject a field we sent, as opposed to failing for real?"""
    blob = f"{type(exc).__name__}: {exc}".lower()
    return "output_format" in blob or "400" in blob or "unexpected" in blob


def _diagnose_cloud(exc: Exception) -> str:
    blob = f"{type(exc).__name__}: {exc}"
    lowered = blob.lower()
    if "api key" in lowered or "401" in lowered or "unauthor" in lowered:
        return (
            "mem0 Cloud rejected the API key.\n"
            "  - Check MEM0_API_KEY in .env against https://app.mem0.ai "
            "(Settings -> API Keys)\n"
            "  - Or set MEMORY_BACKEND=oss to run entirely on this machine.\n\n"
            f"Original error: {blob}"
        )
    if "quota" in lowered or "429" in lowered:
        return (
            "mem0 Cloud quota exceeded on the free Hobby tier.\n"
            "  - Set MEMORY_BACKEND=oss to keep going locally with no cap.\n\n"
            f"Original error: {blob}"
        )
    return blob


# --------------------------------------------------------------------------
# Lazy singleton
# --------------------------------------------------------------------------

_backend: _Backend | None = None


def _build() -> _Backend:
    settings = get_settings()
    validate(settings)
    if settings.memory_backend == "cloud":
        return _CloudBackend(settings)
    return _OssBackend(settings)


def _diagnose(exc: Exception, settings: Settings) -> str:
    """Turn the three failures people actually hit into readable instructions."""
    blob = f"{type(exc).__name__}: {exc}"
    lowered = blob.lower()

    if "already accessed by another instance" in lowered or "storage folder" in lowered:
        return (
            f"Qdrant local mode holds an exclusive lock on {settings.qdrant_path}, "
            f"and another process already has it.\n"
            f"  - Close any other running chat.py / seed_memories.py, or\n"
            f"  - run `docker compose up -d` and set QDRANT_HOST=localhost in .env "
            f"to allow concurrent access.\n\nOriginal error: {blob}"
        )

    if "dimension" in lowered or "vector" in lowered and "dim" in lowered:
        return (
            f"Vector size mismatch. The collection {settings.qdrant_collection!r} was "
            f"created with a different dimension than {settings.embedding_model} "
            f"produces ({settings.embedding_dims}).\n"
            f"  - Delete {settings.qdrant_path} to recreate the collection, or\n"
            f"  - set QDRANT_COLLECTION to a new name in .env.\n\nOriginal error: {blob}"
        )

    if "sentence_transformers" in lowered or "sentence-transformers" in lowered:
        return (
            "Local embeddings need sentence-transformers, which is not installed.\n"
            "  - pip install -r requirements-local.txt   (adds torch, ~1GB), or\n"
            "  - set EMBEDDER_PROVIDER=gemini to embed through the API instead "
            "(no download).\n\n"
            f"Original error: {blob}"
        )

    if "api_key" in lowered or "api key" in lowered or "unauthenticated" in lowered:
        return f"The LLM rejected the credentials while building the memory store.\n\n{blob}"

    return blob


def get_backend() -> _Backend:
    """Return the process-wide backend, building it on first use.

    Lazy on purpose: the oss backend loads all-MiniLM-L6-v2 (~90MB the first
    time, a couple of seconds after that). Cloud mode is effectively instant.
    """
    global _backend
    if _backend is None:
        _backend = _build()
    return _backend


def describe_backend() -> str:
    """One line naming the active memory backend, for the startup banner."""
    return get_backend().label


def warm_up() -> None:
    """Force the slow one-time setup to happen now, with a message on screen.

    Without this, the first user message of a session appears to hang for
    several seconds while torch loads. No-op cost in cloud mode.
    """
    get_backend()


def close() -> None:
    """Release backend resources, notably the Qdrant folder lock.

    Registered with atexit because qdrant-client's __del__ runs during
    interpreter teardown, by which point sys.meta_path is gone and its own
    imports raise. That produces an ugly (though harmless) ImportError
    traceback after every clean exit. Closing deterministically avoids it.
    """
    global _backend
    if _backend is None:
        return
    _backend.close()
    _backend = None


atexit.register(close)


def reset_store() -> None:
    """Drop the singleton, releasing any handles. Mainly for tests."""
    close()


# --------------------------------------------------------------------------
# Public API - identical regardless of which backend is active
# --------------------------------------------------------------------------


def recall(
    query: str,
    user_id: str,
    limit: int | None = None,
    threshold: float | None = None,
) -> list[Recalled]:
    """Semantically search this user's memories.

    In oss mode the query is embedded locally with all-MiniLM-L6-v2 and matched
    against Qdrant by cosine similarity - no LLM call, so it costs nothing and
    uses no quota. In cloud mode mem0's servers do the same work remotely.
    """
    query = (query or "").strip()
    if not query:
        return []

    settings = get_settings()
    top_k = limit if limit is not None else settings.recall_limit
    backend = get_backend()

    # Scoping by user_id is what keeps memories separate between people. Both
    # backends apply it server-side (a Qdrant payload filter / an API param),
    # never as a post-filter, so one user's memories cannot surface in
    # another user's search.
    response = with_retry(
        lambda: backend.search(query, user_id, top_k, threshold),
        what="memory search",
        max_retries=settings.max_retries,
        base_delay=settings.retry_base_delay,
        on_retry=_on_retry,
    )

    return [
        Recalled(
            id=str(row.get("id", "")),
            text=str(row.get("memory", "")),
            score=row.get("score"),
            created_at=row.get("created_at"),
        )
        for row in _rows(response)
        if row.get("memory")
    ]


def remember(
    user_message: str,
    assistant_message: str,
    user_id: str,
) -> list[MemoryOp]:
    """Extract durable facts from one exchange and reconcile them with the store.

    THE SIGNATURE IS THE BUG FIX. This takes two strings, not a message list,
    so it is structurally impossible to hand mem0 the system prompt.

    Why that matters: the system prompt built for this turn contains the
    memories we just recalled. Feed it back to mem0 and the extractor sees
    those facts as brand-new input, re-extracts them, and stores near-duplicates
    of what it already has. Do that every turn and the store fills with copies
    of the same fact, recall quality collapses, and reconciliation can no longer
    tell which duplicate to UPDATE. Plenty of tutorials pass the whole
    conversation here, including the system role. Do not.

    In oss mode this costs 2 LLM calls against your own key: one to extract
    candidate facts, one to decide ADD / UPDATE / DELETE / NONE against what is
    already stored. In cloud mode mem0 runs both on their side, so it costs
    none of your Gemini quota.
    """
    settings = get_settings()
    backend = get_backend()

    # Only these two roles. No system role, ever - see the docstring.
    messages = [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": assistant_message},
    ]

    def _add_once() -> Any:
        with _watch_swallowed_errors() as watcher:
            result = backend.add(messages, user_id)
        if watcher.captured:
            # Re-raise so a rate-limited reconcile is retried rather than
            # silently reported as "nothing to remember".
            raise RetryableError(watcher.captured[0])
        return result

    # On an asynchronous backend we need the "before" picture first, because
    # add() returns before the write has happened. Costs one extra listing per
    # turn, which is the price of showing you what the memory layer decided.
    before = _snapshot(user_id) if backend.async_writes else {}

    response = with_retry(
        _add_once,
        what="memory extraction",
        max_retries=settings.max_retries,
        base_delay=settings.retry_base_delay,
        on_retry=_on_retry,
    )

    if backend.async_writes and _is_pending(_rows(response)):
        return _await_write(user_id, before, settings)

    ops: list[MemoryOp] = []
    for row in _rows(response):
        event = str(row.get("event") or "NONE").upper()
        ops.append(
            MemoryOp(
                event=event,
                text=str(row.get("memory", "")),
                # mem0 reports the superseded text on UPDATE, which is exactly
                # what makes a contradiction visible in the debug output.
                previous=row.get("previous_memory") or row.get("old_memory"),
                id=str(row.get("id")) if row.get("id") else None,
            )
        )
    return ops


def all_memories(user_id: str, limit: int = 200) -> list[Recalled]:
    """Every stored memory for one user, newest-first where mem0 reports dates."""
    settings = get_settings()
    backend = get_backend()

    response = with_retry(
        lambda: backend.get_all(user_id, limit),
        what="memory listing",
        max_retries=settings.max_retries,
        base_delay=settings.retry_base_delay,
        on_retry=_on_retry,
    )

    items = [
        Recalled(
            id=str(row.get("id", "")),
            text=str(row.get("memory", "")),
            score=None,  # get_all does not rank, so there is no score to show
            created_at=row.get("created_at"),
        )
        for row in _rows(response)
        if row.get("memory")
    ]
    # created_at is an ISO-8601 string, so a plain string sort is chronological.
    items.sort(key=lambda m: m.created_at or "", reverse=True)
    return items


def forget_all(user_id: str) -> int:
    """Delete every memory for one user. Returns how many were removed.

    Scoped to user_id on purpose: there is no "wipe everyone" in this API, so a
    multi-user deployment cannot lose another user's data through this path.
    """
    backend = get_backend()
    count = len(all_memories(user_id))
    if count:
        backend.delete_all(user_id)
    return count


def known_users() -> list[str]:
    """Best-effort list of user_ids that have memories, for greeting returnees.

    Only implemented for the oss backend, by reading payloads straight out of
    Qdrant - mem0's OSS Memory class has no "list users" call. The hosted API
    does expose one, but it is not part of the free Hobby surface we rely on
    here, so cloud mode simply returns []. Callers must treat an empty list as
    "unknown", never as "no users exist".
    """
    settings = get_settings()
    if not settings.uses_local_memory:
        return []

    try:
        backend = get_backend()
        client = backend.memory.vector_store.client  # type: ignore[attr-defined]
        seen: set[str] = set()
        offset = None
        while True:
            points, offset = client.scroll(
                collection_name=settings.qdrant_collection,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                user = (point.payload or {}).get("user_id")
                if user:
                    seen.add(str(user))
            if offset is None:
                break
        return sorted(seen)
    except Exception:  # noqa: BLE001 - a nicety; never break startup over it
        return []


__all__ = [
    "Recalled",
    "MemoryOp",
    "MemoryStoreError",
    "QuotaExhausted",
    "ConfigError",
    "recall",
    "remember",
    "all_memories",
    "forget_all",
    "known_users",
    "warm_up",
    "set_notifier",
    "get_backend",
    "describe_backend",
    "reset_store",
    "close",
]
