"""Single source of truth for every provider, model and path in the project.

Nothing else in the codebase reads os.environ directly. If you want to swap an
LLM, a vector store or an embedding model, this is the only file you touch.

Cost note: every provider named here is free. Gemini's free tier needs no
credit card, Ollama runs on your own machine, sentence-transformers runs on
your CPU, and Qdrant local mode is just a folder on disk.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv

# Load .env from the project root regardless of the cwd the user launched from,
# so `python chat.py` behaves the same as `python /full/path/to/chat.py`.
PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

Backend = Literal["gemini", "ollama"]

# Where memories live.
#   "cloud" -> mem0's hosted API. Extraction, embedding and storage all happen
#              on their servers, so nothing is downloaded and no LLM key is
#              needed for the memory layer. Your conversations leave this
#              machine.
#   "oss"   -> everything local: sentence-transformers on CPU + Qdrant on disk.
#              Nothing leaves the machine, and there is no per-account cap.
MemoryBackend = Literal["cloud", "oss"]

# all-MiniLM-L6-v2 emits 384-dimensional vectors. Qdrant fixes a collection's
# vector size at creation time, so this number MUST match the embedder or every
# insert fails. mem0 defaults to 1536 (the OpenAI size) when you do not say
# otherwise, which is exactly the mismatch that bites people who adapt an
# OpenAI-based tutorial to a local embedder.
EMBEDDING_DIMS = 384


class ConfigError(RuntimeError):
    """Raised for missing/invalid configuration, carrying a human-readable fix."""


def _env_str(name: str, default: str) -> str:
    # Treat an empty string in .env the same as "not set": a bare `KEY=` line is
    # a very common way to end up with a silently blank value.
    value = os.getenv(name, "").strip()
    return value or default


def _env_int(name: str, default: int) -> int:
    raw = _env_str(name, str(default))
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc


def _env_bool(name: str, default: bool) -> bool:
    truthy = {"1", "true", "yes", "on"}
    return _env_str(name, "true" if default else "false").lower() in truthy


def _env_path(name: str, default: Path) -> Path:
    """Resolve a path setting, anchoring relatives to the project root.

    Without this, `QDRANT_PATH=./qdrant_data` would follow whatever directory
    you happened to launch from, so running the chat from your home folder
    would silently start a second, empty memory store.
    """
    raw = _env_str(name, "")
    if not raw:
        return default
    candidate = Path(raw).expanduser()
    return candidate if candidate.is_absolute() else (PROJECT_ROOT / candidate).resolve()


@dataclass(frozen=True)
class Settings:
    """Immutable, fully-resolved configuration for one run of the app."""

    # --- where memories are stored (see MemoryBackend above) ---
    memory_backend: MemoryBackend = "cloud"
    mem0_api_key: str = ""

    # --- which LLM answers you, and which LLM mem0 uses to extract facts ---
    backend: Backend = "gemini"

    # Flash-LITE tier on purpose, and the choice is quota-driven rather than
    # quality-driven. Free-tier quota is enforced per project PER MODEL
    # (quotaId GenerateRequestsPerDayPerProjectPerModel-FreeTier), and the
    # allowance differs wildly between models: gemini-2.5-flash reports a
    # limit of just 20 requests/day, which at ~3 calls per chat turn is about
    # six messages before you are locked out for the day. The lite models have
    # far more headroom and still reconcile memory correctly (verified: they
    # emit UPDATE and DELETE, not just ADD).
    #
    # Because the quota is per model, switching GEMINI_MODEL to another entry
    # from `python check_setup.py --list-models` gives you a fresh allowance.
    gemini_model: str = "gemini-3.5-flash-lite"
    gemini_api_key: str = ""

    ollama_model: str = "llama3.1:8b"
    ollama_base_url: str = "http://localhost:11434"

    # --- embeddings: always local, always on CPU, whatever the LLM backend is ---
    # Keeping embeddings independent of `backend` means switching gemini <-> ollama
    # never changes vector dimensions, so an existing collection stays readable.
    embedder_provider: str = "huggingface"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dims: int = EMBEDDING_DIMS

    # --- vector store ---
    qdrant_collection: str = "agent_memory"
    qdrant_path: Path = field(default_factory=lambda: PROJECT_ROOT / "qdrant_data")
    # Set QDRANT_HOST to switch from embedded-file mode to a real server
    # (see docker-compose.yml). Empty string means local mode.
    qdrant_host: str = ""
    qdrant_port: int = 6333

    # mem0 keeps an audit log of every ADD/UPDATE/DELETE in SQLite.
    history_db_path: Path = field(
        default_factory=lambda: PROJECT_ROOT / "mem0_history.db"
    )

    # --- behaviour ---
    recall_limit: int = 5  # how many memories to inject per turn
    debug: bool = True  # print recalled memories and memory operations
    max_retries: int = 5  # for 429 / transient LLM errors
    retry_base_delay: float = 2.0  # seconds, doubled each attempt
    # How long to wait for mem0 Cloud's background worker to finish a queued
    # write before giving up and reporting PENDING. Cloud backend only.
    cloud_write_timeout: float = 12.0

    @property
    def local_qdrant(self) -> bool:
        return not self.qdrant_host

    @property
    def uses_local_memory(self) -> bool:
        """True when we must load sentence-transformers and open Qdrant.

        Gate every import of those on this. In cloud mode they are dead weight:
        a ~90MB model download and several seconds of torch startup for a code
        path that never runs.
        """
        return self.memory_backend == "oss"

    @property
    def llm_model(self) -> str:
        return self.gemini_model if self.backend == "gemini" else self.ollama_model


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Read the environment once and cache the result."""
    backend = _env_str("BACKEND", "gemini").lower()
    if backend not in ("gemini", "ollama"):
        raise ConfigError(
            f"BACKEND must be 'gemini' or 'ollama', got {backend!r}. Check your .env."
        )

    memory_backend = _env_str("MEMORY_BACKEND", "cloud").lower()
    if memory_backend not in ("cloud", "oss"):
        raise ConfigError(
            f"MEMORY_BACKEND must be 'cloud' or 'oss', got {memory_backend!r}. "
            f"Check your .env."
        )

    return Settings(
        memory_backend=memory_backend,  # type: ignore[arg-type]
        mem0_api_key=_env_str("MEM0_API_KEY", ""),
        backend=backend,  # type: ignore[arg-type]
        gemini_model=_env_str("GEMINI_MODEL", "gemini-3.5-flash-lite"),
        # Accept either name: Google's own tooling reads GOOGLE_API_KEY, most
        # tutorials say GEMINI_API_KEY. Requiring exactly one of them is a
        # classic first-run trap, so take whichever is present.
        gemini_api_key=_env_str("GEMINI_API_KEY", "") or _env_str("GOOGLE_API_KEY", ""),
        ollama_model=_env_str("OLLAMA_MODEL", "llama3.1:8b"),
        ollama_base_url=_env_str("OLLAMA_BASE_URL", "http://localhost:11434"),
        embedding_model=_env_str(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        embedding_dims=_env_int("EMBEDDING_DIMS", EMBEDDING_DIMS),
        qdrant_collection=_env_str("QDRANT_COLLECTION", "agent_memory"),
        qdrant_path=_env_path("QDRANT_PATH", PROJECT_ROOT / "qdrant_data"),
        qdrant_host=_env_str("QDRANT_HOST", ""),
        qdrant_port=_env_int("QDRANT_PORT", 6333),
        history_db_path=_env_path("HISTORY_DB_PATH", PROJECT_ROOT / "mem0_history.db"),
        recall_limit=_env_int("RECALL_LIMIT", 5),
        debug=_env_bool("DEBUG", True),
        max_retries=_env_int("MAX_RETRIES", 5),
        cloud_write_timeout=float(_env_int("CLOUD_WRITE_TIMEOUT", 12)),
    )


MISSING_KEY_HELP = """
No Gemini API key found.

  1. Open https://aistudio.google.com/apikey  (sign in with any Google account)
  2. Click "Create API key" - no credit card and no billing setup required
  3. Copy .env.example to .env and paste the key in:

         GEMINI_API_KEY=AIza...

Prefer to stay fully offline instead? Install Ollama (https://ollama.com),
run `ollama pull llama3.1:8b`, then set BACKEND=ollama in your .env.
That path needs no key at all.
""".strip()


MISSING_MEM0_KEY_HELP = """
MEMORY_BACKEND is 'cloud' but no MEM0_API_KEY was found.

  1. Open https://app.mem0.ai  and sign up (the Hobby tier is free and asks
     for no credit card)
  2. Go to Settings -> API Keys and create one
  3. Add it to your .env:

         MEM0_API_KEY=m0-...

Or keep everything on this machine instead - no account, no key, no cap:

         MEMORY_BACKEND=oss

Local mode stores memories in Qdrant on disk and embeds with
sentence-transformers on your CPU.
""".strip()


def validate(settings: Settings) -> None:
    """Fail fast with an actionable message instead of a stack trace later."""
    # The chat reply always needs an LLM, whichever memory backend is in use.
    if settings.backend == "gemini" and not settings.gemini_api_key:
        raise ConfigError(MISSING_KEY_HELP)

    if settings.memory_backend == "cloud" and not settings.mem0_api_key:
        raise ConfigError(MISSING_MEM0_KEY_HELP)

    # Everything below only applies to the local vector store, which cloud mode
    # never touches.
    if not settings.uses_local_memory:
        return

    # Guard the single most likely setup mistake in this whole project.
    if (
        settings.embedding_dims != EMBEDDING_DIMS
        and "MiniLM-L6" in settings.embedding_model
    ):
        raise ConfigError(
            f"EMBEDDING_DIMS={settings.embedding_dims} but {settings.embedding_model} "
            f"produces {EMBEDDING_DIMS}-dim vectors, so every Qdrant insert would "
            f"fail. Remove EMBEDDING_DIMS from .env to use the default."
        )


def _llm_block(settings: Settings) -> dict[str, Any]:
    """The `llm` section of the mem0 config, used for fact extraction."""
    if settings.backend == "gemini":
        return {
            "provider": "gemini",
            "config": {
                "model": settings.gemini_model,
                "api_key": settings.gemini_api_key,
                # Extraction is a structured-output task, not a creative one.
                # Temperature drift here produces junk memories.
                "temperature": 0.1,
                "max_tokens": 2000,
            },
        }
    return {
        "provider": "ollama",
        "config": {
            "model": settings.ollama_model,
            "ollama_base_url": settings.ollama_base_url,
            "temperature": 0.1,
            "max_tokens": 2000,
        },
    }


def _vector_store_block(settings: Settings) -> dict[str, Any]:
    base: dict[str, Any] = {
        "collection_name": settings.qdrant_collection,
        "embedding_model_dims": settings.embedding_dims,
    }
    if settings.local_qdrant:
        # `path` is what makes qdrant-client run embedded, in-process. It takes
        # an exclusive file lock on the folder: one process at a time.
        base["path"] = str(settings.qdrant_path)
        # on_disk=True is NOT an optimisation here, it is what makes memory
        # persist at all. mem0's Qdrant wrapper does this on startup:
        #     if not on_disk: shutil.rmtree(path)
        # and on_disk defaults to False. Leave this off and every launch
        # silently deletes the entire store, which looks exactly like "the
        # agent forgot everything again".
        base["on_disk"] = True
    else:
        base["host"] = settings.qdrant_host
        base["port"] = settings.qdrant_port
    return {"provider": "qdrant", "config": base}


def mem0_config(settings: Settings | None = None) -> dict[str, Any]:
    """Build the dict handed to `mem0.Memory.from_config()`.

    Only meaningful for MEMORY_BACKEND=oss. In cloud mode mem0's servers own
    the embedder, the vector store and the extraction LLM, so none of this is
    sent anywhere.

    Lives here rather than in memory_store.py so that every model and provider
    decision sits in one file. It is plain data, so importing config never
    imports mem0.
    """
    settings = settings or get_settings()
    return {
        "vector_store": _vector_store_block(settings),
        "llm": _llm_block(settings),
        "embedder": {
            "provider": settings.embedder_provider,
            "config": {
                "model": settings.embedding_model,
                "embedding_dims": settings.embedding_dims,
            },
        },
        "history_db_path": str(settings.history_db_path),
        # v1.1 returns memory operations as a dict with a "results" key. v1.0 is
        # deprecated and emits a warning on every call.
        "version": "v1.1",
    }
