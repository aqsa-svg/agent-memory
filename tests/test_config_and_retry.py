"""Tests for configuration resolution and rate-limit handling.

Offline. `with_retry` tests monkeypatch time.sleep so exponential backoff is
exercised without actually waiting.
"""

from __future__ import annotations

import pytest

import config
import retry
from config import ConfigError, Settings
from retry import QuotaExhausted, RetryableError, with_retry


@pytest.fixture(autouse=True)
def clear_settings_cache():
    """get_settings is lru_cached, so each test needs a clean read."""
    config.get_settings.cache_clear()
    yield
    config.get_settings.cache_clear()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_blank_env_var_falls_back_to_default(monkeypatch):
    # A bare `KEY=` line in .env is the classic way to get a silently empty
    # value, and must behave as "not set" rather than as an empty model name.
    monkeypatch.setenv("GEMINI_MODEL", "   ")
    assert config.get_settings().gemini_model == "gemini-3.5-flash-lite"


def test_gemini_key_accepts_either_variable_name(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "from-google-var")
    assert config.get_settings().gemini_api_key == "from-google-var"


def test_invalid_backend_is_rejected(monkeypatch):
    monkeypatch.setenv("BACKEND", "openai")
    with pytest.raises(ConfigError, match="BACKEND must be"):
        config.get_settings()


def test_invalid_memory_backend_is_rejected(monkeypatch):
    monkeypatch.setenv("MEMORY_BACKEND", "postgres")
    with pytest.raises(ConfigError, match="MEMORY_BACKEND must be"):
        config.get_settings()


def test_relative_paths_anchor_to_project_root(monkeypatch):
    # Otherwise launching from another directory silently starts a second,
    # empty memory store.
    monkeypatch.setenv("QDRANT_PATH", "./qdrant_data")
    resolved = config.get_settings().qdrant_path
    assert resolved.is_absolute()
    assert resolved.parent == config.PROJECT_ROOT


def test_missing_gemini_key_gives_actionable_message():
    settings = Settings(backend="gemini", gemini_api_key="", memory_backend="oss")
    with pytest.raises(ConfigError) as exc:
        config.validate(settings)
    assert "aistudio.google.com/apikey" in str(exc.value)
    assert "no credit card" in str(exc.value)


def test_missing_mem0_key_gives_actionable_message():
    settings = Settings(
        backend="ollama", memory_backend="cloud", mem0_api_key=""
    )
    with pytest.raises(ConfigError) as exc:
        config.validate(settings)
    assert "app.mem0.ai" in str(exc.value)
    # It must also offer the no-account escape hatch.
    assert "MEMORY_BACKEND=oss" in str(exc.value)


def test_embedding_dim_mismatch_is_caught_before_qdrant_fails():
    settings = Settings(
        backend="ollama",
        memory_backend="oss",
        embedding_dims=1536,  # the OpenAI default people paste in by accident
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    )
    with pytest.raises(ConfigError, match="384"):
        config.validate(settings)


def test_cloud_mode_skips_local_store_validation():
    # Nonsense dims are irrelevant when there is no local vector store.
    settings = Settings(
        backend="ollama",
        memory_backend="cloud",
        mem0_api_key="m0-x",
        embedding_dims=1536,
    )
    config.validate(settings)  # must not raise


def test_mem0_config_uses_384_dims_and_persists_to_disk():
    cfg = config.mem0_config(Settings(memory_backend="oss", backend="ollama"))
    store = cfg["vector_store"]["config"]
    assert store["embedding_model_dims"] == 384
    # on_disk=False makes mem0 shutil.rmtree the folder on every startup.
    assert store["on_disk"] is True


def test_qdrant_host_switches_from_path_to_server():
    cfg = config.mem0_config(
        Settings(memory_backend="oss", backend="ollama", qdrant_host="localhost")
    )
    store = cfg["vector_store"]["config"]
    assert store["host"] == "localhost"
    assert "path" not in store


# ---------------------------------------------------------------------------
# Rate limits
# ---------------------------------------------------------------------------

PER_MINUTE = Exception(
    "429 RESOURCE_EXHAUSTED. quotaId: "
    "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"
)
PER_DAY = Exception(
    "429 RESOURCE_EXHAUSTED. quotaId: "
    "GenerateRequestsPerDayPerProjectPerModel-FreeTier, quotaValue: 20"
)


def test_rate_limit_detection():
    assert retry.is_rate_limit(PER_MINUTE)
    assert retry.is_rate_limit(PER_DAY)
    assert not retry.is_rate_limit(ValueError("bad json"))


def test_daily_quota_is_distinguished_from_per_minute():
    # Retrying a per-day quota is pointless; retrying per-minute works.
    assert retry.is_daily_quota(PER_DAY)
    assert not retry.is_daily_quota(PER_MINUTE)


def test_daily_quota_raises_immediately_without_retrying(monkeypatch):
    monkeypatch.setattr(retry.time, "sleep", lambda _: None)
    calls = []

    def boom():
        calls.append(1)
        raise PER_DAY

    with pytest.raises(QuotaExhausted, match="daily free-tier quota"):
        with_retry(boom, what="test", max_retries=5)
    assert len(calls) == 1  # no wasted attempts


def test_per_minute_limit_is_retried_then_succeeds(monkeypatch):
    slept: list[float] = []
    monkeypatch.setattr(retry.time, "sleep", slept.append)
    attempts = []

    def flaky():
        attempts.append(1)
        if len(attempts) < 3:
            raise PER_MINUTE
        return "ok"

    assert with_retry(flaky, what="test", max_retries=5, base_delay=2.0) == "ok"
    assert len(attempts) == 3
    # Exponential: roughly 2s then 4s, plus a little jitter.
    assert len(slept) == 2
    assert 2.0 <= slept[0] < 2.6
    assert 4.0 <= slept[1] < 4.6


def test_server_supplied_retry_delay_wins_over_our_backoff(monkeypatch):
    slept: list[float] = []
    monkeypatch.setattr(retry.time, "sleep", slept.append)
    attempts = []

    def flaky():
        attempts.append(1)
        if len(attempts) < 2:
            raise Exception("429 RESOURCE_EXHAUSTED 'retryDelay': '13s'")
        return "ok"

    with_retry(flaky, what="test", max_retries=3, base_delay=2.0)
    assert 13.0 <= slept[0] < 13.6  # honoured 13s, not our 2s


def test_non_retryable_error_surfaces_immediately(monkeypatch):
    monkeypatch.setattr(retry.time, "sleep", lambda _: None)
    attempts = []

    def broken():
        attempts.append(1)
        raise KeyError("a real bug")

    with pytest.raises(KeyError):
        with_retry(broken, what="test", max_retries=5)
    assert len(attempts) == 1  # must not mask genuine bugs behind retries


def test_retryable_error_is_retried(monkeypatch):
    """RetryableError is how we re-raise errors mem0 swallowed internally."""
    monkeypatch.setattr(retry.time, "sleep", lambda _: None)
    attempts = []

    def swallowed():
        attempts.append(1)
        if len(attempts) < 2:
            raise RetryableError("Error in new memory actions response: 429")
        return "ok"

    assert with_retry(swallowed, what="test", max_retries=3) == "ok"
    assert len(attempts) == 2


def test_on_retry_callback_is_told_what_is_happening(monkeypatch):
    monkeypatch.setattr(retry.time, "sleep", lambda _: None)
    seen = []
    attempts = []

    def flaky():
        attempts.append(1)
        if len(attempts) < 2:
            raise PER_MINUTE
        return "ok"

    with_retry(
        flaky,
        what="memory extraction",
        max_retries=3,
        on_retry=lambda what, delay, n, total: seen.append((what, n, total)),
    )
    assert seen == [("memory extraction", 1, 3)]
