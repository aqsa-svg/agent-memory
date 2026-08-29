"""Stage 1 verification: does the configuration load and does the key work?

Run this before anything else:

    python check_setup.py

It prints the fully-resolved configuration, then makes one tiny live call to
the selected backend so you find out now - not three modules deep - whether the
key is valid and the model name exists on the free tier.
"""

from __future__ import annotations

import sys

from config import ConfigError, get_settings, mem0_config, validate

GREEN = "\033[32m"
RED = "\033[31m"
DIM = "\033[2m"
RESET = "\033[0m"


def ok(msg: str) -> None:
    print(f"{GREEN}  OK{RESET}  {msg}")


def fail(msg: str) -> None:
    print(f"{RED}FAIL{RESET}  {msg}")


def _mask(secret: str) -> str:
    if not secret:
        return "(not set)"
    return f"{secret[:6]}...{secret[-4:]} ({len(secret)} chars)"


def check_gemini(model: str, api_key: str) -> bool:
    """One real request. A key that merely *parses* proves nothing."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    try:
        resp = client.models.generate_content(
            model=model,
            contents="Reply with the single word: pong",
            config=types.GenerateContentConfig(
                temperature=0.0,
                # 2.5-series models think before answering, and thinking tokens
                # count against max_output_tokens. Too small a budget returns an
                # empty candidate, which looks like a broken key but is not.
                max_output_tokens=512,
            ),
        )
    except Exception as exc:  # noqa: BLE001 - we want the message, not the class
        detail = str(exc)
        fail(f"Gemini call failed: {detail.splitlines()[0][:200]}")
        if "API_KEY_INVALID" in detail or "API key not valid" in detail:
            print("      The key was rejected. Generate a fresh one at")
            print("      https://aistudio.google.com/apikey")
        elif "404" in detail or "not found" in detail.lower():
            print(f"      Model {model!r} is not available to this key.")
            print("      Run this script with --list-models to see valid names.")
        elif "429" in detail:
            print("      Rate limited. The key works; you have simply run out of")
            print("      free quota for now. Try again in a minute.")
        return False

    text = (getattr(resp, "text", "") or "").strip()
    if not text:
        fail("Gemini returned an empty response (model reachable, no text).")
        return False
    ok(f"Gemini responded: {text[:40]!r}")
    return True


def check_ollama(model: str, base_url: str) -> bool:
    import requests

    try:
        tags = requests.get(f"{base_url}/api/tags", timeout=5).json()
    except Exception as exc:  # noqa: BLE001
        fail(f"Cannot reach Ollama at {base_url}: {exc}")
        print("      Install from https://ollama.com, then `ollama serve`.")
        return False

    installed = [m["name"] for m in tags.get("models", [])]
    ok(f"Ollama reachable, {len(installed)} model(s) installed")
    # Ollama tags carry an explicit :tag; accept a bare name as a prefix match.
    if not any(name == model or name.startswith(f"{model}:") for name in installed):
        fail(f"Model {model!r} not pulled. Run: ollama pull {model}")
        print(f"      Installed: {', '.join(installed) or '(none)'}")
        return False
    ok(f"Model {model!r} is present")
    return True


def list_models(api_key: str) -> None:
    from google import genai

    client = genai.Client(api_key=api_key)
    print("\nModels your key can call with generateContent:\n")
    for m in client.models.list():
        actions = getattr(m, "supported_actions", None) or []
        if "generateContent" in actions:
            print(f"  {(m.name or '').removeprefix('models/')}")


def main() -> int:
    print("\n=== agent-memory :: setup check ===\n")

    try:
        settings = get_settings()
    except ConfigError as exc:
        fail(str(exc))
        return 1
    ok("config.py loaded")

    if "--list-models" in sys.argv:
        if not settings.gemini_api_key:
            fail("Need a GEMINI_API_KEY in .env to list models.")
            return 1
        list_models(settings.gemini_api_key)
        return 0

    print(f"\n{DIM}Resolved configuration{RESET}")
    print(f"  memory backend   {settings.memory_backend}")
    if settings.memory_backend == "cloud":
        print(f"  mem0 key         {_mask(settings.mem0_api_key)}")
    print(f"  llm backend      {settings.backend}")
    print(f"  llm model        {settings.llm_model}")
    if settings.backend == "gemini":
        print(f"  gemini key       {_mask(settings.gemini_api_key)}")
    else:
        print(f"  ollama url       {settings.ollama_base_url}")
    if settings.uses_local_memory:
        print(f"  embedder         {settings.embedding_model}")
        print(f"  embedding dims   {settings.embedding_dims}")
        print(
            "  vector store     "
            + (
                f"qdrant local -> {settings.qdrant_path}"
                if settings.local_qdrant
                else f"qdrant server -> {settings.qdrant_host}:{settings.qdrant_port}"
            )
        )
        print(f"  history db       {settings.history_db_path}")
    else:
        print("  embedder         (none - mem0's servers embed for you)")
        print("  vector store     (none - mem0 hosted)")
    print(f"  recall limit     {settings.recall_limit}")
    print(f"  debug output     {settings.debug}\n")

    try:
        validate(settings)
    except ConfigError as exc:
        fail("Configuration is incomplete:\n")
        print(exc)
        return 1
    ok("configuration validated")

    if settings.uses_local_memory:
        # The dims in the mem0 config are what Qdrant actually creates the
        # collection with, so assert on that dict rather than on Settings.
        cfg = mem0_config(settings)
        dims = cfg["vector_store"]["config"]["embedding_model_dims"]
        if dims != 384:
            fail(f"mem0 config would create a {dims}-dim collection, expected 384")
            return 1
        ok("mem0 config builds, vector store is 384-dim")
    else:
        ok("cloud memory backend - no local embedder or vector store needed")

    healthy = (
        check_gemini(settings.gemini_model, settings.gemini_api_key)
        if settings.backend == "gemini"
        else check_ollama(settings.ollama_model, settings.ollama_base_url)
    )
    if not healthy:
        return 1

    print(f"\n{GREEN}Stage 1 passed.{RESET} Next: python seed_memories.py\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
