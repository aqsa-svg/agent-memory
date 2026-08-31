"""FastAPI backend for the web front end.

    python server.py          # then open http://127.0.0.1:8000
    uvicorn server:app        # equivalent

This is a thin HTTP wrapper. It contains no turn logic of its own - it calls
agent.run_turn(), exactly as chat.py does - and it never imports mem0. That is
the whole point of the layering: the memory implementation can change without
either front end noticing.

Deliberately single-process. Qdrant local mode holds an exclusive file lock on
./qdrant_data, so running this alongside chat.py fails with "already accessed
by another instance". Run `docker compose up -d` and set QDRANT_HOST=localhost
if you want both at once.
"""

from __future__ import annotations

import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

import agent
import demo_guard
import memory_store
from config import ConfigError, get_settings
from demo_guard import DEMO_MODE, guard
from memory_store import MemoryStoreError, QuotaExhausted

HERE = Path(__file__).resolve().parent

# Per-user rolling conversation window, mirroring the CLI's `history` list.
# In-process and therefore lost on restart - which is fine, and rather the
# point: durable knowledge belongs in the memory layer, not in a transcript.
_histories: dict[str, list[dict[str, str]]] = {}

# The memory backend is not built for concurrent use: local Qdrant is
# single-writer, and mem0's client is not documented as thread-safe. Serialising
# turns costs nothing here (they are network-bound and one human is typing) and
# removes a whole class of heisenbug.
_turn_lock = threading.Lock()


class ChatRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=64)
    message: str = Field(min_length=1, max_length=8000)


class ForgetRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=64)


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Build the backend now so the first message is not paying for a ~90MB
    # model load while the user watches a spinner.
    try:
        memory_store.warm_up()
    except (ConfigError, MemoryStoreError) as exc:
        # Do not crash the server: the UI renders the problem far more
        # legibly than a traceback in a terminal the user may not be watching.
        print(f"\nStartup problem:\n\n{exc}\n")
    yield
    memory_store.close()


app = FastAPI(title="agent-memory", lifespan=lifespan)


def _clean_user(raw: str) -> str:
    """Resolve the memory scope for a request.

    In demo mode only server-issued guest tokens are accepted. Anything else -
    including a blank field or somebody typing another visitor's token format
    incorrectly - is rejected, so no visitor can address another's memories.
    """
    user = raw.strip().lower()[:64]
    if DEMO_MODE and not demo_guard.is_valid_visitor_id(user):
        raise HTTPException(
            status_code=400,
            detail="Reload the page to get a demo session.",
        )
    return user


@app.get("/api/status")
def status() -> dict[str, Any]:
    """What the header's status dot and label report."""
    settings = get_settings()
    try:
        payload: dict[str, Any] = {
            "ok": True,
            "memory_backend": settings.memory_backend,
            "memory": memory_store.describe_backend(),
            "llm": f"{settings.backend} / {settings.llm_model}",
        }
    except (ConfigError, MemoryStoreError) as exc:
        return {"ok": False, "error": str(exc)}

    payload["demo_mode"] = DEMO_MODE
    if DEMO_MODE:
        # In demo mode the browser must not choose its own user_id, so the
        # server issues one and the UI hides the field entirely.
        payload["visitor_id"] = demo_guard.new_visitor_id()
        payload["limits"] = guard.stats()
    return payload


@app.get("/api/memories")
def memories(user_id: str) -> dict[str, Any]:
    user_id = _clean_user(user_id)
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required")
    try:
        stored = memory_store.all_memories(user_id)
    except (ConfigError, MemoryStoreError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {
        "memories": [
            {"id": m.id, "text": m.text, "created_at": m.created_at} for m in stored
        ]
    }


@app.post("/api/chat")
def chat(request: ChatRequest) -> dict[str, Any]:
    user_id = _clean_user(request.user_id)
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required")

    allowed, why = guard.check(user_id)
    if not allowed:
        # 429 so the UI renders this in the same calm style as a quota message.
        return JSONResponse(status_code=429, content={"detail": why})

    history = _histories.setdefault(user_id, [])

    try:
        with _turn_lock:
            turn = agent.run_turn(
                request.message.strip(), user_id=user_id, history=history
            )
    except QuotaExhausted as exc:
        # Expected on a free tier, and not a server fault. 429 lets the UI say
        # something calm rather than rendering a generic failure. The visitor
        # did nothing wrong, so give their allowance back.
        guard.refund(user_id)
        return JSONResponse(status_code=429, content={"detail": str(exc)})
    except (ConfigError, MemoryStoreError) as exc:
        guard.refund(user_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "reply": turn.reply,
        "summary": turn.summary(),
        "pending": turn.is_pending,
        "recalled": [
            {"text": m.text, "score": m.score} for m in turn.recalled
        ],
        "ops": [
            {"event": op.event, "text": op.text, "previous": op.previous}
            for op in turn.ops
        ],
    }


@app.post("/api/forget")
def forget(request: ForgetRequest) -> dict[str, Any]:
    user_id = _clean_user(request.user_id)
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required")
    try:
        removed = memory_store.forget_all(user_id)
    except (ConfigError, MemoryStoreError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    _histories.pop(user_id, None)
    return {"deleted": removed}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(HERE / "index.html")


if __name__ == "__main__":
    import uvicorn

    import os

    # Hosting platforms hand the port in as PORT and expect us to listen on all
    # interfaces. Absent that, stay on loopback so a local run does not quietly
    # expose itself to the rest of the network.
    port = int(os.getenv("PORT", "8000"))
    host = "0.0.0.0" if os.getenv("PORT") else "127.0.0.1"
    if host == "127.0.0.1":
        print(f"\n  agent-memory web ui  ->  http://127.0.0.1:{port}\n")
    # reload=False on purpose: the reloader would spawn a second process and
    # both would fight over the Qdrant file lock.
    uvicorn.run(app, host=host, port=port, log_level="warning")
