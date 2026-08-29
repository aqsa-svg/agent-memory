"""Tests for the memory layer's pure logic.

Deliberately offline: no API key, no network, no model download, no quota.
Everything here is either a pure function or uses a fake backend, so the suite
runs in CI and in under a second.

The headline test is test_remember_never_sends_system_prompt - it guards the
one bug this whole project is built around avoiding.
"""

from __future__ import annotations

import pytest

import memory_store
from memory_store import MemoryOp, _diff_to_ops, _is_pending, _rows


# ---------------------------------------------------------------------------
# Response normalisation - the shapes the two backends actually return
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "response, expected_texts",
    [
        # OSS Memory, v1.1
        ({"results": [{"memory": "a"}, {"memory": "b"}]}, ["a", "b"]),
        # MemoryClient v1 endpoints return a bare list
        ([{"memory": "a"}], ["a"]),
        # Some list endpoints use "memories"
        ({"memories": [{"memory": "a"}]}, ["a"]),
        # And some use "data"
        ({"data": [{"memory": "a"}]}, ["a"]),
        # A single memory object returned bare
        ({"memory": "a", "id": "1"}, ["a"]),
        # Empties must not explode
        (None, []),
        ({}, []),
        ([], []),
        ({"results": []}, []),
    ],
)
def test_rows_normalises_every_known_shape(response, expected_texts):
    assert [row.get("memory") for row in _rows(response)] == expected_texts


def test_rows_discards_non_dict_entries():
    # A malformed payload should degrade, not crash the chat loop.
    assert _rows([{"memory": "a"}, "junk", None, 42]) == [{"memory": "a"}]


# ---------------------------------------------------------------------------
# Asynchronous cloud writes
# ---------------------------------------------------------------------------


def test_is_pending_detects_queued_write():
    queued = [{"message": "queued", "status": "PENDING", "event_id": "abc"}]
    assert _is_pending(queued) is True


def test_is_pending_false_for_real_events():
    assert _is_pending([{"id": "1", "memory": "a", "event": "ADD"}]) is False


# ---------------------------------------------------------------------------
# Reconstructing operations by diffing snapshots (cloud backend)
# ---------------------------------------------------------------------------


def test_diff_detects_add():
    ops = _diff_to_ops({}, {"1": "Is a vegetarian"})
    assert [(o.event, o.text) for o in ops] == [("ADD", "Is a vegetarian")]


def test_diff_detects_update_and_keeps_previous():
    ops = _diff_to_ops({"1": "Backend engineer"}, {"1": "Staff engineer"})
    assert len(ops) == 1
    assert ops[0].event == "UPDATE"
    assert ops[0].text == "Staff engineer"
    # Losing `previous` would gut the debug output, so assert on it explicitly.
    assert ops[0].previous == "Backend engineer"


def test_diff_detects_delete():
    ops = _diff_to_ops({"1": "Lives in Bangalore"}, {})
    assert [(o.event, o.text) for o in ops] == [("DELETE", "Lives in Bangalore")]


def test_diff_reports_nothing_when_unchanged():
    snapshot = {"1": "Is a vegetarian"}
    assert _diff_to_ops(snapshot, snapshot) == []


def test_diff_handles_replacement_as_delete_plus_add():
    # mem0 reconciles a moved-city contradiction by dropping the old row and
    # inserting a new one, which must surface as two distinct operations.
    ops = _diff_to_ops({"1": "Lives in Bangalore"}, {"2": "Moved to Berlin"})
    assert {(o.event, o.text) for o in ops} == {
        ("DELETE", "Lives in Bangalore"),
        ("ADD", "Moved to Berlin"),
    }


# ---------------------------------------------------------------------------
# The bug this project exists to avoid
# ---------------------------------------------------------------------------


class _FakeBackend:
    """Records what remember() hands to the backend."""

    label = "fake"
    async_writes = False

    def __init__(self) -> None:
        self.received: list[dict[str, str]] = []

    def add(self, messages, user_id):
        self.received = messages
        return {"results": [{"id": "1", "memory": "x", "event": "ADD"}]}

    def close(self) -> None:
        pass


@pytest.fixture
def fake_backend(monkeypatch):
    backend = _FakeBackend()
    monkeypatch.setattr(memory_store, "get_backend", lambda: backend)
    return backend


def test_remember_never_sends_system_prompt(fake_backend):
    """The whole point. Only user and assistant roles may reach mem0.

    If a system prompt ever leaks in, the memories it contains get re-extracted
    and stored as duplicates on every single turn.
    """
    memory_store.remember("I am vegetarian", "Noted!", user_id="u1")

    roles = [m["role"] for m in fake_backend.received]
    assert roles == ["user", "assistant"]
    assert "system" not in roles


def test_remember_passes_content_through_unchanged(fake_backend):
    memory_store.remember("I am vegetarian", "Noted!", user_id="u1")
    assert fake_backend.received == [
        {"role": "user", "content": "I am vegetarian"},
        {"role": "assistant", "content": "Noted!"},
    ]


def test_remember_maps_events_to_ops(fake_backend):
    ops = memory_store.remember("hi", "hello", user_id="u1")
    assert ops == [MemoryOp(event="ADD", text="x", previous=None, id="1")]


def test_recall_short_circuits_on_empty_query(monkeypatch):
    # Must not cost an embedding call or reach the backend at all.
    monkeypatch.setattr(
        memory_store,
        "get_backend",
        lambda: pytest.fail("recall() hit the backend for an empty query"),
    )
    assert memory_store.recall("   ", user_id="u1") == []
