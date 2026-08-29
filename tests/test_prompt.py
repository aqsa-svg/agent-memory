"""Tests for system-prompt construction.

Offline - build_system_prompt is a pure function.
"""

from __future__ import annotations

import llm
from memory_store import Recalled


def test_memories_are_injected_as_a_list():
    prompt = llm.build_system_prompt(
        [
            Recalled(id="1", text="Is a vegetarian"),
            Recalled(id="2", text="Learning guitar"),
        ]
    )
    assert "- Is a vegetarian" in prompt
    assert "- Learning guitar" in prompt


def test_empty_memories_produce_a_new_user_marker():
    prompt = llm.build_system_prompt([])
    assert llm.NO_MEMORIES in prompt


def test_prompt_tells_the_model_to_trust_the_present_over_memory():
    # Without this instruction the model argues with users who have just
    # corrected themselves, which is the opposite of useful.
    prompt = llm.build_system_prompt([Recalled(id="1", text="Lives in Bangalore")])
    assert "trust what they said now" in prompt


def test_prompt_discourages_reciting_memories_back():
    prompt = llm.build_system_prompt([Recalled(id="1", text="Is a vegetarian")])
    assert "do not recite them back" in prompt
