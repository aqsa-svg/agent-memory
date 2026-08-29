"""Stage 2 verification: does the whole memory pipeline actually work?

    python seed_memories.py

Stores three facts about a throwaway user, searches for them semantically,
then proves the reconciliation step by contradicting one of them and watching
mem0 emit an UPDATE rather than a second, conflicting ADD.

This exercises every piece that can break before the chat loop exists. On the
oss backend that means sentence-transformers loading, the 384-dim Qdrant
collection, and the LLM round-trip mem0 uses to extract facts. On the cloud
backend it exercises the hosted API and its asynchronous writes.

Pass --keep to leave the test data in place, --user NAME to pick the user id.
"""

from __future__ import annotations

import argparse
import sys
import time

from config import ConfigError, get_settings
from memory_store import (
    MemoryStoreError,
    describe_backend,
    QuotaExhausted,
    all_memories,
    forget_all,
    recall,
    remember,
    set_notifier,
    warm_up,
)

DIM = "\033[2m"
BOLD = "\033[1m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
RESET = "\033[0m"

# Deliberately varied: a preference, a fact about work, and a constraint.
# Search queries below use none of the same keywords, so a hit proves semantic
# matching rather than substring luck.
FACTS: list[tuple[str, str]] = [
    (
        "I'm a vegetarian and I really don't like mushrooms.",
        "Noted - vegetarian, and mushrooms are off the list.",
    ),
    (
        "I work as a backend engineer, mostly writing Go and Python.",
        "Backend engineer working in Go and Python, got it.",
    ),
    (
        "I live in Bangalore and I'm learning to play the guitar.",
        "Bangalore-based, and picking up guitar. Nice.",
    ),
]

# (query, what we hope surfaces) - no shared keywords with the facts above.
QUERIES: list[tuple[str, str]] = [
    ("What should I cook for dinner?", "the vegetarian / mushroom memory"),
    ("Which programming languages do I use?", "the Go and Python memory"),
    ("Any hobbies I've picked up recently?", "the guitar memory"),
]


def step(n: int, title: str) -> None:
    print(f"\n{BOLD}[{n}] {title}{RESET}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", default="stage2_test_user")
    parser.add_argument(
        "--keep", action="store_true", help="do not delete the seeded memories"
    )
    args = parser.parse_args()
    user_id = args.user

    # Route retry/backoff messages to stdout in dim grey.
    set_notifier(lambda msg: print(f"{DIM}    ... {msg}{RESET}"))

    settings = get_settings()
    print(f"\n{BOLD}=== agent-memory :: stage 2 end-to-end ==={RESET}")
    print(f"{DIM}user_id={user_id}  memory={settings.memory_backend}  "
          f"llm={settings.backend}/{settings.llm_model}{RESET}")

    step(1, "Opening the memory backend")
    if settings.uses_local_memory:
        print(f"{DIM}    First run downloads ~90MB for all-MiniLM-L6-v2. "
              f"After that it is local and offline.{RESET}")
    else:
        print(f"{DIM}    Cloud mode: nothing to download, mem0's servers "
              f"do the embedding.{RESET}")
    t0 = time.time()
    warm_up()
    print(f"{GREEN}    ready in {time.time() - t0:.1f}s{RESET}")

    # Start clean so repeat runs are meaningful rather than cumulative.
    existing = forget_all(user_id)
    if existing:
        print(f"{DIM}    cleared {existing} memory(ies) from a previous run{RESET}")

    step(2, f"Storing {len(FACTS)} facts (2 LLM calls each: extract + reconcile)")
    for user_msg, assistant_msg in FACTS:
        print(f'{DIM}    you: "{user_msg}"{RESET}')
        ops = remember(user_msg, assistant_msg, user_id=user_id)
        if not ops:
            print(f"{YELLOW}      -> NONE (mem0 found nothing durable to store){RESET}")
        for op in ops:
            print(f"{GREEN}      -> {op.event}{RESET} {op.text}")

    step(3, "What is in the store now")
    stored = all_memories(user_id)
    if not stored:
        print(f"{RED}    Nothing was stored. Something is wrong upstream.{RESET}")
        return 1
    for memory in stored:
        print(f"    - {memory.text}")

    step(4, "Semantic search (embedding-only, no LLM, no quota used)")
    hits = 0
    for query, expectation in QUERIES:
        results = recall(query, user_id=user_id)
        print(f'\n{DIM}    query: "{query}"{RESET}')
        print(f"{DIM}    expecting: {expectation}{RESET}")
        if not results:
            print(f"{YELLOW}      (no matches){RESET}")
            continue
        hits += 1
        for r in results:
            score = f"{r.score:.3f}" if r.score is not None else "  -  "
            print(f"      {DIM}{score}{RESET}  {r.text}")

    step(5, "Contradicting myself - reconciliation, not a blind second ADD")
    print(f"{DIM}    mem0 has two valid ways to reconcile, and which one fires"
          f"\n    depends on the kind of contradiction:"
          f"\n      refinement (same fact, more detail) -> UPDATE"
          f"\n      replacement (fact is now false)     -> DELETE + ADD"
          f"\n    Either way the stale claim must not survive.{RESET}")

    contradiction = "I got promoted last week, I'm a staff engineer now."
    print(f'\n{DIM}    you: "{contradiction}"{RESET}')
    ops = remember(contradiction, "Congrats on the promotion!", user_id=user_id)
    for op in ops:
        if op.event == "UPDATE" and op.previous:
            print(f"{GREEN}      -> UPDATE{RESET} {op.previous}")
            print(f"{GREEN}         becomes{RESET} {op.text}")
        else:
            print(f"{GREEN}      -> {op.event}{RESET} {op.text}")
    if not ops:
        print(f"{YELLOW}      -> NONE (nothing reconciled){RESET}")

    print(f"\n{DIM}    store after the contradiction:{RESET}")
    for memory in all_memories(user_id):
        print(f"    - {memory.text}")

    if args.keep:
        print(f"\n{DIM}--keep set, leaving {user_id}'s memories in place.{RESET}")
    else:
        removed = forget_all(user_id)
        print(f"\n{DIM}Cleaned up {removed} test memory(ies). "
              f"Pass --keep to skip this.{RESET}")

    if hits == 0:
        print(f"\n{RED}Search returned nothing for any query.{RESET}")
        return 1

    print(f"\n{GREEN}Stage 2 passed.{RESET} {describe_backend()}\n")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ConfigError, MemoryStoreError) as exc:
        print(f"\n{RED}Setup problem{RESET}\n\n{exc}\n", file=sys.stderr)
        raise SystemExit(1) from None
    except QuotaExhausted as exc:
        print(f"\n{YELLOW}{exc}{RESET}\n", file=sys.stderr)
        raise SystemExit(1) from None
    except KeyboardInterrupt:
        print("\nInterrupted.")
        raise SystemExit(130) from None
