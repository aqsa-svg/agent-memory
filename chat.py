"""The chat loop.

    python chat.py                 # asks who you are
    python chat.py --user saif     # skips the prompt
    python chat.py --quiet         # start with debug output off

What happens on every turn:

    1. embed your message and semantically search stored memories
    2. inject the top matches into the system prompt
    3. call the LLM for a reply
    4. hand the exchange to the memory layer, which extracts durable facts and
       decides ADD / UPDATE / DELETE / NONE against what is already stored

Note that this file never imports mem0. It only talks to memory_store, so the
whole memory layer could be swapped for an HTTP call without touching the loop.
"""

from __future__ import annotations

import argparse
import sys

from config import ConfigError, get_settings
import llm
import memory_store
from memory_store import (
    MemoryOp,
    MemoryStoreError,
    QuotaExhausted,
    Recalled,
)

# ---------------------------------------------------------------------------
# Terminal styling
# ---------------------------------------------------------------------------

GREY = "\033[90m"  # debug internals - deliberately low contrast
BOLD = "\033[1m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
RESET = "\033[0m"

# Event -> colour. Seeing UPDATE and DELETE stand out is the point of the
# debug output, so they get warm colours and ADD stays quiet.
EVENT_COLOURS = {
    "ADD": GREEN,
    "UPDATE": YELLOW,
    "DELETE": RED,
    "NONE": GREY,
    "PENDING": GREY,
}


def _enable_ansi() -> None:
    """Prepare the console for ANSI colour and non-ASCII text.

    Windows consoles need a nudge before they honour ANSI escapes, and their
    default code page is not UTF-8 - without reconfiguring, an em dash from
    the model renders as a replacement character mid-sentence.
    """
    try:
        import colorama

        colorama.just_fix_windows_console()
    except Exception:  # noqa: BLE001 - worst case, colours show as raw codes
        pass

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001 - not all streams support this
            pass


class Console:
    """All printing goes through here so /quiet has a single switch."""

    def __init__(self, debug: bool) -> None:
        self.debug = debug

    def dim(self, text: str) -> None:
        """Debug internals. Suppressed by /quiet."""
        if self.debug:
            print(f"{GREY}{text}{RESET}")

    def note(self, text: str) -> None:
        """Low-priority messages that stay visible even in quiet mode."""
        print(f"{GREY}{text}{RESET}")

    def show_recall(self, memories: list[Recalled]) -> None:
        if not self.debug:
            return
        if not memories:
            self.dim("  recall: nothing relevant stored yet")
            return
        self.dim(f"  recall: {len(memories)} memory(ies) injected into the prompt")
        for m in memories:
            score = f"{m.score:.3f}" if m.score is not None else "  -  "
            self.dim(f"    [{score}] {m.text}")

    def show_ops(self, ops: list[MemoryOp]) -> None:
        if not self.debug:
            return
        if not ops:
            self.dim("  memory: NONE (nothing durable in that exchange)")
            return
        for op in ops:
            colour = EVENT_COLOURS.get(op.event, GREY)
            if op.event == "UPDATE" and op.previous:
                # The headline case: show what it replaced, so a contradiction
                # is visibly reconciled rather than silently duplicated.
                self.dim(f"  memory: {colour}UPDATE{RESET}{GREY}")
                self.dim(f"            was: {op.previous}")
                self.dim(f"            now: {op.text}")
            else:
                self.dim(f"  memory: {colour}{op.event}{RESET}{GREY} {op.text}")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

HELP = f"""
{BOLD}Commands{RESET}
  /memories   list everything stored about you
  /forget     delete all your memories (asks first)
  /quiet      toggle the dim grey debug output
  /whoami     show the active user and memory backend
  /help       this list
  /exit       quit  (Ctrl-C also works)
""".rstrip()


def cmd_memories(console: Console, user_id: str) -> None:
    memories = memory_store.all_memories(user_id)
    if not memories:
        print(f"{GREY}Nothing stored for {user_id} yet.{RESET}")
        return
    print(f"\n{BOLD}{len(memories)} memory(ies) for {user_id}{RESET}")
    for i, m in enumerate(memories, 1):
        when = f"  {GREY}{m.created_at[:10]}{RESET}" if m.created_at else ""
        print(f"  {i:>2}. {m.text}{when}")
    print()


def cmd_forget(console: Console, user_id: str) -> None:
    count = len(memory_store.all_memories(user_id))
    if not count:
        print(f"{GREY}Nothing stored for {user_id} - nothing to forget.{RESET}")
        return
    # Destructive and irreversible, so require an explicit yes rather than
    # treating a stray Enter as consent.
    answer = input(
        f"{YELLOW}Delete all {count} memory(ies) for {user_id}? "
        f"This cannot be undone. [y/N] {RESET}"
    ).strip().lower()
    if answer != "y":
        print(f"{GREY}Cancelled - nothing was deleted.{RESET}")
        return
    removed = memory_store.forget_all(user_id)
    print(f"{GREEN}Deleted {removed} memory(ies).{RESET}")


def handle_command(raw: str, console: Console, user_id: str) -> bool:
    """Run a /command. Returns False when the loop should exit."""
    command = raw.strip().lower()

    if command in ("/exit", "/quit", "/q"):
        return False
    if command == "/help":
        print(HELP)
    elif command == "/memories":
        cmd_memories(console, user_id)
    elif command == "/forget":
        cmd_forget(console, user_id)
    elif command == "/quiet":
        console.debug = not console.debug
        state = "on" if console.debug else "off"
        print(f"{GREY}Debug output {state}.{RESET}")
    elif command == "/whoami":
        print(f"{GREY}user: {user_id}{RESET}")
        print(f"{GREY}memory: {memory_store.describe_backend()}{RESET}")
        settings = get_settings()
        print(f"{GREY}replies: {settings.backend} / {settings.llm_model}{RESET}")
    else:
        print(f"{GREY}Unknown command {command!r}. Try /help.{RESET}")
    return True


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------


def ask_user_id(supplied: str | None) -> str:
    """Identify the human. Memories are scoped to this for the whole session."""
    if supplied:
        return supplied.strip().lower()

    known = memory_store.known_users()
    if known:
        print(f"{GREY}Returning users: {', '.join(known)}{RESET}")

    while True:
        name = input("Who are you? ").strip().lower()
        if name and not name.startswith("/"):
            return name
        print(f"{GREY}Please enter a name - it scopes your memories.{RESET}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Chat agent with long-term memory.")
    parser.add_argument("--user", help="user id to scope memories by")
    parser.add_argument(
        "--quiet", action="store_true", help="start with debug output off"
    )
    args = parser.parse_args()

    _enable_ansi()
    settings = get_settings()
    console = Console(debug=settings.debug and not args.quiet)

    # Route retry/backoff messages from both layers into the console. These
    # stay visible even in quiet mode: a 30s wait with no explanation looks
    # like a hang.
    memory_store.set_notifier(console.note)
    llm.set_notifier(console.note)

    print(f"\n{BOLD}agent-memory{RESET}")

    # Touching the backend triggers config validation and, in oss mode, the
    # model load. Do it before asking for a name so failures surface early.
    print(f"{GREY}Starting memory backend...{RESET}")
    memory_store.warm_up()
    print(f"{GREY}memory:  {memory_store.describe_backend()}{RESET}")
    print(f"{GREY}replies: {settings.backend} / {settings.llm_model}{RESET}")

    user_id = ask_user_id(args.user)

    stored = len(memory_store.all_memories(user_id))
    if stored:
        print(f"\n{CYAN}Welcome back, {user_id}.{RESET} "
              f"{GREY}I remember {stored} thing(s) about you.{RESET}")
    else:
        print(f"\n{CYAN}Hello, {user_id}.{RESET} "
              f"{GREY}Nothing stored about you yet.{RESET}")
    print(f"{GREY}Type /help for commands, /exit to quit.{RESET}\n")

    # Short rolling window of the live conversation. Memory handles anything
    # older, which is the entire point - we are not stuffing the transcript in.
    history: list[dict[str, str]] = []

    while True:
        try:
            user_message = input(f"{BOLD}you>{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_message:
            continue
        if user_message.startswith("/"):
            if not handle_command(user_message, console, user_id):
                break
            continue

        try:
            # 1. semantic search over what we already know
            memories = memory_store.recall(user_message, user_id=user_id)
            console.show_recall(memories)

            # 2. inject those memories into the system prompt
            system_prompt = llm.build_system_prompt(memories)

            # 3. ask the LLM. history holds user/assistant turns only; the
            #    system prompt is passed out-of-band and never joins it.
            history.append({"role": "user", "content": user_message})
            reply = llm.generate_reply(system_prompt, history, settings)
            history.append({"role": "assistant", "content": reply})
            # Keep the window small: memory is the long-term store, not this.
            history[:] = history[-8:]

            print(f"\n{CYAN}agent>{RESET} {reply}\n")

            # 4. write the exchange back. ONLY the two message strings - the
            #    system prompt must not go in, or the memories we just recalled
            #    would be re-extracted and stored again as duplicates.
            ops = memory_store.remember(user_message, reply, user_id=user_id)
            console.show_ops(ops)

        except QuotaExhausted as exc:
            print(f"\n{YELLOW}{exc}{RESET}\n")
        except (ConfigError, MemoryStoreError) as exc:
            print(f"\n{RED}{exc}{RESET}\n")
            return 1
        except KeyboardInterrupt:
            # Ctrl-C during a slow call cancels the turn, not the session.
            print(f"\n{GREY}(cancelled){RESET}\n")
            continue

    print(f"{GREY}Bye. Your memories are saved - run me again and I'll "
          f"still know you.{RESET}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ConfigError, MemoryStoreError) as exc:
        # Startup failures: print the guidance, not a traceback.
        print(f"\n{RED}Setup problem{RESET}\n\n{exc}\n", file=sys.stderr)
        raise SystemExit(1) from None
