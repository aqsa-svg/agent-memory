# agent-memory

A chat agent that remembers you between sessions. Tell it something, close the
terminal, come back days later, and it still knows.

It costs nothing to run. Gemini's free tier needs no credit card, Ollama runs
on your own machine, embeddings run on your CPU, and the vector store is a
folder on disk.

```
you> I'm a vegetarian and I'm learning guitar.
  memory: ADD User is a vegetarian and is learning to play the guitar

  ... quit, reboot, come back on Thursday ...

you> What should I cook for dinner tonight?
  recall: 1 memory(ies) injected into the prompt
    [0.139] User is a vegetarian and is learning to play the guitar

agent> How about a roasted butternut squash and sage risotto? Or a chickpea
       and spinach coconut curry, which comes together in about 20 minutes.
```

Nothing in that second session mentioned vegetarianism. It came from memory.

And when you contradict yourself, it doesn't accumulate both versions — it
reconciles:

```
you> I got promoted last week, I'm a staff engineer now.
  recall: 2 memory(ies) injected into the prompt
    [0.512] Works as a backend engineer
    [0.188] Mostly writes Go and Python

agent> Congratulations on the promotion!

  memory: UPDATE
            was: Works as a backend engineer
            now: Works as a staff engineer
```

One engineering memory afterwards, not two contradictory ones. That `UPDATE`
is the part most memory demos never show.

---

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows.  macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env              # then paste your key(s) in

python check_setup.py             # 1. does the config load and the key work?
python seed_memories.py           # 2. does the memory pipeline work end to end?
python chat.py                    # 3. talk to it
```

There are two front ends. The terminal one above, and a web UI:

```bash
python server.py                  # then open http://127.0.0.1:8000
```

Both call the same `agent.run_turn()`, so they cannot drift apart. Run one at a
time — see [Concurrency](#concurrency-and-when-you-need-docker) for why.

Requires Python 3.11+.

`check_setup.py` makes one real API call, so a typo in your key fails here with
a readable message instead of six modules deep. If the model name is rejected,
`python check_setup.py --list-models` prints exactly what your key can call.

---

## Choosing a memory backend

Set `MEMORY_BACKEND` in `.env`. This is the biggest decision in the project.

| | `cloud` (default) | `oss` |
|---|---|---|
| Where memories live | mem0's servers | `./qdrant_data` on this machine |
| Setup | an API key | a ~90MB one-time model download |
| Your Gemini quota | used for replies only | used for replies **and** memory |
| LLM calls per turn | ~1 | ~3 |
| Privacy | conversations sent to mem0 | nothing leaves the machine |
| Limits | free Hobby tier cap | none |
| Contradictions | **additive only — stale facts survive** | properly reconciled (UPDATE / DELETE) |

That last row is the one to read twice.

**Cloud** is simpler and much cheaper on your Gemini quota, because mem0 runs
extraction on their side. But its current behaviour is additive: tell it you
moved from Bangalore to Berlin and you end up holding *both* facts. Verified
on this project:

```
store: ['User moved to Berlin around July 2026, ending their residence in Bangalore',
        'User lives in Bangalore',          <-- stale, still there
        'User is a vegetarian']
```

**oss** reconciles properly, because this project pins `mem0ai==0.1.118`:

```
UPDATE | Works as a staff engineer | prev: Works as a backend engineer
DELETE | Lives in Bangalore
ADD    | Moved to Berlin last month
```

If watching memory *correct itself* is the point for you, use `MEMORY_BACKEND=oss`.
If you want the lowest-friction setup and mostly additive facts, cloud is fine.

Cloud writes are also **asynchronous** — `add()` returns `{"status": "PENDING"}`
and the memory materialises a moment later. `memory_store.py` handles this by
snapshotting before the write and diffing after, so the debug output looks the
same on both backends.

---

## What happens on each turn

Four steps, in `chat.py`:

```
                    ┌──────────────────────────────────────┐
   your message ───►│ 1. embed + semantic search           │  no LLM call
                    │    memory_store.recall()             │  costs nothing
                    └───────────────┬──────────────────────┘
                                    │ top 5 matches
                    ┌───────────────▼──────────────────────┐
                    │ 2. inject into the system prompt     │
                    │    llm.build_system_prompt()         │
                    └───────────────┬──────────────────────┘
                                    │
                    ┌───────────────▼──────────────────────┐
                    │ 3. call the LLM for a reply          │  1 LLM call
                    │    llm.generate_reply()              │
                    └───────────────┬──────────────────────┘
                                    │ user + assistant only
                    ┌───────────────▼──────────────────────┐
                    │ 4. extract facts, reconcile          │  2 LLM calls (oss)
                    │    memory_store.remember()           │  0 (cloud)
                    └──────────────────────────────────────┘
                         ADD / UPDATE / DELETE / NONE
```

**Step 1** embeds your message with `all-MiniLM-L6-v2` and cosine-matches it
against your stored memories. No LLM, so it is free and uses no quota. The
scores you see in the debug output are that cosine similarity.

**Step 2** renders the matches into the system prompt. Only the top matches,
not your whole history — that is the difference between memory and a transcript.

**Step 3** sends the system prompt plus a short rolling window of recent turns.

**Step 4** hands the exchange to mem0, which extracts durable facts and decides
what to do about each one against what it already knows.

### The bug most tutorials have

In step 4, pass **only the user and assistant messages**. Never the system prompt.

The system prompt contains the memories you just recalled in step 1. Feed it
back and mem0 re-extracts those same facts as though they were new, storing
near-duplicates of what it already has. Do that every turn and the store fills
with copies, recall quality degrades, and reconciliation can no longer tell
which duplicate to update.

This project makes the mistake impossible to make. `remember()` takes two
strings, not a message list:

```python
def remember(user_message: str, assistant_message: str, user_id: str) -> list[MemoryOp]:
```

There is no parameter a system prompt could go into.

---

## Watching it work

Debug output is on by default, in dim grey. `/quiet` toggles it.

```
you> I got promoted, I'm a staff engineer now.
  recall: 2 memory(ies) injected into the prompt
    [0.512] Works as a backend engineer
    [0.188] Mostly writes Go and Python

agent> Congratulations! ...

  memory: UPDATE
            was: Works as a backend engineer
            now: Works as a staff engineer
```

That `UPDATE` is the whole point. The agent did not accumulate a contradiction —
it replaced the stale fact.

Two shapes of reconciliation, both correct:

- **refinement** (same fact, more detail) → `UPDATE`
- **replacement** (old fact now false) → `DELETE` + `ADD`

### Commands

| | |
|---|---|
| `/memories` | list everything stored about you |
| `/forget` | delete it all (asks first) |
| `/quiet` | toggle the debug output |
| `/whoami` | show active user and backend |
| `/help` | command list |
| `/exit` | quit |

---

## A worked example

Paste these in, one per line. Quit between sessions to prove persistence is
real rather than in-process state.

**Session one** — `python chat.py --user demo`

```
I'm a vegetarian and I really don't like mushrooms.
I work as a backend engineer, mostly writing Go and Python.
/memories
/exit
```

**Session two** — start it again: `python chat.py --user demo`

```
What should I cook tonight?
```

It should suggest something vegetarian and mushroom-free without you
mentioning either. Watch `recall:` to see which memory fired and with what score.

```
I got promoted last week, I'm a staff engineer now.
```

On `oss` you should see `UPDATE`, with the old text shown above the new one.

```
/memories
```

There should be **one** engineering memory, not two contradictory ones.

---

## Rate limits and your real daily budget

Gemini's free tier is quota-limited, and the limit is **per model**:

```
quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier
```

The allowance varies enormously between models. Measured on a real free key
while building this, `gemini-2.5-flash` allows only **20 requests/day**.

On the `oss` backend one chat turn costs about **three** LLM calls — one reply,
plus two for mem0's extraction and reconciliation. So:

> **your daily message budget ≈ daily quota ÷ 3**

At 20/day that is roughly **six messages**, which is why this project does not
default to that model. Two ways to get more room:

1. **Use a lite model.** The default is `gemini-3.5-flash-lite`, which has far
   more headroom. Trade-off: it is weaker at reconciliation and sometimes
   returns `NONE` where `gemini-3.5-flash` correctly fires `UPDATE`. If you
   care most about seeing memory correct itself, set
   `GEMINI_MODEL=gemini-3.5-flash`.
2. **Switch models when one runs dry.** Quota is per model, so changing
   `GEMINI_MODEL` gives you a fresh allowance. `check_setup.py --list-models`
   shows the options.

On `cloud`, memory costs you no Gemini quota at all — roughly one call per
turn — but the mem0 free tier has its own cap.

When a 429 arrives, `retry.py` backs off exponentially and honours the server's
own `retryDelay`. It distinguishes the two cases:

- **per-minute** → waits and retries, printing what it is doing
- **per-day** → stops immediately with a readable message, because waiting
  cannot help

```
memory extraction: daily free-tier quota exhausted. It resets on Google's
clock (midnight Pacific). Switch to BACKEND=ollama in your .env to keep
working offline in the meantime.
```

No stack trace.

---

## Running fully offline

```bash
# https://ollama.com
ollama pull llama3.1:8b
```

```ini
BACKEND=ollama
MEMORY_BACKEND=oss
```

No key, no network, no quota. Embeddings were already local, so only the LLM
changes. Note embeddings stay on `all-MiniLM-L6-v2` regardless of backend, so
switching between Gemini and Ollama never changes your vector dimensions and
your existing memories stay readable.

---

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

```
44 passed in 0.20s
```

The suite is entirely offline — no API key, no network, no model download, no
quota. It covers the logic that is easy to break and expensive to debug:

- **`remember()` never sends the system prompt.** The single most important
  test in the project. A fake backend records exactly what reaches mem0 and
  asserts the roles are `["user", "assistant"]` and nothing else.
- **Response normalisation** across all six shapes the two backends return
  (`{"results": …}`, a bare list, `{"memories": …}`, `{"data": …}`, a lone
  object, and empties).
- **Snapshot diffing** for cloud's asynchronous writes — that ADD, UPDATE,
  DELETE and replacement-as-DELETE+ADD are each reconstructed correctly.
- **Rate-limit classification** — that a per-day quota raises immediately
  instead of burning five pointless retries, that per-minute backs off
  exponentially, and that the server's own `retryDelay` overrides our guess.
- **Config guards** — that a 1536-dim setting is rejected before Qdrant fails
  on every insert, and that `on_disk` is `True` so mem0 cannot `rmtree` your
  memories on startup.

---

## Layout

```
config.py          every provider, model and path. The only file reading env vars.
memory_store.py    the only file that imports mem0. recall/remember/all_memories/forget_all
agent.py           one turn of conversation, shared by both front ends
llm.py             the reply model (gemini or ollama)
retry.py           exponential backoff, 429 classification
chat.py            terminal front end: the loop, debug output, /commands
server.py          web front end: FastAPI wrapper around agent.run_turn()
index.html         the web UI
check_setup.py     stage 1 verification
seed_memories.py   stage 2 verification
tests/             offline test suite (no key, no network)
docker-compose.yml optional real Qdrant server
```

Neither front end imports mem0. Both call `agent.run_turn()`, which calls
`memory_store` — so the memory implementation can change without either UI
noticing, and the two UIs cannot drift apart.

The web UI mirrors the terminal's debug output: a drawer under each reply,
collapsed by default, headed with a summary like `2 recalled · 1 update`.
Open it and a superseded fact appears struck through directly above what
replaced it.

---

## Concurrency, and when you need Docker

The `oss` backend runs Qdrant in local mode: the engine is embedded in-process
and writes to `./qdrant_data`. No server, no Docker.

The catch is that local mode takes an **exclusive file lock** on that folder.
One process at a time. A second `chat.py` fails with *"already accessed by
another instance"*.

When you need concurrency — a web API, a worker, two terminals — run a real
server:

```bash
docker compose up -d
```

Then make one change in `.env`:

```ini
QDRANT_HOST=localhost
```

`config.py` sees a non-empty `QDRANT_HOST` and builds the vector-store config
with `host`/`port` instead of `path`. Nothing else changes.

Memories in `./qdrant_data` do **not** migrate automatically — the server
starts empty.

---

## Configuration

All of it lives in `.env`. See `.env.example` for the annotated version.

| Variable | Default | Notes |
|---|---|---|
| `MEMORY_BACKEND` | `cloud` | `cloud` or `oss` |
| `MEM0_API_KEY` | — | cloud only. Free Hobby tier, no card |
| `BACKEND` | `gemini` | `gemini` or `ollama`, for replies |
| `GEMINI_API_KEY` | — | free, no card |
| `GEMINI_MODEL` | `gemini-3.5-flash-lite` | quota is per model |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | oss only |
| `EMBEDDING_DIMS` | `384` | **must match the model** |
| `QDRANT_PATH` | `./qdrant_data` | oss local mode |
| `QDRANT_HOST` | empty | set to switch to a server |
| `RECALL_LIMIT` | `5` | memories injected per turn |
| `DEBUG` | `true` | `/quiet` toggles at runtime |
| `MAX_RETRIES` | `5` | for 429s |

### Two settings that will silently ruin your day

**`EMBEDDING_DIMS` must match your embedding model.** `all-MiniLM-L6-v2`
produces 384-dimensional vectors. Qdrant fixes a collection's vector size at
creation, so a mismatch fails every insert. mem0 assumes 1536 (the OpenAI size)
if you say nothing — the exact trap in every OpenAI tutorial adapted to a local
embedder. `config.py` validates this and refuses to start.

**`on_disk=True` is not an optimisation.** mem0's Qdrant wrapper does this:

```python
if not on_disk:
    shutil.rmtree(path)
```

and `on_disk` defaults to `False`. Leave it off and every launch silently
deletes your entire memory store — which looks exactly like "the agent forgot
everything again". `config.py` sets it explicitly.

---

## Notes on the dependencies

- **`mem0ai` is pinned to `0.1.118`, deliberately.** mem0 2.x rewrote `add()`
  around an "additive extraction" prompt whose stated rule is *"Your sole
  operation is ADD"* — it never emits UPDATE or DELETE. 0.1.118 is the last
  release with the classic two-call extract-then-reconcile path. Do not bump it
  without re-reading `mem0/configs/prompts.py`.
- **`google-genai`, not `google-generativeai`.** mem0 does
  `from google import genai` and raises `ImportError` on the older package,
  which Google also stopped supporting in Nov 2025. Same free tier, same key.
- **`openai` appears in your install log.** It is a hard dependency of `mem0ai`
  itself, imported at module level by mem0's HuggingFace embedder for an
  optional self-hosted-inference path this project never enables. No client is
  constructed, no key is read, no request is made.

---

## What to build next

Roughly in order of value:

1. **Run both front ends at once.** Local Qdrant's file lock currently allows
   only one process, so `chat.py` and `server.py` cannot both run. `docker
   compose up -d` plus `QDRANT_HOST=localhost` fixes it — the code needs no
   change.
2. **Show *why* a memory was recalled.** You have the cosine scores already.
   Surfacing near-misses just below the cut-off teaches you a lot about where
   `RECALL_LIMIT` and the score threshold should sit.
3. **Memory decay.** Nothing currently ages out. A "last accessed" timestamp
   plus a periodic sweep would stop one-off remarks outliving real preferences.
4. **Let the user correct memory directly.** `/forget <n>` to drop a single
   bad extraction. The primitives exist (`mem0` exposes per-id delete); only
   the command is missing.
5. **Test reconciliation properly.** `seed_memories.py` is a smoke test. Because
   the reconcile step is an LLM judgement call, its reliability varies by model
   and by how much is already stored — a fixture set of contradiction pairs run
   across models would tell you which model to trust.
6. **Group memories by agent or session.** mem0 supports `agent_id` and
   `run_id` alongside `user_id`. That is the road to per-project memory rather
   than one flat pile per person.
