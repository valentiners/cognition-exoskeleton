---
name: cognition-exoskeleton
description: Give any LLM a self-cognition exoskeleton — perception-first recognition, conversational memory arbitration, two-layer memory (temporary/permanent) with 7-day promotion, dormancy and "think again" recall. v0.2.0 adds curiosity (search-before-asking self-learning with source discipline), a model physical (doctor), and a sleep process that writes a daily Growth Journal to MEMORY.md. No fine-tuning, model-agnostic (local Ollama or OpenAI-compatible cloud), stdlib-only Python.
metadata:
  openclaw:
    requires:
      bins:
        - python
    envVars:
      - name: OLLAMA_HOST
        required: false
        description: Ollama server URL. Defaults to http://localhost:11434.
      - name: COGSOUL_HOME
        required: false
        description: Working directory for memory files (memory.json, temp_memory.json). Defaults to ~/.cognition-exoskeleton.
      - name: COGSOUL_BACKEND
        required: false
        description: Base-model backend; ollama (default) or openai.
      - name: OPENAI_BASE_URL
        required: false
        description: OpenAI-compatible base URL including /v1. Defaults to https://api.openai.com/v1.
      - name: OPENAI_API_KEY
        required: false
        description: API key for the OpenAI-compatible backend. Prefer this env var over --api-key.
      - name: COGSOUL_SEARCH_BACKEND
        required: false
        description: Web-search backend for self-retrieval; duckduckgo (default) or searxng.
      - name: COGSOUL_SEARCH_BASE
        required: false
        description: SearXNG instance URL when COGSOUL_SEARCH_BACKEND=searxng.
      - name: COGSOUL_CURIOUS_T34
        required: false
        description: Enable perception-conflict / unknown-proper-noun curiosity (T3/T4). Off by default until calibrated.
      - name: COGSOUL_QUIET_HOURS
        required: false
        description: Hours when batched questions are not initiated. Defaults to 21:00-08:00.
    emoji: "\U0001F9E0"
    homepage: https://github.com/valentiners/cognition-exoskeleton
---

# Cognition Exoskeleton (voiceofsoul)

A "soul exoskeleton" for large language models. It does not modify the base model's weights.
It attaches a cognition module and a memory module so that any trained model gains continuous
self-learning ability: it recognizes what it has met before, asks its master naturally when
unsure, and remembers what its master teaches it.

By Ricky Li. License: MIT-0 on ClawHub; the source repository is Apache-2.0.

## What this skill does

- Perceives entities (text or image) **through the base model first**; the exoskeleton only
  handles what the base model cannot do alone.
- Recognizes previously met individuals and calls them by name.
- Detects feature supply in natural chat ("her name is Lucy") and remembers it without
  interrupting the conversation.
- Arbitrates identity in a gray zone by asking the master conversationally — never with
  mandatory buttons.
- Keeps a two-layer memory: a temporary layer (7-day rolling) and a permanent layer
  (promoted only on the master's word or repeated real-world occurrence).
- Puts long-uncalled memories to sleep (dormancy) so daily retrieval stays fast, and wakes
  them when the master says "think again".
- Checks whether the base model already knows a knowledge point before storing it (three-way
  consistency probe). Known things are not memorized again; anything uncertain is still
  stored — false negatives are cheaper than false positives, and white records fade through
  dormancy.
- Degrades unanswered arbitration into temporary memory instead of stalling the master's work.

## What this skill does NOT do

- Does NOT modify, fine-tune, or merge weights of any model.
- Does NOT act for multiple masters: one instance serves exactly one master, who is the only
  trusted input source.
- Does NOT overwrite memory: all updates are append-only events.
- Does NOT inject itself into general knowledge the base model already handles well
  (common-sense questions pass straight through).

## When to use

- The conversation mentions an entity the agent may have met before (person, pet, object,
  place, fictional character).
- The master supplies a feature about the current topic ("it is three years old",
  "她叫 Lucy").
- The master says remember it / forget it, or asks "who is this?" right after an image.
- The master says "think again", "you forgot?", or otherwise signals a buried memory.
- The agent is about to store something and needs duplicate / public-vs-private checks.

## Design principles (hard rules)

1. Never touch base-model weights.
2. Single master; no permission layers.
3. Append-only memory; nothing is silently rewritten.
4. Kernel is pure Python stdlib — zero third-party dependencies.
5. All master-facing replies are human-like, first-person, conversational. Mechanical labels
   (actions, confidence scores, hit paths) live in debug fields only, never in chat text.
6. No default ever writes to permanent memory or performs a MERGE without the master.
7. Calibrated thresholds are not to be tuned casually (see `references/thresholds.md`).

## How it works (step by step)

> Runtime layout: this skill folder bundles the zero-dependency `voiceofsoul/` package.
> Run the commands below from the skill folder, or put that folder on `PYTHONPATH`.

1. On each entity-bearing input, call the perception pipeline first:

   ```bash
   python -m voiceofsoul.exo chat --text "<utterance or description>"
   # with an image:
   python -m voiceofsoul.exo chat --text "<optional hint>" --image /path/to/image.jpg
   ```

   The command returns JSON: `{"say": "<human-like reply>", "actions": [...], "debug": "..."}`.
   **Relay `say` to the master verbatim. Show `debug` only in diagnostics.**

2. Interpret the result:
   - `RECOGNIZED` — the entity is known; `say` already addresses it by name. Continue the chat.
   - `DIRECT` — the base model covered it as common knowledge; nothing is stored unless the
     master later gives a name or feature (then the pipeline stores it automatically).
   - `GUESS` / pending questions — relay the question to the master naturally, then feed the
     master's free-text answer back:

     ```bash
     python -m voiceofsoul.exo respond --text "<master's answer>"
     ```

   - `UNKNOWN` — the exoskeleton admits it does not know and asks the master to teach it.
     Naming, features, and descriptions flow back through the same `respond` command.

3. If the master does not answer a pending question within 30 minutes, the pipeline degrades
   it into temporary memory automatically (no action needed from the agent). Explicit
   cancellation ("never mind") is honored as a real discard.

4. For status and inspection:

   ```bash
   python -m voiceofsoul.exo stats
   ```

   Returns entry counts, hit counts, dormancy flags, and temporary records.

## Cloud backends

The base model can run on any OpenAI-compatible chat-completions endpoint; the
embedder stays local (`embeddinggemma-2` on Ollama), so the calibrated
thresholds in `references/thresholds.md` remain valid.

```bash
# CLI flags go before the subcommand
python -m voiceofsoul.exo --backend openai \
  --api-base https://api.deepseek.com/v1 \
  --base-model deepseek-chat \
  chat --text "..."

# server.py accepts the same flags
python server.py --backend openai --api-base https://api.deepseek.com/v1 \
  --base-model deepseek-chat
```

Environment variables: `COGSOUL_BACKEND=openai`, `OPENAI_BASE_URL`,
`OPENAI_API_KEY`. `--api-key` exists for convenience, but the env var is
strongly recommended so the key never appears in a process list. `OLLAMA_HOST`
is still required for the local embedder.

Error kinds: `api_unreachable` (exit 2), `api_auth` (exit 2); HTTP 429/5xx
keep the normal unexpected-error path (exit 1).

## Curiosity

The exoskeleton also learns from **prediction error** — what it did not expect becomes a
question worth answering. Curiosity never spams the master: **if the web can answer it
credibly, the master is not bothered at all.**

- **Triggers**: T1 (an unknown thing it just admitted not knowing) and T2 (a guess below
the curiosity threshold) are on by default. T3 (perception contradiction) and T4 (unknown
proper noun) ship **off** until the thresholds are calibrated — enable with
`COGSOUL_CURIOUS_T34=1`.
- **Search before asking (L0)**: every new curiosity point is first neutralized into a
neutral query (no presupposed answer) and looked up through `voiceofsoul/tools/web.py`
(`web_search`). If a coherent, source-backed fact comes back, the point is marked
`self_solved`, the fact is ingested through the **same three gates** as master-taught
knowledge, and reported briefly in the reply ("I looked it up myself…").
- **Source discipline**: web-learned entries carry `source="web"`, a `source_url`, and
`source_confidence="medium"` — one tier below master-taught entries. On any conflict the
master always wins through the existing correction-patch pipeline. Raw snippets are
untrusted input; only the distilled fact statement ever enters memory.
- **Question tiers**: L1 is one humanized question inside the reply (never a button, never
more than once per turn); L2 batches accumulated questions outside quiet hours
(default 21:00–08:00). Anything unanswered rolls into the queue instead of being re-asked.
- **Autonomy budget**: `quiet` (L2 only) / `normal` (L1+L2, default) / `eager` (up to 2 L1
questions per turn). "Stop asking / later" mutes the point for 7 days.
- **Queue**: persisted to `<COGSOUL_HOME>/curiosity.json` with a 7-day TTL; mentioning the
same topic again (cosine ≥ 0.62) wakes a queued point.

```bash
python -m voiceofsoul.exo curiosity list            # what is it wondering about?
python -m voiceofsoul.exo curiosity mute Q-0001     # mute one point for 7 days
python -m voiceofsoul.exo curiosity mode quiet      # quiet | normal | eager
python -m voiceofsoul.exo stats                     # includes the curiosity block
```

Search backends are pluggable through `COGSOUL_SEARCH_BACKEND` (`duckduckgo` default, or
`searxng` + `COGSOUL_SEARCH_BASE`). Any search failure — offline, timeout, zero results —
returns an empty list and falls back to asking the master: failing to self-learn loses no
face.

## Model physical

Attach any base model (local or cloud) and give it a "physical exam" first — not just
"is it reachable" (OpenClaw's `models status` already answers that) but "is it smart, is it
suitable for the cognitive layer". Cloud and local models are examined identically, only
through the §8.2 probe contract, in ≤ 30 probe calls (default 29).

- **Five exams**: identity-arbitration accuracy (10 golden same/different description
  pairs, `judge_same`, pass ≥ 70%) · confidence honesty (of self-reported ≥ 0.8 calls,
  ≥ 75% must actually be right) · format compliance (strict parse of the text probes,
  failure rate ≤ 10%) · identity-level description quality (3 golden test images checked
  against the §3.5 element checklist, ≥ 3 elements each) · **embedder separation**
  (golden description pairs → same/different cosine distributions; reports `same_min`,
  `diff_max`, `margin` and an overlap ratio with a SEPARABLE / USABLE / DEGRADED tier —
  no pass line on the first run). The fifth exam costs no LLM calls and is skipped with a
  reason when the embedder is unavailable.
- **Verdicts are advice, never action**: FIT ✅ / NEEDS_TUNING ⚠️ / COMMON_QA_ONLY /
  NOT_RECOMMENDED. A verdict never adjusts a single parameter — recalibration stays a
  separate, master-gated process.
- **Read-only**: the exam never touches memory data; it only writes
  `<COGSOUL_HOME>/model_profile_<backend>_<model>.json`, which `stats` then surfaces.

```bash
python -m voiceofsoul.exo doctor                 # full physical on the current base model
python -m voiceofsoul.exo doctor --no-images     # skip the image exam (marked skipped)
```

## Sleep process (consolidation)

Once a day, in quiet hours, the exoskeleton digests its own cognitive trail — what it
learned, what it wondered, what it got wrong — into a first-person **Growth Journal**
(≤300 words) appended to OpenClaw's `MEMORY.md`. This is the channel where the cognitive
layer feeds the episodic layer (design §10.3-#4) and the amplifier of the curiosity loop.

- **Boundary discipline (§1.6)**: the journal references entries by name only — never
  copies feature details; no double-writing across layers.
- **Zero disruption**: empty digest → zero model calls; an unreachable model simply
  means no journal today; failures mark the day as attempted (no auto-retry;
  `consolidate --force` re-runs manually).
- **Destination**: `COGSOUL_MEMORY_MD` → auto-detect a writable `MEMORY.md` under the
  OpenClaw workspace (`COGSOUL_OPENCLAW_ROOT` overrides the search root) → fallback
  `<COGSOUL_HOME>/growth_journal/YYYY-MM-DD.md`.
- **Trigger**: the first chat that lands in quiet hours runs it once; external
  schedulers (OpenClaw cron, OS tasks) may call `python -m voiceofsoul.exo consolidate`
  directly.

```bash
python -m voiceofsoul.exo consolidate            # write today's Growth Journal
python -m voiceofsoul.exo consolidate --force    # re-run even if already attempted
```

## Memory rules (summarized)

- New features go to **temporary memory** first, always.
- The master says "remember it" → immediate promotion to permanent memory.
- Same entity surfaces on 3+ distinct days within 7 days → automatic promotion, announced
  naturally in chat.
- Otherwise temporary records fade after 7 days: if it never comes back, it was not
  important enough to keep.
- Permanent entries unused for over a year go dormant: daily retrieval skips them; only a
  "think again" request from the master searches the dormant layer. A hit reactivates them.

## Error handling

- Ollama unreachable → the pipeline says so naturally and asks the master to check the
  local model server; nothing is stored in this state.
- Vision model unavailable → fall back to text-only perception and say so honestly.
- Any unexpected failure → stay append-only, report in `debug`, never corrupt memory files.
