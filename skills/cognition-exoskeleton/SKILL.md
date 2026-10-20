---
name: cognition-exoskeleton
description: Give any LLM a self-cognition exoskeleton — perception-first recognition, conversational memory arbitration, two-layer memory (temporary/permanent) with 7-day promotion, dormancy and "think again" recall. No fine-tuning, model-agnostic, stdlib-only Python.
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
