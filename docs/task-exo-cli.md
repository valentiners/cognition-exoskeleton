# Development Task: `voiceofsoul.exo` CLI Entry

> Task brief for the coding agent (OpenClaw). 开发任务书（给代码 agent 的工单）。
> Read first, in this order: `AGENTS.md` (charter & hard rules) → `docs/外挂认知模块与记忆模块方案_v4.0.md` (design constitution, bilingual) → `skills/cognition-exoskeleton/SKILL.md` (the contract this task implements).

## 1. Goal

Implement the CLI entry promised in SKILL.md so that any agent (OpenClaw included) can drive the
exoskeleton from the command line:

```bash
python -m voiceofsoul.exo chat --text "<utterance or description>" [--image /path/to/image.jpg]
python -m voiceofsoul.exo respond --text "<master's free-text answer>"
python -m voiceofsoul.exo stats
```

Every command prints a single JSON document to stdout:

```json
{"say": "<human-like reply, relay verbatim to the master>", "actions": [], "debug": "<diagnostics only>"}
```

`stats` prints the same payload as the test service's `GET /api/stats`.

## 2. Architecture requirements（架构要求）

1. **Reuse, do not duplicate.** All orchestration logic already lives in `server.py`
   (`make_handlers`: topic cooldown, pending-timeout degradation, recall wake-up, temp-memory
   pipeline). Extract that session orchestration into a new kernel-level module
   (suggested: `voiceofsoul/session.py`) that **both** `server.py` (HTTP wrapper only) and
   `voiceofsoul/exo.py` (CLI wrapper) import. One brain, two bodies. 一个大脑，两个壳。
2. **Session state must survive process exits.** The CLI is invoked once per message, so
   `STATE` (pending with timestamps, topic, model selection) persists to
   `<COGSOUL_HOME>/session.json`; every invocation loads it, runs one turn, saves it.
   Memory files (`memory.json`, `temp_memory.json`) already persist — keep using them.
3. **Concurrency**: primitive version may accept a last-writer-wins race on `session.json`;
   document the limitation. (SQLite/file-locking is out of scope.)
4. **Kernel purity**: `voiceofsoul/` stays stdlib-only. argparse/json/pathlib are fine.
   The CLI module lives inside the package (`voiceofsoul/exo.py`, `__main__.py` shim).

## 3. Configuration（配置）

| Input | Env var | Flag | Default |
|---|---|---|---|
| Ollama host | `OLLAMA_HOST` | `--ollama` | `http://localhost:11434` |
| Working directory | `COGSOUL_HOME` | `--home` | `~/.cognition-exoskeleton` (created on demand) |
| Base model | — | `--base-model` | `gemma4:latest` |
| Embedding model | — | `--embedding-model` | `embeddinggemma-2:latest` |

Switching the embedding model via flag rebuilds all vectors (existing behavior, §10.1).

## 4. Hard rules（铁律，违反即返工）

From AGENTS.md — restated because they bite:

1. Never modify model weights; base model is called over HTTP through the adapter only;
2. Single-master design; no auth/permission layers;
3. Append-only state changes;
4. `voiceofsoul/` zero third-party dependencies (stdlib only);
5. **Calibrated parameters untouched**: embedding=embeddinggemma-2, mem_th=0.78,
   base_th=0.51, TempMemory.MATCH_TH=0.62, TTL_DAYS=7, PROMOTE_DAYS=3 (basis: design §9);
6. No runtime data (memory/session files) committed to git;
7. All master-facing text human-like, first-person; mechanics narration stays in `debug`;
8. UTF-8 everywhere — Windows GBK console transcoding has bitten us before (design §11.2):
   read/write files and stdout with explicit `encoding="utf-8"`.

## 5. Error behavior（错误行为）

- Ollama unreachable → exit code `2`, stderr carries a JSON error
  `{"error": "ollama_unreachable", "detail": "..."}`; nothing stored;
- Bad arguments → exit code `2`, usage on stderr;
- Any unexpected exception → exit code `1`, traceback on stderr, memory files must remain
  valid JSON (never corrupt on failure).

## 6. Non-goals（明确不做）

- No `perceive_video` implementation (contract only, design §3.6);
- No UI work; no changes to `web/index.html`;
- No packaging/pyroject.toml (separate task);
- No new thresholds, no design changes. **If anything in the design doc is unclear, stop and
  ask Kimi/Ricky — do not guess** (AGENTS.md discipline).

## 7. Acceptance criteria（验收标准）

1. The three SKILL.md commands work verbatim against a running local Ollama;
2. **Cross-process session continuity**: `chat` returns an UNKNOWN/naming pending → a separate
   `respond` invocation completes naming and the entry persists to `memory.json`;
3. `server.py` behavior unchanged: existing suites still pass —
   `_test_lucy.py`, `_test_dormant_e2e.py`, `_test_degrade.py`, `_test_context.py`,
   `_test_temp_memory.py`, `_test_dormant.py`;
4. New test script `_test_exo_cli.py` (git-ignored, follows the `_test_*` convention) covering:
   chat JSON shape, respond-after-pending across processes, stats parity with the HTTP service,
   Ollama-down error path (mock by pointing OLLAMA_HOST at a dead port);
5. `start.bat` untouched and still exclusive (two-layer lock).

## 8. Open questions for Kimi/Ricky（开工前先确认的开放点）

1. `session.json` last-writer-wins acceptable for primitive? ( assumed yes per §2.3 )
2. `--image` accepts a filesystem path only, not base64? ( SKILL.md shows a path )
3. `actions` array: keep the server's optional button payloads, or empty in CLI v1?

---

## Appendix A. Review checklist（审阅清单 — for Kimi & Ricky）

When the coding agent delivers, we review against this list:

- [ ] Contract fidelity: commands, flags, JSON shape match SKILL.md §"How it works" exactly
- [ ] One-brain-two-bodies: no duplicated orchestration logic between `server.py` and `exo.py`
- [ ] Hard rules 1–8 (§4) all hold; `voiceofsoul/` still stdlib-only (`pip install` nothing)
- [ ] Calibrated parameters untouched (diff against design §9 table)
- [ ] Session continuity proven by test evidence (§7.2), not just code reading
- [ ] Error paths: dead-Ollama test actually ran; memory files valid after forced failure
- [ ] Full existing test suites re-run and green
- [ ] No design document was modified by the coding agent (it may only *ask*, never edit)
- [ ] Open questions (§8) were asked, not silently decided
