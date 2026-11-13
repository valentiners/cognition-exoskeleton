# Development Task: Curiosity Module（好奇心模块 + 自主检索通道）

> Task brief for the coding agent (OpenClaw). 开发任务书（给代码 agent 的工单）。
> Read first, in this order: `AGENTS.md` (charter & hard rules) →
> `docs/外挂认知模块与记忆模块方案_v5.0.md` — **the design constitution**, especially
> §4.6 (triggers T1–T4) / §4.7 (tiers L0–L2 & autonomy budget) / §4.8 (curiosity queue & closed loop) /
> §4.9 (self-retrieval channel & source discipline) / §8.2 (probe contract, #10 `curiosity_scan`) /
> §8.3 (`web_search` tool contract) →
> `docs/task-exo-cli.md` + `docs/task-cloud-adapter.md` (precedents: entry points and adapter
> discipline this task must not break).

## 1. Goal

v4.0 and earlier the exoskeleton learns **passively** — the master teaches, it remembers.
This task gives it **curiosity**: prediction-error-driven learning signals (design §4.6),
question tiers with an autonomy budget (§4.7), a persistent curiosity queue with a
self-evolution closed loop (§4.8), and a "search before asking" self-retrieval channel
with strict source discipline (§4.9 / §8.3).

The master must never feel spammed: **if the web can answer it credibly, the master is
not bothered at all**（查得到且可信，就不打扰主人）.

## 2. Scope & components（交付物）

1. **`voiceofsoul/tools/web.py`** (new) — the §8.3 tool contract, the only file that
   touches a search backend:
   ```python
   def web_search(query: str, lang: str = "zh") -> list[dict]:
       """[{title, url, snippet}, ...]; ANY failure (no network / timeout / zero results) -> []."""
   ```
   stdlib-only (urllib). DuckDuckGo HTML endpoint as the default backend, SearXNG-compatible
   API as a documented pluggable alternative (env `COGSOUL_SEARCH_BACKEND`,
   `COGSOUL_SEARCH_BASE`). Discipline identical to the adapter layer: swapping the search
   source = editing only this file, kernel untouched.
2. **Probe #10 `curiosity_scan`** in BOTH adapters (`ollama.py`, `openai_compat.py`),
   sharing prompt templates/parsers in `adapters/prompts.py` (never fork prompts):
   ```python
   def curiosity_scan(self, unified_desc: str, context: str = "") -> list[dict] | None
   # returns [] = not curious; None = unsure -> ALWAYS treated as not curious (宁缺毋滥);
   # each item: {"point": <str>, "kind": "contradiction"|"unknown_entity"|...}
   ```
3. **`voiceofsoul/curiosity.py`** (new) — `CuriosityQueue`, persisted to
   `<COGSOUL_HOME>/curiosity.json`:
   ```python
   # item: {qid, question, source_trigger(T1~T4), stimulus(当时的事物描述),
   #        created_at, status(open|self_solved|answered|muted|expired),
   #        tier_due(L1|L2), hit_count, last_hit_at, solved_source, solved_url}
   class CuriosityQueue:
       def add(point, trigger, stimulus) -> qid
       def due_for_l1(now) -> item | None      # 一轮对话最多 1 次
       def due_for_l2(now) -> list[item]       # 攒批呈现
       def touch(text_embedding) -> [qid]      # 主人提起同主题 -> 唤醒 (§4.8)
       def mark(qid, status, **kw)
       def expire() -> [qid]                   # TTL 7 天 + muted 7 天自动解禁
       def stats() -> dict
   ```
   Pure date/vector logic — no LLM calls (same discipline as `temp_memory.py`).
4. **Session orchestration** (`session.py`) — the curiosity hooks (see §3).
5. **CLI** (`exo.py`) — `stats` output gains the curiosity block (design §4.8:
   累计/已解答/过期/静音); add `exo curiosity list|mute <qid>` for manual ops.
6. **Docs sync** — `skills/cognition-exoskeleton/SKILL.md` gains a "Curiosity" section;
   bundled `skills/cognition-exoskeleton/voiceofsoul/` copy stays byte-identical
   (modulo line endings) with the main package.

## 3. Session orchestration hooks（编排要求）

Order of operations inside a chat turn (all new behavior degrades gracefully — any
failure in the curiosity path must never break the base reply):

1. After `perceive()` returns:
   - **T1**: action UNKNOWN with naming pending → create a curiosity point linked to the
     pending topic (the existing 求教 flow itself is the question; the point tracks it in
     the queue so "master ignored it" does NOT drop it — §4.8);
   - **T2**: action GUESS with `conf < CURIOUS_TH` → create a curiosity point
     ("我猜它是 X，但不太确定……" 的张力即学习信号).
2. **L0 先查后问**（§4.9, runs for every new point, both T1/T2 and T3/T4）:
   neutralize the point into a search query (prompt template in `prompts.py` — the query
   must carry **no preset conclusion**, 防确认偏误) → `web_search()` → base model reads
   snippets and extracts a fact statement. If a coherent, source-backed fact is extracted:
   `mark(qid, self_solved, solved_source, solved_url)` and ingest **through the existing
   §4.3/§4.4 learning pipeline** (`learn_entry`/`merge_into` + the three gates §5) with
   `source="web"` and the URL attached — never a new ingestion channel. Then a brief
   humanized report is appended to the assistant reply ("我自己查了下……"). If search
   fails / returns nothing / the model finds the snippets untrustworthy → fall back to
   the L1/L2 question path. **自学失败不丢人，问主人不丢份。**
3. **T3/T4** via `curiosity_scan` (§8.2 #10): wired in but **default OFF** — config flag
   `curiosity_scan_enabled` (env `COGSOUL_CURIOUS_T34`, default false) until the §6.5
   calibration lands. When disabled, `curiosity_scan` is never even called.
4. **Budget & tiers**（§4.7）before appending any question to the reply:
   - autonomy mode `quiet` (only L2) / `normal` (L1+L2, default) / `eager` (L1 up to 2
     per turn) — persisted in session state, switchable via `exo curiosity mode <m>`;
   - **anti-triggers**（§4.6）: master is mid task-flow (session has any unresolved
     pending), or the same stimulus was asked within 7 days and went unanswered → no
     on-the-spot question;
   - **L1**: at most one humanized light question per assistant turn, non-blocking, free
     text only — **no buttons** (AGENTS rule 6). Master doesn't answer → never re-ask in
     the same conversation, let it roll into the queue;
   - **L2**: batch presentation only outside quiet hours (default 21:00–08:00 master-local;
     env-overridable `COGSOUL_QUIET_HOURS=21:00-08:00`): one opening line + the question
     list, master answers whichever ones they like; each answered question routes through
     the existing naming/feature-supply pipeline (§4.3/§4.4), never a new channel;
   - master says "别问了/以后再说" → `mark(qid, muted)` for 7 days (reuse the temp-memory
     expiry philosophy).
5. **Wake-up**（§4.8）: on every chat utterance, embed once (local embedder, cheap) and
   `queue.touch(embedding)`; any open/muted item with cosine ≥ 0.62 (`TempMemory.MATCH_TH`,
   reuse — do NOT invent a new threshold) wakes up and may be re-presented naturally.
6. **T3 perception contradictions**: when a recognized entry's features conflict with the
   master's wording (the T3 scenario), the contradiction itself is recorded via the
   existing `learn_patch`/mistake pipeline — curiosity never bypasses the three gates.

## 4. Parameters（参数纪律）

| Param | Value | Status |
|---|---|---|
| `CURIOUS_TH` | 0.35 | **placeholder, 待标定 §6.5** — T2 trigger threshold; document in design §9 on request |
| same-topic wake threshold | reuse `MATCH_TH = 0.62` | calibrated, untouched |
| `TTL_DAYS = 7` (queue expiry & mute window) | reuse | calibrated, untouched |
| quiet hours | 21:00–08:00 default, env-overridable | product default, not a calibrated threshold |

Nothing in the design §9 calibrated-parameter table changes. New parameters arrive as
**explicitly-marked placeholders**, not silent calibrations.

## 5. Hard rules（铁律，违反即返工）

1. `voiceofsoul/` stays **stdlib-only**; zero new third-party deps (AGENTS rule 4);
2. **No buttons, ever** — curiosity is conversational free text, humanized and colloquial
   (AGENTS rule 6); questions must be neutrally worded, no preset answer embedded;
3. **Source discipline**（§4.9, 铁律级）: web-learned entries carry `source="web"` + URL,
   initial confidence one tier below master-taught entries; on conflict with master input,
   **master always wins** via the existing correction-patch pipeline (§5.3); raw search
   snippets are untrusted input — only the base-model-distilled fact statement enters the
   memory/instruction channel;
4. Curiosity-produced entries are **homologous with master-taught entries** — same
   pipeline, same three gates, never a lowered standard (§4.8);
5. Any failure in search/extraction/queue I/O degrades to "just don't ask" — the base
   chat reply is never blocked or broken by curiosity machinery;
6. API-key hygiene from task-cloud-adapter holds; search traffic carries no credentials;
7. No design-document edits by the coding agent — unclear points go to Kimi/Ricky
   (AGENTS discipline); no commits of run data (AGENTS rule 7).

## 6. Non-goals（明确不做）

- **Model physical (§8.4 `doctor`)** — separate task, do not start it here;
- **Consolidation scheduler (§10.3)** — separate task;
- No changes to calibrated thresholds; no cloud embedder;
- No web UI changes — curiosity questions ride inside the existing reply text
  (a debug field in the chat JSON is fine, buttons are not);
- No multi-user, no permissions, no async/background workers — everything happens
  in-turn, synchronously.

## 7. Acceptance criteria（验收标准）

1. **Zero regression**: `_test_lucy.py`, `_test_dormant_e2e.py`, `_test_degrade.py`,
   `_test_context.py`, `_test_identity.py`, `_test_temp_memory.py`, `_test_dormant.py`,
   `_test_exo_cli.py`, `_test_openai_adapter.py` all green, default config
   (`curiosity_scan_enabled=false`, mode=normal);
2. **New mock suite `_test_curiosity.py`** (git-ignored, `_test_*` convention), no real
   network: stub HTTP server fakes the search backend → verifies: search failure →
   question path (L1 budget respected); canned snippets → `self_solved` + entry ingested
   via the existing pipeline with `source="web"` and URL; TTL expiry; mute 7 days then
   auto-release; wake-up on ≥0.62 same-topic utterance; anti-trigger suppression
   (pending unresolved → no L1); stats counters;
3. **Real-model e2e on Ricky's machine** (manual evidence, transcript attached):
   an UNKNOWN teach-and-recognize loop still works unchanged; a T2 low-confidence GUESS
   produces at most one humanized question; the autonomy mode switch takes effect;
4. `curiosity.json` survives process restart; `exo stats` shows the curiosity block;
5. `skills/` bundled copy synced (line-ending deltas only); `SKILL.md` valid per its
   schema check; `start.bat` untouched and still exclusive.

## 8. Open questions — proposed defaults, awaiting Kimi/Ricky confirmation

1. T1 points duplicate the existing pending-naming flow; we **link** them (queue tracks,
   pipeline unchanged) rather than replace the pending mechanism — proposed;
2. L2 batch opening line is generated from a fixed template + the question list
   (no LLM call for the opener) — proposed;
3. `exo curiosity` subcommands: `list`, `mute <qid>`, `mode <quiet|normal|eager>` —
   proposed minimal surface;
4. `CURIOUS_TH = 0.35` placeholder — explicitly pending §6.5 calibration;
5. The web UI and `server.py` HTTP surface gain no new endpoints this task — the chat
   JSON may include a `curiosity` debug field only.

---

## Appendix A. Review checklist（审阅清单 — for Kimi & Ricky）

- [ ] §4.6/§4.7/§4.8/§4.9 of the design doc faithfully implemented, incl. all four source-discipline rules
- [ ] Probe #10 implemented in BOTH adapters, prompts shared in `prompts.py`, default OFF
- [ ] `voiceofsoul/` stdlib-only; no new deps; `web_search` failures all → `[]`
- [ ] Calibrated parameters untouched (diff against design §9 table); `CURIOUS_TH` marked placeholder
- [ ] Queue item schema matches §4.8 exactly; persistence across restart proven
- [ ] Mock test evidence (§7.2) actually ran; no real network needed
- [ ] Existing suites re-run green under default config (§7.1)
- [ ] No buttons anywhere; every master-facing string humanized (AGENTS rule 6)
- [ ] No design document was modified by the coding agent
- [ ] `skills/` copy synced; `SKILL.md` schema-valid; `start.bat` untouched
