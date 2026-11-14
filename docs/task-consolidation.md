# Development Task: Consolidation Scheduler（巩固调度器，"睡眠进程"，`voiceofsoul/growth.py`）

> Task brief for the coding agent. 开发任务书（给代码 agent 的工单）。
> Read first, in this order: `AGENTS.md` (charter & hard rules) →
> `docs/外挂认知模块与记忆模块方案_v5.0.md` — **the design constitution**, §10.3-#4
> (this task) + §1.6 (three-layer division — **the journal feeds the episodic layer
> without double-writing it**) + §4.7/§4.8 (quiet hours & the curiosity loop this
> task amplifies) → `docs/task-curiosity.md` (quiet-hours conventions, `_test_*`
> conventions, degradation philosophy).

## 1. Goal

The exoskeleton learns all day but never reflects. This task adds the **sleep
process** (design §10.3-#4): during quiet hours, digest the day's cognitive events
into a human-readable **《成长日记》(Growth Journal)** and append it to OpenClaw's
`MEMORY.md` — the channel through which the **cognitive layer feeds the episodic
layer**, and the amplifier of the curiosity loop (§4.8): what it wondered, what it
self-learned, what the master taught it, what it got wrong.

One run per day, at most. Zero disruption: if anything fails, the day simply goes
without a journal — never an error in the master's face.

```bash
python -m voiceofsoul.exo consolidate          # run now (manual trigger)
```

## 2. Scope & components（交付物）

1. **`voiceofsoul/growth.py`** (new, stdlib-only):
   ```python
   def collect_digest(home, memory, temp, curiosity) -> dict
       # 当日素材：今天新建/更新的 entries、今天的补丁与纠错、
       # 今天的好奇事件（self_solved/answered/expired/muted）、今天的误认记录
   def compose_journal(base, digest, *, date) -> str | None
       # 基础模型把素材写成一篇 ≤300 字的第一人称拟人日记（中文）；
       # 不可达/超时/素材为空 → None（今天就不写，不丢人）
   def consolidate(home, base, session, *, now=None) -> dict
       # collect → compose → 落盘（见 §3）→ 写 last_consolidation.json → 返回摘要
   def last_consolidation(home) -> dict | None   # stats 展示用
   ```
2. **`voiceofsoul/exo.py`** — `consolidate` subcommand (prints the run summary JSON).
3. **Auto-trigger** — in `Session.chat` entry: if now ∈ quiet hours (reuse
   `DEFAULT_QUIET_HOURS` / `COGSOUL_QUIET_HOURS` parsing already in session.py),
   no consolidation yet today (`last_consolidation.json` date ≠ today), and the
   digest is non-empty → run once, **before** the reply is composed. Mark the run
   regardless of outcome (a failed run must not retry every turn).
4. **`stats`** gains `last_consolidation` (date / journal path / event counts).
5. **Docs sync** — `SKILL.md` gains a "Sleep process (consolidation)" section;
   bundled `skills/.../voiceofsoul/` copy stays byte-identical (modulo line endings).

## 3. Journal output & the MEMORY.md contract（写回契约，§1.6 边界）

- **Destination order**: env `COGSOUL_MEMORY_MD` (explicit path) → auto-detect a
  readable-and-writable `MEMORY.md` beside known OpenClaw workspace roots →
  fallback `<COGSOUL_HOME>/growth_journal/YYYY-MM-DD.md` (always available).
- **Append-only**, one clearly-delimited section per day:
  ```markdown
  ## 成长日记 · 2026-11-14
  <journal prose>
  ```
  If the section for today already exists, **replace that section's prose**
  (same-day re-runs update, never duplicate).
- **互引不双写（§1.6 规则 3）**：日记以名字引用认知条目（"今天我认识了
  「豆豆」"），**绝不复制特征明细**；情节流水归 OpenClaw，本进程只写感悟层。
- **Humanized prose only** — the journal is read by the master: first person,
  colloquial, no JSON, no mechanical labels, no confidence numbers (AGENTS rule 6).

## 4. Hard rules（铁律，违反即返工）

1. `voiceofsoul/` stays **stdlib-only**; zero new third-party deps;
2. **Base-model call budget ≤ 3 per run** (1 composition call + ≤2 retries on parse
   failure); a run with zero digest material makes **zero** calls;
3. **Never blocks or breaks chat**: any failure (base unreachable, IO, parse)
   degrades to "no journal today"; auto-trigger failures are logged to stderr only;
4. **Idempotent**: max one successful journal per calendar day; a failed attempt
   marks the day as attempted (retry only via manual `consolidate --force`);
5. **Append-only writes** (AGENTS rule 3): the journal section is the only mutable
   region; everything else (last_consolidation.json, fallback files) is replaced
   atomically (write tmp + rename);
6. **Boundary discipline (§1.6)**: never copies entry feature details into the
   journal; never creates MEMORY.md if none exists at the resolved path —
   fall back to the local journal dir;
7. Master-facing strings humanized; **no buttons**; no design-document edits by the
   coding agent.

## 5. Non-goals（明确不做）

- No scheduling daemon/cron inside the kernel — triggering is: manual CLI +
  once-per-day auto-check at chat entry (quiet hours). External schedulers
  (OpenClaw cron, OS task scheduler) may call `exo consolidate` freely;
- No episode-layer reading (we do **not** parse OpenClaw `memory/*.md` logs in v1 —
  the digest draws on the cognitive layer's own trail only);
- No retroactive journals for past days; no multi-day batching;
- No changes to `cognition.py` / `memory.py` / `temp_memory.py` / `curiosity.py`
  internals (read their state, don't modify them — consolidation is **read-only**
  over the cognitive store);
- No web UI changes.

## 6. Acceptance criteria（验收标准）

1. `_test_consolidation.py` (git-ignored `_test_*` convention), **zero network**:
   fake base + fake clock proves — empty digest → zero base calls, no journal;
   happy path → journal file written with today's section; MEMORY.md resolution
   order (env → detected → fallback); same-day re-run updates the section instead
   of duplicating; failed composition marks the day attempted (no auto-retry loop);
   auto-trigger fires only in quiet hours, only once, only with material;
   `stats` shows `last_consolidation`;
2. Existing suites all green: `_test_doctor.py`, `_test_curiosity.py`,
   `_test_exo_cli.py`, `_test_openai_adapter.py`, `_test_temp_memory.py`,
   `_test_dormant.py`;
3. Manual evidence on Ricky's machine: `python -m voiceofsoul.exo consolidate`
   after a day with real activity → journal section readable in the resolved
   MEMORY.md (or fallback path), transcript attached;
4. `skills/` copy synced; `SKILL.md` valid; `start.bat` untouched.

## 7. Open questions — RESOLVED by Kimi/Ricky（已拍板，按此实现）

1. Journal language: **Chinese** (the master's language), ~≤300 字;
2. Auto-trigger point: **first chat of the day that lands in quiet hours**;
3. Digest scope (v1): entries created/updated today + today's patches/mistakes +
   today's curiosity events — **no reading of OpenClaw daily logs**;
4. `--force` flag re-runs today's journal even if already attempted;
5. The journal addresses the master directly ("今天你又教了我…") — growth visible.
