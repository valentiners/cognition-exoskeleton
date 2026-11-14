# Development Task: Model Physical（模型体检，`voiceofsoul/doctor.py`）

> Task brief for the coding agent. 开发任务书（给代码 agent 的工单）。
> Read first, in this order: `AGENTS.md` (charter & hard rules) →
> `docs/外挂认知模块与记忆模块方案_v5.0.md` — **the design constitution**, §8.4
> (model physical) + §8.2 (probe contract) + §3.5 (image-description element checklist) →
> `docs/task-curiosity.md` (precedent: how a v5.0 module task is scoped & verified).

## 1. Goal

When any base model (local or cloud) is attached, run a "physical exam" that judges
**capability strength and cognitive-layer fitness** — cloud and local models treated
identically. OpenClaw's built-in `models status` only diagnoses reachability; this
diagnoses "is it smart, is it suitable". Entry point (design §8.4):

```bash
python -m voiceofsoul.exo doctor [--backend ollama|openai] [--api-base ...] [--base-model ...]
```

The verdict is **advice only** — a verdict never adjusts a single parameter
(§6.5 + the master's decision still gate every parameter change).

## 2. Scope & components（交付物）

1. **`voiceofsoul/golden/`** — the golden test set (repo assets, loaded at runtime):
   - `judge_pairs.json` — 10 pairs of identity descriptions `{a, b, same}` (5 same /
     5 different, including boundary cases: same person with changed clothing or
     hairstyle; different people sharing hair color), used by exam 1;
   - `calibration.json` — 5 factual questions `{q, expect_substrings[]}` with known
     answers, used by exam 2;
   - `images/` — 3 golden test images (PNG) with deliberately distinct identity-level
     features, used by exam 4.
2. **`voiceofsoul/doctor.py`** — `run_physical(base, *, images_dir=None) -> report`
   (stdlib-only; talks to the base model **only** through the §8.2 probe contract).
3. **`voiceofsoul/exo.py`** — `doctor` subcommand: probes backend reachability exactly
   like `chat` does (ollama_unreachable / api_unreachable / api_auth → exit 2), runs the
   exam, prints the report JSON to stdout.
4. **Report persistence** — `<COGSOUL_HOME>/model_profile_<backend>_<model>.json`;
   `stats` output surfaces the latest profile (verdict + per-exam scores).
5. **Docs sync** — `skills/cognition-exoskeleton/SKILL.md` gains a "Model physical"
   section; bundled `skills/.../voiceofsoul/` copy stays byte-identical
   (modulo line endings).

## 3. The four exams（体检四项，全部走既有探测点，零新增契约）

| # | Exam | Probes used | Pass bar | Failure meaning |
|---|---|---|---|---|
| 1 | judge_accuracy | `judge_same` ×10 (golden pairs) | ≥ 70% correct | unfit for individual recognition |
| 2 | conf_calibration | `infer_confidence` + `answer_question` ×5 each | of self-reported ≥0.8 items, ≥ 75% actually correct | calibration drift; arbitration will escalate often |
| 3 | fmt_compliance | strict validation of the text probes' return values | ≤ 10% parse failures | unreliable machine-readable channel |
| 4 | desc_quality | `perceive_image` ×3 (golden images) vs §3.5 checklist (发色/瞳色/脸型/服装配色/配饰/显著标记) | each image's description covers ≥ 3 checklist elements | category-level descriptions → individual misrecognition (Lucy lesson) |

- **Cost discipline**（§8.4）：total probe calls ≤ 30 — 10 (exam 1) + 10 (exam 2) +
  6 (exam 3's own probes: `infer`, `describe`, `is_public_knowledge`,
  `classify_followup`, `extract_feature`, `curiosity_query`) + 3 (exam 4) = **29**.
  Exam 3 validates formats of calls already made by exams 1–2 where possible;
  `perceive_image` returning `None` on a non-vision model marks exam 4 "skipped",
  never "failed".
- **Four-tier verdict**：**适配 ✅**（全部达标，默认阈值可用）/ **可用需调阈值 ⚠️**
  （任一未达标 → 转 §6.5 重标定）/ **仅适合常识问答**（exam 1 < 70%——禁个体识别场景）/
  **不建议 ❌**（格式遵从 catastrophic 失败或探测点根本不可用）。
  报告含认知层适配度评分与"是否建议云端兜底"的建议字段。
- **Read-only 铁律**：exam 全程只读 + 探测调用，**不修改任何记忆数据**——
  doctor 路径不得构造 MemoryModule / TempMemory / Session，只写
  `model_profile_*.json` 报告文件。

## 4. Hard rules（铁律，违反即返工）

1. `voiceofsoul/` stays **stdlib-only**; zero new third-party deps;
2. Probes only via the §8.2 contract — no adapter-private calls, no new probe methods
   (`curiosity_query` already exists from the curiosity task; it is exam-3's 9th text probe);
3. **Never auto-adjust parameters** — a verdict is advice; every threshold change still
   needs §6.5 + the master's decision (design §8.4 铁律);
4. Calibrated thresholds (design §9) untouched; the doctor only *measures*, it never
   *writes* anything but the report file;
5. `perceive_image == None` (non-vision model) → exam 4 "skipped", not "failed";
   missing golden images → exam 4 "skipped" with reason;
6. Any exam throwing is caught and recorded as `error` in that exam's detail —
   a broken exam never aborts the whole physical;
7. API key never logged (inherits cloud-adapter hygiene).

## 5. Non-goals（明确不做）

- No threshold re-calibration (§6.5 is a separate, master-gated process);
- No model routing / auto-switching — the report merely *carries* a
  "cloud_fallback_suggested" field for a future routing task;
- No changes to `cognition.py` / `session.py` chat paths; no web UI changes;
- No video exams (probe `perceive_video` stays untested until a video adapter exists);
- No design-document edits by the coding agent.

## 6. Acceptance criteria（验收标准）

1. `python -m voiceofsoul.exo doctor` on Ricky's machine (local Ollama, gemma4)
   completes and writes a real `model_profile_ollama_gemma4:latest.json`; transcript
   attached as evidence;
2. Mock suite `_test_doctor.py` (git-ignored `_test_*` convention), **zero network**:
   scripted fake base proves — report schema; exam-1 verdict boundary at exactly 70%;
   exam-2 calibration math; exam-3 parse-failure counting; exam-4 skipped vs failed on
   `None`; a crashing exam degrades to `error` without aborting; **exactly 29 probe
   calls counted** (asserted); report file persistence + `stats` surfacing;
3. Backends treated identically: `--backend openai` reaches the same four exams
   (mock evidence; real cloud optional);
4. `stats` shows the latest profile after a doctor run;
5. All existing suites still green (`_test_curiosity.py`, `_test_exo_cli.py`,
   `_test_openai_adapter.py`, `_test_temp_memory.py`, `_test_dormant.py`);
   `skills/` copy synced; `start.bat` untouched.

## 7. Open questions — RESOLVED by Kimi/Ricky（已拍板，按此实现）

1. Golden assets ship inside the repo (`voiceofsoul/golden/`), loaded relative to the
   package — one source of truth for every exam, cloud and local;
2. Exam 4's checklist coverage is computed by a deterministic keyword/synonym matcher over
   the §3.5 element list (no extra LLM call, no extra probe);
3. Exam sizes fixed at 10 / 5 / 6 / 3 = 29 probe calls (≤ 30 cloud budget, design §8.4);
4. Verdict thresholds exactly as design §8.4: 70% / 75% / 10% / ≥3 checklist elements;
5. `stats` surfaces only the latest profile's verdict + per-exam pass/fail, not the
   full report.
