# Draft — GitHub Release v0.2.0

## 标题（Title）

**v0.2.0 — the model gets a life of its own** · 认知外挂的第一个"活着"的版本

> 备选：`v0.2.0 — Curiosity, Self-Checkup & First Growth Journal`

---

## 正文（Body，English 主述 + 中文点睛）

**What if your local model companion could learn on its own — without retraining a single weight?**

Cognition Exoskeleton is a drop-in "cognitive soul" layer for any LLM. It never touches the base model's weights; instead it wraps it with a cognition module and a memory module, so a frozen model can keep learning from daily life — recognizing faces, remembering names, and (as of this release) getting curious on its own.

`v0.2.0` turns the exoskeleton from "a memory that never forgets" into **a model that has a life of its own**: it wonders, it self-checks, and it writes its own growth journal while you sleep.
（从"不遗忘的记忆"进化为"有自己生活的模型"：它会好奇、会体检、还会在你睡着时写日记。）

## What's New

### 🔍 Curiosity Module — it starts asking questions on its own
触发源 T1–T4（主人展示、主人提及、回答缺口、聊天漂移），L0 先查现有记忆再问，L1/L2 提问分级，自主额度三档防骚扰，好奇队列 7 天 TTL。T3/T4 默认关闭（`COGSOUL_CURIOUS_T34=1` 开启）。
The model no longer waits to be taught. When something in the conversation doesn't add up — an unexplained object, a name it half-remembers — it can now ask *you* first, at a polite, throttled pace.

### 🌐 Autonomous Web Retrieval — learn what the owner never told it
stdlib-only web search (DuckDuckGo by default, SearXNG pluggable). Hard discipline baked in: knowledge from the web is downgraded one tier, always carries its source URL, and the owner's word always wins. 来源纪律：网络知识降一档 + 必附出处 + 主人永远优先。

### 🩺 Model Physical (`exo doctor`) — can the model judge itself?
A four-exam self-checkup with a golden set (10 same-person pairs + 5 confidence probes + 3 AI-generated test images), capped at exactly 29 probe calls — one under the cloud API budget. Real-world result on `ollama gemma4`:
`judge 0.8 · calibration 0.8 · format 1.0 · description 0.75 → NEEDS_TUNING ⚠️ (fitness 0.83)`

The most revealing finding: the **description-quality** exam caught gemma4 *missing a small scar and thick eyebrows* in the test photo, drifting to category-level descriptions ("an anime girl") instead of identity-level ones. That is — quantified — the exact root cause of the famous **Lucy misidentification** from our earliest field tests (§2.1). The model failed the test in precisely the way we had observed it failing us in real life. 体检抓到的正是当年 Lucy 误认的量化根源。

### 😴 Sleep Consolidation (`exo consolidate`) — growth journal, written while you sleep
A quiet-hours process that collects the day's novelties, composes a **growth journal** (probe #11), and writes it back to OpenClaw's `MEMORY.md` (cross-referenced, never double-written, §1.6). At most one entry per day; failure counts as "tried", never nags. 每天最多一篇，失败也安静退场。

**Real end-to-end test** — I taught it "that weird thing is called 噗噜噗噜", gave it the name, and let gemma4 write its first journal entry:

> "今天好像认识了一个新朋友，叫"噗噜噗噜"。说实话，我不太确定它到底是个什么东西，但它给我的感觉挺新奇的……"

A frozen, 9-billion-parameter model — writing about its day like a kid coming home from school. That's the whole point of this project. 冻结的模型，写出了放学回家般的日记。

## Stories from the field（实测花絮）

- **The T2 dead path.** With `CURIOUS_TH=0.35`, trigger source T2 *could never fire*: GUESS only appears at conf ≥ 0.51 or memory-match ≥ 0.70. We caught it in review, and Ricky ruled: raise the threshold to **0.60** (placeholder until §6.5 calibration). A classic case of a "safe-looking default" that silently disabled a feature. 阈值看着保守，实则把功能焊死了。
- **Two real bugs found by the physical exam itself:** the answer verifier choked on LaTeX (`$\text{h}_2\text{o}$`) and thousands separators — fixed with `_keynorm`; and the fitness score could be pushed outside 0–1 by group counts — now normalized by coverage.

## Try it

```bash
exo doctor                # self-checkup your model → model_profile_*.json
exo consolidate           # manually trigger a growth journal entry
exo consolidate --force   # ignore the once-a-day idempotency guard
exo chat                # T1–T4 curiosity + web retrieval in conversation
```

Cloud APIs (OpenAI-compatible) work out of the box: `--backend openai --api-base ... --api-key ...`.

## Thanks

Designed with Ricky Li. Built in collaboration between humans and AI — Ricky's ideas, Kimi's implementation, one very patient local gemma4. 署名 Ricky Li。

---

**Compare:** [v0.1.0...v0.2.0](https://github.com/valentiners/cognition-exoskeleton/compare/v0.1.0...v0.2.0)
