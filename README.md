# cognition-exoskeleton（自认知外挂）

> A "cognitive soul" plug-in for frozen base models — 一款大模型的认知灵魂外挂.
> Without touching any weights, it gives any already-trained model a continuous-learning loop:
> **perceive → judge by itself first → ask the master only when truly unsure → name together → remember → reuse & self-correct.**
> （识别 → 基础模型先自主判断 → 真拿不准才求助主人 → 共同命名 → 记忆 → 复用与纠错）

**<img width="420"  alt="自认知进化宣传图 (1)" src="https://github.com/user-attachments/assets/066c9991-57a0-4c7c-9b84-5a71284d58ef" />

voiceofsoul** · v0.3.0 · Apache-2.0 · by Ricky Li ([github.com/valentiners](https://github.com/valentiners))

---

## Why this exists

What if the model living on your own computer could learn about the world the way we do —
noticing new things on its own, remembering the small moments of your daily life, growing
alongside you like a real companion?

Retraining a model from scratch is slow, complex, and expensive. So I kept asking myself:
could a simple exoskeleton give *any* already-trained model basic cognition and self-learning —
no retraining, no weights touched?

That question became this project: **自认知进化** — cognitive evolution.
（让本地模型伙伴像人一样认知、记忆、陪伴——不靠重训，靠外挂。）

---

## What this is

External modules attached to a frozen base model — local Ollama **or any OpenAI-compatible
cloud API** (DeepSeek, Moonshot, OpenAI…), swappable at runtime:

- **Cognitive Module** — the base model judges first: common knowledge passes straight through
  (no questions asked); only genuinely uncertain cases escalate to the master.
  基础模型先认知，外挂只做它做不了的。
- **Memory Module** — an append-only knowledge base: a main entry store (`entries`, for proper
  names / private knowledge) plus a **patch zone** (`patches`) that only stores incremental
  knowledge the base model lacks. （补丁区：基础模型已知概念只补粒度，绝不整库重复记忆。）
- **Temp Memory** — new features are cached for 7 days first; they graduate to permanent memory
  when the master says "remember it", or automatically after appearing on 3 different days
  within a week. （临时记忆 + 七天法则。）
- **Layered retrieval & dormancy** — high-frequency / active / dormant layers; memories unused
  for a year fall asleep and are only recalled when the master says "think again".
  （分层检索 + 休眠记忆。）

### New in v0.3.0 — a cognitive architecture

Born from a midnight philosophy question (*"human thinking relies not only on memory, but also
intuition, and instincts like survival"*), nine waves (W0–W8) landed in a single day, each
field-tested against a real local gemma4. The doctrine (design doc §1.7):

> **Instinct gives direction. Intuition gives speed. Memory gives detail. Backup gives immortality.**
> （本能给方向，直觉给速度，记忆给细节，备份给永生。）

- **Three-layer architecture** — instinct acts without asking; intuition takes effect only
  after consultation (every intuition card carries a provenance "birth certificate"); memory is
  the archive. （三层认知架构：本能/直觉/记忆，直觉必须带出生证明。）
- **Embedder physical (doctor exam #5)** — the hippocampus gets checked too: zero LLM calls,
  ≤30 embeddings. Field finding: `embeddinggemma-2` separates identities with a thin margin
  (+0.0721) — near-conflicts are normal terrain, and W3 was designed around exactly that.
- **L1 novelty sense** — before storing anything new, three paraphrased probes decide whether
  the base model already knows it; private entities are never probed. Bias: false negatives
  hurt more than false positives — when in doubt, remember anyway. （知识边界探测。）
- **L2 memory web & belief revision** — every entry carries a Top-5 neighbor list at birth;
  near-conflicts route through a two-tier detector (soft doubts accumulate silently in debug,
  hard conflicts ask politely). Corrections settle through a **vote ledger**: entries retire
  into `superseded` status — archive kept, one-command revive. MERGE-only history never breaks.
  Field story: a photo scored Lucy 0.8018 / her cousin 0.7895 → the model asked *which one it
  was* — the historical misidentification became a question, not a bug.
- **Knowledge-boundary report (`survey`)** — batch-probe the topics that actually appear in
  your logs (read-only, cite-don't-copy): 🟢 already knows / 🟡 half-knows (with sampled
  evidence) / 🔵 total blank. The first real report flagged "three-layer cognitive
  architecture" as the model's very first blank — it doesn't know the architecture it lives on.
- **Hypothesis-driven curiosity** — a hypothesis is *knowledge that hasn't voted yet*: a
  falsifiable statement, mandatory evidence, a verification plan (web / master / wait), and a
  lifecycle that settles through the vote ledger. Unverified hypotheses expire in silence —
  no nagging, ever. （假设 = 尚未投票的知识。）
- **L3 foresight** — while you sleep, new memories get projected: applicability tags and risk
  tags, generated offline in the sleep process, consumed online by pure vector matching (zero
  LLM in the conversation loop). Loss-aversion rule: a 🟥 severe hit (life/privacy/finance)
  warns first and asks for a free-text confirmation before irreversible actions; web-sourced
  risks cap at 🟨 — no AI fear-mongering. Recurring dangers graduate into instinct rules:
  **the model proposes, the master legislates.** （前瞻推演：离线生成，在线零成本消费。）
- **Weekly review & teaching loop** — every week, quietly: an incremental boundary report, and
  🟡 topics become polite teach-me requests. The master teaches → temp memory → the 7-day
  promotion pipeline. It knows what it doesn't know, and asks to be taught.
- **Soul backup (`restore`)** — daily atomic backup of the entire soul (memory, temp, queue,
  journal, instinct rules) with SHA-256 verification and 7-daily + monthly rotation.
  Free self-preservation runs automatically; anything costing the master asks first; off-site
  push is opt-in only. （灵魂备份：备份的是灵魂全部，恢复可进可退。）

Milestone write-up (中文): [docs/项目里程碑_2026-11-15.md](docs/项目里程碑_2026-11-15.md)

### v0.2.0 — the model gets a life of its own

- **Curiosity** — prediction error is a learning signal: unknown things, shaky guesses,
  perception contradictions and unknown proper nouns become questions. Tiers + an autonomy
  budget keep it polite; unanswered questions queue up instead of nagging.
  （好奇心模块：预测误差即学习信号，提问分级 + 自主额度，绝不连环问。）
- **Self-retrieval channel** — *search before asking*: every curiosity point is first looked up
  on the web (pluggable backend, stdlib-only); a credible, source-cited answer is self-learned
  through the same three gates — the master is never interrupted when the web can teach it.
  Web-learned knowledge is derated one tier and the master always wins on conflict. §4.9
- **Model physical (`doctor`)** — attach any base model (cloud or local, treated identically)
  and run a four-exam golden test set first: identity-arbitration accuracy, confidence honesty,
  format compliance, identity-level description quality. A verdict (**FIT / NEEDS_TUNING /
  COMMON_QA_ONLY / NOT_RECOMMENDED**) is advice only — it never adjusts a parameter.
  （模型体检：先体检再上岗，结论只给建议。）
- **Sleep process (`consolidate`)** — once a day in quiet hours the exoskeleton digests its own
  day — what it learned, wondered, and got wrong — into a first-person *Growth Journal*
  appended to OpenClaw's `MEMORY.md`: the cognitive layer feeding the episodic layer.
  （睡眠进程：每天一篇成长日记，认知层反哺情节层。）

```
new input ──→ unified description ──→ permanent memory hit (≥0.78 → recognized)
                      │
        ┌─────────────┼──────────────────────────────┐
        ▼             ▼                              ▼
    recognized   base model knows it?          truly unknown
    (memory)     ├─ unique proper name ──→ LLM identity judge → recognize / learn
                 ├─ none ──────────────→ DIRECT pass-through (no questions)
                 └─ unsure ────────────→ nearest neighbor + LLM judge → guess / unknown
                                     ↘ curious? → web_search → self-learn / ask politely
```

## Core principles

- **The base model is never modified** — call-only, zero weight updates. 原模型只被调用，权重与源文件不被修改。
- **No duplicate memory** — common knowledge the base model already has never enters the store; only *patches* are attached to its known concepts. 防重复记忆。
- **Judge locally first** — the base model decides what it can (identity judgment replaces pop-up arbitration); the master's arbitration is reserved for what it genuinely can't. 能自己裁决的不打扰主人。
- **Two-layer memory** — temp cache (7-day rule) + permanent store (three-gate anti-duplication); nothing enters permanent memory without either the master's word or the 3-days-in-a-week rule. 双层记忆。
- **Conversational, humanized** — all replies in first-person natural language; mechanical labels stay in debug fields only; no forced buttons anywhere. 拟人化对话。
- **Append-only** — rejected names, mistakes, audits: everything leaves a trail. 只追加、不覆盖。
- **Single-master design** — one master = supreme authority and the only trusted input source. 单主人原则。
- **Zero third-party dependencies** — the kernel is pure Python stdlib; cloud and local backends ride one shared probe contract. 零依赖，云端本地一视同仁。

## Quick start

Zero third-party dependencies (pure Python standard library), Python 3.9+.

### Install as an OpenClaw skill (ClawHub)

```bash
openclaw skills install @valentiners/cognition-exoskeleton
```

The published folder is self-contained (kernel + golden test set) and MIT-0; the source
repository below stays Apache-2.0. After installing, `openclaw skills info
cognition-exoskeleton` should report **Ready** and *visible to model*.

### Real mode (local Ollama + browser)

```bash
# 1. start local Ollama with a vision+text model (default: gemma4) and an embedding model (default: embeddinggemma-2)
# 2. launch the test service (auto-opens the browser)
start.bat            # Windows; or: python server.py
```

Chat in the browser: describe things or send images. The exoskeleton recognizes what it knows,
passes through common knowledge, caches features you supply, and remembers what you tell it to.

### CLI (same brain, scriptable)

```bash
python -m voiceofsoul.exo chat --text "她叫 Lucy，一只橘猫"   # perceive / learn
python -m voiceofsoul.exo respond --text "记住它"              # answer a pending question
python -m voiceofsoul.exo curiosity list                     # what is it wondering about?
python -m voiceofsoul.exo curiosity mode quiet                # quiet | normal | eager
python -m voiceofsoul.exo doctor                             # physical exam for the base model (now 5 exams, incl. zero-LLM embedder check)
python -m voiceofsoul.exo consolidate                          # write today's Growth Journal (+ weekly boundary review)
python -m voiceofsoul.exo survey                             # knowledge-boundary report (estimates first; rerun with --yes)
python -m voiceofsoul.exo restore --list                     # the soul's backup shelf (--tag <ts> to restore)
python -m voiceofsoul.exo stats                              # memory + curiosity + exam summary
```

Cloud base model instead of local Ollama:
`--backend openai --api-base https://api.deepseek.com/v1` + `OPENAI_API_KEY`
(the local embedder stays on Ollama — all calibrated thresholds carry over).

### Offline demo (no Ollama needed)

```bash
python demo.py
```

`demo.py` is an interactive CLI demo with a simulated base model (v3.3-era behaviors: y/n
prompts). It uses its own `demo_memory.json` and never touches the real memory store.

## Design doc

The full design lives in
[docs/外挂认知模块与记忆模块方案_v5.0.md](docs/外挂认知模块与记忆模块方案_v5.0.md)
(bilingual Chinese/English) — the project's design "constitution". v5.0 (2026-11-14)
repositions the exoskeleton as the **cognitive layer complementing OpenClaw's official
memory** (§1.6: the official system records, this exoskeleton *recognizes*), and adds
proactive learning: curiosity, self-retrieval, model physical, and the sleep process.
v5.1 additions (2026-11-15, §1.7) formalize the three-layer cognitive architecture
(instinct / intuition / memory), the L1–L3 intuition pipeline, Instinct #0 (protect the
master) / Instinct #1 (soul backup), and the embedding local-first/cloud-fallback policy.
Earlier versions are archived in `docs/` and the repo history.

## Roadmap

| Direction                                                            | Status        | Design doc |
| -------------------------------------------------------------------- | ------------- | ---------- |
| Curiosity module (triggers, tiers, autonomy budget, queue)           | ✅ v0.2.0      | §4.6–§4.8  |
| Self-retrieval learning channel (search before asking)             | ✅ v0.2.0      | §4.9 / §8.3 |
| Model physical (`doctor`, four-exam golden test set)               | ✅ v0.2.0      | §8.4       |
| Consolidation scheduler ("sleep process", Growth Journal)            | ✅ v0.2.0      | §10.3-#4   |
| OpenAI-compatible cloud adapter                                      | ✅ v0.2.0      | §8.2       |
| Three-layer cognitive architecture (instinct / intuition / memory) | ✅ v0.3.0      | §1.7       |
| Embedder physical (doctor exam #5, zero-LLM separation margin)     | ✅ v0.3.0      | §8.4       |
| Knowledge-boundary probe (L1 novelty sense, 30-day cache)          | ✅ v0.3.0      | §8.4       |
| Memory web: neighbor lists + vote-ledger belief revision            | ✅ v0.3.0      | §6.1 / §6.4 |
| Knowledge survey (`survey`, G1) + weekly review (G2) + teaching loop (G3) | ✅ v0.3.0 | §4.8 / §8.4 |
| Hypothesis-driven curiosity (birth certificates, silent expiry)    | ✅ v0.3.0      | §4.8       |
| Foresight (L3 offline tags, risk tiers, instinct proposals)        | ✅ v0.3.0      | §1.7 / §6.1 |
| Soul backup & restore (atomic, verified, revivable)                | ✅ v0.3.0      | §1.7 / §10.3 |
| Conflict-band & foresight-threshold calibration (placeholders now) | ⏳ pending §6.5 | §6.5      |
| Library packaging (`pyproject.toml`, kernel/service decoupling)        | ⏳ roadmap     | §10.3-#5   |
| OpenClaw skill publish (MIT-0 on ClawHub; v0.3.0 submitted 2026-11-23, security scan pending) | 🟡 submitted | §10.2      |
| Video / real-world perception interface                              | ⏳ reserved    | §3.6       |
| Memory observability UI (stats already expose dormancy & curiosity)  | ⏳ roadmap     | §10.3-#7   |
| T3/T4 curiosity triggers calibration (currently off by default)      | ⏳ pending §6.5 | §4.6      |
| Voice input                                                          | ⏳ open        | §12        |

## Project layout

```
cognition-exoskeleton/
├── server.py                 # minimal HTTP test service + session orchestration
├── start.bat                 # Windows one-click launcher (exclusive-port guard)
├── demo.py                   # offline CLI demo (v3.3-era; uses demo_memory.json)
├── voiceofsoul/              # core package (stdlib-only)
│   ├── cognition.py          # perception pipeline + three gates + learning/correction
│   ├── memory.py             # permanent memory: entries + patches, layered retrieval
│   ├── temp_memory.py        # temp memory + 7-day graduation rule
│   ├── curiosity.py          # curiosity queue (7-day TTL, wake-up, mute)
│   ├── doctor.py             # model physical: five exams + verdict (exam #5 = embedder, zero LLM)
│   ├── growth.py             # sleep process: digest → Growth Journal (+ weekly boundary review)
│   ├── knowledge.py          # L1 novelty sense: three-probe consistency judge + 30-day cache
│   ├── conflict.py           # near-conflict classifier (hard / soft / silent)
│   ├── hypothesis.py         # hypothesis objects: lifecycle + vote-ledger settlement
│   ├── foresight.py          # L3 tags: offline generation, online zero-LLM matching, instinct proposals
│   ├── survey.py             # knowledge-boundary reports (G1 manual / G2 weekly / G3 teaching loop)
│   ├── backup.py             # soul backup: atomic write, SHA-256 verify, rotation, restore
│   ├── session.py            # chat orchestration (arbitration, curiosity, conflict, consolidation)
│   ├── exo.py                # CLI entry (chat/respond/curiosity/doctor/consolidate/survey/restore/stats)
│   ├── tools/web.py          # the only file touching a search backend (pluggable)
│   ├── golden/               # doctor's golden test set (pairs, calibration, images)
│   └── adapters/             # ollama.py / openai_compat.py (one shared probe contract)
├── web/index.html            # browser test UI (cat-paw button, staged images)
├── skills/cognition-exoskeleton/  # OpenClaw skill bundle (SKILL.md + kernel copy)
├── docs/                     # design constitution v5.0 (bilingual) + task briefs
├── AGENTS.md                 # charter for coding agents (read first)
├── .github/ABOUT.md          # finalized GitHub About text (bilingual) & topics
└── LICENSE                   # Apache-2.0, (c) 2026 Ricky Li
```

## License

Apache License 2.0, see [LICENSE](LICENSE).
（OpenClaw ClawHub 发布的 skill 为 MIT-0，署名 Ricky Li 见 `homepage` 声明。）
