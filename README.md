# cognition-exoskeleton（自认知外挂）

> A "cognitive soul" plug-in for frozen base models — 一款大模型的认知灵魂外挂.
> Without touching any weights, it gives any already-trained model a continuous-learning loop:
> **perceive → judge by itself first → ask the master only when truly unsure → name together → remember → reuse & self-correct.**
> （识别 → 基础模型先自主判断 → 真拿不准才求助主人 → 共同命名 → 记忆 → 复用与纠错）

**voiceofsoul** · primitive version (v0.1.0) · Apache-2.0 · by Ricky Li ([github.com/valentiners](https://github.com/valentiners))

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

External modules attached to a frozen base model (local Ollama, swappable):

- **Cognitive Module** — base model judges first: common knowledge passes straight through (no questions asked); only genuinely uncertain cases are escalated to the master. 基础模型先认知，外挂只做它做不了的。
- **Memory Module** — an append-only knowledge base: a main entry store (`entries`, for proper names / private knowledge) plus a **patch zone** (`patches`) that only stores incremental knowledge the base model lacks. （补丁区：基础模型已知概念只补粒度，绝不整库重复记忆。）
- **Temp Memory** — new features are cached for 7 days first; they graduate to permanent memory when the master says "remember it", or automatically after appearing on 3 different days within a week. （临时记忆 + 七天法则。）
- **Conversational by design** — no forced button choices; the master talks naturally ("她叫 Lucy" / "记住它" / "忘掉"), and free text answers every pending flow. 拟人化、无按钮、自由文本即回答。

```
new input ──→ unified description ──→ permanent memory hit (≥0.78 → recognized)
                      │
        ┌─────────────┼──────────────────────────────┐
        ▼             ▼                              ▼
    recognized   base model knows it?          truly unknown
    (memory)     ├─ unique proper name ──→ LLM identity judge → recognize / learn
                 ├─ none ──────────────→ DIRECT pass-through (no questions)
                 └─ unsure ────────────→ nearest neighbor + LLM judge → guess / unknown
```

## Core principles

- **The base model is never modified** — call-only, zero weight updates. 原模型只被调用，权重与源文件不被修改。
- **No duplicate memory** — common knowledge the base model already has never enters the store; only *patches* are attached to its known concepts. 防重复记忆。
- **Judge locally first** — the base model decides what it can (identity judgment replaces pop-up arbitration); the master's arbitration is reserved for what it genuinely can't. 能自己裁决的不打扰主人。
- **Two-layer memory** — temp cache (7-day rule) + permanent store (three-gate anti-duplication); nothing enters permanent memory without either the master's word or the 3-days-in-a-week rule. 双层记忆。
- **Conversational, humanized** — all replies in first-person natural language; mechanical labels stay in debug fields only. 拟人化对话。
- **Append-only** — rejected names, mistakes, audits: everything leaves a trail. 只追加、不覆盖。
- **Single-master design** — one master = supreme authority and the only trusted input source. 单主人原则。

## Quick start

Zero third-party dependencies (pure Python standard library), Python 3.9+.

### Real mode (local Ollama + browser)

```bash
# 1. start local Ollama with a vision+text model (default: gemma4) and an embedding model (default: embeddinggemma-2)
# 2. launch the test service (auto-opens the browser)
start.bat            # Windows; or: python server.py
```

Chat in the browser: describe things or send images. The exoskeleton recognizes what it knows, passes through common knowledge, caches features you supply, and remembers what you tell it to.

### Offline demo (no Ollama needed)

```bash
python demo.py
```

`demo.py` is an interactive CLI demo with a simulated base model (v3.3-era behaviors: y/n prompts). It uses its own `demo_memory.json` and never touches the real memory store.

## Design doc

The full design lives in [docs/外挂认知模块与记忆模块方案_v4.0.md](docs/外挂认知模块与记忆模块方案_v4.0.md) (bilingual Chinese/English) — the project's design "constitution". It consolidates everything through v3.4's browser-tested iterations: base-model-first perception, conversational interaction without buttons, two-layer memory with the 7-day rule, dormancy & layered retrieval, arbitration-timeout degradation, the video perception interface, and the OpenClaw skill path. (v3.4 and earlier versions are archived in the repo history.)

## Roadmap

Primitive version already ships (via local Ollama): base-model-first perception with LLM identity judgment, three-gate anti-duplication, two-layer memory (temp 7-day rule + permanent store), patch zone, conversational humanized UI (cat-paw send button, cancelable in-flight requests), model hot-switching with vector rebuild, image perception, CLI offline demo.

| Direction | Status | Design doc section |
|---|---|---|
| Library packaging (`pyproject.toml`, importable `voiceofsoul`) | ⏳ roadmap | §9.9 |
| OpenClaw skill-plugin integration (spec TBD) | ⏳ roadmap | §9.9 |
| Video / real-world perception interface | ⏳ reserved | §9.9 |
| Memory observability UI (inspect / intervene temp & permanent) | ⏳ roadmap | §9.9 |
| Async arbitration (suspend vs. degrade when master is away) | ⏳ open | §9.9 |
| Upgrade `demo.py` to v3.4 behaviors (temp memory, conversational) | ⏳ roadmap | §9.1/§9.2 |
| Lazy patch auditing (zero downtime on base-model swap) | ⏳ roadmap | 4.5 |
| Voice input | ⏳ open issues | §8 |

## Project layout

```
cognition-exoskeleton/
├── server.py              # minimal HTTP test service + session orchestration
├── start.bat              # Windows one-click launcher (exclusive-port guard)
├── demo.py                # offline CLI demo (v3.3-era; uses demo_memory.json)
├── voiceofsoul/           # core package (stdlib-only)
│   ├── cognition.py       # perception pipeline + three gates + learning
│   ├── memory.py          # permanent memory: entries + patches
│   ├── temp_memory.py     # temp memory + 7-day graduation rule
│   ├── embedding.py       # hashing baseline embedder (offline demo only)
│   └── adapters/ollama.py # the only file that talks to Ollama
├── web/index.html         # browser test UI (cat-paw button, staged images)
├── _test_temp_memory.py   # unit tests for the 7-day rule (mocked dates)
├── docs/                  # design doc (方案 v3.4, Chinese) — chapter 9 is authoritative
├── examples/              # recorded CLI sample session (v3.3-era)
├── AGENTS.md              # charter for coding agents (read first)
├── .github/ABOUT.md       # finalized GitHub About text (bilingual) & topics
└── LICENSE                # Apache-2.0, (c) 2026 Ricky Li
```

## License

Apache License 2.0, see [LICENSE](LICENSE).
