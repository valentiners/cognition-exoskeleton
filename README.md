# cognition-exoskeleton（自认知外挂）

> A "cognitive soul" plug-in for frozen base models — 一款大模型的认知灵魂外挂.
> Without touching any weights, it gives any already-trained model a continuous-learning loop:
> **perceive → ask the master when unsure → name together → store → reuse & self-correct.**
> （识别 → 不确定就求助主人 → 共同命名 → 存库 → 复用与纠错）

**voiceofsoul** · primitive version (v0.1.0) · Apache-2.0

---

## What this is

Two external modules attached to a frozen base model:

- **Cognitive Module** — compares every new input against the base model's compressed knowledge and the memory store, decides: recognize, guess, arbitrate, or admit unknown.
- **Memory Module** — an append-only knowledge base: a main entry store (`entries`, for proper names / private knowledge) plus a **patch zone** (`patches`) that only stores incremental knowledge the base model lacks. （补丁区：基础模型已知概念只补粒度，绝不整库重复记忆。）

```
new input ──→ unified description ──→ memory search (>=70% → confirmed hit)
                      │
        ┌─────────────┼──────────────────┐
        ▼             ▼                  ▼
     >=70%       51%~70% gray zone   <51%
   confirmed     master arbitrates   vs. base model
      hit        MERGE/NEW/SUBCLASS  (>51% guess / <=51% admit unknown → naming)
```

## Core principles

- **The base model is never modified** — call-only, zero weight updates. 原模型只被调用，权重与源文件不被修改。
- **No duplicate memory** — common knowledge the base model already has never enters the store; only *patches* are attached to its known concepts. 防重复记忆。
- **Master arbitration is non-delegable** — at every fork (gray zone, naming, patch conflicts) the master decides. 主人仲裁不可让渡。
- **Append-only** — rejected names, mistakes, audits: everything leaves a trail. 只追加、不覆盖。
- **Single-master design** — one master = supreme authority and the only trusted input source. 单主人原则。

## Quick start

Zero dependencies (pure Python standard library), Python 3.9+:

```bash
git clone https://github.com/<your-org>/cognition-exoskeleton.git
cd cognition-exoskeleton
python demo.py
```

`demo.py` is an interactive CLI demo: you play the **master** （主人）, and a simulated base model (with a little built-in common knowledge) drives the full loop — guess → confirm → patch/entry → re-recognize → record mistakes. Memory persists to `memory.json` in the working directory.

### Session commands

| Input | Effect |
|---|---|
| any descriptive text | recognition flow (recognize / arbitrate / guess / unknown) |
| `mistake <entry_id> <mistook_as> [new_feature]`（中文等效：`错误 <ID> <错认为> [特征]`） | record a recognition mistake (append-only) |
| `stats`（`统计`） | memory store statistics |
| `exit` / `quit`（`退出`） | end session |

## Example

See [examples/sample_session.md](examples/sample_session.md) for a recorded run with annotations mapping each step to the design doc.

## Design doc

The full design lives in [docs/外挂认知模块与记忆模块方案_v3.4.md](docs/外挂认知模块与记忆模块方案_v3.4.md) (Chinese) — two-tier thresholds & gray-zone arbitration, dual-channel multimodal feature extraction, dual-gate anti-duplication, the patch zone, zero-downtime patch auditing on base-model swap, and health-based forgetting. 设计文档为中文，是本项目的设计"宪法"。

## Roadmap

Primitive version already ships: two-tier thresholds + master arbitration, dual-gate anti-duplication, patch zone (`feature_addition` / `correction`), append-only storage, JSON persistence, CLI demo.

| Direction | Status | Design doc section |
|---|---|---|
| Lazy patch auditing (zero downtime on base-model swap) | ⏳ roadmap | 4.5 |
| Inter-patch conflict arbitration (`priority`) | ⏳ roadmap | 4.5.6 |
| REDUNDANT perturbation stability test | ⏳ roadmap | 4.5.3 |
| Per-model threshold calibration buckets | ⏳ roadmap | 5.3 |
| Real embedding model backend (current hashing baseline is offline-demo only) | ⏳ roadmap | 3.2 |
| Image / multimodal input | ⏳ roadmap | §3 |
| Voice input | ⏳ roadmap | open issues |

## Project layout

```
cognition-exoskeleton/
├── demo.py                # interactive CLI demo (you are the master)
├── voiceofsoul/           # core package
│   ├── embedding.py       # hashing baseline embedder (zero-dep; swappable for real models)
│   ├── memory.py          # memory module: entries + patches, JSON persistence
│   └── cognition.py       # cognitive module: perceive / arbitrate / learn / record mistakes
├── docs/                  # design doc (方案 v3.4, Chinese)
├── examples/              # recorded sample session
├── .github/ABOUT.md       # finalized GitHub About text (bilingual) & topics
└── LICENSE                # Apache-2.0, (c) 2026 Ricky Li
```

## License

Apache License 2.0, see [LICENSE](LICENSE).
