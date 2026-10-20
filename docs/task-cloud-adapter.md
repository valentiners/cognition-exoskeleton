# Development Task: Cloud API Adapter (`voiceofsoul/adapters/openai_compat.py`)

> Task brief for the coding agent (OpenClaw). 开发任务书（给代码 agent 的工单）。
> Read first, in this order: `AGENTS.md` (charter & hard rules) →
> `docs/外挂认知模块与记忆模块方案_v4.0.md` (design constitution, bilingual) →
> `docs/task-exo-cli.md` (precedent task: the CLI entry this task must not break) →
> `skills/cognition-exoskeleton/SKILL.md` (the contract this task extends).

## 1. Goal

Let the exoskeleton run its **base model** on any OpenAI-compatible cloud API (OpenAI,
DeepSeek, Moonshot, vLLM, Ollama's own `/v1`, …) so the skill works on hosts **without a
local inference stack**:

```bash
# new flags on both existing entries (server.py and voiceofsoul.exo)
--backend {ollama,openai}     # default ollama; env COGSOUL_BACKEND
--api-base  https://api.deepseek.com/v1   # env OPENAI_BASE_URL
--api-key   sk-...                        # env OPENAI_API_KEY (env strongly preferred)
python -m voiceofsoul.exo --backend openai chat --text "<utterance>"
```

Scope is **base model only**. The embedder stays local (`embeddinggemma-2` via Ollama) —
this preserves every calibrated threshold untouched (§4 rule 5, design §9).

## 2. Architecture requirements（架构要求）

1. **New adapter, same contract.** `voiceofsoul/adapters/openai_compat.py` exposes
   `OpenAICompatBaseModel` implementing exactly the same 9 probe methods as
   `OllamaBaseModel`: `infer`, `infer_confidence`, `describe`, `is_public_knowledge`,
   `classify_followup`, `extract_feature`, `answer_question`, `judge_same`,
   `perceive_image`. The kernel (`cognition.py`, `session.py`) must not change.
2. **One transport, shared brain.** Extract the prompt templates and response parsers
   shared by both adapters into `voiceofsoul/adapters/prompts.py`; each adapter only
   differs in transport (Ollama `/api/generate` vs OpenAI `POST {base}/chat/completions`,
   stdlib `urllib` only). Never fork the prompts — drift between adapters would fork the
   cognition itself.
3. **Factory over hardcoding.** Add `make_base_model(backend, *, model, host, api_base,
   api_key)` to `voiceofsoul/adapters/__init__.py`; `server.py` and `exo.py` stop
   importing adapter classes directly and go through the factory.
4. **Vision is optional at runtime.** `perceive_image` sends a `data:` URL in an
   `image_url` content part; if the cloud model is not vision-capable, return `None`
   (the kernel already handles this gracefully — do not invent a new failure mode).
5. **Probing per backend.** `ensure_ollama` runs when `backend=ollama`. For
   `backend=openai` probe with a 1-token chat completion instead; the embedding probe
   (Ollama) always runs, because the embedder is local in this task.
6. **Kernel purity**: `voiceofsoul/` stays stdlib-only (`urllib`, `json`, `base64`).
   No SDK, no `pip install`. 零第三方依赖。

## 3. Configuration（配置）

| Input | Env var | Flag | Default |
|---|---|---|---|
| Backend | `COGSOUL_BACKEND` | `--backend` | `ollama` |
| OpenAI-compatible base URL | `OPENAI_BASE_URL` | `--api-base` | `https://api.openai.com/v1` |
| API key | `OPENAI_API_KEY` | `--api-key` | none (required when backend=openai) |
| Ollama host (embedder) | `OLLAMA_HOST` | `--ollama` | `http://localhost:11434` |
| Cloud model name | — | `--base-model` | `gemma4:latest` → document that cloud names differ |

Existing `--base-model` / `--embedding-model` semantics are unchanged; embedding model
selection remains Ollama-only.

## 4. Hard rules（铁律，违反即返工）

Same 8 as `task-exo-cli.md` §4, plus:

9. **The API key never appears in logs, error messages, session.json, or stdout/stderr
   payloads** — only ever read from env/flag into the adapter instance;
10. **Calibrated thresholds are for the local embedder and do not change** — a cloud
    embedder is explicitly out of scope (would trigger design §9 re-calibration);
11. Cloud adapter must behave deterministically: request `temperature: 0.2` (matching the
    Ollama adapter), no streaming, no client-side retries beyond the existing single
    attempt — fail loudly, the kernel's error paths already handle failure.

## 5. Error behavior（错误行为）

- Backend unreachable / connection error → exit code `2`, stderr JSON
  `{"error": "api_unreachable", "detail": "..."}` (ollama backend keeps the existing
  `ollama_unreachable` kind);
- HTTP 401/403 → exit code `2`, `{"error": "api_auth", "detail": "..."}`;
- HTTP 429/5xx → exit code `1`, traceback on stderr, memory files must remain valid JSON;
- Missing key with `--backend openai` → exit code `2`, usage on stderr;
- All other exit-code semantics unchanged (argparse misuse → 2, unexpected → 1).

## 6. Non-goals（明确不做）

- **No cloud embedder** — separate task; requires re-calibration per design §9 and a
  full-library re-embed (§3.3 mechanism already supports it);
- No streaming, no tool-calling, no retry/rate-limit policies;
- No changes to `cognition.py` / `session.py` / `memory.py` / `temp_memory.py`;
- No UI work; no changes to `web/index.html`;
- No new thresholds, no design-document edits. **If the design doc is unclear, stop and
  ask Kimi/Ricky — do not guess** (AGENTS.md discipline).

## 7. Acceptance criteria（验收标准）

1. **Zero regression**: with default `--backend ollama`, every existing suite passes
   unchanged — `_test_lucy.py`, `_test_dormant_e2e.py`, `_test_degrade.py`,
   `_test_context.py`, `_test_identity.py`, `_test_temp_memory.py`, `_test_dormant.py`;
2. **Mock-based cloud tests**: new `_test_openai_adapter.py` (git-ignored, `_test_*`
   convention) spins a stdlib stub HTTP server speaking `/v1/chat/completions` and
   verifies: all 9 probe methods parse correctly from canned responses; a full
   chat → pending → respond naming loop persists to `memory.json`; dead stub → exit 2
   with `api_unreachable`; 401 → exit 2 with `api_auth`; **no real API key needed**;
3. **Mixed mode works**: `--backend openai` with local Ollama embedder runs a real
   end-to-end Chinese conversation on Ricky's machine (manual evidence: transcript of
   one UNKNOWN → teach → recognize loop);
4. `skills/cognition-exoskeleton/SKILL.md` gains a short "Cloud backends" section
   (English, with the new flags); the bundled `skills/cognition-exoskeleton/voiceofsoul/`
   copy is synced with the main package — both copies must be byte-identical except
   line endings;
5. `start.bat` untouched and still exclusive.

## 8. Open questions — RESOLVED by Kimi/Ricky（已拍板，按此实现）

1. **Scope is base-model-only**; embedder stays local `embeddinggemma-2` → all §9
   thresholds carry over unchanged. Cloud embedder = future task;
2. Backend selection via `--backend` + `COGSOUL_BACKEND`; key via `OPENAI_API_KEY` env
   first — the flag exists for convenience but the help text must recommend the env var;
3. Prompt templates and parsers are **shared** in `adapters/prompts.py` — one cognitive
   behavior, two transports;
4. `/api/models` and `/api/switch` with `backend=openai`: may return an empty model list
   and current selection without crashing; embedding switching stays Ollama-only;
5. Error taxonomy: exit codes unchanged; new error kinds `api_unreachable` / `api_auth`.

---

## Appendix A. Review checklist（审阅清单 — for Kimi & Ricky）

- [ ] Contract fidelity: all 9 probe methods implemented, kernel files untouched (git diff)
- [ ] Prompt parity: `adapters/prompts.py` is the single source; no duplicated prompt text
- [ ] Hard rules 1–11 all hold; `voiceofsoul/` still stdlib-only; key never logged
- [ ] Calibrated parameters untouched (diff against design §9 table)
- [ ] Cloud path proven by mock-server test evidence (§7.2), not just code reading
- [ ] Error paths: dead-stub and 401 tests actually ran; memory files valid after failure
- [ ] Full existing test suites re-run and green under default ollama backend (§7.1)
- [ ] No design document was modified by the coding agent (it may only *ask*, never edit)
- [ ] Open questions (§8) were followed, not silently re-decided
- [ ] `skills/` bundled copy synced with `voiceofsoul/` (only line-ending deltas)
