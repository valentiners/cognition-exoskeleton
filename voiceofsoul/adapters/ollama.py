"""Ollama adapter: local base model plus local embeddinggemma-2 embedder.

- OllamaEmbedder   -> memory vectors (design §3.2: decoupled from the base model)
- OllamaBaseModel  -> the 9-probe base-model contract
- list_models / ensure_ollama -> Ollama transport helpers

Prompts and parsers live in prompts.py so this adapter and the OpenAI-compatible
adapter cannot drift. Stdlib only.
"""

from __future__ import annotations

import json
import math
import urllib.request

from . import prompts


def _post(host: str, path: str, payload: dict, timeout: float = 120.0) -> dict:
    req = urllib.request.Request(
        host.rstrip("/") + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get_json(host: str, path: str, timeout: float = 30.0) -> dict:
    with urllib.request.urlopen(host.rstrip("/") + path, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def ensure_ollama(host: str, timeout: float = 5.0) -> None:
    """Probe Ollama; raises OSError when unreachable (CLI maps to exit 2)."""
    _get_json(host, "/api/tags", timeout=timeout)


class OllamaEmbedder:
    """Independent external encoder (Ollama embedding model)."""

    def __init__(self, model: str = "embeddinggemma-2:latest",
                 host: str = "http://localhost:11434"):
        self.model = model
        self.host = host
        self.model_tag = f"ollama:{model}"

    def embed(self, text: str) -> list[float]:
        d = _post(self.host, "/api/embeddings",
                  {"model": self.model, "prompt": text})
        vec = [float(x) for x in d["embedding"]]
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    @staticmethod
    def cosine(a: list[float], b: list[float]) -> float:
        return sum(x * y for x, y in zip(a, b))


# Ollama /api/tags only lists names; /api/show capabilities classify them.
_MODEL_CACHE: dict[str, list] = {}


def list_models(host: str) -> dict:
    """List Ollama base models / encoders for the HTTP model picker."""
    bases, embeds = [], []
    for m in _get_json(host, "/api/tags").get("models", []):
        name = m["name"]
        caps = _MODEL_CACHE.get(name)
        if caps is None:
            try:
                req = urllib.request.Request(
                    host.rstrip("/") + "/api/show",
                    data=json.dumps({"name": name}).encode(),
                    headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=30) as resp:
                    caps = json.loads(resp.read().decode()).get("capabilities", [])
            except OSError:
                caps = []
            _MODEL_CACHE[name] = caps
        if "embedding" in caps:
            embeds.append(name)
        elif "completion" in caps:
            bases.append(name)
    return {"base": sorted(bases), "embedding": sorted(embeds)}


class OllamaBaseModel:
    """Base model is only called; weights are never modified."""

    def __init__(self, model: str = "gemma4:latest",
                 host: str = "http://localhost:11434", think: bool = False):
        self.model = model
        self.host = host
        self.think = think

    # ---------- transport ----------
    def _generate(self, prompt: str, timeout: float = 120.0) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "think": self.think,
            "options": {"temperature": 0.2},
        }
        d = _post(self.host, "/api/generate", payload, timeout=timeout)
        return (d.get("response") or "").strip()

    # ---------- 9 probe methods ----------
    def infer_confidence(self, text: str) -> float:
        try:
            raw = self._generate(prompts.confidence_prompt(text))
        except (OSError, KeyError):
            return 0.0
        return prompts.parse_confidence(raw)

    def infer(self, text: str) -> str:
        try:
            raw = self._generate(prompts.infer_prompt(text))
        except OSError:
            return "未知事物"
        return prompts.parse_infer(raw)

    def describe(self, name: str) -> str | None:
        """Legacy gate-2 probe: can the base model describe this name?"""
        try:
            raw = self._generate(prompts.describe_prompt(name))
        except OSError:
            return None
        return prompts.parse_describe(raw)

    def is_public_knowledge(self, name: str, context: str = "") -> bool | None:
        try:
            raw = self._generate(prompts.public_knowledge_prompt(name, context))
        except OSError:
            return None
        return prompts.parse_public_knowledge(raw)

    def classify_followup(self, context: str, utterance: str) -> tuple:
        try:
            raw = self._generate(prompts.followup_prompt(context, utterance),
                                 timeout=120)
        except OSError:
            raw = ""
        return prompts.parse_followup(raw, utterance)

    def extract_feature(self, context: str, utterance: str) -> tuple[str, str] | None:
        result = self.classify_followup(context, utterance)
        return (result[1], result[2]) if result[0] == "feature" else None

    def answer_question(self, question: str, context: str = "") -> str | None:
        try:
            raw = self._generate(prompts.answer_prompt(question, context),
                                 timeout=180)
        except OSError:
            return None
        return prompts.parse_answer(raw)

    def judge_same(self, desc_a: str, desc_b: str) -> bool | None:
        try:
            raw = self._generate(prompts.judge_same_prompt(desc_a, desc_b))
        except OSError:
            return None
        return prompts.parse_judge_same(raw)

    def perceive_image(self, image_b64: str, hint: str = "") -> str | None:
        prompt = prompts.image_prompt(hint)
        payload = {
            "model": self.model,
            "prompt": prompt,
            "images": [image_b64],
            "stream": False,
            "think": self.think,
            "options": {"temperature": 0.2},
        }
        try:
            ans = _post(self.host, "/api/generate", payload,
                        timeout=300).get("response") or None
        except (OSError, KeyError):
            return None
        if not ans:
            return None
        return prompts.clean_image_description(ans)
