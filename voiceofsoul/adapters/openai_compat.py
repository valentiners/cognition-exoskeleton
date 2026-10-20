"""OpenAI-compatible chat-completions base-model adapter.

Transport only: every prompt and parser comes from
voiceofsoul/adapters/prompts.py, exactly like the Ollama adapter. The
embedder is intentionally not here — this task keeps the local Ollama
embeddinggemma-2 so the calibrated thresholds in design §9 stay valid.

Stdlib only: urllib, json, base64.
"""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request

from . import prompts

DEFAULT_API_BASE = "https://api.openai.com/v1"


class ApiError(RuntimeError):
    """Base class for adapter transport failures."""


class ApiUnreachableError(ApiError):
    """Connection refused / DNS / timeout: the endpoint could not answer."""


class ApiAuthError(ApiError):
    """HTTP 401/403: missing or rejected credentials."""


class ApiHTTPError(ApiError):
    """Any other non-2xx response (429/5xx should surface as exit 1)."""

    def __init__(self, status: int, detail: str = ""):
        self.status = status
        message = f"HTTP {status}"
        if detail:
            message += f": {detail}"
        super().__init__(message)


def _redact(text: str, api_key: str | None) -> str:
    if api_key and api_key in text:
        return text.replace(api_key, "***")
    return text


def _post_json(url: str, payload: dict, api_key: str | None,
               timeout: float) -> dict:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = _redact(exc.read().decode("utf-8", errors="replace")[:300],
                             api_key)
        except OSError:
            pass
        if exc.code in (401, 403):
            # Never echo the provider's auth body: it can contain key fragments.
            raise ApiAuthError(f"HTTP {exc.code}") from None
        raise ApiHTTPError(exc.code, detail) from None
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ApiUnreachableError(_redact(str(exc), api_key)) from None


def _guess_image_mime(image_b64: str) -> str:
    try:
        head = base64.b64decode(image_b64[:64], validate=False)
    except (ValueError, TypeError):
        return "image/jpeg"
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if head.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if head.startswith(b"GIF87a") or head.startswith(b"GIF89a"):
        return "image/gif"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"


class OpenAICompatBaseModel:
    """Base model over POST {api_base}/chat/completions."""

    def __init__(self, model: str = "gemma4:latest",
                 api_base: str = DEFAULT_API_BASE,
                 api_key: str | None = None):
        self.model = model
        self.api_base = (api_base or DEFAULT_API_BASE).rstrip("/")
        self.api_key = api_key
        self.endpoint = f"{self.api_base}/chat/completions"

    # ---------- transport ----------
    def _chat(self, content, timeout: float = 120.0,
              max_tokens: int | None = None) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.2,
            "stream": False,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        data = _post_json(self.endpoint, payload, self.api_key, timeout)
        choices = data.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        raw = message.get("content", "")
        if isinstance(raw, list):
            parts = []
            for part in raw:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    parts.append(part["text"])
            raw = "".join(parts)
        return str(raw or "").strip()

    def _text(self, prompt: str, timeout: float = 120.0) -> str:
        return self._chat(prompt, timeout=timeout)

    # ---------- 9 probe methods (same contract as OllamaBaseModel) ----------
    def infer_confidence(self, text: str) -> float:
        raw = self._text(prompts.confidence_prompt(text))
        return prompts.parse_confidence(raw)

    def infer(self, text: str) -> str:
        raw = self._text(prompts.infer_prompt(text))
        return prompts.parse_infer(raw)

    def describe(self, name: str) -> str | None:
        raw = self._text(prompts.describe_prompt(name))
        return prompts.parse_describe(raw)

    def is_public_knowledge(self, name: str, context: str = "") -> bool | None:
        raw = self._text(prompts.public_knowledge_prompt(name, context))
        return prompts.parse_public_knowledge(raw)

    def classify_followup(self, context: str, utterance: str) -> tuple:
        raw = self._text(prompts.followup_prompt(context, utterance))
        return prompts.parse_followup(raw, utterance)

    def extract_feature(self, context: str, utterance: str) -> tuple[str, str] | None:
        result = self.classify_followup(context, utterance)
        return (result[1], result[2]) if result[0] == "feature" else None

    def answer_question(self, question: str, context: str = "") -> str | None:
        raw = self._text(prompts.answer_prompt(question, context), timeout=180)
        return prompts.parse_answer(raw)

    def judge_same(self, desc_a: str, desc_b: str) -> bool | None:
        raw = self._text(prompts.judge_same_prompt(desc_a, desc_b))
        return prompts.parse_judge_same(raw)

    def perceive_image(self, image_b64: str, hint: str = "") -> str | None:
        if image_b64.startswith("data:"):
            image_url = image_b64
        else:
            mime = _guess_image_mime(image_b64)
            image_url = f"data:{mime};base64,{image_b64}"
        content = [
            {"type": "text", "text": prompts.image_prompt(hint)},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]
        try:
            raw = self._chat(content, timeout=300)
        except ApiHTTPError as exc:
            # A text-only model commonly rejects the image part with 400/422.
            # The kernel already has a graceful "I cannot see it" path.
            if exc.status in (400, 422):
                return None
            raise
        if not raw:
            return None
        return prompts.clean_image_description(raw)


def ensure_openai(api_base: str = DEFAULT_API_BASE,
                  api_key: str | None = None,
                  model: str = "gemma4:latest",
                  timeout: float = 5.0) -> None:
    """Probe a 1-token chat completion; errors are classified for the CLI."""
    base = (api_base or DEFAULT_API_BASE).rstrip("/")
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "ping"}],
        "temperature": 0.2,
        "stream": False,
        "max_tokens": 1,
    }
    _post_json(f"{base}/chat/completions", payload, api_key, timeout)


__all__ = [
    "ApiError", "ApiUnreachableError", "ApiAuthError", "ApiHTTPError",
    "OpenAICompatBaseModel", "ensure_openai", "DEFAULT_API_BASE",
]
