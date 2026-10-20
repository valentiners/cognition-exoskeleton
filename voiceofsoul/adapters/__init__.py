"""Base-model adapter layer.

Kernel code never imports a concrete adapter class. It asks this factory for a
base model by backend name; the embedder remains local Ollama in this task.
"""

from .ollama import (OllamaBaseModel, OllamaEmbedder, ensure_ollama,
                     list_models)
from .openai_compat import (ApiAuthError, ApiError, ApiHTTPError,
                            ApiUnreachableError, OpenAICompatBaseModel,
                            ensure_openai)

DEFAULT_BACKEND = "ollama"


def make_base_model(backend: str, *, model: str,
                    host: str, api_base: str | None = None,
                    api_key: str | None = None):
    """Create the base model for the selected transport.

    backend: "ollama" or "openai" (OpenAI-compatible chat completions).
    """
    name = (backend or DEFAULT_BACKEND).strip().lower()
    if name == "ollama":
        return OllamaBaseModel(model=model, host=host)
    if name == "openai":
        return OpenAICompatBaseModel(model=model, api_base=api_base or
                                     "https://api.openai.com/v1",
                                     api_key=api_key)
    raise ValueError(f"unknown backend: {backend!r} (expected ollama or openai)")


__all__ = [
    "OllamaBaseModel", "OllamaEmbedder", "ensure_ollama", "list_models",
    "OpenAICompatBaseModel", "ensure_openai",
    "ApiError", "ApiUnreachableError", "ApiAuthError", "ApiHTTPError",
    "make_base_model", "DEFAULT_BACKEND",
]
