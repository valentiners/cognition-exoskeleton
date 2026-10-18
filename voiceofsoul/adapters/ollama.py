"""Ollama 适配器：本地 Ollama 服务同时提供基础模型与独立编码器。

- OllamaEmbedder   → 记忆模块的向量来源（设计文档 3.2：与基础模型解耦）
- OllamaBaseModel  → 基础模型探测点：infer / infer_confidence / describe

只依赖标准库（urllib），无第三方依赖。
"""

from __future__ import annotations

import json
import math
import urllib.request


def _post(host: str, path: str, payload: dict, timeout: float = 120.0) -> dict:
    req = urllib.request.Request(
        host.rstrip("/") + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


class OllamaEmbedder:
    """独立外部编码器（经由 Ollama 的 embedding 模型，如 mxbai-embed-large）。"""

    def __init__(self, model: str = "mxbai-embed-large",
                 host: str = "http://localhost:11434"):
        self.model = model
        self.host = host
        self.model_tag = f"ollama:{model}"

    def embed(self, text: str) -> list[float]:
        d = _post(self.host, "/api/embeddings", {"model": self.model, "prompt": text})
        vec = [float(x) for x in d["embedding"]]
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    @staticmethod
    def cosine(a: list[float], b: list[float]) -> float:
        return sum(x * y for x, y in zip(a, b))


class OllamaBaseModel:
    """基础模型只被调用，权重不被修改（设计文档核心原则 1）。

    三个探测点对应设计文档中的"基础模型压缩知识"：
    - infer(text)            最可能的通名猜测
    - infer_confidence(text) 识别把握程度 0~1
    - describe(name)         零样本完整描述；不认识返回 None（闸口 2 的依据）
    """

    def __init__(self, model: str = "Qwen3.8:latest",
                 host: str = "http://localhost:11434", think: bool = False):
        self.model = model
        self.host = host
        self.think = think  # 关闭 thinking，探测类 prompt 要快

    # ---------- 内部 ----------
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

    # ---------- 探测点 ----------
    def infer_confidence(self, text: str) -> float:
        prompt = (
            "请只输出一个 0 到 1 之间的小数（例如 0.72），表示你对识别以下事物的把握程度。\n"
            "如果你完全不认识它，输出 0。不要输出任何其他文字。\n"
            f"描述：{text}"
        )
        try:
            raw = self._generate(prompt).replace("%", "").strip()
            return min(max(float(raw), 0.0), 1.0)
        except (ValueError, KeyError, OSError):
            return 0.0

    def infer(self, text: str) -> str:
        prompt = (
            "以下是对某个事物的描述。请只输出一个最可能的通用名称（2 到 6 个字），"
            "例如「猫」「汽车」。如果你不认识，只输出「未知」。不要输出任何其他文字。\n"
            f"描述：{text}"
        )
        try:
            ans = self._generate(prompt)
            return ans if ans and ans != "未知" else "未知事物"
        except OSError:
            return "未知事物"

    def describe(self, name: str) -> str | None:
        """闸口 2 自检：基础模型能否零样本给出完整、稳定的描述。"""
        prompt = (
            f"请用一两句话描述「{name}」是什么。如果你完全不认识这个词，"
            f"只输出「不认识」。不要输出任何其他文字。"
        )
        try:
            ans = self._generate(prompt)
        except OSError:
            return None
        if any(k in ans for k in ("不认识", "无法", "不知道", "不确定")) or not ans:
            return None
        return ans
