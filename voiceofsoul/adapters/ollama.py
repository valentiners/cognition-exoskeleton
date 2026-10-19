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

    def __init__(self, model: str = "gemma4:latest",
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
        """旧版闸口 2 自检：基础模型能否零样本给出完整、稳定的描述。"""
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

    def is_public_knowledge(self, name: str, context: str = "") -> bool | None:
        """闸口 2 自检（结合上下文）：该事物是公共知识还是私人特定。

        解决"常见人名/通名误判"：不问模型认不认识这个词，而问这个事物
        是广为人知的公共概念，还是只特定于主人个人圈子的私人事物。
        返回 True（公共）/False（私人）；模型拿不准或调用失败返回 None。
        """
        prompt = (
            f"有人想把「{name}」作为一个需要长期记住的特定事物。"
            f"相关背景：{context or '（暂无更多描述）'}\n"
            "请问它属于广为人知的公共知识/通用概念，"
            "还是只特定于某个人或小圈子的私人事物？只输出「公共」或「私人」。"
        )
        try:
            ans = self._generate(prompt)
        except OSError:
            return None
        if "私人" in ans:
            return False
        if "公共" in ans:
            return True
        return None

    def extract_feature(self, context: str, utterance: str) -> tuple[str, str] | None:
        """对话上下文中的特征供给检测。

        主人刚聊完某个事物，新消息可能是在补充它的信息（"她叫Lucy""它三岁"），
        也可能是提问、闲聊或新话题。是补充 → 返回 (属性, 内容)；否则返回 None。
        """
        import re
        prompt = (
            "背景：主人刚才向我展示/描述了这样一个事物：\n"
            f"{context}\n\n"
            f"主人的新消息：{utterance}\n\n"
            "如果这条消息是在补充这个事物的具体信息（名字、称呼、属性、特征等），"
            "请用一行「属性: 内容」的格式输出（例如「名字: Lucy」「年龄: 三岁」），"
            "不要输出任何其他文字。\n"
            "如果它是提问、闲聊、新话题或与该事物无关，只输出「无关」。"
        )
        try:
            ans = self._generate(prompt, timeout=120).strip().strip("。.").strip("「」")
        except OSError:
            return None
        if not ans or "无关" in ans:
            return None
        m = re.match(r"^([^:：]{1,12})[:：](.+)$", ans)
        if not m:
            return None
        attr, val = m.group(1).strip().strip("「」"), m.group(2).strip().strip("「」")
        if not attr or not val:
            return None
        if attr.lower() in ("名字", "name", "称呼", "姓名"):
            attr = "名字"
        return attr, val

    def answer_question(self, question: str) -> str | None:
        """常识问答直通：基础模型能覆盖的知识直接回答，不进入记忆流程。

        对应设计文档 4.1 主规则——事务能被基础模型压缩知识完整覆盖 → 不建条目。
        """
        prompt = (
            "请用两三句话简洁准确地回答下面的问题。如果这是你的常识范围内的知识，"
            "直接回答；只有当你确实完全不掌握相关知识时才回答「不了解」。\n"
            f"问题：{question}"
        )
        try:
            ans = self._generate(prompt, timeout=180)
        except OSError:
            return None
        if not ans or ans.strip() in ("不了解", "不知道"):
            return None
        return ans

    def judge_same(self, desc_a: str, desc_b: str) -> bool | None:
        """同一性裁决：两段描述是否指向同一事物。

        用于替代"灰色地带事事问主人"——向量相似度噪声大，由基础模型做最终裁决。
        返回 True/False；模型拿不准或调用失败返回 None（交由上层走未知流程）。
        """
        prompt = (
            "以下两段描述是否最可能指向同一个具体事物？只输出「是」或「否」。\n"
            f"描述一：{desc_a}\n描述二：{desc_b}"
        )
        try:
            ans = self._generate(prompt)
        except OSError:
            return None
        if "否" in ans:
            return False
        if "是" in ans:
            return True
        return None

    def perceive_image(self, image_b64: str, hint: str = "") -> str | None:
        """感知通道 A（设计文档 3.1）：图像 → 视觉模型 → 统一结构化描述。

        输出的是"描述文字"，后续比对/命名/存库流程与文本输入完全同路。
        """
        prompt = (
            "请观察图片，用一段简洁的中文文字描述图中的主体事物，"
            "涵盖颜色、外形、大小、类型等可辨认的特征。"
            "只输出描述本身，不要输出寒暄、解释或任何其他文字。"
        )
        if hint:
            prompt += f"\n主人的补充说明：{hint}"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "images": [image_b64],
            "stream": False,
            "think": self.think,
            "options": {"temperature": 0.2},
        }
        try:
            return _post(self.host, "/api/generate", payload,
                         timeout=300).get("response") or None
        except (OSError, KeyError):
            return None
