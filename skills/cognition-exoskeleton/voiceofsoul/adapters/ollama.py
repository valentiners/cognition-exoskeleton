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


def _get_json(host: str, path: str, timeout: float = 30.0) -> dict:
    with urllib.request.urlopen(host.rstrip("/") + path, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def ensure_ollama(host: str, timeout: float = 5.0) -> None:
    """探测 Ollama 是否可达；不可达时抛出 OSError（CLI 映射为退出码 2）。"""
    _get_json(host, "/api/tags", timeout=timeout)


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


# Ollama /api/tags 只给名单，模型分类要靠 /api/show 的 capabilities 字段：
# 含 "embedding" → embedding 模型；含 "completion" → 基础模型。结果缓存。
_MODEL_CACHE: dict[str, list] = {}


def list_models(host: str) -> dict:
    """列出可用基础模型 / 编码器（供 HTTP 层的模型下拉与 CLI 诊断）。"""
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

    def classify_followup(self, context: str, utterance: str) -> tuple:
        """对话跟进分类：主人这句话想干什么？

        返回 ("feature", 属性, 内容) / ("remember",) / ("forget",) / ("none",)。
        不弹按钮、不用关键词硬猜——让基础模型像聊天一样理解主人的意图，
        调用失败时退回关键词兜底。上下文是主人刚才聊的事物描述。
        """
        import re
        prompt = (
            "背景：主人刚才和我聊到这样一个事物：\n"
            f"{context}\n\n"
            f"主人的新消息：{utterance}\n\n"
            "请判断这条消息属于哪种情况，只输出一行结论：\n"
            "- 如果是在补充这个事物的信息（名字、称呼、属性、特征等）"
            "→ 输出「属性: 内容」（例如「名字: Lucy」「年龄: 三岁」）\n"
            "- 如果是在要求我永远记住这个事物 → 只输出「记住」\n"
            "- 如果是在要求我忘掉它、不用记 → 只输出「忘掉」\n"
            "- 如果是提问、闲聊、新话题或与该事物无关 → 只输出「无关」"
        )
        ans = ""
        try:
            ans = self._generate(prompt, timeout=120).strip().strip("。.")
        except OSError:
            pass
        m = re.match(r"^([^:：]{1,12})[:：](.+)$", ans)
        if m and "无关" not in ans:
            attr = m.group(1).strip().strip("「」")
            val = m.group(2).strip().strip("「」")
            # 模型偶尔套娃输出"属性: 名字: 小明"——剥掉外层泛化属性名再拆一次
            if attr in ("属性", "特征", "信息") and re.search(r"[:：]", val):
                m2 = re.match(r"^([^:：]{1,12})[:：](.+)$", val)
                if m2:
                    attr, val = m2.group(1).strip().strip("「」"), m2.group(2).strip().strip("「」")
            if attr.lower() in ("名字", "name", "称呼", "姓名"):
                attr = "名字"
            if attr and val:
                return ("feature", attr, val)
        if "忘掉" in ans or "不用记" in ans:
            return ("forget",)
        if "记住" in ans or "记着" in ans:
            return ("remember",)
        if "无关" in ans:
            return ("none",)
        # —— 关键词兜底（LLM 调用失败或答非所问时）——
        if re.search(r"忘掉|不用记|别记|forget", utterance, re.IGNORECASE):
            return ("forget",)
        if re.search(r"记住|记着|别忘|记一下|remember", utterance, re.IGNORECASE):
            return ("remember",)
        m = re.match(r".*(?:名字是|名叫|叫|name is|name's)\s*[「']?([^「』'，,。.!！?？\s]{1,20})",
                     utterance, re.IGNORECASE)
        if m:
            return ("feature", "名字", m.group(1))
        return ("none",)

    def extract_feature(self, context: str, utterance: str) -> tuple[str, str] | None:
        """兼容旧接口：只取特征供给语义。"""
        r = self.classify_followup(context, utterance)
        return (r[1], r[2]) if r[0] == "feature" else None

    def answer_question(self, question: str, context: str = "") -> str | None:
        """常识问答直通：基础模型能覆盖的知识直接回答，不进入记忆流程。

        对应设计文档 4.1 主规则——事务能被基础模型压缩知识完整覆盖 → 不建条目。
        context 为对话中刚聊过的事物（含已缓存特征）——"他是谁？"这类指代问题
        没有上下文会答崩（实测 gemma4 会反过来索要上下文，聊天断裂）。
        """
        prompt = (
            "请用两三句话简洁准确地回答下面的问题。如果这是你的常识范围内的知识，"
            "直接回答；只有当你确实完全不掌握相关知识时才回答「不了解」。\n"
            f"问题：{question}"
        )
        if context:
            prompt += (f"\n背景：主人这句话问的很可能是刚聊过的这个事物——{context}"
                       f"\n请结合背景作答；如果问题与背景无关，就按常识直接回答。")
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
            "请观察图片，用一段简洁的中文文字描述图中的主体事物。\n"
            "要求抓住能把这个个体与同类其他个体区分开的细节，例如：\n"
            "- 人物：发型与发色、瞳色、脸型、肤色、服装款式与配色、配饰、显著标记（痣/纹身/伤痕）、表情与气质\n"
            "- 动物：毛色与花纹、体型、耳/尾/眼的形态、显著标记\n"
            "- 物品：材质、颜色、形状、磨损或独特之处\n"
            "先一句话说类别（如「一位年轻女性」），再罗列上述区分性细节。"
            "只输出描述本身：不要寒暄、不要解释，也不要回答主人附带的任何问题。"
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
            ans = _post(self.host, "/api/generate", payload,
                        timeout=300).get("response") or None
        except (OSError, KeyError):
            return None
        if not ans:
            return None
        # 剥掉 markdown 装饰（**、#、- 列表符等），模型常带格式输出，污染入库 embedding
        import re as _re
        ans = _re.sub(r"\*\*|__|`|^#+\s*", "", ans, flags=_re.MULTILINE)
        ans = _re.sub(r"^\s*[-•]\s*", "", ans, flags=_re.MULTILINE)
        return ans.strip()
