"""Shared prompt templates and response parsers for base-model adapters.

Both the local Ollama adapter and the OpenAI-compatible adapter must speak the
same cognition. Prompts and parsers therefore live here exactly once; adapters
only differ in transport. Do not fork prompt text into an adapter.
"""

from __future__ import annotations

import re


# ---------------------------------------------------------------- prompts

def confidence_prompt(text: str) -> str:
    return (
        "请只输出一个 0 到 1 之间的小数（例如 0.72），表示你对识别以下事物的把握程度。\n"
        "如果你完全不认识它，输出 0。不要输出任何其他文字。\n"
        f"描述：{text}"
    )


def infer_prompt(text: str) -> str:
    return (
        "以下是对某个事物的描述。请只输出一个最可能的通用名称（2 到 6 个字），"
        "例如「猫」「汽车」。如果你不认识，只输出「未知」。不要输出任何其他文字。\n"
        f"描述：{text}"
    )


def describe_prompt(name: str) -> str:
    return (
        f"请用一两句话描述「{name}」是什么。如果你完全不认识这个词，"
        f"只输出「不认识」。不要输出任何其他文字。"
    )


def public_knowledge_prompt(name: str, context: str = "") -> str:
    return (
        f"有人想把「{name}」作为一个需要长期记住的特定事物。"
        f"相关背景：{context or '（暂无更多描述）'}\n"
        "请问它属于广为人知的公共知识/通用概念，"
        "还是只特定于某个人或小圈子的私人事物？只输出「公共」或「私人」。"
    )


def followup_prompt(context: str, utterance: str) -> str:
    return (
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


def answer_prompt(question: str, context: str = "") -> str:
    prompt = (
        "请用两三句话简洁准确地回答下面的问题。如果这是你的常识范围内的知识，"
        "直接回答；只有当你确实完全不掌握相关知识时才回答「不了解」。\n"
        f"问题：{question}"
    )
    if context:
        prompt += (f"\n背景：主人这句话问的很可能是刚聊过的这个事物——{context}"
                   f"\n请结合背景作答；如果问题与背景无关，就按常识直接回答。")
    return prompt


def judge_same_prompt(desc_a: str, desc_b: str) -> str:
    return (
        "以下两段描述是否最可能指向同一个具体事物？只输出「是」或「否」。\n"
        f"描述一：{desc_a}\n描述二：{desc_b}"
    )


def image_prompt(hint: str = "") -> str:
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
    return prompt


# ---------------------------------------------------------------- parsers

def parse_confidence(raw: str | None) -> float:
    try:
        value = float((raw or "").replace("%", "").strip())
        return min(max(value, 0.0), 1.0)
    except (TypeError, ValueError):
        return 0.0


def parse_infer(raw: str | None) -> str:
    ans = (raw or "").strip()
    return ans if ans and ans != "未知" else "未知事物"


def parse_describe(raw: str | None) -> str | None:
    ans = (raw or "").strip()
    if any(k in ans for k in ("不认识", "无法", "不知道", "不确定")) or not ans:
        return None
    return ans


def parse_public_knowledge(raw: str | None) -> bool | None:
    ans = (raw or "").strip()
    if "私人" in ans:
        return False
    if "公共" in ans:
        return True
    return None


def parse_followup(raw: str | None, utterance: str) -> tuple:
    ans = (raw or "").strip().strip("。.")
    m = re.match(r"^([^:：]{1,12})[:：](.+)$", ans)
    if m and "无关" not in ans:
        attr = m.group(1).strip().strip("「」")
        val = m.group(2).strip().strip("「」")
        # 模型偶尔套娃输出"属性: 名字: 小明"——剥掉外层泛化属性名再拆一次
        if attr in ("属性", "特征", "信息") and re.search(r"[:：]", val):
            m2 = re.match(r"^([^:：]{1,12})[:：](.+)$", val)
            if m2:
                attr, val = (m2.group(1).strip().strip("「」"),
                             m2.group(2).strip().strip("「」"))
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


def parse_answer(raw: str | None) -> str | None:
    ans = (raw or "").strip()
    if not ans or ans in ("不了解", "不知道"):
        return None
    return ans


def parse_judge_same(raw: str | None) -> bool | None:
    ans = (raw or "").strip()
    if "否" in ans:
        return False
    if "是" in ans:
        return True
    return None


def clean_image_description(raw: str | None) -> str:
    """Strip markdown decoration that pollutes stored embeddings."""
    ans = raw or ""
    ans = re.sub(r"\*\*|__|`|^#+\s*", "", ans, flags=re.MULTILINE)
    ans = re.sub(r"^\s*[-•]\s*", "", ans, flags=re.MULTILINE)
    return ans.strip()
