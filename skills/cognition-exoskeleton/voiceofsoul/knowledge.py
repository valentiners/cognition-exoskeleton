"""知识边界探测（探测点 #13 / W2 · L1 新旧感）。

回答一个问题：**这个知识点，基础模型的权重里本来就有吗？**
判断不靠直接问“你会不会”（幻觉重灾区），而靠**行为一致性**：同一知识点用
三种不同措辞各问一次，看三次答复的关键槽位是否稳定一致。

判定哲学（Ricky 拍板 2026-11-15，任务书 W2）：

- **假阴性危害大于假阳性**——宁可白记，不可漏记；
  只有「三答一致且不含迟疑措辞」才判 ``KNOWN``（不入库）；
  ``UNCERTAIN`` / ``NEW`` 一律照常走学习流程。
- **UNCERTAIN 档（两答一致）归入“偏向记”一侧**，照常学习；
- **禁止为“纠错白记”设计任何新机制**：白记条目靠 §6.4 休眠自然消亡。

本模块只做纯判断与缓存，不调用 LLM（探测调用在适配层 / 会话编排层）。
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

from .adapters.prompts import parse_knowledge_answer
from .doctor import _keynorm      # 复用医生修 LaTeX / 千分位时的归一化

TTL_DAYS = 30          # 探测结论缓存 30 天（任务书 §2.4；不是 §9 标定值）
KNOWN = "KNOWN"        # 三答一致 → 本来就会，不入库
UNCERTAIN = "UNCERTAIN"  # 两答一致 / 有迟疑 / 探测失败 → 照常学习（偏向记）
NEW = "NEW"            # 三答矛盾或都不知道 → 照常学习

HEDGE_WORDS = ("可能", "大概", "也许", "或许", "好像", "似乎",
               "应该是", "估计", "不确定", "说不准", "可能吧")


def is_hedged(text: str | None) -> bool:
    """答复里带迟疑措辞 → 不算“笃定知道”（三答一致也不给 KNOWN）。"""
    ans = text or ""
    return any(w in ans for w in HEDGE_WORDS)


def _compat(a: str, b: str):
    """槽位值是否兼容：归一后相等，或短的是长串的子串。缺一侧返回 None。"""
    a, b = _keynorm(a), _keynorm(b)
    if not a or not b:
        return None
    if a == b:
        return True
    if len(a) >= 2 and len(b) >= 2 and (a in b or b in a):
        return True
    return False


def _agree(a: dict, b: dict) -> bool:
    """两条答复是否指向同一个知识点。

    优先级：名称 → 类别 → 关键事实。名称相同但类别明确矛盾（动物 vs 植物）
    算不一致；名称缺失时就比较类别，避免“同义不同措辞”被误判为矛盾。
    """
    if a["unsure"] or b["unsure"]:
        return False
    name = _compat(a["name"], b["name"])
    category = _compat(a["category"], b["category"])
    if name is True:
        return category is not False
    if name is False:
        return False
    if category is True:
        return True
    if category is False:
        return False
    return _compat(a["value"], b["value"]) is True


def judge_answers(answers: list | None) -> dict:
    """三答一致性判定；返回 verdict / consistent / hedged / reason / answers。

    - 三答两两一致且无迟疑 → KNOWN
    - 任意一对一致（两答一致）→ UNCERTAIN
    - 互不一致 / 均答不知道 → NEW
    - 探测不完整（不足三答）→ UNCERTAIN（偏向记，绝不因故障漏记）
    """
    answers = list(answers or [])
    parsed = [parse_knowledge_answer(a) for a in answers]
    if len(answers) >= 3:
        pairs = [(0, 1), (0, 2), (1, 2)]
        agrees = [_agree(parsed[i], parsed[j]) for i, j in pairs]
        top = 3 if all(agrees) else 2 if any(agrees) else 1
    else:
        top = 0

    hedged = any(is_hedged(item["raw"]) for item in parsed)
    if len(answers) < 3:
        verdict, reason = UNCERTAIN, "incomplete"
    elif all(item["unsure"] for item in parsed):
        verdict, reason = NEW, "all_unsure"
    elif top == 3 and not hedged:
        verdict, reason = KNOWN, "three_agree"
    elif top >= 2:
        verdict, reason = UNCERTAIN, "two_agree"
    else:
        verdict, reason = NEW, "contradictory"

    return {
        "verdict": verdict,
        "consistent": top,
        "hedged": hedged,
        "reason": reason,
        "answers": [item["raw"][:200] for item in parsed],
    }


def _parse_ts(value) -> datetime | None:
    try:
        return datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None


class KnowledgeProbeCache:
    """探测结论缓存：``<COGSOUL_HOME>/knowledge_probe_cache.json``，TTL 30 天。

    同一知识点（key 归一化后）期内重复提及不重复探测；统计计数也落在这里，
    供后续标定看 KNOWN/UNCERTAIN/NEW 分布。原子写，损坏不拖垮对话。
    """

    def __init__(self, path):
        self.path = Path(path)
        self.items: dict[str, dict] = {}
        self.stats: dict[str, int] = {
            "probes": 0, "known": 0, "uncertain": 0, "new": 0,
            "cache_hits": 0, "expired": 0,
        }
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                self.items = data.get("items", {}) or {}
                for key, val in (data.get("stats") or {}).items():
                    if isinstance(val, int):
                        self.stats[key] = val
            except (OSError, ValueError, TypeError):
                self.items = {}

    def _persist(self):
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps({"items": self.items, "stats": self.stats},
                                  ensure_ascii=False, indent=2),
                       encoding="utf-8")
        tmp.replace(self.path)

    def get(self, key: str, now: datetime | None = None) -> dict | None:
        item = self.items.get(key)
        if not item:
            return None
        now = now or datetime.now()
        expires = _parse_ts(item.get("expires_at"))
        if expires is not None and now >= expires:
            self.stats["expired"] += 1
            self._persist()
            return None
        item["hits"] = int(item.get("hits", 0)) + 1
        self.stats["cache_hits"] += 1
        self._persist()
        return dict(item)

    def put(self, key: str, result: dict, now: datetime | None = None):
        now = now or datetime.now()
        verdict = result.get("verdict", UNCERTAIN)
        self.items[key] = {
            "verdict": verdict,
            "created_at": now.isoformat(timespec="seconds"),
            "expires_at": (now + timedelta(days=TTL_DAYS)).isoformat(
                timespec="seconds"),
            "answers": result.get("answers", []),
            "hits": 0,
        }
        self.stats["probes"] += 1
        if verdict == KNOWN:
            self.stats["known"] += 1
        elif verdict == NEW:
            self.stats["new"] += 1
        else:
            self.stats["uncertain"] += 1
        self._persist()
        return dict(self.items[key])

    def stats_snapshot(self) -> dict:
        st = dict(self.stats)
        st["total"] = st["known"] + st["uncertain"] + st["new"]
        return st


__all__ = ["judge_answers", "is_hedged", "KnowledgeProbeCache",
           "KNOWN", "UNCERTAIN", "NEW", "TTL_DAYS", "HEDGE_WORDS"]
