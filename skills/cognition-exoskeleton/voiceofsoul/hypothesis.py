"""W5 假设驱动型好奇 —— 假设对象、生命周期、票箱结算。

一句话：**猜测不再是一句话，而是一张挂着证据的假设**；确认/证伪都走 W3 的
票箱（`memory.apply_vote`），记忆、直觉、假设共用同一套信用体系。

铁律（任务书 W5）：

- **假设 = 尚未投票的知识**：假设期不写永久记忆；确认转正才 +正票，证伪才负票/纠错补丁；
- **出生证明**：无 `evidence` 的猜测不许升级为假设；转正条目带
  “本条目由假设 H-xxxx 经确认编译”；
- **零新增结算逻辑**：结算只调用 `memory.apply_vote` / `save_patch` / `learn_entry` 既有路径；
- **expired 纯静默**：过期不提示、不进日记（Ricky 拍板：未来信息量大，节约资源）；
- 范围克制：不做主动实验、不做过期汇总、不动 T2 阈值 0.60。

本模块保持纯逻辑（无 LLM、无网络）；队列持久化与提问在 curiosity/session。
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta

TTL_DAYS = 7                  # 与好奇队列/临时记忆同哲学（既有机制）
MIN_LEN = 6
MAX_LEN = 80
PLANS = ("web", "master", "wait")
STATUSES = ("proposed", "verifying", "confirmed", "refuted", "expired")

# 陈述过泛：没有具体断言对象，只有“像/某种/之类”
_VAGUE = ("某种", "某个", "某些", "之类", "一种东西", "一类东西", "像个")
_PREDICATE = re.compile(
    r"(是|不是|而非|属于|等于|就是|和|与|同一个人|同一类|姐妹|兄弟|超过|低于|大于|小于)")

_TRANSITIONS = {
    "proposed": {"verifying", "confirmed", "refuted", "expired"},
    "verifying": {"confirmed", "refuted", "expired"},
    "confirmed": set(),
    "refuted": set(),
    "expired": set(),
}


def is_falsifiable(statement: str | None) -> bool:
    """可证伪性粗检：长度、具体断言、不过泛（不是语义级判断，只挡明显垃圾）。"""
    s = (statement or "").strip()
    if not (MIN_LEN <= len(s) <= MAX_LEN):
        return False
    if any(word in s for word in _VAGUE):
        return False
    return bool(_PREDICATE.search(s))


def build(statement: str, confidence: float, evidence: list,
          plan: str = "wait", *, target_entry_id: str | None = None,
          subject: str = "") -> dict:
    """校验并构造假设对象；无 evidence / 陈述过泛 / plan 非法 → ValueError。"""
    if not is_falsifiable(statement):
        raise ValueError("statement is not falsifiable")
    if plan not in PLANS:
        raise ValueError(f"unknown plan: {plan!r}")
    ev = [e for e in (evidence or []) if isinstance(e, dict)]
    if not ev:
        raise ValueError("evidence is required (birth certificate)")
    for item in ev:
        if not item.get("type") or not (item.get("ref") or item.get("note")):
            raise ValueError("evidence items need type and ref/note")
    try:
        conf = min(max(float(confidence), 0.0), 1.0)
    except (TypeError, ValueError):
        conf = 0.0
    now = datetime.now().isoformat(timespec="seconds")
    return {
        "hid": "",                    # 由队列分配 H-xxxx
        "statement": statement.strip(),
        "confidence": round(conf, 4),
        "evidence": ev,
        "plan": plan,
        "hypo_status": "proposed",
        "target_entry_id": target_entry_id,
        "subject": subject or "",
        "created_at": now,
        "updated_at": now,
        "result": "",
        "entry_id": None,
        "solved_source": None,
    }


def transition(hypo: dict, new_status: str, **fields) -> dict:
    """状态机；非法迁移直接报错（假设生命周期全留痕）。"""
    current = hypo.get("hypo_status", "proposed")
    if new_status not in STATUSES:
        raise ValueError(f"unknown hypothesis status: {new_status!r}")
    if new_status not in _TRANSITIONS.get(current, set()):
        raise ValueError(f"illegal transition {current} -> {new_status}")
    hypo["hypo_status"] = new_status
    hypo["updated_at"] = datetime.now().isoformat(timespec="seconds")
    hypo.update(fields)
    return hypo


def is_expired(hypo: dict, now: datetime | None = None) -> bool:
    created = hypo.get("created_at")
    try:
        start = datetime.fromisoformat(str(created))
    except (TypeError, ValueError):
        return False
    now = now or datetime.now()
    return now >= start + timedelta(days=TTL_DAYS)


def attach_birth_certificate(memory, entry_id: str, hypo: dict):
    """给转正条目挂出生证明（只追加，不改既有字段语义）。"""
    entry = memory.entries.get(entry_id)
    if not entry:
        return
    hid = hypo.get("hid", "")
    entry.setdefault("provenance", {})["compiled_from"] = hid
    entry["provenance"]["birth_certificate"] = {
        "hypothesis": hid,
        "statement": hypo.get("statement", ""),
        "evidence": hypo.get("evidence", []),
    }
    entry.setdefault("formation", {}).setdefault("chat_log", []).append(
        f"本条目由假设 {hid} 经确认编译")
    memory.update(entry_id, {"event": "假设转正", "hypothesis": hid,
                             "statement": hypo.get("statement", "")})


def confirm(hypo: dict, memory, *, entry_id: str | None = None,
            source: str = "master") -> dict:
    """确认：有目标条目 → +1 正票；新条目 → 挂出生证明。不复制结算逻辑。"""
    hid = hypo.get("hid", "")
    target = entry_id or hypo.get("target_entry_id")
    if target:
        memory.apply_vote(target, +1, note=f"假设 {hid} 确认")
        if entry_id and entry_id != hypo.get("target_entry_id"):
            attach_birth_certificate(memory, entry_id, hypo)
    return transition(hypo, "confirmed", result="confirmed",
                      entry_id=target, solved_source=source)


def refute(hypo: dict, memory, *, note: str = "", source: str = "master") -> dict:
    """证伪：有目标条目 → -1 负票 + 纠错补丁；无条目 → 只留纠错补丁。"""
    hid = hypo.get("hid", "")
    target = hypo.get("target_entry_id")
    if target and target in memory.entries:
        memory.apply_vote(target, -1, note=f"假设 {hid} 证伪{('：' + note) if note else ''}")
        name = memory.entries[target].get("name", target)
        memory.save_patch({
            "target": name,
            "type": "correction",
            "content": {"假设": hypo.get("statement", ""), "结论": "证伪",
                        "备注": note},
            "source_entry": target,
            "weight": -1,
        })
    else:
        memory.save_patch({
            "target": hypo.get("subject") or hypo.get("statement", "")[:20],
            "type": "correction",
            "content": {"假设": hypo.get("statement", ""), "结论": "证伪",
                        "备注": note},
            "weight": -1,
        })
    return transition(hypo, "refuted", result="refuted",
                      solved_source=source)


__all__ = ["build", "transition", "is_falsifiable", "is_expired",
           "confirm", "refute", "attach_birth_certificate",
           "TTL_DAYS", "PLANS", "STATUSES"]
