"""L2 近似冲突侦测（W3，2026-11-15）。

设计依据：方案 §1.7 直觉 L2（关联感）+ W3 任务书 §2.2。

两级判定（两个阈值都是**占位值**，首跑只记录分布，标定走 §6.5）：

- **硬冲突**：新观察同时命中两条不同永久条目（均 ≥ mem_th），差距 < GAP；
- **软存疑**：只命中一条（另一条紧咬但未过 mem_th），差距 < GAP 且第二名 ≥ SOFT_SECOND；
  只记 debug、不进对话；同一对累计 3 次自动升级为硬冲突。

⚠️ 任务书 §2.2 的表格（“均 ≥0.78 且差距<0.05 → 硬”）与 §4.2 的三组验收样例
（0.80/0.795 期望为软存疑）在 0.80/0.795 上互相矛盾。本实现以**可执行的验收样例**为准：
高位紧咬（第二名 ≥0.79，属于 W1 实测的“常态地形”）先软存疑累计；刚过线的第二命中
（0.78 ≤ 第二名 < 0.79，更像重复/真冲突）直接硬冲突。若 Kimi/Ricky 确认表格优先，
改 `classify_conflict` 一个分支即可。
"""

from __future__ import annotations

GAP = 0.05           # 占位：前两名差距阈值（任务书 §2.2）
SOFT_SECOND = 0.79   # 占位：软存疑的第二名下限（任务书 §2.2）
ESCALATE_AFTER = 3   # 同对软存疑累计次数 → 升级硬冲突（任务书 §2.2）

HARD = "hard"
SOFT = "soft"


def classify_conflict(primary: float | None, secondary: float | None,
                      mem_th: float = 0.78, gap: float = GAP,
                      soft_second: float = SOFT_SECOND) -> dict | None:
    """返回 {"level": "hard"|"soft", "gap": ..., "primary": ..., "secondary": ...} 或 None。"""
    if primary is None or secondary is None:
        return None
    if primary < mem_th:
        return None
    diff = round(primary - secondary, 4)
    if diff >= gap:
        return None
    base = {"primary": round(primary, 4), "secondary": round(secondary, 4),
            "gap": diff}
    if secondary >= soft_second:
        return {**base, "level": SOFT, "reason": "near_tie"}
    if secondary >= mem_th:
        return {**base, "level": HARD, "reason": "double_hit"}
    return None


def pair_key(a: str, b: str) -> str:
    """同对条目的稳定键（与顺序无关）。"""
    return "|".join(sorted([str(a), str(b)]))
