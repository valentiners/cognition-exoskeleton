"""记忆模块（设计文档第 5 节）。

两个存储区：
- entries：主条目库（专名 / 私人知识），参与首轮向量检索；
- patches：基础模型知识补丁区（feature_addition / correction），
  不参与首轮检索，仅在基础模型识别命中后作为增强/纠偏层被调用。

核心原则：只追加、不覆盖 —— update 一律 append 到 updates / audit_history。
单主人原则：不设权限层，主人是唯一可信输入源。

分层检索（设计文档 9.10）：高频层 / 活跃层 / 休眠层。
休眠是查询时派生的判定（last_hit_at 距今 > DORMANT_DAYS），不设状态位、
不搬库存储；命中即 touch 刷新 last_hit_at，条目自然复活。
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

DORMANT_DAYS = 365      # 超过这么久未被调用 → 休眠（9.10，概念默认待标定）
FREQ_WINDOW_DAYS = 90   # 高频层：近期窗口
FREQ_MIN_HITS = 3       # 高频层：窗口内最少命中（primitive 用累计数近似）


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _days_since(iso: str | None) -> float | None:
    """距今天数；无法解析返回 None（视为活跃，保守不误休眠）。"""
    if not iso:
        return None
    try:
        return (datetime.now() - datetime.fromisoformat(iso)).total_seconds() / 86400
    except ValueError:
        return None


def _last_active_iso(entry: dict) -> str | None:
    """条目最近一次被调用的时间；从未命中过以创建时间起算。"""
    return entry.get("health", {}).get("last_hit_at") or \
        entry.get("formation", {}).get("created_at")


def is_dormant(entry: dict, days: int = DORMANT_DAYS) -> bool:
    d = _days_since(_last_active_iso(entry))
    return d is not None and d > days


def is_frequent(entry: dict, window_days: int = FREQ_WINDOW_DAYS,
                min_hits: int = FREQ_MIN_HITS) -> bool:
    """高频层判定：近期有命中且累计命中达标（primitive 近似，正式版窗口内计数）。"""
    h = entry.get("health", {})
    if h.get("hit_count", 0) < min_hits:
        return False
    d = _days_since(h.get("last_hit_at"))
    return d is not None and d <= window_days


class MemoryModule:
    def __init__(self, path: str | Path = "memory.json", embedder=None):
        self.path = Path(path)
        self.embedder = embedder
        data = self._load()
        self.entries: dict[str, dict] = data.get("entries", {})
        self.patches: list[dict] = data.get("patches", [])

    # ---------- 持久化 ----------
    def _load(self) -> dict:
        if self.path.exists():
            return json.loads(self.path.read_text(encoding="utf-8"))
        return {}

    def _persist(self):
        self.path.write_text(
            json.dumps({"entries": self.entries, "patches": self.patches},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ---------- 主条目 ----------
    def save(self, entry: dict) -> str:
        """新建主条目（调用方须已通过第 4 节防重复判定与闸口）。"""
        entry_id = f"M-{len(self.entries) + 1:04d}"
        entry["id"] = entry_id
        entry.setdefault("updates", [])
        entry.setdefault("threshold_meta", {}).setdefault("by_base_model", {})
        entry.setdefault("health", {"hit_count": 0, "last_hit_at": None})
        self.entries[entry_id] = entry
        self._persist()
        return entry_id

    def update(self, entry_id: str, update_record: dict):
        """只追加更新：错误事件、重命名、MERGE 增补，全部进 updates 留痕。"""
        update_record = dict(update_record)
        update_record["time"] = _now()
        self.entries[entry_id]["updates"].append(update_record)
        self._persist()

    def search(self, query_text: str, threshold: float = 0.70,
               layer: str = "all") -> list[tuple[str, float]]:
        """分层向量检索，返回 [(entry_id, score)]，按相似度降序。

        layer（设计文档 9.10）：
        - "frequent"：非休眠且高频（L1 日常首选）
        - "active"：全部非休眠（L2 静默扩检）
        - "dormant"：仅休眠条目（L3，仅主人"再想想"时）
        - "all"：全库（默认，兼容旧调用；日常识别请显式传层）
        与基础模型解耦：向量来自独立编码器（设计文档 3.2）。
        """
        q = self.embedder.embed(query_text)
        results = []
        for eid, e in self.entries.items():
            if layer == "frequent" and (is_dormant(e) or not is_frequent(e)):
                continue
            if layer == "active" and is_dormant(e):
                continue
            if layer == "dormant" and not is_dormant(e):
                continue
            emb = e.get("features", {}).get("embedding")
            if not emb:
                continue
            score = self.embedder.cosine(q, emb)
            if score >= threshold:
                results.append((eid, round(score, 4)))
        return sorted(results, key=lambda x: -x[1])

    def best_match(self, query_text: str,
                   include_dormant: bool = False) -> tuple[str, float] | None:
        """全库（默认排除休眠）找最接近的条目，供灰色地带仲裁与闸口 3 查重。

        查重/仲裁不能因条目休眠就放行重复建库——默认排除休眠只是日常识别
        的语义；需要全库时显式 include_dormant=True（设计文档 9.10）。
        """
        q = self.embedder.embed(query_text)
        best, best_score = None, -1.0
        for eid, e in self.entries.items():
            if not include_dormant and is_dormant(e):
                continue
            emb = e.get("features", {}).get("embedding")
            if not emb:
                continue
            s = self.embedder.cosine(q, emb)
            if s > best_score:
                best, best_score = eid, s
        return (best, round(best_score, 4)) if best else None

    def touch(self, entry_id: str):
        """命中后记 health（遗忘巡检/休眠判定的数据源，设计文档 6.3 / 9.10）。

        必须落盘：last_hit_at 若随重启丢失，休眠判定会系统性失效。
        """
        h = self.entries[entry_id].setdefault("health", {"hit_count": 0, "last_hit_at": None})
        h["hit_count"] += 1
        h["last_hit_at"] = _now()
        self._persist()

    # ---------- 补丁区 ----------
    def save_patch(self, patch: dict) -> str:
        """存基础模型知识补丁（闸口 1/2 的输出口）。"""
        patch_id = f"P-{len(self.patches) + 1:04d}"
        patch["id"] = patch_id
        patch.setdefault("priority", 0)
        patch.setdefault("status", "active")  # active / archived / suspended
        patch.setdefault("audit_history", [])
        patch.setdefault("updates", [])
        patch["created_at"] = _now()
        self.patches.append(patch)
        self._persist()
        return patch_id

    def update_patch(self, patch_id: str, update_record: dict):
        for p in self.patches:
            if p["id"] == patch_id:
                update_record = dict(update_record)
                update_record["time"] = _now()
                p["updates"].append(update_record)
                self._persist()
                return
        raise KeyError(patch_id)

    def find_active_patches(self, target_concept: str) -> list[dict]:
        """按 target 取生效中的补丁（纠偏层调用入口）。

        primitive 版匹配规则：target 相同，或补丁 target 是概念串的子串/超串。
        """
        hits = [p for p in self.patches
                if p["status"] == "active"
                and (p["target"] in target_concept or target_concept in p["target"])]
        # priority 大者优先（设计文档 4.5.6 的三级裁决，primitive 版取一级）
        return sorted(hits, key=lambda p: -p.get("priority", 0))

    # ---------- 统计 ----------
    def stats(self) -> dict:
        return {
            "entries": len(self.entries),
            "patches": len(self.patches),
            "patches_active": sum(1 for p in self.patches if p["status"] == "active"),
        }
