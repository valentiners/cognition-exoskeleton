"""记忆模块（设计文档第 5 节）。

两个存储区：
- entries：主条目库（专名 / 私人知识），参与首轮向量检索；
- patches：基础模型知识补丁区（feature_addition / correction），
  不参与首轮检索，仅在基础模型识别命中后作为增强/纠偏层被调用。

核心原则：只追加、不覆盖 —— update 一律 append 到 updates / audit_history。
单主人原则：不设权限层，主人是唯一可信输入源。
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


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

    def search(self, query_text: str, threshold: float = 0.70) -> list[tuple[str, float]]:
        """首轮向量检索，返回 [(entry_id, score)]，按相似度降序。

        与基础模型解耦：向量来自独立编码器（设计文档 3.2）。
        """
        q = self.embedder.embed(query_text)
        results = []
        for eid, e in self.entries.items():
            emb = e.get("features", {}).get("embedding")
            if not emb:
                continue
            score = self.embedder.cosine(q, emb)
            if score >= threshold:
                results.append((eid, round(score, 4)))
        return sorted(results, key=lambda x: -x[1])

    def best_match(self, query_text: str) -> tuple[str, float] | None:
        """全库找最接近的条目（不限阈值），供灰色地带仲裁与闸口 3 查重。"""
        q = self.embedder.embed(query_text)
        best, best_score = None, -1.0
        for eid, e in self.entries.items():
            emb = e.get("features", {}).get("embedding")
            if not emb:
                continue
            s = self.embedder.cosine(q, emb)
            if s > best_score:
                best, best_score = eid, s
        return (best, round(best_score, 4)) if best else None

    def touch(self, entry_id: str):
        """命中后记 health（遗忘巡检的数据源，设计文档 6.3）。"""
        h = self.entries[entry_id].setdefault("health", {"hit_count": 0, "last_hit_at": None})
        h["hit_count"] += 1
        h["last_hit_at"] = _now()

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
