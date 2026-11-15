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
FREQ_WINDOW_DAYS = 90   # 高频层：近窗口期内有命中即算（不设次数门槛，9.10 修订）

# ---------- L2 织网与信念修正（W3，2026-11-15）----------
NEIGHBOR_TOP = 5        # 写入时固化的邻居清单条数（任务书 §2.1）
VOTE_FLOOR = 1          # 占位：净置信度 < 1 → superseded（一条纠错票即退休）
ACTIVE = "active"
SUPERSEDED = "superseded"


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


def is_frequent(entry: dict, window_days: int = FREQ_WINDOW_DAYS) -> bool:
    """高频层判定：近窗口期内被调用过（主人 2026-10-20 修订：不设次数门槛）。

    理由：命中一次就说明主人近期还在提它，检索优先给它；
    次数权重交给未来的检索排序优化，分层过滤只管"近期是否活着"。
    """
    d = _days_since(entry.get("health", {}).get("last_hit_at"))
    return d is not None and d <= window_days


class MemoryModule:
    def __init__(self, path: str | Path = "memory.json", embedder=None):
        self.path = Path(path)
        self.embedder = embedder
        data = self._load()
        self.entries: dict[str, dict] = data.get("entries", {})
        self.patches: list[dict] = data.get("patches", [])
        for entry in self.entries.values():
            self._normalize_entry(entry)

    # ---------- 向后兼容 ----------
    @staticmethod
    def _normalize_entry(entry: dict):
        """旧条目缺字段按默认补齐（只补本地结构，不自行改写磁盘）。"""
        entry.setdefault("status", ACTIVE)
        entry.setdefault("neighbors", [])
        votes = entry.setdefault("votes", {})
        votes.setdefault("positive", 1)   # 建条目即一张正票
        votes.setdefault("negative", 0)
        meta = entry.setdefault("neighbors_meta", {})
        meta.setdefault("embedder_version",
                        entry.get("embedding_meta", {}).get("model", ""))
        meta.setdefault("computed_at", None)
        meta.setdefault("valid", bool(entry.get("neighbors")))

    @staticmethod
    def _net(entry: dict) -> int:
        votes = entry.get("votes") or {}
        return int(votes.get("positive", 1)) - int(votes.get("negative", 0))

    def _retrievable(self, entry: dict) -> bool:
        """检索资格：未退休且净置信度不低于地板。"""
        return (entry.get("status", ACTIVE) != SUPERSEDED
                and self._net(entry) >= VOTE_FLOOR)

    def effective_confidence(self, entry_id: str) -> int:
        """实时结算有效置信度（正票 − 负票），不预计算缓存（任务书 §3.4）。"""
        entry = self.entries.get(entry_id)
        return self._net(entry) if entry else 0

    # ---------- 持久化 ----------
    def _load(self) -> dict:
        if self.path.exists():
            return json.loads(self.path.read_text(encoding="utf-8"))
        return {}

    def _persist(self):
        # 原子写：任何一步失败都不会留下半截 JSON（任务书 §5）。
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(
            json.dumps({"entries": self.entries, "patches": self.patches},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp.replace(self.path)

    # ---------- 主条目 ----------
    def save(self, entry: dict) -> str:
        """新建主条目（调用方须已通过第 4 节防重复判定与闸口）。"""
        entry_id = f"M-{len(self.entries) + 1:04d}"
        entry["id"] = entry_id
        entry.setdefault("updates", [])
        entry.setdefault("threshold_meta", {}).setdefault("by_base_model", {})
        entry.setdefault("health", {"hit_count": 0, "last_hit_at": None})
        self._normalize_entry(entry)
        self.entries[entry_id] = entry
        # L2 织网（任务书 §2.1）：写入时一次性固化 Top-5 邻居，不维护实时索引。
        self._build_neighbors(entry_id)
        self._persist()
        return entry_id

    def _build_neighbors(self, entry_id: str):
        """只算一次：全库 Top-N 邻居 + 当时的编码器版本（§3.3 / 任务书 §2.1）。"""
        entry = self.entries.get(entry_id)
        if not entry or not self.embedder:
            return
        if not hasattr(self.embedder, "cosine"):
            return
        emb = entry.get("features", {}).get("embedding")
        if not emb:
            return
        scored = []
        for oid, other in self.entries.items():
            if oid == entry_id or not self._retrievable(other):
                continue
            other_emb = other.get("features", {}).get("embedding")
            if not other_emb:
                continue
            scored.append((oid, round(float(self.embedder.cosine(emb, other_emb)), 4)))
        scored.sort(key=lambda item: -item[1])
        entry["neighbors"] = [{"entry_id": oid, "similarity": sim}
                              for oid, sim in scored[:NEIGHBOR_TOP]]
        entry["neighbors_meta"] = {
            "embedder_version": getattr(self.embedder, "model_tag", ""),
            "computed_at": _now(),
            "valid": True,
        }

    def invalidate_neighbors(self):
        """换编码器后旧邻居清单标注失效（不自动重建，§3.3 / 任务书 §2.1）。"""
        for entry in self.entries.values():
            entry.setdefault("neighbors_meta", {})["valid"] = False
        self._persist()

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
            if not self._retrievable(e):
                continue
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
            if not self._retrievable(e):
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

    # ---------- L2 织网：排名与一跳邻居补检 ----------
    def ranked_matches(self, query_text: str, layer: str = "active",
                       limit: int = 2) -> list[tuple[str, float]]:
        """不分阈值的 Top-N 排名（冲突侦测用：需要第二名分数）。"""
        q = self.embedder.embed(query_text)
        scored = []
        for eid, e in self.entries.items():
            if not self._retrievable(e):
                continue
            if layer == "active" and is_dormant(e):
                continue
            if layer == "frequent" and (is_dormant(e) or not is_frequent(e)):
                continue
            emb = e.get("features", {}).get("embedding")
            if not emb:
                continue
            scored.append((eid, round(float(self.embedder.cosine(q, emb)), 4)))
        scored.sort(key=lambda item: -item[1])
        return scored[:limit]

    def neighbor_probe(self, query_text: str, threshold: float,
                       layer: str = "active") -> list[tuple[str, float]]:
        """检索第二梯队（最小版，任务书 §2.5）：只跳一次。

        直接检索未命中时，以当前层最相近条目为锚，检查它的邻居清单；
        邻居清单版本失效或编码器不一致 → 不解释（§3.3）。
        绝不从邻居再往外跳（防级联放大）。
        """
        q = self.embedder.embed(query_text)
        anchor, best = None, -1.0
        for eid, e in self.entries.items():
            if not self._retrievable(e):
                continue
            if layer == "active" and is_dormant(e):
                continue
            if layer == "frequent" and (is_dormant(e) or not is_frequent(e)):
                continue
            emb = e.get("features", {}).get("embedding")
            if not emb:
                continue
            score = float(self.embedder.cosine(q, emb))
            if score > best:
                anchor, best = e, score
        if not anchor:
            return []
        meta = anchor.get("neighbors_meta") or {}
        if not meta.get("valid"):
            return []
        version = meta.get("embedder_version")
        if version and version != getattr(self.embedder, "model_tag", version):
            return []          # 换过编码器：旧邻居清单不解释
        hits = []
        for nb in anchor.get("neighbors", []):
            other = self.entries.get(nb.get("entry_id"))
            if not other or not self._retrievable(other):
                continue
            emb = other.get("features", {}).get("embedding")
            if not emb:
                continue
            score = round(float(self.embedder.cosine(q, emb)), 4)
            if score >= threshold:
                hits.append((other["id"], score))
        hits.sort(key=lambda item: -item[1])
        return hits

    # ---------- 信念修正：否决票 / 状态反转 / 复活 ----------
    def apply_vote(self, entry_id: str, weight: int, note: str = "") -> dict:
        """给条目投一票（正/负），实时结算净置信度；跌破地板则转 superseded。"""
        entry = self.entries.get(entry_id)
        if not entry:
            raise KeyError(entry_id)
        votes = entry.setdefault("votes", {"positive": 1, "negative": 0})
        if weight >= 0:
            votes["positive"] = int(votes.get("positive", 0)) + weight
        else:
            votes["negative"] = int(votes.get("negative", 0)) - weight
        net = self._net(entry)
        if net < VOTE_FLOOR:
            entry["status"] = SUPERSEDED
        elif entry.get("status") == SUPERSEDED:
            entry["status"] = ACTIVE
        self.update(entry_id, {"event": "置信票", "weight": weight,
                               "note": note, "net": net,
                               "status": entry.get("status", ACTIVE)})
        return {"entry_id": entry_id, "net": net,
                "status": entry.get("status", ACTIVE)}

    def revive(self, entry_id: str) -> dict:
        """复活已退休条目：补一张正票回到地板以上（档案仍在，可再纠错）。"""
        entry = self.entries.get(entry_id)
        if not entry:
            raise KeyError(entry_id)
        if entry.get("status") != SUPERSEDED and self._net(entry) >= VOTE_FLOOR:
            return {"entry_id": entry_id, "net": self._net(entry),
                    "status": entry.get("status", ACTIVE)}
        return self.apply_vote(entry_id, +1, note="主人复活")

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
            "superseded": sum(1 for e in self.entries.values()
                              if e.get("status") == SUPERSEDED),
        }
