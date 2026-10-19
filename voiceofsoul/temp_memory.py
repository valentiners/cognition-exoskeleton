"""临时记忆（七天法则）。

新事物的特征一律先暂存于此，而不是立刻进永久记忆：
- 主人明确要求记住 → 立即晋升永久记忆（server 层处理）；
- 主人没说 → 留在临时记忆，超过七天未再出现自动遗忘；
- 七天内不同日期出现 ≥3 次 → 自动晋升永久记忆（记忆巩固，像真实生命体）。

与永久记忆（MemoryModule）完全解耦，只依赖外部编码器生成向量。
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path


class TempMemory:
    TTL_DAYS = 7          # 超过七天未出现 → 遗忘
    PROMOTE_DAYS = 3      # 七天内不同日期出现 ≥3 次 → 自动晋升
    MATCH_TH = 0.62       # 临时记忆命中阈值（低于永久记忆的 0.78，宽松些）

    def __init__(self, path, embedder):
        self.path = Path(path)
        self.embedder = embedder
        self.records: dict[str, dict] = {}
        self._n = 0
        if self.path.exists():
            d = json.load(open(self.path, encoding="utf-8"))
            self.records = d.get("records", {})
            self._n = d.get("counter", 0)

    def _persist(self):
        json.dump({"counter": self._n, "records": self.records},
                  open(self.path, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)

    # ---------- 生命周期 ----------
    def expire(self) -> list[str]:
        """清理超过 TTL 未再出现的记录，返回被遗忘的 id 列表。"""
        today = date.today()
        gone = [rid for rid, r in self.records.items()
                if (today - date.fromisoformat(r["last_seen"])).days > self.TTL_DAYS]
        for rid in gone:
            del self.records[rid]
        if gone:
            self._persist()
        return gone

    def add(self, desc: str, features: dict | None = None) -> str:
        """新事物的特征暂存，返回临时记录 id。"""
        self._n += 1
        rid = f"T-{self._n:04d}"
        today = date.today().isoformat()
        self.records[rid] = {
            "id": rid, "desc": desc, "features": features or {},
            "embedding": self.embedder.embed(desc),
            "first_seen": today, "last_seen": today, "days": [today],
        }
        self._persist()
        return rid

    def update_features(self, rid: str, attr: str, value: str):
        r = self.records.get(rid)
        if not r:
            return
        r["features"][attr] = value
        self._persist()

    def touch(self, rid: str) -> bool:
        """该事物今天又出现了。返回 True 表示触发自动晋升（七天窗口内 ≥3 个不同日期）。"""
        r = self.records.get(rid)
        if not r:
            return False
        t = date.today().isoformat()
        r["last_seen"] = t
        if t not in r["days"]:
            r["days"].append(t)
        cutoff = (date.today() - timedelta(days=self.TTL_DAYS - 1)).isoformat()
        r["days"] = [d for d in r["days"] if d >= cutoff]
        self._persist()
        return len(r["days"]) >= self.PROMOTE_DAYS

    def best_match(self, text: str, threshold: float | None = None):
        """向量近邻，返回 (id, score) 或 None。"""
        if not self.records:
            return None
        vec = self.embedder.embed(text)
        th = threshold if threshold is not None else self.MATCH_TH
        best_id, best_s = None, th
        for rid, r in self.records.items():
            s = self.embedder.cosine(vec, r["embedding"])
            if s >= best_s:
                best_id, best_s = rid, s
        return (best_id, round(best_s, 4)) if best_id else None

    def get(self, rid: str) -> dict | None:
        return self.records.get(rid)

    def remove(self, rid: str):
        self.records.pop(rid, None)
        self._persist()

    def stats(self) -> list[dict]:
        return [{"id": r["id"], "desc": r["desc"][:60], "features": r["features"],
                 "days": r["days"], "first_seen": r["first_seen"]}
                for r in self.records.values()]
