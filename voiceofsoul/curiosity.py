"""好奇队列（设计文档 §4.8）。

好奇心不是只问一句就算：当场没问、问了没答、以及主人当时忙没空理的
好奇点，都要留下来，七天内有缘再续、过期自然遗忘。

本模块只做**日期与向量逻辑**，不调用任何 LLM（与 temp_memory.py 同纪律）：

- 持久化到 ``<COGSOUL_HOME>/curiosity.json``，原子写；
- TTL 7 天（与临时记忆同哲学）：超期未解 → ``expired``；
- 静音 7 天自动解禁（主人说「别问了／以后再说」→ ``muted``）；
- 主人任何渠道提起同主题（余弦 ≥ 0.62，复用 TempMemory.MATCH_TH）
  → 命中即唤醒：``hit_count`` 加一，``last_hit_at`` 刷新。

条目结构（任务书 §2.3 / 设计 §4.8）::

    {qid, question, source_trigger(T1~T4), stimulus, created_at,
     status(open|self_solved|answered|muted|expired),
     tier_due(L1|L2), hit_count, last_hit_at, solved_source, solved_url}

另有两个内部字段，与条目结构不冲突：``embedding``（唤醒用的向量，同
TempMemory 的做法）与 ``muted_at``（静音起点，7 天解禁用）。
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

TTL_DAYS = 7          # 与 TempMemory.TTL_DAYS 同值（标定值，勿动）
MATCH_TH = 0.62       # 复用 TempMemory.MATCH_TH（标定值，勿动）
STATUSES = ("open", "self_solved", "answered", "muted", "expired",
            "refuted")   # refuted = 假设被证伪（W5）
TRIGGERS = ("T1", "T2", "T3", "T4", "CONFLICT",
            "HYPOTHESIS", "TEACH_SUGGESTION")   # CONFLICT=L2 冲突；HYPOTHESIS=W5；TEACH_SUGGESTION=G3


def _now_iso(now: datetime | None = None) -> str:
    return (now or datetime.now()).isoformat(timespec="seconds")


def _parse(ts: str | None) -> datetime | None:
    try:
        return datetime.fromisoformat(str(ts))
    except (TypeError, ValueError):
        return None


class CuriosityQueue:
    """单主人好奇队列；进程重启后从 curiosity.json 恢复。"""

    def __init__(self, path, embedder=None):
        self.path = Path(path)
        self.embedder = embedder
        self.items: dict[str, dict] = {}
        self._n = 0
        self._h = 0
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                self.items = data.get("items", {}) or {}
                self._n = int(data.get("counter", 0) or 0)
                self._h = int(data.get("hypo_counter", 0) or 0)
            except (OSError, ValueError, TypeError):
                # 队列损坏不能拖垮对话：从空队列起步，磁盘旧文件不删。
                self.items, self._n, self._h = {}, 0, 0

    # ---------- 持久化 ----------
    def _persist(self):
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps({"counter": self._n, "hypo_counter": self._h,
                                  "items": self.items},
                                  ensure_ascii=False, indent=2),
                       encoding="utf-8")
        tmp.replace(self.path)

    def next_hypothesis_id(self) -> str:
        """假设编号（H-xxxx），与好奇点编号同文件持久化。"""
        self._h += 1
        self._persist()
        return f"H-{self._h:04d}"

    # ---------- 向量 ----------
    def _embed(self, text: str) -> list[float] | None:
        if not self.embedder or not text:
            return None
        try:
            return self.embedder.embed(text)
        except Exception:      # 编码器暂时不可用 → 该条不参与唤醒，绝不阻断对话
            return None

    def _cosine(self, a, b) -> float:
        try:
            return float(self.embedder.cosine(a, b))
        except Exception:
            return 0.0

    def find_similar(self, text: str) -> dict | None:
        """找同主题的在册条目（≥ MATCH_TH），用于防重复入队。"""
        vec = self._embed(text)
        if not vec:
            return None
        best, best_score = None, MATCH_TH
        for item in self.items.values():
            if item.get("status") == "expired" or not item.get("embedding"):
                continue
            score = self._cosine(vec, item["embedding"])
            if score >= best_score:
                best, best_score = item, score
        return best

    # ---------- 生命周期 ----------
    def add(self, point: str, trigger: str, stimulus: str,
            dedupe: bool = True) -> str:
        """新增一个好奇点；同主题已在册则只唤醒旧条目（dedupe=False 可跳过）。"""
        point = (point or "").strip()
        if not point:
            raise ValueError("curiosity point must not be empty")
        if trigger not in TRIGGERS:
            raise ValueError(f"unknown trigger: {trigger!r}")
        existing = self.find_similar(stimulus or point) if dedupe else None
        if existing:
            existing["hit_count"] = int(existing.get("hit_count", 0)) + 1
            existing["last_hit_at"] = _now_iso()
            self._persist()
            return existing["qid"]

        now = _now_iso()
        self._n += 1
        qid = f"Q-{self._n:04d}"
        self.items[qid] = {
            "qid": qid,
            "question": point,
            "source_trigger": trigger,
            "stimulus": (stimulus or point).strip(),
            "created_at": now,
            "status": "open",
            "tier_due": "L1",
            "hit_count": 0,
            "last_hit_at": None,
            "solved_source": None,
            "solved_url": None,
            "embedding": self._embed(stimulus or point),
        }
        self._persist()
        return qid

    def get(self, qid: str) -> dict | None:
        return self.items.get(qid)

    def mark(self, qid: str, status: str, **kw):
        """状态流转；额外字段用关键字传入（如 solved_source / tier_due）。"""
        if status not in STATUSES:
            raise ValueError(f"unknown status: {status!r}")
        item = self.items.get(qid)
        if not item:
            raise KeyError(qid)
        item["status"] = status
        if status == "muted" and "muted_at" not in kw:
            item["muted_at"] = _now_iso()
        elif status != "muted":
            item.pop("muted_at", None)
        for key, value in kw.items():
            item[key] = value
        self._persist()
        return item

    def due_for_l1(self, now: datetime | None = None) -> dict | None:
        """本轮可以当场轻问的第一条（一轮最多 1 次；eager 档由调用方多取）。"""
        now = now or datetime.now()
        for item in sorted(self.items.values(), key=lambda i: i["created_at"]):
            if item.get("status") != "open" or item.get("tier_due") != "L1":
                continue
            created = _parse(item.get("created_at"))
            if created and (now - created).days > TTL_DAYS:
                continue
            return item
        return None

    def due_for_l2(self, now: datetime | None = None) -> list[dict]:
        """攒批呈现用：全部待问的 L2 条目（按入队时间）。"""
        now = now or datetime.now()
        out = []
        for item in sorted(self.items.values(), key=lambda i: i["created_at"]):
            if item.get("status") != "open" or item.get("tier_due") != "L2":
                continue
            created = _parse(item.get("created_at"))
            if created and (now - created).days > TTL_DAYS:
                continue
            out.append(item)
        return out

    def touch(self, text_embedding: list[float] | None) -> list[str]:
        """主人又说到了同一个主题 → 唤醒在册条目（§4.8）。"""
        if not text_embedding:
            return []
        woken, changed = [], False
        for item in self.items.values():
            if item.get("status") not in ("open", "muted"):
                continue
            emb = item.get("embedding")
            if not emb:
                continue
            if self._cosine(text_embedding, emb) < MATCH_TH:
                continue
            item["hit_count"] = int(item.get("hit_count", 0)) + 1
            item["last_hit_at"] = _now_iso()
            changed = True
            if item.get("status") == "open":
                woken.append(item["qid"])
        if changed:
            self._persist()
        return woken

    def expire(self, now: datetime | None = None) -> list[str]:
        """TTL 到期：未解的好奇过期留痕；静音满 7 天自动解禁（转回 L2 攒批）。"""
        now = now or datetime.now()
        changed = []
        for item in self.items.values():
            created = _parse(item.get("created_at"))
            if item.get("status") == "open" and created \
                    and (now - created).days > TTL_DAYS:
                item["status"] = "expired"
                changed.append(item["qid"])
                continue
            if item.get("status") == "muted":
                since = _parse(item.get("muted_at")) or created
                if since and (now - since).days > TTL_DAYS:
                    item["status"] = "open"
                    item["tier_due"] = "L2"   # 解禁后先进攒批，不立刻连环追问
                    item.pop("muted_at", None)
                    changed.append(item["qid"])
        if changed:
            self._persist()
        return changed

    # ---------- 统计 ----------
    def stats(self) -> dict:
        """累计 / 已解答 / 过期 / 静音（设计 §4.8：让成长可见）。"""
        total = len(self.items)
        out = {"total": total}
        for status in STATUSES:
            out[status] = sum(1 for i in self.items.values()
                              if i.get("status") == status)
        out["answered_total"] = out["answered"] + out["self_solved"]
        return out

    def list(self, status: str | None = None) -> list[dict]:
        """CLI 用：去掉向量后的条目视图。"""
        out = []
        for item in sorted(self.items.values(), key=lambda i: i["created_at"]):
            if status and item.get("status") != status:
                continue
            out.append({k: v for k, v in item.items() if k != "embedding"})
        return out


__all__ = ["CuriosityQueue", "TTL_DAYS", "MATCH_TH", "STATUSES", "TRIGGERS"]
