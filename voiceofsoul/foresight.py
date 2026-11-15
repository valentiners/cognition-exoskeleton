"""W6 L3 前瞻推演感 —— 离线生成 · 在线消费 · 分级警示 · 升格送审。

架构铁律：**推演是 LLM 重活，放睡眠进程离线做；在线只做向量匹配，零 LLM、不卡对话。**

- 标签挂在单条记忆上（不做跨条目组合推演）；
- 风险标签分级：🟥 严重（人身/隐私/财务，唯一允许打断）/ 🟨 注意 / 🟦 提示；
- **来源加严**：`source=web` 的风险标签最高只能 🟨，🟥 必须有本地依据；
- **损失厌恶**：风险与机会同时命中，警示先行；
- **升格送审**：同一风险模式跨 ≥3 天且 ≥3 次命中 → 生成本能规则草案；模型只有提案权，
  主人自由文本批准后才写入 `instinct_rules.json`（本项目本能层首个实际载体）；
- TTL 90 天是占位值，只记录不强制（低摩擦方案）；§9 已标定阈值一律不动。
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path

from .adapters import prompts
from .doctor import _keynorm

TYPES = ("risk", "applicability")
SEVERITIES = ("red", "yellow", "blue")
SOURCES = ("master", "local", "web")
RISK_ORDER = {"red": 3, "yellow": 2, "blue": 1}
TTL_DAYS = 90            # 占位：只记录 expires_at，不强制过期
MATCH_TH = 0.62          # 占位：复用临时记忆阈值，不新增标定
PROMOTE_DAYS = 3
PROMOTE_HITS = 3
REJECT_COOLDOWN_DAYS = 30

SEVERITY_LABEL = {"red": "🟥", "yellow": "🟨", "blue": "🟦"}


def cap_severity(severity: str | None, source: str) -> str | None:
    """来源加严：web 风险最高 🟨（防“AI 版危言耸听”）。"""
    if severity == "red" and source == "web":
        return "yellow"
    return severity


def build_tag(entry_id: str, kind: str, scene: str, *, severity=None,
              source: str = "master", evidence: list, irreversible: bool = False,
              scene_embedding=None, now: datetime | None = None) -> dict:
    """校验并构造标签；无 evidence / 非法字段 → ValueError。"""
    if kind not in TYPES:
        raise ValueError(f"unknown tag type: {kind!r}")
    if source not in SOURCES:
        raise ValueError(f"unknown source: {source!r}")
    scene = (scene or "").strip()
    if not scene or len(scene) > 80:
        raise ValueError("scene is required (<=80 chars)")
    ev = [e for e in (evidence or []) if isinstance(e, dict)]
    if not ev:
        raise ValueError("evidence is required (birth certificate)")
    if kind == "risk":
        if severity not in SEVERITIES:
            raise ValueError(f"unknown severity: {severity!r}")
        severity = cap_severity(severity, source)
    else:
        severity = None
    now = now or datetime.now()
    return {
        "tag_id": "",
        "entry_id": entry_id,
        "type": kind,
        "scene": scene,
        "severity": severity,
        "source": source,
        "irreversible": bool(irreversible) if kind == "risk" else False,
        "evidence": ev,
        "status": "active",
        "created_at": now.isoformat(timespec="seconds"),
        "expires_at": (now + timedelta(days=TTL_DAYS)).isoformat(
            timespec="seconds"),
        "scene_embedding": scene_embedding,
    }


def active_tags(memory) -> list[dict]:
    out = []
    for entry in memory.entries.values():
        if entry.get("status") == "superseded":
            continue
        for tag in entry.get("foresight", []) or []:
            if tag.get("status", "active") == "active":
                out.append(tag)
    return out


def match_tags(context_vec, embedder, tags: list[dict],
               threshold: float = MATCH_TH) -> list[dict]:
    """在线匹配：本地向量余弦，零 LLM。"""
    if not context_vec or not embedder:
        return []
    hits = []
    for tag in tags:
        emb = tag.get("scene_embedding")
        if not emb:
            continue
        try:
            score = float(embedder.cosine(context_vec, emb))
        except Exception:
            continue
        if score >= threshold:
            hits.append({"tag": tag, "similarity": round(score, 4)})
    hits.sort(key=lambda h: (-RISK_ORDER.get(h["tag"].get("severity"), 0),
                             -h["similarity"]))
    return hits


def route(hits: list[dict]) -> dict | None:
    """损失厌恶：只要有风险命中，警示先行；机会向让位。"""
    risks = [h for h in hits if h["tag"]["type"] == "risk"]
    if risks:
        return risks[0]
    apps = [h for h in hits if h["tag"]["type"] == "applicability"]
    return apps[0] if apps else None


def warning_text(tag: dict) -> str:
    """拟人化警示；不出现“标签/推演/严重级”等机制自述。"""
    reason = (tag.get("reason") or "").strip().rstrip("。.!！") \
        or tag.get("scene", "")
    if tag["type"] == "applicability":
        return f"对了，{tag.get('scene', '')}的时候也许用得上——{reason}"
    sev = tag.get("severity")
    if sev == "red":
        if tag.get("irreversible"):
            return (f"等等，这件事我得拦你一下——{reason}。"
                    "你要真打算做，再跟我说一遍「确定」，我才放心。")
        return f"等等，这个我得提醒你一下——{reason}。"
    if sev == "yellow":
        return f"顺便提醒你一句——{reason}"
    return f"小提示：{reason}"


def _today(now: datetime | None = None) -> str:
    return (now or datetime.now()).date().isoformat()


class ForesightState:
    """命中计数、展示记录、升格提案与拒后冷却（``foresight_state.json``）。"""

    def __init__(self, path):
        self.path = Path(path)
        self.data = {"patterns": {}, "proposals": {}, "shown": {},
                     "proposal_counter": 0}
        if self.path.exists():
            try:
                loaded = json.loads(self.path.read_text(encoding="utf-8"))
                for key in self.data:
                    if key in loaded:
                        self.data[key] = loaded[key]
            except (OSError, ValueError, TypeError):
                pass

    def _persist(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        tmp.replace(self.path)

    # ---------- 展示节流 ----------
    def shown_today(self, tag_id: str, now: datetime | None = None) -> bool:
        return self.data["shown"].get(tag_id) == _today(now)

    def mark_shown(self, tag_id: str, now: datetime | None = None):
        self.data["shown"][tag_id] = _today(now)
        self._persist()

    # ---------- 命中与升格 ----------
    def register_hit(self, tag: dict, now: datetime | None = None) -> dict:
        today = _today(now)
        key = _keynorm(tag.get("scene", ""))
        pat = self.data["patterns"].setdefault(
            key, {"dates": [], "count": 0, "rejected_until": None,
                  "proposal_id": None})
        if today not in pat["dates"]:
            pat["dates"].append(today)
        pat["count"] = int(pat.get("count", 0)) + 1
        self._persist()
        return pat

    def due_proposal(self, tag: dict, now: datetime | None = None) -> dict | None:
        """≥3 天且 ≥3 次、未被拒冷却 → 生成（或复用）本能规则草案。"""
        now = now or datetime.now()
        today = now.date()
        key = _keynorm(tag.get("scene", ""))
        pat = self.data["patterns"].get(key)
        if not pat:
            return None
        if len(set(pat["dates"])) < PROMOTE_DAYS or pat["count"] < PROMOTE_HITS:
            return None
        until = pat.get("rejected_until")
        if until:
            try:
                if today < date.fromisoformat(until):
                    return None
            except ValueError:
                pass
        pid = pat.get("proposal_id")
        if pid and self.data["proposals"].get(pid, {}).get("status") == "proposed":
            return self.data["proposals"][pid]
        if pid and self.data["proposals"].get(pid, {}).get("status") == "approved":
            return None
        self.data["proposal_counter"] = int(
            self.data.get("proposal_counter", 0)) + 1
        pid = f"P-{self.data['proposal_counter']:04d}"
        draft = {
            "pid": pid,
            "pattern": key,
            "if": f"场景像「{tag.get('scene', '')}」",
            "then": f"先提醒主人：{tag.get('reason') or tag.get('scene', '')}",
            "from_tag": tag.get("tag_id", ""),
            "severity": tag.get("severity"),
            "status": "proposed",
            "created_at": now.isoformat(timespec="seconds"),
        }
        self.data["proposals"][pid] = draft
        pat["proposal_id"] = pid
        self._persist()
        return dict(draft)

    def decide_proposal(self, pid: str, approved: bool,
                        now: datetime | None = None):
        now = now or datetime.now()
        draft = self.data["proposals"].get(pid)
        if not draft:
            raise KeyError(pid)
        draft["status"] = "approved" if approved else "rejected"
        draft["decided_at"] = now.isoformat(timespec="seconds")
        pat = self.data["patterns"].get(draft.get("pattern"))
        if pat and not approved:
            pat["rejected_until"] = (now.date() + timedelta(
                days=REJECT_COOLDOWN_DAYS)).isoformat()
        if pat and approved:
            pat["rejected_until"] = None
        self._persist()
        return draft


class InstinctStore:
    """本能规则存储（``instinct_rules.json``）——只接收主人批准过的规则。"""

    def __init__(self, path):
        self.path = Path(path)
        self.data = {"version": 1, "counter": 0, "rules": []}
        if self.path.exists():
            try:
                loaded = json.loads(self.path.read_text(encoding="utf-8"))
                for key in self.data:
                    if key in loaded:
                        self.data[key] = loaded[key]
            except (OSError, ValueError, TypeError):
                pass

    def _persist(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        tmp.replace(self.path)

    def approve(self, draft: dict, now: datetime | None = None) -> dict:
        """主人批准后写入（唯一写入路径）。"""
        now = now or datetime.now()
        for rule in self.data["rules"]:
            if rule.get("from_proposal") == draft.get("pid") \
                    and rule.get("status") == "active":
                return rule
        self.data["counter"] = int(self.data.get("counter", 0)) + 1
        rule = {
            "id": f"R-{self.data['counter']:04d}",
            "if": draft.get("if", ""),
            "then": draft.get("then", ""),
            "from_proposal": draft.get("pid", ""),
            "from_tag": draft.get("from_tag", ""),
            "status": "active",
            "approved_at": now.isoformat(timespec="seconds"),
            "history": [{"event": "approved",
                         "time": now.isoformat(timespec="seconds")}],
        }
        self.data["rules"].append(rule)
        self._persist()
        return rule

    def rollback(self, rule_id: str, now: datetime | None = None) -> dict:
        now = now or datetime.now()
        for rule in self.data["rules"]:
            if rule.get("id") == rule_id:
                rule["status"] = "rolled_back"
                rule.setdefault("history", []).append(
                    {"event": "rolled_back",
                     "time": now.isoformat(timespec="seconds")})
                self._persist()
                return rule
        raise KeyError(rule_id)

    def active_rules(self) -> list[dict]:
        return [r for r in self.data["rules"] if r.get("status") == "active"]


def generate_for_today(base, memory, embedder, now: datetime | None = None) -> dict:
    """离线生成：给当日新记忆打标签；同步、沉默、可重复（已试过就跳过）。"""
    now = now or datetime.now()
    today = now.date().isoformat()
    stats = {"entries": 0, "tags": 0, "calls": 0, "failed": 0}
    if memory is None:
        return stats
    fn = getattr(base, "foresight_tags", None)
    if not fn:
        return stats
    for eid, entry in list(memory.entries.items()):
        if entry.get("status") == "superseded":
            continue
        created = str(entry.get("formation", {}).get("created_at", ""))
        if not created.startswith(today):
            continue
        if entry.get("foresight_meta", {}).get("generated_at"):
            continue          # 幂等：同条目只生成一次
        name = entry.get("name", "")
        desc = entry.get("features", {}).get("描述", "")
        if not name or not desc:
            continue
        stats["entries"] += 1
        raw = None
        try:
            raw = fn(name, desc)
            stats["calls"] += 1
        except Exception:
            stats["failed"] += 1
        parsed = prompts.parse_foresight_tags(raw) if raw else []
        source = "web" if entry.get("provenance", {}).get("source") == "web" \
            else "master"
        tags = []
        for item in parsed:
            try:
                vec = embedder.embed(item["scene"]) if embedder else None
                tag = build_tag(
                    eid, item["type"], item["scene"],
                    severity=item.get("severity"), source=source,
                    evidence=[{"type": "memory", "ref": eid, "note": name},
                              {"type": "reason", "ref": item.get("reason", "")}],
                    irreversible=item.get("irreversible", False),
                    scene_embedding=vec, now=now)
            except (ValueError, Exception):
                continue
            tag["tag_id"] = f"F-{eid}-{len(entry.get('foresight', [])) + 1}"
            tag["reason"] = item.get("reason", "")
            tags.append(tag)
        if tags:
            entry.setdefault("foresight", []).extend(tags)
            stats["tags"] += len(tags)
            memory.update(eid, {"event": "前瞻标签",
                                "tags": [t["tag_id"] for t in tags]})
        entry["foresight_meta"] = {
            "generated_at": now.isoformat(timespec="seconds"),
            "model": getattr(base, "model", ""),
            "tags": len(tags),
        }
        memory._persist()
    return stats


__all__ = ["build_tag", "active_tags", "match_tags", "route", "warning_text",
           "ForesightState", "InstinctStore", "generate_for_today",
           "cap_severity", "TYPES", "SEVERITIES", "SOURCES", "TTL_DAYS",
           "MATCH_TH", "PROMOTE_DAYS", "PROMOTE_HITS"]
