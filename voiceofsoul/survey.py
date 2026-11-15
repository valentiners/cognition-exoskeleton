"""W4 知识边界报告 G1 —— `exo survey`（W2 的批量版）。

一句话：**围绕主人生活圈，抽样测一测“哪些知识它本来就会、哪些一知半解、哪些完全空白”。**
诚实定位：抽样报告，不是全域普查；只测主人实际聊到的东西。

纪律（任务书 W4）：

- **零改动复用 W2**：三问一致性判定在 `knowledge.py`，结论缓存 `KnowledgeProbeCache`
  直接抵扣，适配器 `knowledge_probe` 原样调用；本模块只做批量编排与报告渲染。
- **只读** OpenClaw 日志 `memory/YYYY-MM-DD.md`，不写、不改、不复制原文（§1.6 互引不双写）。
- **先预估、后确认**：不带 ``--yes`` 只返回预估，不消耗任何探测调用。
- **独立预算**：survey 是主人手动批处理，不占日常自主额度；调用数只进报告与 debug。
- 报告落盘 ``<COGSOUL_HOME>/output/``（运行数据，不入 git）。
"""

from __future__ import annotations

import json
import os
from datetime import date, datetime, timedelta
from pathlib import Path

from .adapters import prompts
from .doctor import _keynorm
from .knowledge import (KNOWN, NEW, UNCERTAIN, KnowledgeProbeCache,
                        judge_answers)

DEFAULT_DAYS = 30
DEFAULT_MAX_TOPICS = 15
MAX_TOPICS_CAP = 15          # 任务书拍板：主题上限 15
LOG_CHAR_CAP = 4000          # 单个日志摘录上限
TOTAL_CHAR_CAP = 20000       # 提取 prompt 的总长度上限
RECENCY_HALF_LIFE_DAYS = 7.0  # 排序用新近度（占位口径，不属 §9 标定）

TIERS = {
    KNOWN: ("🟢", "本来就会"),
    UNCERTAIN: ("🟡", "一知半解·幻觉高发"),
    NEW: ("🔵", "完全空白"),
}


def openclaw_root() -> Path:
    """OpenClaw 工作区根：测试用 COGSOUL_OPENCLAW_ROOT 隔离真实日志。"""
    root = os.environ.get("COGSOUL_OPENCLAW_ROOT")
    if root:
        return Path(root).expanduser()
    return Path.home() / ".openclaw" / "workspace"


def load_logs(days: int = DEFAULT_DAYS, root: Path | None = None,
              today: date | None = None) -> list[tuple[str, str]]:
    """只读最近 N 天的 memory/YYYY-MM-DD.md；返回 [(date, text)]（旧→新）。"""
    root = root or openclaw_root()
    today = today or date.today()
    out = []
    for i in range(max(1, days)):
        d = today - timedelta(days=i)
        path = root / "memory" / f"{d.isoformat()}.md"
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        out.append((d.isoformat(), text[:LOG_CHAR_CAP]))
    return sorted(out)


def _log_blob(logs: list[tuple[str, str]]) -> str:
    parts = [f"## {day}\n{text}" for day, text in logs]
    return "\n\n".join(parts)[:TOTAL_CHAR_CAP]


def extract_topics(base, logs: list[tuple[str, str]],
                   days: int) -> list[dict]:
    """一次调用提取主题；模型把私人主题也列出，解析时按类型过滤。"""
    if not logs:
        return []
    prompt = prompts.survey_topics_prompt(_log_blob(logs), days)
    try:
        if hasattr(base, "survey_topics"):
            raw = base.survey_topics(_log_blob(logs), days)
        else:
            # 兼容旧适配器：没有通用补全口时退回 answer_question
            raw = base.answer_question(prompt)
    except Exception:
        return []
    pairs = prompts.parse_survey_topics(raw)
    known_dates = {day for day, _ in logs}
    # 只保留真实存在的日志日期（防止模型编造出处）
    return [p for p in pairs if p["date"] in known_dates]


def rank_topics(pairs: list[dict], today: date | None = None,
                max_topics: int = DEFAULT_MAX_TOPICS) -> list[dict]:
    """频率 × 新近度加权排序；主题上限 15。"""
    today = today or date.today()
    cap = min(max(1, int(max_topics)), MAX_TOPICS_CAP)
    agg: dict[str, dict] = {}
    for item in pairs:
        key = _keynorm(item.get("topic", ""))
        if not key:
            continue
        rec = agg.setdefault(key, {"topic": item["topic"].strip(),
                                   "mentions": 0, "dates": set()})
        rec["mentions"] += 1
        rec["dates"].add(item["date"])
    ranked = []
    for rec in agg.values():
        dates = sorted(rec["dates"])
        last = date.fromisoformat(dates[-1])
        days_ago = max(0, (today - last).days)
        recency = 1.0 / (1.0 + days_ago / RECENCY_HALF_LIFE_DAYS)
        ranked.append({
            "topic": rec["topic"],
            "mentions": rec["mentions"],
            "last_seen": dates[-1],
            "sources": dates,
            "recency": round(recency, 4),
            "score": round(rec["mentions"] * recency, 4),
        })
    ranked.sort(key=lambda r: (-r["score"], -r["mentions"],
                               r["last_seen"], r["topic"]))
    return ranked[:cap]


def _peek_cache(path: Path, key: str, now: datetime) -> dict | None:
    """只读预览缓存（不改计数、不落盘）；供预估使用。"""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    item = (data.get("items") or {}).get(key)
    if not item:
        return None
    try:
        expires = datetime.fromisoformat(str(item.get("expires_at")))
    except (TypeError, ValueError):
        return item
    return item if now < expires else None


def build_plan(topics: list[dict], cache_path: Path,
               now: datetime | None = None) -> dict:
    """预估：主题数 × 3 − 缓存命中；返回逐主题计划与净调用数。"""
    now = now or datetime.now()
    plan = []
    cached = 0
    for item in topics:
        hit = _peek_cache(cache_path, _keynorm(item["topic"]), now)
        if hit:
            cached += 1
        plan.append({**item, "cached": bool(hit),
                     "est_calls": 0 if hit else 3})
    return {
        "topics": len(topics),
        "cache_hits": cached,
        "probe_calls": sum(p["est_calls"] for p in plan),
        "extraction_calls": 1,
        "total_calls": 1 + sum(p["est_calls"] for p in plan),
        "items": plan,
    }


def _suggestion(topic: str, verdict: str) -> str | None:
    if verdict != UNCERTAIN:
        return None
    return (f"建议用一个明确的定义 + 一个例子把「{topic}」讲清楚；"
            "正式教学闭环见 G3，本建议仅供参考。")


def _short(text: str, limit: int = 90) -> str:
    text = " ".join(str(text or "").split())
    return text[:limit] + ("…" if len(text) > limit else "")


def _report_date(now: datetime) -> str:
    return now.strftime("%Y-%m-%d")


def render_markdown(report: dict) -> str:
    topics = report["topics"]
    by = {tier: [t for t in topics if t["verdict"] == tier]
          for tier in (UNCERTAIN, NEW, KNOWN)}
    counts = report["counts"]
    lines = [
        f"# 🧠 知识边界报告 · {report['generated_at'][:10]}",
        "",
        f"模型: {report['model']['base']} · 编码器: {report['model']['embedder']}"
        f" · 探测 {report['probe_calls']} 次（缓存命中 {report['cache_hits']} 次）",
        f"概览: 🟢{counts[KNOWN]}  🟡{counts[UNCERTAIN]}  🔵{counts[NEW]}",
        "",
    ]
    if by[UNCERTAIN]:
        lines += ["## 🟡 重点关注（幻觉高发，建议教学）", ""]
        for i, t in enumerate(by[UNCERTAIN], 1):
            lines.append(f"{i}. **{t['topic']}** — 三答一致度 {t['consistent']}/3"
                         f"（{_short(t['evidence'][0], 60) if t['evidence'] else '无答复'}）")
            for ans in t["evidence"][:3]:
                lines.append(f"   - {_short(ans)}")
            if t.get("suggestion"):
                lines.append(f"   💡 教学建议（仅供参考）: {t['suggestion']}")
            lines.append("")
    if by[NEW]:
        lines += ["## 🔵 完全空白", ""]
        for i, t in enumerate(by[NEW], 1):
            lines.append(f"{i}. **{t['topic']}** — 来源 {', '.join(t['sources'])}")
        lines.append("")
    if by[KNOWN]:
        lines += ["## 🟢 本来就会", ""]
        for i, t in enumerate(by[KNOWN], 1):
            lines.append(f"{i}. **{t['topic']}** — 来源 {', '.join(t['sources'])}")
        lines.append("")
    lines.append("📎 主题出处: " + "；".join(
        f"{t['topic']}（{', '.join(t['sources'])}）" for t in topics))
    lines.append("")
    lines.append("> 这是抽样报告，不是全域普查；只覆盖最近聊到的主题。"
                 "报告头记录了模型与编码器版本，换模型后旧报告视为失效。")
    return "\n".join(lines)


def _write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def run_survey(base, home, *, days: int = DEFAULT_DAYS,
               max_topics: int = DEFAULT_MAX_TOPICS,
               confirm: bool = False, root: Path | None = None,
               embedder_model: str = "",
               now: datetime | None = None) -> dict:
    """完整流水线：读日志 → 提取 → 排序 → 预估 →（确认后）探测 → 报告。"""
    now = now or datetime.now()
    root = root or openclaw_root()
    cache_path = Path(home) / "knowledge_probe_cache.json"

    logs = load_logs(days, root, today=now.date())
    if not logs:
        return {"status": "no_logs", "root": str(root), "days": days}

    pairs = extract_topics(base, logs, days)
    topics = rank_topics(pairs, today=now.date(), max_topics=max_topics)
    if not topics:
        return {"status": "no_topics", "root": str(root), "days": days,
                "extraction_calls": 1}

    plan = build_plan(topics, cache_path, now=now)
    if not confirm:
        return {"status": "needs_confirm", "root": str(root), "days": days,
                "max_topics": min(max(1, int(max_topics)), MAX_TOPICS_CAP),
                "estimate": plan}

    cache = KnowledgeProbeCache(cache_path)
    results = []
    probe_calls = 0
    cache_hits = 0
    for item in topics:
        key = _keynorm(item["topic"])
        cached = cache.get(key)
        if cached:
            verdict = cached.get("verdict", UNCERTAIN)
            evidence = list(cached.get("answers") or [])
            consistent = None
            cached_flag = True
            cache_hits += 1
        else:
            try:
                answers = base.knowledge_probe(item["topic"])
            except Exception as exc:
                results.append({**item, "verdict": UNCERTAIN,
                                "evidence": [], "error": str(exc)[:120],
                                "cached": False, "suggestion": None})
                continue
            judged = judge_answers(answers)
            cache.put(key, judged)
            verdict = judged["verdict"]
            evidence = judged["answers"]
            consistent = judged["consistent"]
            cached_flag = False
            probe_calls += 3
        emoji, label = TIERS.get(verdict, TIERS[UNCERTAIN])
        results.append({
            **item,
            "verdict": verdict,
            "emoji": emoji,
            "label": label,
            "consistent": consistent,
            "evidence": evidence,
            "cached": cached_flag,
            "suggestion": _suggestion(item["topic"], verdict),
        })

    counts = {KNOWN: 0, UNCERTAIN: 0, NEW: 0}
    for t in results:
        counts[t["verdict"]] = counts.get(t["verdict"], 0) + 1
    report = {
        "schema": "cognition-exoskeleton/knowledge_survey@1",
        "generated_at": now.isoformat(timespec="seconds"),
        "days": days,
        "max_topics": min(max(1, int(max_topics)), MAX_TOPICS_CAP),
        "model": {"base": getattr(base, "model", "") or "",
                  "embedder": embedder_model},
        "extraction_calls": 1,
        "probe_calls": probe_calls,
        "cache_hits": cache_hits,
        "total_calls": 1 + probe_calls,
        "counts": counts,
        "log_root": str(root),
        "cache_path": str(cache_path),
        "topics": results,
    }
    out_dir = Path(home) / "output"
    stamp = _report_date(now).replace("-", "")
    md_path = out_dir / f"knowledge_survey_{stamp}.md"
    json_path = out_dir / f"knowledge_survey_{stamp}.json"
    _write(json_path, json.dumps(report, ensure_ascii=False, indent=2))
    _write(md_path, render_markdown(report))
    return {"status": "ok", "report": report,
            "markdown_path": str(md_path), "json_path": str(json_path)}


__all__ = ["run_survey", "load_logs", "extract_topics", "rank_topics",
           "build_plan", "render_markdown", "openclaw_root",
           "DEFAULT_DAYS", "DEFAULT_MAX_TOPICS", "MAX_TOPICS_CAP"]
