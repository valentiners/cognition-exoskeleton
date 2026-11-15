"""voiceofsoul.growth —— 巩固调度器（"睡眠进程"，设计 §10.3-#4）。

安静时段把认知层当天的活动整理成一篇第一人称《成长日记》，追加到
OpenClaw 的 MEMORY.md（认知层反哺情节层的通道，§1.6 边界：互引不双写）。

铁律：
- 幂等：每天最多一篇；失败也算尝试过（自动触发不重试，--force 手动重来）；
- 零打扰：素材为空 → 零模型调用；模型不可达 → 今天没有日记，不丢人；
- 只读认知库：只读取 entries/temp/curiosity 状态，一行不改；
- 每次运行基础模型调用 ≤ 1 次（解析不合格由 parse_journal 就地否决）。
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

JOURNAL_HEADER = "## 成长日记 · {date}"
MARKER_NAME = "last_consolidation.json"


# ---------- 小工具 ----------

def _today(now: datetime | None = None) -> str:
    return (now or datetime.now()).date().isoformat()


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _load_marker(home: Path) -> dict:
    p = Path(home) / MARKER_NAME
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_marker(home: Path, marker: dict) -> None:
    _atomic_write(Path(home) / MARKER_NAME,
                  json.dumps(marker, ensure_ascii=False, indent=2))


def last_consolidation(home) -> dict | None:
    """最近一次巩固记录（stats 展示用）。"""
    m = _load_marker(Path(home))
    return m or None


# ---------- 素材收集（只读） ----------

def collect_digest(memory, temp, curiosity, *, now: datetime | None = None) -> dict:
    """当天认知素材：新学/更新的实体、补丁与纠错、好奇事件、误认。

    互引不双写（§1.6）：只取名字与事件类型，绝不带特征明细。"""
    today = _today(now)
    learned, updated = [], []
    for e in memory.entries.values():
        if str(e.get("formation", {}).get("created_at", "")).startswith(today):
            learned.append(e.get("name", "?"))
        elif any(str(u.get("time", "")).startswith(today)
                 for u in e.get("updates", [])):
            updated.append(e.get("name", "?"))
    mistakes = [e.get("name", "?") for e in memory.entries.values()
                if str(e.get("threshold_meta", {})
                        .get("last_error_at", "")).startswith(today)]
    patches_today = sum(1 for p in memory.patches
                        if str(p.get("created_at", "")).startswith(today))
    curious_today = {}
    for item in curiosity.items.values():
        if str(item.get("created_at", "")).startswith(today):
            st = item.get("status", "open")
            # W5：假设过期纯静默，不进日记
            if item.get("source_trigger") == "HYPOTHESIS" and st == "expired":
                continue
            curious_today[st] = curious_today.get(st, 0) + 1
    temp_today = sum(1 for r in temp.records.values()
                     if str(r.get("first_seen", "")).startswith(today))
    return {"learned": learned, "updated": updated,
            "mistakes": mistakes, "patches_today": patches_today,
            "curiosity_today": curious_today, "temp_today": temp_today}


def _has_material(digest: dict) -> bool:
    return bool(digest["learned"] or digest["updated"] or
                digest["mistakes"] or digest["patches_today"] or
                digest["curiosity_today"] or digest["temp_today"])


def _material_text(digest: dict) -> str:
    """素材 → 给基础模型的紧凑中文清单（已去敏）。"""
    lines = []
    if digest["learned"]:
        lines.append("今天我新认识了：" + "、".join(f"「{n}」" for n in digest["learned"]))
    if digest["updated"]:
        lines.append("我又补充了解了：" + "、".join(f"「{n}」" for n in digest["updated"]))
    if digest["patches_today"]:
        lines.append(f"我给已有的知识打了 {digest['patches_today']} 个小补丁")
    ct = digest["curiosity_today"]
    if ct:
        parts = []
        if ct.get("self_solved"):
            parts.append(f"{ct['self_solved']} 个自己查明白了")
        if ct.get("answered"):
            parts.append(f"{ct['answered']} 个是你教我的")
        if ct.get("expired"):
            parts.append(f"{ct['expired']} 个过了七天没解开，先放下了")
        if ct.get("muted"):
            parts.append(f"{ct['muted']} 个你让我先别问")
        if ct.get("open"):
            parts.append(f"{ct['open']} 个还挂在心里")
        lines.append("今天我好奇的事：" + "、".join(parts))
    if digest["mistakes"]:
        lines.append("我认错了：" + "、".join(f"「{n}」" for n in digest["mistakes"])
                      + "，以后会更小心")
    if digest["temp_today"]:
        lines.append(f"还有 {digest['temp_today']} 个新事物先寄放在短期记忆里")
    return "\n".join(lines)


# ---------- 日记落盘（MEMORY.md 契约，§1.6） ----------

def resolve_journal_target(home) -> Path:
    """目的地优先级：COGSOUL_MEMORY_MD → 自动探测 OpenClaw 工作区 → 本地回落。

    自动探测根可用 COGSOUL_OPENCLAW_ROOT 覆盖（测试隔离 / 非常规安装）；
    设为不存在路径即跳过自动探测。"""
    env = os.environ.get("COGSOUL_MEMORY_MD", "").strip()
    if env:
        p = Path(env)
        if p.is_file():
            return p
    roots = [Path.home() / ".openclaw" / "workspace",
             Path.home() / ".openclaw"]
    override = os.environ.get("COGSOUL_OPENCLAW_ROOT", "").strip()
    if override:
        roots = [Path(override)]
    for root in roots:
        try:
            if root.is_dir():
                hits = sorted(root.rglob("MEMORY.md"))[:1]
                if hits:
                    return hits[0]
        except OSError:
            continue
    return Path(home) / "growth_journal"


def _append_or_replace_section(md_path: Path, header: str, prose: str) -> None:
    """追加或同段更新（当天重跑不重复追加）；全部原子写。"""
    text = md_path.read_text(encoding="utf-8") if md_path.is_file() else ""
    section = f"{header}\n{prose.strip()}\n"
    if header in text:
        head, rest = text.split(header, 1)
        # 旧段：从 header 之后到下一个 "## " 或文末
        nxt = rest.find("\n## ", 1)
        if nxt == -1:
            new_text = head + section
        else:
            new_text = head + section + rest[nxt + 1:]
    else:
        sep = "\n" if text and not text.endswith("\n\n") else ""
        new_text = text + sep + "\n" + section if text else section
    _atomic_write(md_path, new_text)


# ---------- 主流程 ----------

def consolidate(home, base, *, memory=None, temp=None, curiosity=None,
                now: datetime | None = None, force: bool = False) -> dict:
    """跑一次巩固。返回摘要 dict；任何失败都不抛异常（铁律：零打扰）。"""
    home = Path(home)
    today = _today(now)
    marker = _load_marker(home)
    # 幂等铁律：失败也算尝试过——自动触发不重试，手动 --force 才能重来
    if not force and marker.get("date") == today:
        return {"date": today, "ran": False, "reason": "already_done",
                "journal_path": marker.get("journal_path")}

    digest = {"learned": [], "updated": [], "mistakes": [],
              "patches_today": 0, "curiosity_today": {}, "temp_today": 0}
    try:
        if memory is not None:
            digest = collect_digest(memory, temp, curiosity, now=now)
    except Exception as exc:
        print(f"[睡眠进程] 素材收集失败：{exc}", file=sys.stderr)

    journal = None
    if _has_material(digest):
        compose = getattr(base, "compose_journal", None)
        if compose:
            try:
                journal = compose(_material_text(digest))
            except Exception as exc:
                print(f"[睡眠进程] 日记写作失败：{exc}", file=sys.stderr)
    result = {"date": today, "ran": True, "digest": digest,
              "ok": journal is not None}
    if journal is None:
        result["reason"] = "no_material" if not _has_material(digest) \
            else "compose_failed"
    else:
        try:
            target_dir = resolve_journal_target(home)
            if target_dir.suffix.lower() == ".md":
                md_path = target_dir
            else:
                target_dir.mkdir(parents=True, exist_ok=True)
                md_path = target_dir / f"{today}.md"
            _append_or_replace_section(
                md_path, JOURNAL_HEADER.format(date=today), journal)
            result["journal_path"] = str(md_path)
        except Exception as exc:
            result["ok"] = False
            result["reason"] = f"write_failed: {exc}"
            print(f"[睡眠进程] 日记落盘失败：{exc}", file=sys.stderr)
    _save_marker(home, result)
    return result
