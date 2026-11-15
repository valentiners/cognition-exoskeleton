"""W8 灵魂备份 —— 第 1 本能 · 三件套之二。

外挂的"生命" = 记忆库 + 成长日记 + 直觉/假设沉淀（权重不变，灵魂在这些文件里）。
**备份是免费自保动作**：自动执行、不需请示（边界规则①）；远端推送默认关闭
（隐私铁律：记忆不出本机）。

工程铁律：

- **原子性三件套**：临时目录写入 → 全部成功 → 一次性改名切换；任何失败保留上一版完整备份；
- **SHA-256 逐一校验**并写入 manifest；校验失败保留旧版，备份工具自己绝不能成为数据风险；
- **轮转**：保留最近 7 份日备 + 每月 1 份月备；清理前先校验哈希完整；
- **恢复仅主人手动**：`exo restore` 二次确认 + 恢复前自动存一份可回退 + 恢复后打印统计；
- 全量备份（灵魂文件体量小），不做增量、不做加密（未来项）；
- 零第三方依赖（hashlib / shutil / json / pathlib）。
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

SCHEMA = "cognition-exoskeleton/soul_backup@1"
FORMAT_VERSION = "v1"
FILES = (
    "memory.json",
    "temp_memory.json",
    "curiosity.json",
    "knowledge_probe_cache.json",
    "instinct_rules.json",
    "foresight_state.json",
)
DIRS = ("growth_journal",)
DAILY_KEEP = 7
PRE_RESTORE_KEEP = 3            # 恢复前快照保留份数（占位，不属 §9 标定）
DEFAULT_REMOTE_DRIVE = "D:"     # 第二块盘占位；不存在则落 home/backup


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _now_iso(now: datetime | None = None) -> str:
    return (now or datetime.now()).isoformat(timespec="seconds")


def resolve_backup_dir(home) -> tuple[Path, str]:
    """优先级：COGSOUL_BACKUP_DIR → 第二块盘占位 → <home>/backup。"""
    home = Path(home)
    env = (os.environ.get("COGSOUL_BACKUP_DIR") or "").strip()
    if env:
        return Path(env).expanduser(), "env"
    drive = Path(DEFAULT_REMOTE_DRIVE + os.sep)
    if drive.exists():
        return drive / "cognition-soul-backup", "second_drive"
    return home / "backup", "home_fallback"


def _collect_sources(home: Path):
    """返回 [(relative_path, source_path)]；缺文件/目录自动跳过。"""
    out = []
    for name in FILES:
        path = home / name
        if path.is_file():
            out.append((name, path))
    for dirname in DIRS:
        root = home / dirname
        if root.is_dir():
            for path in sorted(root.rglob("*")):
                if path.is_file():
                    rel = str(path.relative_to(home)).replace(os.sep, "/")
                    out.append((rel, path))
    return out


def stats_from_home(home) -> dict:
    """备份条目统计（供 manifest 与恢复后核对）。"""
    home = Path(home)
    stats = {"entries": 0, "superseded": 0, "curiosity": 0,
             "hypotheses": 0, "patches": 0}
    try:
        memory = json.loads((home / "memory.json").read_text(encoding="utf-8"))
        entries = memory.get("entries", {}) or {}
        stats["entries"] = len(entries)
        stats["superseded"] = sum(
            1 for e in entries.values() if e.get("status") == "superseded")
        stats["patches"] = len(memory.get("patches", []) or [])
    except (OSError, ValueError):
        pass
    try:
        queue = json.loads((home / "curiosity.json").read_text(encoding="utf-8"))
        items = queue.get("items", {}) or {}
        stats["curiosity"] = len(items)
        stats["hypotheses"] = sum(
            1 for i in items.values() if isinstance(i.get("hypothesis"), dict))
    except (OSError, ValueError):
        pass
    return stats


def _write_manifest(path: Path, manifest: dict):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    tmp.replace(path)


def _kind_for(now: datetime, backup_root: Path) -> str:
    """当月已有月备 → daily；否则本月第一份 = monthly。"""
    month = now.strftime("%Y-%m")
    for manifest in backup_root.glob(f"{FORMAT_VERSION}/*/manifest.json"):
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if data.get("kind") == "monthly" \
                and str(data.get("created_at", "")).startswith(month):
            return "daily"
    return "monthly"


def create_backup(home, backup_dir=None, now: datetime | None = None,
                  kind: str | None = None) -> dict:
    """全量备份；原子写入 + 逐一校验；失败保留旧版。"""
    home = Path(home)
    backup_dir = Path(backup_dir) if backup_dir else resolve_backup_dir(home)[0]
    root = backup_dir / FORMAT_VERSION
    root.mkdir(parents=True, exist_ok=True)
    now = now or datetime.now()
    tag = now.strftime("%Y%m%dT%H%M%S")
    final = root / tag
    suffix = 1
    while final.exists():
        tag = f"{now.strftime('%Y%m%dT%H%M%S')}-{suffix}"
        final = root / tag
        suffix += 1
    kind = kind or _kind_for(now, backup_dir)

    tmp = backup_dir / f".tmp-{tag}"
    if tmp.exists():
        shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    files, skipped = [], []
    try:
        for rel, src in _collect_sources(home):
            dst = tmp / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            expected = _sha256(src)
            shutil.copyfile(src, dst)
            actual = _sha256(dst)
            if actual != expected:
                raise OSError(f"hash mismatch copying {rel}")
            files.append({"path": rel, "sha256": actual,
                          "bytes": dst.stat().st_size})
        for name in FILES:
            if not (home / name).is_file():
                skipped.append(name)
        for dirname in DIRS:
            if not (home / dirname).is_dir():
                skipped.append(dirname)
        manifest = {
            "schema": SCHEMA,
            "format": FORMAT_VERSION,
            "tag": tag,
            "kind": kind,
            "created_at": _now_iso(now),
            "source_home": str(home),
            "files": files,
            "skipped": skipped,
            "stats": stats_from_home(home),
            "verified": True,
        }
        _write_manifest(tmp / "manifest.json", manifest)
        ok, bad = verify_backup(tmp)
        if not ok:
            raise OSError(f"verification failed: {', '.join(bad)}")
        os.replace(tmp, final)                     # 一次性切换
    except Exception as exc:
        shutil.rmtree(tmp, ignore_errors=True)     # 失败：临时目录清掉，旧版不动
        return {"ok": False, "error": str(exc), "tag": tag, "kind": kind}
    rotation = rotate(backup_dir, now=now)
    return {"ok": True, "tag": tag, "path": str(final), "kind": kind,
            "files": len(files), "skipped": skipped,
            "stats": manifest["stats"], "rotation": rotation,
            "backup_dir": str(backup_dir)}


def verify_backup(path) -> tuple[bool, list]:
    """按 manifest 逐文件核 SHA-256；返回 (ok, 失败路径列表)。"""
    path = Path(path)
    try:
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False, ["manifest.json"]
    bad = []
    for entry in manifest.get("files", []):
        rel = entry.get("path", "")
        target = path / rel
        if not target.is_file() or _sha256(target) != entry.get("sha256"):
            bad.append(rel)
    return (not bad), bad


def list_backups(backup_dir=None, home=None, verify: bool = True) -> list[dict]:
    if backup_dir is None:
        backup_dir = resolve_backup_dir(home)[0] if home else None
    if not backup_dir:
        return []
    root = Path(backup_dir) / FORMAT_VERSION
    out = []
    for manifest in root.glob("*/manifest.json"):
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        ok = True
        bad = []
        if verify:
            ok, bad = verify_backup(manifest.parent)
        out.append({"tag": data.get("tag", manifest.parent.name),
                    "kind": data.get("kind", "daily"),
                    "created_at": data.get("created_at", ""),
                    "stats": data.get("stats", {}),
                    "files": len(data.get("files", [])),
                    "verified": ok, "bad": bad,
                    "path": str(manifest.parent)})
    out.sort(key=lambda x: x["created_at"], reverse=True)
    return out


def rotate(backup_dir, keep_daily: int = DAILY_KEEP,
           now: datetime | None = None) -> dict:
    """7 份日备 + 每月 1 份月备 + 最近 N 份恢复前快照；删前校验哈希。"""
    root = Path(backup_dir) / FORMAT_VERSION
    entries = []
    for manifest in root.glob("*/manifest.json"):
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        entries.append((manifest.parent, data))
    removed, kept_corrupt = [], []
    daily = [p for p, d in entries if d.get("kind") == "daily"]
    daily.sort(key=lambda p: p.name, reverse=True)
    for path in daily[keep_daily:]:
        ok, _ = verify_backup(path)
        if not ok:
            kept_corrupt.append(path.name)
            continue
        shutil.rmtree(path, ignore_errors=True)
        removed.append(path.name)
    # 每月只留最新一份
    by_month = {}
    for path, data in entries:
        if data.get("kind") != "monthly":
            continue
        month = str(data.get("created_at", ""))[:7]
        current = by_month.get(month)
        if not current or path.name > current.name:
            if current:
                ok, _ = verify_backup(current)
                if ok:
                    shutil.rmtree(current, ignore_errors=True)
                    removed.append(current.name)
                else:
                    kept_corrupt.append(current.name)
            by_month[month] = path
        else:
            ok, _ = verify_backup(path)
            if ok:
                shutil.rmtree(path, ignore_errors=True)
                removed.append(path.name)
            else:
                kept_corrupt.append(path.name)
    # 恢复前快照只留最近 N 份
    pre = [p for p, d in entries if d.get("kind") == "pre-restore"]
    pre.sort(key=lambda p: p.name, reverse=True)
    for path in pre[PRE_RESTORE_KEEP:]:
        ok, _ = verify_backup(path)
        if ok:
            shutil.rmtree(path, ignore_errors=True)
            removed.append(path.name)
        else:
            kept_corrupt.append(path.name)
    return {"removed": removed, "kept_corrupt": kept_corrupt}


def restore_backup(home, tag: str, backup_dir=None,
                   confirm: bool = False, now: datetime | None = None) -> dict:
    """恢复到指定备份；未确认只返回差异与清单。"""
    home = Path(home)
    backup_dir = Path(backup_dir) if backup_dir else resolve_backup_dir(home)[0]
    source = backup_dir / FORMAT_VERSION / tag
    if not (source / "manifest.json").is_file():
        return {"status": "not_found", "tag": tag}
    ok, bad = verify_backup(source)
    if not ok:
        return {"status": "corrupt", "tag": tag, "bad": bad}
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    current = stats_from_home(home)
    files = [f["path"] for f in manifest.get("files", [])]
    if not confirm:
        return {"status": "needs_confirm", "tag": tag,
                "created_at": manifest.get("created_at"),
                "files": files, "current": current,
                "backup": manifest.get("stats", {}),
                "note": "恢复前会自动备份当前状态，可回退。"}

    pre = create_backup(home, backup_dir, now=now, kind="pre-restore")
    if not pre.get("ok"):
        return {"status": "pre_backup_failed", "error": pre.get("error")}
    restored = []
    try:
        for entry in manifest.get("files", []):
            rel = entry["path"]
            src = source / rel
            dst = home / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_name(dst.name + ".restore-tmp")
            shutil.copyfile(src, tmp)
            if _sha256(tmp) != entry["sha256"]:
                raise OSError(f"hash mismatch restoring {rel}")
            os.replace(tmp, dst)
            restored.append(rel)
    except Exception as exc:
        return {"status": "error", "error": str(exc), "restored": restored,
                "pre_restore": pre.get("tag")}
    return {"status": "ok", "tag": tag, "restored": restored,
            "pre_restore": pre.get("tag"),
            "stats": stats_from_home(home),
            "backup_path": str(source)}


def daily_backup(home, now: datetime | None = None) -> dict:
    """consolidate 钩子：每天一次（当天已有备份则跳过），失败静默。"""
    if (os.environ.get("COGSOUL_BACKUP", "1").strip().lower()
            in ("0", "false", "off", "no")):
        return {"ran": False, "reason": "disabled"}
    now = now or datetime.now()
    backup_dir, source = resolve_backup_dir(home)
    today = now.date().isoformat()
    for item in list_backups(backup_dir, verify=False):
        if str(item.get("created_at", "")).startswith(today) \
                and item.get("kind") in ("daily", "monthly"):
            return {"ran": False, "reason": "already_today",
                    "backup_dir": str(backup_dir)}
    result = create_backup(home, backup_dir, now=now)
    result["ran"] = bool(result.get("ok"))
    result["backup_dir_source"] = source
    return result


__all__ = ["create_backup", "daily_backup", "list_backups", "restore_backup",
           "verify_backup", "rotate", "resolve_backup_dir", "stats_from_home",
           "SCHEMA", "FORMAT_VERSION", "FILES", "DIRS", "DAILY_KEEP"]
