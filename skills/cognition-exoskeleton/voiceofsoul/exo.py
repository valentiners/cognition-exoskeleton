"""voiceofsoul.exo —— 自认知外挂的命令行入口。

用法：
    python -m voiceofsoul.exo chat --text "..." [--image /path/to/image.jpg]
    python -m voiceofsoul.exo respond --text "..."
    python -m voiceofsoul.exo stats

每条命令只向 stdout 输出一个 JSON 文档：
    {"say": "...", "actions": [], "debug": "..."}

设计约定（任务书 §1~§5）：
- 复用 voiceofsoul.session.Session，HTTP 服务与 CLI 共用一个大脑；
- 每次调用加载 <COGSOUL_HOME>/session.json，跑一轮，再原子保存；
- Ollama 不可达 → 退出码 2，stderr 输出 {"error": "ollama_unreachable", ...}；
- 参数错误 → 退出码 2，argparse 用法写 stderr；
- 其他异常 → 退出码 1，traceback 写 stderr，记忆文件保持有效 JSON。
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import traceback
from pathlib import Path

from .adapters import (ApiAuthError, ApiUnreachableError, OllamaEmbedder,
                       ensure_ollama, ensure_openai, make_base_model)
from .cognition import CognitiveModule
from .memory import MemoryModule
from .session import DEFAULT_HOME, Session, load_state
from .temp_memory import TempMemory

DEFAULT_OLLAMA = "http://localhost:11434"
DEFAULT_BASE_MODEL = "gemma4:latest"
DEFAULT_EMBEDDING_MODEL = "embeddinggemma-2:latest"
DEFAULT_BACKEND = "ollama"
DEFAULT_API_BASE = "https://api.openai.com/v1"


def _utf8_stdio():
    """Windows 控制台默认 GBK；显式切 UTF-8，避免中文 JSON 被转码。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def _print_json(payload) -> None:
    json.dump(payload, sys.stdout, ensure_ascii=False)
    sys.stdout.write("\n")
    sys.stdout.flush()


def _print_error(kind: str, detail: str) -> None:
    json.dump({"error": kind, "detail": detail}, sys.stderr, ensure_ascii=False)
    sys.stderr.write("\n")
    sys.stderr.flush()


def _add_common(parser: argparse.ArgumentParser, suppress: bool = False) -> None:
    """把全局配置加到一个 parser；子命令里用 SUPPRESS 避免覆盖父级默认值。"""
    default = (lambda value: argparse.SUPPRESS) if suppress else (lambda value: value)
    parser.add_argument("--ollama", default=default(
        os.environ.get("OLLAMA_HOST", DEFAULT_OLLAMA)),
        help="Ollama 服务地址（默认取 OLLAMA_HOST；编码器始终走这里）")
    parser.add_argument("--backend", choices=("ollama", "openai"),
                        default=default(os.environ.get("COGSOUL_BACKEND",
                                                       DEFAULT_BACKEND)),
                        help="基础模型后端（默认 ollama；也可用 COGSOUL_BACKEND）")
    parser.add_argument("--api-base", default=default(
        os.environ.get("OPENAI_BASE_URL", DEFAULT_API_BASE)),
        help="OpenAI-compatible base URL（包含 /v1；默认取 OPENAI_BASE_URL）")
    parser.add_argument("--api-key",
                        default=argparse.SUPPRESS if suppress else None,
                        help="API key；强烈建议改用 OPENAI_API_KEY 环境变量，避免泄露")
    parser.add_argument("--home", default=default(
        os.environ.get("COGSOUL_HOME", str(DEFAULT_HOME))),
        help="工作目录（session.json / memory.json / temp_memory.json）")
    # 模型选择不设 argparse 默认值：未显式传参时要沿用 session.json 里
    # 上一次的选择，再由 _build_session 回落到常量默认值。
    parser.add_argument("--base-model",
                        default=argparse.SUPPRESS if suppress else None,
                        help="基础模型（默认 gemma4:latest）")
    parser.add_argument("--embedding-model",
                        default=argparse.SUPPRESS if suppress else None,
                        help="编码器（默认 embeddinggemma-2:latest）")
    # 识别阈值：标定值见设计文档 §9，勿随意改动（与 server.py 保持一致）。
    parser.add_argument("--mem-th", type=float, default=default(0.78),
                        help="记忆库命中阈值（标定值 0.78）")
    parser.add_argument("--base-th", type=float, default=default(0.51),
                        help="基础模型猜测阈值（标定值 0.51）")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m voiceofsoul.exo",
        description="自认知外挂 CLI：识别、教学、记忆与统计。")
    _add_common(parser)
    sub = parser.add_subparsers(dest="command", required=True)

    p_chat = sub.add_parser("chat", help="感知一条输入（文本 / 图片）")
    _add_common(p_chat, suppress=True)
    p_chat.add_argument("--text", default="", help="主人说的话或事物描述")
    p_chat.add_argument("--image", default=None,
                        help="图片文件路径（不是 base64）")

    p_resp = sub.add_parser("respond", help="回答上一个未决问题")
    _add_common(p_resp, suppress=True)
    p_resp.add_argument("--text", required=True, help="主人的自由文本回答")

    sub.add_parser("stats", help="输出与 GET /api/stats 相同的统计 JSON")
    return parser


def _resolve_config(args) -> dict:
    """Resolve env/flag/session defaults before any network probe."""
    home = Path(os.path.expanduser(args.home))
    state = load_state(home)
    ollama = args.ollama or state.get("ollama_host") or DEFAULT_OLLAMA
    # 未显式传 --base-model / --embedding-model 时，沿用上次会话的选择。
    base_model = args.base_model or state.get("base_model") or DEFAULT_BASE_MODEL
    emb_model = (args.embedding_model or state.get("embedding_model")
                 or DEFAULT_EMBEDDING_MODEL)
    backend = (args.backend or os.environ.get("COGSOUL_BACKEND")
               or DEFAULT_BACKEND).strip().lower()
    api_base = (args.api_base or os.environ.get("OPENAI_BASE_URL")
                or DEFAULT_API_BASE)
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    return {"home": home, "state": state, "ollama": ollama,
            "base_model": base_model, "embedding_model": emb_model,
            "backend": backend, "api_base": api_base, "api_key": api_key}


def _build_session(args, cfg: dict | None = None) -> Session:
    cfg = cfg or _resolve_config(args)
    home, state = cfg["home"], cfg["state"]
    ollama, base_model = cfg["ollama"], cfg["base_model"]
    emb_model = cfg["embedding_model"]

    embedder = OllamaEmbedder(model=emb_model, host=ollama)
    base = make_base_model(cfg["backend"], model=base_model, host=ollama,
                           api_base=cfg["api_base"], api_key=cfg["api_key"])
    memory = MemoryModule(home / "memory.json", embedder)
    temp = TempMemory(home / "temp_memory.json", embedder)
    cog = CognitiveModule(base, memory, embedder,
                          base_th=args.base_th, mem_th=args.mem_th)
    session = Session(cog, temp, home=home, state=state,
                      base_model=base_model, embedding_model=emb_model,
                      ollama_host=ollama)

    # 设计文档 §3.3 / §10.1：换编码器后一次性重建全部向量。
    prev_emb = state.get("embedding_model")
    stale_meta = any(
        e.get("embedding_meta", {}).get("model") not in (None, embedder.model_tag)
        for e in memory.entries.values())
    if (args.embedding_model and prev_emb and args.embedding_model != prev_emb) \
            or stale_meta:
        session.reembed_all()
        session._save()
    return session


def main(argv: list[str] | None = None) -> int:
    _utf8_stdio()
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "chat" and not args.text.strip() and not args.image:
        parser.error("chat 需要 --text 或 --image")
    if args.command == "chat" and args.image and not Path(args.image).is_file():
        parser.error(f"图片文件不存在：{args.image}")

    cfg = _resolve_config(args)
    if cfg["backend"] == "openai" and not cfg["api_key"]:
        parser.error("--backend openai 需要 OPENAI_API_KEY 或 --api-key")

    try:
        if args.command in ("chat", "respond"):
            # 编码器始终在本地 Ollama：先探测它，再探测云端后端。
            try:
                ensure_ollama(cfg["ollama"])
            except OSError as exc:
                _print_error("ollama_unreachable", str(exc))
                return 2
            if cfg["backend"] == "openai":
                try:
                    ensure_openai(cfg["api_base"], cfg["api_key"],
                                  cfg["base_model"])
                except ApiAuthError as exc:
                    _print_error("api_auth", str(exc))
                    return 2
                except ApiUnreachableError as exc:
                    _print_error("api_unreachable", str(exc))
                    return 2

        session = _build_session(args, cfg)

        if args.command == "chat":
            image_b64 = None
            if args.image:
                image_b64 = base64.b64encode(
                    Path(args.image).read_bytes()).decode("ascii")
            _print_json(session.chat(args.text.strip(), image_b64))
        elif args.command == "respond":
            _print_json(session.respond(args.text.strip()))
        else:
            _print_json(session.stats())
        return 0
    except ApiAuthError as exc:
        _print_error("api_auth", str(exc))
        return 2
    except ApiUnreachableError as exc:
        _print_error("api_unreachable", str(exc))
        return 2
    except OSError as exc:
        # 预检之后 Ollama 掉线，或本地文件系统错误；按“不可达”路径返回 2。
        _print_error("ollama_unreachable", str(exc))
        return 2
    except Exception:
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
