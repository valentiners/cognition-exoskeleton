#!/usr/bin/env python3
"""自认知外挂 · 浏览器测试服务（stdlib-only，无第三方依赖）。

用法：
    python server.py            # 默认 http://localhost:8600
    python server.py --port 8080

主人在浏览器里与外挂对话，完成识别 / 仲裁 / 教学闭环。
这是"部署架构"讨论中的最小验证：Ollama(基础模型+embedding) + voiceofsoul 内核 + 浏览器 UI。

本文件只保留 HTTP 壳：会话编排在 voiceofsoul/session.py，与
voiceofsoul/exo.py（CLI）共用同一个大脑。
"""

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from voiceofsoul import CognitiveModule, MemoryModule
from voiceofsoul.adapters import (OllamaEmbedder, list_models,
                                  make_base_model)
from voiceofsoul.session import DEFAULT_HOME, Session
from voiceofsoul.temp_memory import TempMemory

WEB_DIR = Path(__file__).parent / "web"
EXIT_VALUES = {"退出", "exit", "quit"}


class Handler(BaseHTTPRequestHandler):
    chat = respond = stats_fn = models_fn = switch_fn = None  # 由 main 注入

    def log_message(self, *args):  # 静默日志
        pass

    def _headers(self, code: int, length: int, content_type: str):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Access-Control-Allow-Origin", "*")  # 允许直接双击打开页面
        self.end_headers()

    def _json(self, payload: dict, code: int = 200):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._headers(code, len(body), "application/json; charset=utf-8")
        self.wfile.write(body)

    def do_OPTIONS(self):  # CORS 预检
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
            body = html.encode("utf-8")
            self._headers(200, len(body), "text/html; charset=utf-8")
            self.wfile.write(body)
        elif self.path == "/api/stats":
            self._json(self.stats_fn())
        elif self.path == "/api/models":
            self._json(self.models_fn())
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        try:
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"]))
                                 .decode("utf-8"))
        except (ValueError, KeyError):
            return self._json({"error": "bad request"}, 400)
        if self.path == "/api/chat":
            text = (payload.get("text") or "").strip()
            image = (payload.get("image") or "").strip()
            # data URL → 纯 base64
            if image.startswith("data:"):
                image = image.split(",", 1)[1] if "," in image else ""
            if not text and not image:
                return self._json({"error": "empty input"}, 400)
            if text in EXIT_VALUES:
                return self._json({"say": "会话不退出（服务常驻），直接继续描述新事物即可。",
                                   "actions": [], "debug": ""})
            self._json(self.chat(text, image or None))
        elif self.path == "/api/respond":
            self._json(self.respond((payload.get("value") or "").strip()))
        elif self.path == "/api/switch":
            self._json(self.switch_fn(payload))
        else:
            self._json({"error": "not found"}, 404)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8600)
    ap.add_argument("--base-model", default="gemma4:latest")
    ap.add_argument("--embedding-model", default="embeddinggemma-2:latest")
    ap.add_argument("--ollama", default="http://localhost:11434")
    ap.add_argument("--backend", choices=("ollama", "openai"),
                    default=os.environ.get("COGSOUL_BACKEND", "ollama"))
    ap.add_argument("--api-base", default=os.environ.get(
        "OPENAI_BASE_URL", "https://api.openai.com/v1"),
        help="OpenAI-compatible base URL（包含 /v1）")
    ap.add_argument("--api-key", default=None,
                    help="API key；建议用 OPENAI_API_KEY 环境变量，避免泄露")
    ap.add_argument("--home", default=os.environ.get("COGSOUL_HOME", str(DEFAULT_HOME)),
                    help="会话状态目录（session.json）")
    ap.add_argument("--memory", default="memory.json")
    ap.add_argument("--temp-memory", default="temp_memory.json",
                    help="临时记忆文件（七天法则：七天内不同日期出现≥3次自动晋升永久）")
    ap.add_argument("--mem-th", type=float, default=0.78,
                    help="记忆库命中阈值（需按所用编码器标定，设计文档 §9）")
    ap.add_argument("--base-th", type=float, default=0.51,
                    help="基础模型猜测阈值")
    args = ap.parse_args()

    backend = (args.backend or "ollama").strip().lower()
    api_base = args.api_base or "https://api.openai.com/v1"
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if backend == "openai" and not api_key:
        ap.error("--backend openai 需要 OPENAI_API_KEY 或 --api-key")

    embedder = OllamaEmbedder(model=args.embedding_model, host=args.ollama)
    base = make_base_model(backend, model=args.base_model, host=args.ollama,
                           api_base=api_base, api_key=api_key)
    memory = MemoryModule(args.memory, embedder)
    temp = TempMemory(args.temp_memory, embedder)
    cog = CognitiveModule(base, memory, embedder,
                          base_th=args.base_th, mem_th=args.mem_th)

    session = Session(cog, temp, home=args.home,
                      base_model=args.base_model,
                      embedding_model=args.embedding_model,
                      ollama_host=args.ollama)
    Handler.chat = staticmethod(session.chat)
    Handler.respond = staticmethod(session.respond)
    Handler.stats_fn = staticmethod(session.stats)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)

    # 模型列表与热切换（HTTP 层只转发，编排在 Session）
    def models_handler():
        try:
            if backend == "ollama":
                d = list_models(args.ollama)
            else:
                d = {"base": [],
                     "embedding": list_models(args.ollama).get("embedding", [])}
        except OSError:
            d = {"base": [], "embedding": []}
        d["current"] = {"base": session.state["base_model"],
                        "embedding": session.state["embedding_model"],
                        "backend": backend}
        return d

    def switch_handler(payload: dict):
        base_name = (payload.get("base") or "").strip() or None
        emb = (payload.get("embedding") or "").strip() or None
        try:
            if backend == "openai":
                msgs = []
                if base_name:
                    cog.base = make_base_model(
                        "openai", model=base_name, host=args.ollama,
                        api_base=api_base, api_key=api_key)
                    session.state["base_model"] = base_name
                    msgs.append(f"基础模型 → {base_name}")
                if emb:
                    msgs.append(session.switch_models(None, emb))
                if msgs:
                    session._save()
                return {"ok": True,
                        "msg": "；".join(msgs) if msgs else "模型未变化"}
            msg = session.switch_models(base_name, emb)
            return {"ok": True, "msg": msg}
        except OSError as e:
            return {"ok": False, "msg": f"切换失败：{e}"}

    Handler.models_fn = staticmethod(models_handler)
    Handler.switch_fn = staticmethod(switch_handler)
    print(f"自认知外挂测试服务已启动： http://localhost:{args.port}")
    print(f"  后端: {backend} | 基础模型: {args.base_model} | 编码器: {args.embedding_model}")
    if backend == "openai":
        print(f"  API: {api_base}")
    print("按 Ctrl+C 停止")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
