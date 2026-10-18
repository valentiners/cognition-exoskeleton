#!/usr/bin/env python3
"""自认知外挂 · 浏览器测试服务（stdlib-only，无第三方依赖）。

用法：
    python server.py            # 默认 http://localhost:8600
    python server.py --port 8080

主人在浏览器里与外挂对话，完成识别 / 仲裁 / 教学闭环。
这是"部署架构"讨论中的最小验证：Ollama(基础模型+embedding) + voiceofsoul 内核 + 浏览器 UI。
"""

import argparse
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from voiceofsoul import CognitiveModule, MemoryModule
from voiceofsoul.adapters import OllamaBaseModel, OllamaEmbedder

WEB_DIR = Path(__file__).parent / "web"
EXIT_VALUES = {"退出", "exit", "quit"}

# ------------------------------------------------------------------ 全局状态
# 单主人原则：一个服务实例只服务一个主人，会话状态即全局状态。
STATE = {"pending": None}  # 待主人裁决的状态机


def make_handlers(cog: CognitiveModule):
    """返回 (chat(text), respond(value), stats()) 三个处理器。"""

    def finish_entry(name: str, desc: str) -> str:
        """闸口 2（专名/通名自检）+ 闸口 3（建库前去重）。"""
        if cog.is_common_knowledge(name):
            pid = cog.learn_patch(name, {"描述": desc}, patch_type="feature_addition")
            return (f"「{name}」是基础模型已知的通名 → 只存补丁 {pid}，不建条目（防重复）")
        best = cog.memory.best_match(desc)
        if best and best[1] >= cog.mem_th:
            cog.merge_into(best[0], desc, note="闸口 3 查重命中 MERGE")
            return f"与既有条目 {best[0]} 过于相似（{best[1]}）→ MERGE，不建重复条目"
        overlap = next((p for p in cog.memory.patches
                        if p["target"] in name or name in p["target"]), None)
        if overlap:
            cog.memory.update_patch(overlap["id"],
                                    {"event": "内容合并", "note": f"与新建「{name}」重合"})
            return f"与补丁 {overlap['id']} 的 target 重合 → 合并进补丁"
        eid = cog.learn_entry(name, desc, ai_proposed=name)
        return f"专名「{name}」已建主条目 {eid} ✓"

    def say(text: str, actions=None, debug: str = "") -> dict:
        return {"say": text, "actions": actions or [], "debug": debug}

    def chat(text: str) -> dict:
        act = cog.perceive(text)
        kind = act["action"]
        if kind == "RECOGNIZED":
            return say(f"我认出来了——这是「{act['name']}」✓（记忆库命中，置信度 {act['confidence']}）",
                       debug=f"action={kind}")
        if kind == "ARBITRATE":
            STATE["pending"] = {"kind": "arbitrate", "text": text,
                                "candidate_id": act["candidate_id"]}
            return say(f"这个和之前认识的「{act['candidate_name']}」有点像"
                       f"（相似度 {act['similarity']}），是同一个吗？",
                       actions=[{"label": "是同一个", "value": "yes"},
                                {"label": "不是", "value": "no"},
                                {"label": "是它的幼体/子类", "value": "child"}],
                       debug=f"action={kind}")
        if kind == "GUESS":
            STATE["pending"] = {"kind": "guess", "text": text, "guess": act["guess"]}
            return say(f"我猜这是{act['guess']}？（基础模型置信度 {act['confidence']}）",
                       actions=[{"label": "对", "value": "yes"},
                                {"label": "不对", "value": "no"}],
                       debug=f"action={kind}")
        STATE["pending"] = {"kind": "naming", "text": text}
        return say("我承认无法识别这个事物。请主人告诉我它是什么（名字 + 特征描述）",
                   debug=f"action={kind}")

    def respond(value: str) -> dict:
        p = STATE.get("pending")
        if not p:
            return say("当前没有待裁决的事项，请直接描述新事物。")
        kind = p["kind"]

        if kind == "arbitrate":
            STATE["pending"] = None
            if value == "yes":
                cog.merge_into(p["candidate_id"], p["text"], note="灰色地带仲裁 MERGE")
                return say(f"已按主人的裁决 MERGE 进 {p['candidate_id']}（只追加留痕）")
            if value == "child":
                STATE["pending"] = {"kind": "naming", "text": p["text"],
                                    "parent": p["candidate_id"]}
                return say("好的，请主人给这个幼体/子类起名字并描述特征")
            STATE["pending"] = {"kind": "naming", "text": p["text"]}
            return say("好的，那请主人告诉我它是什么（名字 + 特征描述）")

        if kind == "guess":
            if value == "no":
                STATE["pending"] = None
                return say("猜测被主人否定，丢弃不入库。")
            # 闸口 1：GUESS 分支确认回环（设计文档 4.2）
            STATE["pending"] = {"kind": "gate1", "text": p["text"], "guess": p["guess"]}
            return say(f"这就是普通的{p['guess']}，对吗？还是有特别之处想记住？\n"
                       f"（直接回复：「普通」丢弃 / 特征描述存补丁 / "
                       f"「叫<名字>」建专名条目）")

        if kind == "gate1":
            text, guess = p["text"], p["guess"]
            v = value.strip()
            if v in ("普通", "ordinary", "就是普通的"):
                STATE["pending"] = None
                return say("常识不入库（防重复原则），流程结束。")
            m = re.match(r"^(叫|named\s*)\s*(.+)$", v, re.IGNORECASE)
            if m:
                STATE["pending"] = {"kind": "named_desc", "name": m.group(2).strip(),
                                    "guess": guess}
                return say(f"请主人补充「{m.group(2).strip()}」的特征描述")
            pid = cog.learn_patch(guess, {"特征": v}, patch_type="feature_addition")
            STATE["pending"] = None
            return say(f"已存增量补丁 {pid} → target={guess}"
                       f"（基础模型已知此概念，外挂只补粒度）")

        if kind == "named_desc":
            STATE["pending"] = None
            name = p["name"]
            desc = f"{name}，{value.strip()}"
            if p.get("parent"):
                eid = cog.learn_entry(name, desc,
                                      relation={"type": "subtype_of",
                                                "parent": p["parent"]})
                return say(f"已建子类条目 {eid}，挂在 {p['parent']} 下 ✓")
            return say(finish_entry(name, desc))

        if kind == "naming":
            STATE["pending"] = None
            desc = value.strip()
            name = re.split(r"[,，]", desc)[0].strip()
            if p.get("parent"):
                eid = cog.learn_entry(name, desc,
                                      relation={"type": "subtype_of",
                                                "parent": p["parent"]})
                return say(f"已建子类条目 {eid}，挂在 {p['parent']} 下 ✓")
            return say(finish_entry(name, desc))

        STATE["pending"] = None
        return say("未知状态，已重置。请继续描述新事物。")

    def stats() -> dict:
        s = cog.memory.stats()
        entries = [{"id": eid, "name": e["name"],
                    "hits": e.get("health", {}).get("hit_count", 0),
                    "errors": e.get("threshold_meta", {}).get("mistake_count", 0)}
                   for eid, e in cog.memory.entries.items()]
        patches = [{"id": p["id"], "target": p["target"], "type": p["type"],
                    "status": p["status"], "content": p["content"]}
                   for p in cog.memory.patches]
        return {"stats": s, "entries": entries, "patches": patches,
                "pending": STATE["pending"]}

    return chat, respond, stats


# ------------------------------------------------------------------ HTTP 层

class Handler(BaseHTTPRequestHandler):
    chat = respond = stats_fn = None  # 由 main 注入

    def log_message(self, *args):  # 静默日志
        pass

    def _json(self, payload: dict, code: int = 200):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
            body = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/stats":
            self._json(self.stats_fn())
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
            if not text:
                return self._json({"error": "empty text"}, 400)
            if text in EXIT_VALUES:
                return self._json({"say": "会话不退出（服务常驻），直接继续描述新事物即可。",
                                   "actions": [], "debug": ""})
            self._json(self.chat(text))
        elif self.path == "/api/respond":
            self._json(self.respond((payload.get("value") or "").strip()))
        else:
            self._json({"error": "not found"}, 404)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8600)
    ap.add_argument("--base-model", default="Qwen3.8:latest")
    ap.add_argument("--embedding-model", default="mxbai-embed-large")
    ap.add_argument("--ollama", default="http://localhost:11434")
    ap.add_argument("--memory", default="memory.json")
    ap.add_argument("--mem-th", type=float, default=0.70,
                    help="记忆库命中阈值（需按所用编码器标定，设计文档 5.3）")
    ap.add_argument("--base-th", type=float, default=0.51,
                    help="基础模型猜测阈值")
    args = ap.parse_args()

    embedder = OllamaEmbedder(model=args.embedding_model, host=args.ollama)
    base = OllamaBaseModel(model=args.base_model, host=args.ollama)
    memory = MemoryModule(args.memory, embedder)
    cog = CognitiveModule(base, memory, embedder,
                          base_th=args.base_th, mem_th=args.mem_th)

    Handler.chat, Handler.respond, Handler.stats_fn = (
        staticmethod(f) for f in make_handlers(cog))
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"自认知外挂测试服务已启动： http://localhost:{args.port}")
    print(f"  基础模型: {args.base_model} | 编码器: {args.embedding_model}")
    print("按 Ctrl+C 停止")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
