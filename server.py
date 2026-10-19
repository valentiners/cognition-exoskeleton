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
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from voiceofsoul import CognitiveModule, MemoryModule
from voiceofsoul.adapters import OllamaBaseModel, OllamaEmbedder
from voiceofsoul.temp_memory import TempMemory

WEB_DIR = Path(__file__).parent / "web"
EXIT_VALUES = {"退出", "exit", "quit"}
# 常识问句特征：直通基础模型回答，不进识别/记忆流程
QA_PATTERN = re.compile(r"(是什么|什么是|为什么|怎么|如何|哪里|谁|吗|呢|么|？|\?)")

# ------------------------------------------------------------------ 全局状态
# 单主人原则：一个服务实例只服务一个主人，会话状态即全局状态。
STATE = {"pending": None, "topic": None, "base_model": "", "embedding_model": ""}


def make_handlers(cog: CognitiveModule, temp):
    """返回 (chat(text), respond(value), stats()) 三个处理器。

    temp：临时记忆（七天法则）——新特征先暂存，主人要求或七天三次才晋升永久。
    """

    def finish_entry(name: str, desc: str, modality: str = "text") -> str:
        """闸口 2（公共/私人自检）+ 闸口 3（建库前去重）。"""
        if cog.is_common_knowledge(name, desc):
            pid = cog.learn_patch(name, {"描述": desc}, patch_type="feature_addition")
            return f"「{name}」听起来是大家都知道的公共知识，我就不占专属记忆了，只留个小补丁备查～"
        best = cog.memory.best_match(desc)
        if best and best[1] >= cog.mem_th:
            cog.merge_into(best[0], desc, note="闸口 3 查重命中 MERGE")
            return "这个和我已有的记忆太像了，我就把它并进原来的条目，不重复占地方。"
        overlap = next((p for p in cog.memory.patches
                        if p["target"] in name or name in p["target"]), None)
        if overlap:
            cog.memory.update_patch(overlap["id"],
                                    {"event": "内容合并", "note": f"与新建「{name}」重合"})
            return "和已有的补丁重合了，我把它合过去了～"
        cog.learn_entry(name, desc, ai_proposed=name, source_modality=modality)
        return f"「{name}」我记住啦，以后见到就认得 ✓"

    def finish_named(topic: dict, name: str) -> str:
        """兼容旧流程：把话题里已缓存的特征拼成描述走三道闸口。"""
        feats = topic.get("features", {})
        extra = "；".join(f"{k}：{v}" for k, v in feats.items())
        desc = topic.get("desc", "")
        desc = f"{desc}；{extra}" if extra else desc
        return finish_entry(name, desc, modality=topic.get("modality", "text"))

    def say(text: str, actions=None, debug: str = "") -> dict:
        return {"say": text, "actions": actions or [], "debug": debug}

    # ---------- 临时记忆 → 永久记忆的晋升 ----------
    def promote_temp(tid: str, name: str | None = None) -> str:
        """把一条临时记忆晋升进永久记忆（走闸口 2/3）。缺名字时转而追问主人。"""
        rec = temp.get(tid)
        if not rec:
            return "咦，这条记忆好像已经不在了……"
        feats = rec["features"]
        name = (name or feats.get("名字") or "").strip()
        if not name:
            STATE["pending"] = {"kind": "promote_name", "temp_id": tid}
            return "好，我要把它永远记住。那我该怎么称呼它？告诉我名字就好。"
        desc = rec["desc"]
        extra = "；".join(f"{k}：{v}" for k, v in feats.items())
        if extra:
            desc = f"{desc}；{extra}"
        temp.remove(tid)
        STATE["topic"] = None
        return finish_entry(name, desc, modality=rec.get("modality", "text"))

    def handle_remember(topic: dict) -> str:
        """主人要求记住当前话题：有临时记录就晋升，没有就先暂存再晋升。"""
        tid = topic.get("temp_id")
        if not tid:
            tid = temp.add(topic["desc"], modality=topic.get("modality", "text"))
            topic["temp_id"] = tid
        if topic.get("kind") == "recognized" and topic.get("entry_id"):
            rec = temp.get(tid)
            if rec:
                e = cog.memory.entries[topic["entry_id"]]
                e["features"].update(rec["features"])
                extra = "；".join(f"{k}：{v}" for k, v in rec["features"].items())
                e["features"]["embedding"] = cog.embedder.embed(
                    e["features"].get("描述", "") + ("；" + extra if extra else ""))
                cog.memory._persist()
                temp.remove(tid)
                return f"好～都写进「{e['name']}」的长期记忆里了 ✓"
        return promote_temp(tid)

    def fresh_topic(topic):
        """话题超过半小时没续上就作废（鲜活生命体不会无限盯着旧话题）。"""
        import time as _t
        return topic and (_t.time() - topic.get("at", 0)) < 1800

    def chat(text: str, image_b64: str | None = None) -> dict:
        # 0a. 晋升追问命名的回答："它叫什么"的自由文本应答
        p = STATE.get("pending")
        if p and p.get("kind") == "promote_name" and text.strip() and not image_b64:
            STATE["pending"] = None
            return say(promote_temp(p["temp_id"], name=text.strip()))

        # 0b. 未决流程的自由文本应答：主人的一句话就是回答，不用按钮
        p = STATE.get("pending")
        if p and not image_b64 and text.strip():
            if re.search(r"^(算了|取消|没事了)$", text.strip()):
                STATE["pending"] = None
                STATE["topic"] = None
                return say("好～那就当没这回事。")
            STATE["pending"] = None  # 交给 respond 处理；它内部会重建需要的 pending
            return respond(text.strip())

        STATE["pending"] = None  # 主人发来新输入，未决裁决自动取消
        gone = temp.expire()
        if gone:
            print(f"[临时记忆] 遗忘了 {len(gone)} 件久未提起的事：{', '.join(gone)}")

        # 常识问答直通：纯问题不进识别/记忆流程，基础模型能答就直接答
        # （设计文档 4.1 主规则：能被基础模型压缩知识完整覆盖 → 不建条目）
        if not image_b64 and QA_PATTERN.search(text):
            ans = getattr(cog.base, "answer_question", lambda _: None)(text)
            if ans:
                return say(ans, debug="action=QA_PASS（基础模型覆盖，不走记忆流程）")

        # 对话跟进：主人在聊刚才那个事物吗？——像聊天一样判断意图，不弹按钮
        import time as _t
        topic = STATE.get("topic")
        if fresh_topic(topic) and not image_b64 and not QA_PATTERN.search(text):
            verdict = getattr(cog.base, "classify_followup", lambda c, u: ("none",))(
                topic["desc"], text)
            if verdict[0] == "remember":
                return say(handle_remember(topic))
            if verdict[0] == "forget":
                tid = topic.get("temp_id")
                if tid:
                    temp.remove(tid)
                STATE["topic"] = None
                return say("好，我这就忘掉，当我没听过～")
            if verdict[0] == "feature":
                attr, value = verdict[1], verdict[2]
                tid = topic.get("temp_id")
                if not tid:
                    tid = temp.add(topic["desc"],
                                   modality=topic.get("modality", "text"))
                    topic["temp_id"] = tid
                temp.update_features(tid, attr, value)
                if attr == "名字":
                    return say(f"哦～原来叫「{value}」呀，我悄悄记下了。")
                return say(f"嗯嗯，{attr}是「{value}」，我悄悄记下了。")
        elif topic and not fresh_topic(topic):
            STATE["topic"] = None  # 旧话题自然冷却

        # 感知通道 A：图像先经视觉模型转为统一描述，再走同一条比对流程
        desc = None
        if image_b64:
            desc = cog.base.perceive_image(image_b64, text)
            if not desc:
                return say("唔，这张图我没看明白……能换一张，或者用文字告诉我你看到的是什么吗？")
            text = f"{text}，{desc}" if text else desc
        modality = "image" if image_b64 else "text"  # 感知通道，随条目入库留痕

        # 临时记忆命中：新事物七天法则——今天又见到了吗？
        trec = temp.best_match(text)
        if trec:
            tid, score = trec
            rec = temp.get(tid)
            promoted = temp.touch(tid)
            STATE["topic"] = {"kind": "temp", "temp_id": tid, "desc": rec["desc"],
                              "modality": rec.get("modality", "text"), "at": _t.time()}
            if promoted:  # 七天内不同日期出现≥3次 → 自动晋升永久记忆
                nm = rec["features"].get("名字")
                msg = promote_temp(tid, name=nm)
                lead = (f"咦，这不就是{'「' + nm + '」' if nm else '之前聊过的那个'}嘛！"
                        f"它这几天老是出现，我就把它正式记住了——")
                return say(lead + msg,
                           debug=f"action=TEMP_PROMOTE sim={score}")
            nm = rec["features"].get("名字")
            greet = (f"这不是{nm}嘛，我记得的呀～" if nm
                     else "这个我记得，我们之前聊过的～")
            return say(greet, debug=f"action=TEMP_HIT days={len(rec['days'])} sim={score}")

        act = cog.perceive(text)
        kind = act["action"]
        if kind == "RECOGNIZED":
            STATE["topic"] = {"kind": "recognized", "desc": text, "modality": modality,
                              "entry_id": act["entry_id"], "at": _t.time()}
            return say(f"嗯……这个是「{act['name']}」呀，我认得的～",
                       debug=f"action={kind} via={act.get('via')} conf={act['confidence']}")
        if kind == "DIRECT":
            STATE["topic"] = {"kind": "direct", "desc": text,
                              "modality": modality, "at": _t.time()}
            if desc:  # 图片输入：先讲"我看到一张图片"，再自然描述
                return say(f"我看到一张图片——{desc}\n"
                           f"这类事物我本来就认识，就不占用我的长期记忆啦。"
                           f"不过，要是它对你有什么特别的意义，悄悄告诉我它的名字，"
                           f"我会认真记住的。",
                           debug=f"action={kind} conf={act['confidence']}")
            return say(f"这个我本来就认识：{act['guess']}～\n"
                       f"如果它对你有特别的意义，告诉我它的名字和特征，"
                       f"我会把它变成长期记忆。",
                       debug=f"action={kind} conf={act['confidence']}")
        if kind == "ARBITRATE":
            STATE["topic"] = None
            STATE["pending"] = {"kind": "arbitrate", "text": text,
                                "candidate_id": act["candidate_id"],
                                "modality": modality}
            return say("唔，这个和我之前认识的「{}」有点像……它们是同一个吗？".format(
                           act["candidate_name"]),
                       actions=[{"label": "是同一个", "value": "yes"},
                                {"label": "不是", "value": "no"},
                                {"label": "是它的幼体/子类", "value": "child"}],
                       debug=f"action={kind} sim={act['similarity']}")
        if kind == "GUESS":
            STATE["topic"] = {"kind": "guess", "desc": text, "modality": modality,
                              "guess": act["guess"], "at": _t.time()}
            STATE["pending"] = {"kind": "guess", "text": text, "guess": act["guess"],
                                "modality": modality}
            return say(f"嗯……让我猜猜，这是{act['guess']}？",
                       actions=[{"label": "对", "value": "yes"},
                                {"label": "不对", "value": "no"}],
                       debug=f"action={kind} conf={act['confidence']}")
        STATE["topic"] = {"kind": "unknown", "desc": text,
                          "modality": modality, "at": _t.time()}
        STATE["pending"] = {"kind": "naming", "text": text, "modality": modality}
        return say("唔……这个我确实不认识。你愿意教教我吗？"
                   "告诉我它叫什么、长什么样就好～",
                   debug=f"action={kind}")

    def respond(value: str) -> dict:
        p = STATE.get("pending")
        if not p:
            return say("嗯？现在没有等我拿主意的事情呀，你直接说就好。")
        kind = p["kind"]
        v = value.strip()

        # 自由文本的模糊理解：主人的自然表达 → 内部裁决值
        def as_no(s):
            return bool(re.search(r"不|否|没|错|no\b|nope|wrong", s, re.IGNORECASE))
        def as_yes(s):
            return bool(re.search(r"是|对|嗯|好|没错|yes|yeah|yep", s, re.IGNORECASE))

        if kind == "arbitrate":
            if value == "child" or re.search(r"子类|幼体|分支|小的", v):
                STATE["pending"] = {"kind": "naming", "text": p["text"],
                                    "parent": p["candidate_id"],
                                    "modality": p.get("modality", "text")}
                return say("明白，那它是新的一支。给它起个名字，再简单说说特征吧。")
            if value == "yes" or (as_yes(v) and not as_no(v)):
                STATE["pending"] = None
                cog.merge_into(p["candidate_id"], p["text"], note="灰色地带仲裁 MERGE")
                return say("好，那我就把它并到原来的记忆里，只补充、不改动旧的～")
            if value == "no" or as_no(v):
                STATE["pending"] = None
                STATE["pending"] = {"kind": "naming", "text": p["text"],
                                    "modality": p.get("modality", "text")}
                return say("好，那它是另一个事物。告诉我它叫什么、长什么样吧。")
            STATE["pending"] = p  # 没听懂，再问一次
            return say("唔……你是说这是同一个，还是另一个呀？")

        if kind == "guess":
            if value == "no" or (as_no(v) and not as_yes(v)):
                STATE["pending"] = None
                return say("啊，猜错了……那先不记，等你想教我的时候再说。")
            # 闸口 1：GUESS 分支确认回环（设计文档 4.2）
            STATE["pending"] = {"kind": "gate1", "text": p["text"], "guess": p["guess"],
                                "modality": p.get("modality", "text")}
            return say(f"那这就是普通的{p['guess']}，对吗？还是它有特别之处？\n"
                       f"（直接回我：「普通」/ 特征描述 / 「叫<名字>」）")

        if kind == "gate1":
            text, guess = p["text"], p["guess"]
            if v in ("普通", "ordinary", "就是普通的") or \
                    (as_yes(v) and not as_no(v) and len(v) <= 4):
                STATE["pending"] = None
                return say("好～常识就不占用我的记忆了。")
            m = re.match(r"^(叫|named\s*)\s*(.+)$", v, re.IGNORECASE)
            if m:
                STATE["pending"] = {"kind": "named_desc", "name": m.group(2).strip(),
                                    "guess": guess,
                                    "modality": p.get("modality", "text")}
                return say(f"好呀，那「{m.group(2).strip()}」是什么样的呢？简单说说特征就行。")
            pid = cog.learn_patch(guess, {"特征": v}, patch_type="feature_addition")
            STATE["pending"] = None
            return say(f"记下啦，作为小补丁（{pid}）挂在「{guess}」下面～")

        if kind == "named_desc":
            STATE["pending"] = None
            STATE["topic"] = None
            name = p["name"]
            desc = f"{name}，{value.strip()}"
            if p.get("parent"):
                cog.learn_entry(name, desc,
                                relation={"type": "subtype_of",
                                          "parent": p["parent"]},
                                source_modality=p.get("modality", "text"))
                return say(f"记住啦，它是挂在原来那条下面的新一支 ✓")
            return say(finish_entry(name, desc, modality=p.get("modality", "text")))

        if kind == "naming":
            STATE["pending"] = None
            STATE["topic"] = None
            desc = value.strip()
            name = re.split(r"[,，]", desc)[0].strip()
            if p.get("parent"):
                cog.learn_entry(name, desc,
                                relation={"type": "subtype_of",
                                          "parent": p["parent"]},
                                source_modality=p.get("modality", "text"))
                return say("记住啦，它是挂在原来那条下面的新一支 ✓")
            return say(finish_entry(name, desc, modality=p.get("modality", "text")))

        STATE["pending"] = None
        return say("唔，我刚才有点懵，已经理清了。我们继续～")

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
                "temp": temp.stats(), "pending": STATE["pending"]}

    return chat, respond, stats


# ------------------------------------------------------------------ HTTP 层

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


# ------------------------------------------------------------------ 模型管理
# Ollama /api/tags 只给名单，模型分类要靠 /api/show 的 capabilities 字段：
# 含 "embedding" → embedding 模型；含 "completion" → 基础模型。结果缓存。
_MODEL_CACHE: dict[str, list] = {}


def _get_json(host: str, path: str, timeout: float = 30.0) -> dict:
    with urllib.request.urlopen(host.rstrip("/") + path, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def list_models(host: str) -> dict:
    bases, embeds = [], []
    for m in _get_json(host, "/api/tags").get("models", []):
        name = m["name"]
        caps = _MODEL_CACHE.get(name)
        if caps is None:
            try:
                req = urllib.request.Request(
                    host.rstrip("/") + "/api/show",
                    data=json.dumps({"name": name}).encode(),
                    headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=30) as r:
                    caps = json.loads(r.read().decode()).get("capabilities", [])
            except OSError:
                caps = []
            _MODEL_CACHE[name] = caps
        if "embedding" in caps:
            embeds.append(name)
        elif "completion" in caps:
            bases.append(name)
    return {"base": sorted(bases), "embedding": sorted(embeds)}


def switch_models(cog, temp, host: str, base: str | None, embedding: str | None) -> str:
    """运行时热切换模型。切换 embedding 时按设计文档 3.3 重建全部记忆向量。"""
    msgs = []
    if base:
        cog.base = OllamaBaseModel(model=base, host=host)
        STATE["base_model"] = base
        msgs.append(f"基础模型 → {base}")
    if embedding:
        new_emb = OllamaEmbedder(model=embedding, host=host)
        cog.embedder = new_emb
        cog.memory.embedder = new_emb
        temp.embedder = new_emb
        STATE["embedding_model"] = embedding
        n = 0
        for e in cog.memory.entries.values():
            desc = e.get("features", {}).get("描述")
            if desc:
                e["features"]["embedding"] = new_emb.embed(desc)
                e.setdefault("embedding_meta", {})["model"] = new_emb.model_tag
                n += 1
        cog.memory._persist()
        m = 0
        for r in temp.records.values():
            r["embedding"] = new_emb.embed(r["desc"])
            m += 1
        temp._persist()
        msgs.append(f"embedding 模型 → {embedding}（已重建 {n} 条永久 + {m} 条临时记忆向量）")
    return "；".join(msgs) if msgs else "模型未变化"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8600)
    ap.add_argument("--base-model", default="gemma4:latest")
    ap.add_argument("--embedding-model", default="embeddinggemma-2:latest")
    ap.add_argument("--ollama", default="http://localhost:11434")
    ap.add_argument("--memory", default="memory.json")
    ap.add_argument("--temp-memory", default="temp_memory.json",
                    help="临时记忆文件（七天法则：七天内不同日期出现≥3次自动晋升永久）")
    ap.add_argument("--mem-th", type=float, default=0.78,
                    help="记忆库命中阈值（需按所用编码器标定，设计文档 5.3）")
    ap.add_argument("--base-th", type=float, default=0.51,
                    help="基础模型猜测阈值")
    args = ap.parse_args()

    embedder = OllamaEmbedder(model=args.embedding_model, host=args.ollama)
    base = OllamaBaseModel(model=args.base_model, host=args.ollama)
    memory = MemoryModule(args.memory, embedder)
    temp = TempMemory(args.temp_memory, embedder)
    cog = CognitiveModule(base, memory, embedder,
                          base_th=args.base_th, mem_th=args.mem_th)
    STATE["base_model"], STATE["embedding_model"] = args.base_model, args.embedding_model

    Handler.chat, Handler.respond, Handler.stats_fn = (
        staticmethod(f) for f in make_handlers(cog, temp))
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)

    # 模型列表与热切换（挂在 Handler 类上，保持与 chat/respond 同一注入方式）
    def models_handler():
        d = list_models(args.ollama)
        d["current"] = {"base": STATE["base_model"],
                        "embedding": STATE["embedding_model"]}
        return d

    def switch_handler(payload: dict):
        base = (payload.get("base") or "").strip() or None
        emb = (payload.get("embedding") or "").strip() or None
        try:
            msg = switch_models(cog, temp, args.ollama, base, emb)
            return {"ok": True, "msg": msg}
        except OSError as e:
            return {"ok": False, "msg": f"切换失败：{e}"}

    Handler.models_fn = staticmethod(models_handler)
    Handler.switch_fn = staticmethod(switch_handler)
    print(f"自认知外挂测试服务已启动： http://localhost:{args.port}")
    print(f"  基础模型: {args.base_model} | 编码器: {args.embedding_model}")
    print("按 Ctrl+C 停止")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
