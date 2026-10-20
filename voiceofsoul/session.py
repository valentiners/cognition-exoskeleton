"""会话编排（一个大脑，两个壳）。

HTTP 测试服务（server.py）与 CLI（voiceofsoul/exo.py）共用本模块：
- 话题 30 分钟冷却；
- 未决流程（naming / guess / gate1 / arbitrate / promote_name）的自由文本路由；
- 仲裁挂起超时 → 临时记忆降级（设计文档 §4.5）；
- 临时记忆七天法则、特征供给、识别与问答直通。

状态持久化到 <COGSOUL_HOME>/session.json：CLI 每次消息都是一个新进程，
所以每次调用都「加载 → 跑一轮 → 保存」。并发只做 last-writer-wins，
不引入文件锁（任务书 §8.1 已拍板）；多个进程同时写可能互相覆盖，
这是 primitive 版本的已知限制。

内核纪律：本模块只使用标准库；所有 Ollama HTTP 调用都留在
voiceofsoul/adapters/ollama.py。本模块通过 cog.base / cog.embedder
抽象点工作。
"""

from __future__ import annotations

import json
import re
import sys
import time as _t
from pathlib import Path

from .memory import is_dormant

# 常识问句特征：直通基础模型回答，不进识别/记忆流程
QA_PATTERN = re.compile(r"(是什么|什么是|为什么|怎么|如何|哪里|谁|吗|呢|么|？|\?)")
# "再想想"唤醒休眠记忆（§6.4 L3）：必须在问答直通之前判定——
# "你忘了吗"含"吗"会被问句规则截胡
RECALL_PATTERN = re.compile(r"(再想想|好好想想|仔细想想|你忘了吗|你忘了)")
# 仲裁挂起超时（与话题冷却同口径）：超时未答 → 降级暂存临时记忆（§4.5）
PENDING_TTL = 1800
DEGRADE_KINDS = {"naming", "guess", "gate1", "arbitrate"}  # promote_name 不适用
TOPIC_TTL = 1800
DEFAULT_HOME = Path.home() / ".cognition-exoskeleton"


def load_state(home: str | Path | None = None) -> dict:
    """只读加载 session.json；不存在或损坏时返回空状态。

    CLI 在构造适配器之前需要知道上次选择的模型，因此提供这个轻量入口。
    """
    path = Path(home).expanduser() if home else DEFAULT_HOME
    p = path / "session.json"
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


class Session:
    """单主人会话状态机；服务器与 CLI 共用。"""

    def __init__(self, cog, temp, home: str | Path | None = None,
                 state: dict | None = None,
                 base_model: str = "", embedding_model: str = "",
                 ollama_host: str = ""):
        self.cog = cog
        self.temp = temp
        self.home = Path(home).expanduser() if home else DEFAULT_HOME
        self.home.mkdir(parents=True, exist_ok=True)
        self.path = self.home / "session.json"
        self.state = dict(state) if state is not None else self._load()
        self.state.setdefault("pending", None)
        self.state.setdefault("topic", None)

        self.ollama_host = (ollama_host or self.state.get("ollama_host")
                            or "http://localhost:11434")
        self.state["ollama_host"] = self.ollama_host
        if base_model:
            self.state["base_model"] = base_model
        elif not self.state.get("base_model"):
            self.state["base_model"] = getattr(cog.base, "model", "")
        if embedding_model:
            self.state["embedding_model"] = embedding_model
        elif not self.state.get("embedding_model"):
            self.state["embedding_model"] = getattr(cog.embedder, "model_tag", "")

    # ---------- 持久化 ----------
    def _load(self) -> dict:
        if not self.path.exists():
            return {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            # 运行状态损坏时留档再重建，不阻塞下一轮对话。
            try:
                self.path.replace(self.path.with_name(
                    f"session.json.corrupt-{int(_t.time())}"))
            except OSError:
                pass
            return {}

    def _save(self):
        """原子写 session.json；并发只保证 last-writer-wins。"""
        payload = {
            "pending": self.state.get("pending"),
            "topic": self.state.get("topic"),
            "base_model": self.state.get("base_model", ""),
            "embedding_model": self.state.get("embedding_model", ""),
            "ollama_host": self.state.get("ollama_host", ""),
        }
        tmp = self.path.with_name("session.json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        tmp.replace(self.path)

    # ---------- 回复形状 ----------
    @staticmethod
    def _say(text: str, actions=None, debug: str = "") -> dict:
        return {"say": text, "actions": actions or [], "debug": debug}

    # ---------- 闸口 2 / 闸口 3 ----------
    def _finish_entry(self, name: str, desc: str, modality: str = "text") -> str:
        """闸口 2（公共/私人自检）+ 闸口 3（建库前去重）。"""
        if self.cog.is_common_knowledge(name, desc):
            self.cog.learn_patch(name, {"描述": desc}, patch_type="feature_addition")
            return f"「{name}」听起来是大家都知道的公共知识，我就不占专属记忆了，只留个小补丁备查～"
        best = self.cog.memory.best_match(desc, include_dormant=True)  # 查重不受休眠限制
        if best and best[1] >= self.cog.mem_th:
            self.cog.merge_into(best[0], desc, note="闸口 3 查重命中 MERGE")
            return "这个和我已有的记忆太像了，我就把它并进原来的条目，不重复占地方。"
        overlap = next((p for p in self.cog.memory.patches
                        if p["target"] in name or name in p["target"]), None)
        if overlap:
            self.cog.memory.update_patch(overlap["id"],
                                         {"event": "内容合并",
                                          "note": f"与新建「{name}」重合"})
            return "和已有的补丁重合了，我把它合过去了～"
        self.cog.learn_entry(name, desc, ai_proposed=name,
                             source_modality=modality)
        return f"「{name}」我记住啦，以后见到就认得 ✓"

    def _finish_named(self, topic: dict, name: str) -> str:
        """兼容旧流程：把话题里已缓存的特征拼成描述走三道闸口。"""
        feats = topic.get("features", {})
        extra = "；".join(f"{k}：{v}" for k, v in feats.items())
        desc = topic.get("desc", "")
        desc = f"{desc}；{extra}" if extra else desc
        return self._finish_entry(name, desc, modality=topic.get("modality", "text"))

    # ---------- 临时记忆 → 永久记忆的晋升 ----------
    def _promote_temp(self, tid: str, name: str | None = None) -> str:
        """把一条临时记忆晋升进永久记忆（走闸口 2/3）。缺名字时转而追问主人。"""
        rec = self.temp.get(tid)
        if not rec:
            return "咦，这条记忆好像已经不在了……"
        feats = rec["features"]
        name = (name or feats.get("名字") or "").strip()
        if not name:
            self.state["pending"] = {"kind": "promote_name", "temp_id": tid,
                                     "at": _t.time()}
            return "好，我要把它永远记住。那我该怎么称呼它？告诉我名字就好。"
        desc = rec["desc"]
        extra = "；".join(f"{k}：{v}" for k, v in feats.items())
        if extra:
            desc = f"{desc}；{extra}"
        self.temp.remove(tid)
        self.state["topic"] = None
        return self._finish_entry(name, desc, modality=rec.get("modality", "text"))

    def _handle_remember(self, topic: dict) -> str:
        """主人要求记住当前话题：有临时记录就晋升，没有就先暂存再晋升。"""
        tid = topic.get("temp_id")
        if not tid:
            tid = self.temp.add(topic["desc"], modality=topic.get("modality", "text"))
            topic["temp_id"] = tid
        if topic.get("kind") == "recognized" and topic.get("entry_id"):
            rec = self.temp.get(tid)
            if rec:
                e = self.cog.memory.entries[topic["entry_id"]]
                e["features"].update(rec["features"])
                extra = "；".join(f"{k}：{v}" for k, v in rec["features"].items())
                e["features"]["embedding"] = self.cog.embedder.embed(
                    e["features"].get("描述", "") + ("；" + extra if extra else ""))
                self.cog.memory._persist()
                self.temp.remove(tid)
                return f"好～都写进「{e['name']}」的长期记忆里了 ✓"
        return self._promote_temp(tid)

    def _fresh_topic(self, topic) -> bool:
        """话题超过半小时没续上就作废（鲜活生命体不会无限盯着旧话题）。"""
        return bool(topic) and (_t.time() - topic.get("at", 0)) < TOPIC_TTL

    # ---------- 仲裁超时降级（§4.5） ----------
    @staticmethod
    def _pending_expired(p) -> bool:
        return bool(p) and bool(p.get("at")) and (_t.time() - p["at"]) > PENDING_TTL

    def _degrade_pending(self, p):
        """超时未答的仲裁挂起项 → 静默写入临时记忆，七天法则兜底。

        铁律：只暂存、不进永久库、不做 MERGE；主人明确取消不走这里（真丢弃）。
        诊断信息写 stderr，保证 CLI 的 stdout 只有一份 JSON。
        """
        if not p or p.get("kind") not in DEGRADE_KINDS or not p.get("text"):
            return
        if self.temp.best_match(p["text"]):  # 同一件事的近似项已在册，不重复暂存
            return
        tid = self.temp.add(p["text"], modality=p.get("modality", "text"))
        print(f"[仲裁降级] 挂起项超时未答，已暂存临时记忆 {tid}：{p['text'][:40]}",
              file=sys.stderr)

    # ---------- 一轮对话 ----------
    def chat(self, text: str, image_b64: str | None = None) -> dict:
        try:
            result = self._chat(text, image_b64)
        except Exception:
            # 失败时丢弃本轮内存态，磁盘上的 session.json 保持上一轮有效状态。
            self.state = self._load()
            raise
        self._save()
        return result

    def _chat(self, text: str, image_b64: str | None = None) -> dict:
        # 0a. 晋升追问命名的回答："它叫什么"的自由文本应答
        p = self.state.get("pending")
        if p and p.get("kind") == "promote_name" and text.strip() and not image_b64:
            self.state["pending"] = None
            return self._say(self._promote_temp(p["temp_id"], name=text.strip()))

        # 0a5. "再想想"：唤醒沉睡的记忆（§6.4）——分层翻到休眠层，命中即复活
        p = self.state.get("pending")
        topic_now = self.state.get("topic")
        if not image_b64 and RECALL_PATTERN.search(text) and \
                (p or self._fresh_topic(topic_now)):
            src = p or topic_now
            stimulus = src.get("text") or src.get("desc", "")
            act = self.cog.recall(stimulus)
            if act["action"] == "RECOGNIZED":
                self.state["pending"] = None
                self.state["topic"] = {"kind": "recognized", "desc": stimulus,
                                       "modality": src.get("modality", "text"),
                                       "entry_id": act["entry_id"], "at": _t.time()}
                if act.get("layer") == "dormant":
                    return self._say(
                        f"啊……我想起来了！这是「{act['name']}」呀～",
                        debug=f"action=RECALL_DORMANT sim={act['confidence']}")
                return self._say(
                    f"嗯……这个是「{act['name']}」呀，我认得的～",
                    debug=f"action=RECALL sim={act['confidence']} layer={act.get('layer')}")
            if act["action"] == "GUESS":
                return self._say(
                    f"唔……我有点印象，又不太敢确定——是「{act['guess']}」吗？",
                    debug=f"action=RECALL_GUESS sim={act['confidence']} layer={act.get('layer')}")
            return self._say("唔……我把记忆的角落都翻了一遍，真的没有印象……"
                             "要不你提醒我一下？",
                             debug="action=RECALL_MISS（未决流程保持原状）")

        # 0b. 未决流程的自由文本应答：主人的一句话就是回答，不用按钮
        # （纯问句除外——那是新问题，不是回答，交给下方问答/识别流程）
        p = self.state.get("pending")
        if p and not image_b64 and text.strip() and not QA_PATTERN.search(text):
            if re.search(r"^(算了|取消|没事了)$", text.strip()):
                self.state["pending"] = None
                self.state["topic"] = None
                return self._say("好～那就当没这回事。")
            if self._pending_expired(p):  # 超时未答 → 降级暂存，答复不再路由进旧流程
                self._degrade_pending(p)
                self.state["pending"] = None  # 落入下方，按新输入正常处理
            else:
                return self._respond(text.strip())  # respond 内部读取并清理 pending

        if self._pending_expired(self.state.get("pending")):  # 超时挂起先降级
            self._degrade_pending(self.state["pending"])
        self.state["pending"] = None  # 主人发来新输入，未决裁决自动取消
        gone = self.temp.expire()
        if gone:
            print(f"[临时记忆] 遗忘了 {len(gone)} 件久未提起的事：{', '.join(gone)}",
                  file=sys.stderr)

        # 常识问答直通：纯问题不进识别/记忆流程，基础模型能答就直接答。
        # 有新鲜话题时带上上下文——"他是谁？"这类指代问题脱离语境会答崩（实测教训）。
        if not image_b64 and QA_PATTERN.search(text):
            topic = self.state.get("topic")
            ctx = ""
            if self._fresh_topic(topic):
                ctx = topic.get("desc", "")
                if topic.get("kind") == "recognized" and topic.get("entry_id"):
                    e = self.cog.memory.entries.get(topic["entry_id"])
                    if e:
                        ctx = f"它叫「{e['name']}」。" + ctx
                elif topic.get("temp_id"):
                    rec = self.temp.get(topic["temp_id"])
                    if rec and rec.get("features"):
                        extra = "；".join(f"{k}：{v}" for k, v in rec["features"].items())
                        ctx = f"{ctx}；{extra}"
            ans = getattr(self.cog.base, "answer_question",
                          lambda q, c="": None)(text, ctx)
            if ans:
                return self._say(ans, debug="action=QA_PASS（基础模型覆盖，不走记忆流程）")

        # 对话跟进：主人在聊刚才那个事物吗？——像聊天一样判断意图，不弹按钮
        topic = self.state.get("topic")
        if self._fresh_topic(topic) and not image_b64 and not QA_PATTERN.search(text):
            verdict = getattr(self.cog.base, "classify_followup",
                              lambda c, u: ("none",))(topic["desc"], text)
            if verdict[0] == "remember":
                return self._say(self._handle_remember(topic))
            if verdict[0] == "forget":
                tid = topic.get("temp_id")
                if tid:
                    self.temp.remove(tid)
                self.state["topic"] = None
                return self._say("好，我这就忘掉，当我没听过～")
            if verdict[0] == "feature":
                attr, value = verdict[1], verdict[2]
                tid = topic.get("temp_id")
                if not tid:
                    tid = self.temp.add(topic["desc"],
                                        modality=topic.get("modality", "text"))
                    topic["temp_id"] = tid
                self.temp.update_features(tid, attr, value)
                if attr == "名字":
                    return self._say(f"哦，他叫{value}呀，我记下了。")
                return self._say(f"嗯嗯，{attr}是「{value}」，我记下了。")
        elif topic and not self._fresh_topic(topic):
            self.state["topic"] = None  # 旧话题自然冷却

        # 感知通道 A：图像先经视觉模型转为统一描述，再走同一条比对流程
        desc = None
        if image_b64:
            hint = text if text and not QA_PATTERN.search(text) else ""
            desc = self.cog.base.perceive_image(image_b64, hint)
            if not desc:
                return self._say("唔，这张图我没看明白……能换一张，或者用文字告诉我你看到的是什么吗？")
            text = f"{text}，{desc}" if text else desc
        modality = "image" if image_b64 else "text"  # 感知通道，随条目入库留痕

        # 临时记忆命中：新事物七天法则——今天又见到了吗？
        trec = self.temp.best_match(text)
        if trec:
            tid, score = trec
            rec = self.temp.get(tid)
            # 图像暂存的个体同样要过同一性复核，防同类不同个体误命中（与永久条目同规则）
            if rec.get("modality", "text") == "image":
                judge = getattr(self.cog.base, "judge_same",
                                lambda a, b: None)(text, rec["desc"])
                if judge is False:
                    trec = None  # 不是同一人 → 按新事物继续
                # judge None → 拿不准，按命中处理但不当场叫名字
            if trec:
                promoted = self.temp.touch(tid)
                self.state["topic"] = {"kind": "temp", "temp_id": tid,
                                       "desc": rec["desc"],
                                       "modality": rec.get("modality", "text"),
                                       "at": _t.time()}
                if promoted:  # 七天内不同日期出现≥3次 → 自动晋升永久记忆
                    nm = rec["features"].get("名字")
                    msg = self._promote_temp(tid, name=nm)
                    lead = (f"咦，这不就是{'「' + nm + '」' if nm else '之前聊过的那个'}嘛！"
                            f"它这几天老是出现，我就把它正式记住了——")
                    return self._say(lead + msg,
                                     debug=f"action=TEMP_PROMOTE sim={score}")
                nm = rec["features"].get("名字")
                greet = (f"这不是{nm}嘛，我记得的呀～" if nm
                         else "这个我记得，我们之前聊过的～")
                return self._say(greet, debug=f"action=TEMP_HIT days={len(rec['days'])} sim={score}")

        act = self.cog.perceive(text)
        kind = act["action"]
        if kind == "RECOGNIZED":
            self.state["topic"] = {"kind": "recognized", "desc": text,
                                   "modality": modality,
                                   "entry_id": act["entry_id"], "at": _t.time()}
            return self._say(f"嗯……这个是「{act['name']}」呀，我认得的～",
                             debug=f"action={kind} via={act.get('via')} conf={act['confidence']}")
        if kind == "DIRECT":
            self.state["topic"] = {"kind": "direct", "desc": text,
                                   "modality": modality, "at": _t.time()}
            if desc:  # 图片输入：先讲"我看到一张图片"，再自然描述
                return self._say(f"我看到一张图片——{desc}",
                                 debug=f"action={kind} conf={act['confidence']}；"
                                       f"常识直通。若主人随后给出名字，走特征供给→记忆流程")
            return self._say(f"哦，这是{act['guess']}呀～",
                             debug=f"action={kind} conf={act['confidence']}；"
                                   f"常识直通。若主人随后给出名字，走特征供给→记忆流程")
        if kind == "ARBITRATE":
            self.state["topic"] = None
            self.state["pending"] = {"kind": "arbitrate", "text": text,
                                     "candidate_id": act["candidate_id"],
                                     "modality": modality, "at": _t.time()}
            return self._say("唔，这个和我之前认识的「{}」有点像……它们是同一个吗？".format(
                                 act["candidate_name"]),
                             actions=[{"label": "是同一个", "value": "yes"},
                                      {"label": "不是", "value": "no"},
                                      {"label": "是它的幼体/子类", "value": "child"}],
                             debug=f"action={kind} sim={act['similarity']}")
        if kind == "GUESS":
            self.state["topic"] = {"kind": "guess", "desc": text,
                                   "modality": modality, "guess": act["guess"],
                                   "at": _t.time()}
            self.state["pending"] = {"kind": "guess", "text": text,
                                     "guess": act["guess"], "modality": modality,
                                     "at": _t.time()}
            return self._say(f"嗯……让我猜猜，这是{act['guess']}？",
                             actions=[{"label": "对", "value": "yes"},
                                      {"label": "不对", "value": "no"}],
                             debug=f"action={kind} conf={act['confidence']}")
        self.state["topic"] = {"kind": "unknown", "desc": text,
                               "modality": modality, "at": _t.time()}
        self.state["pending"] = {"kind": "naming", "text": text,
                                 "modality": modality, "at": _t.time()}
        return self._say("唔……这个我确实不认识。你愿意教教我吗？"
                         "告诉我它叫什么、长什么样就好～",
                         debug=f"action={kind}")

    # ---------- 回答未决流程 ----------
    def respond(self, value: str) -> dict:
        try:
            result = self._respond(value)
        except Exception:
            self.state = self._load()
            raise
        self._save()
        return result

    def _respond(self, value: str) -> dict:
        p = self.state.get("pending")
        if not p:
            return self._say("嗯？现在没有等我拿主意的事情呀，你直接说就好。")
        kind = p["kind"]
        v = value.strip()

        # 自由文本的模糊理解：主人的自然表达 → 内部裁决值
        def as_no(s):
            return bool(re.search(r"不|否|没|错|no\b|nope|wrong", s, re.IGNORECASE))

        def as_yes(s):
            return bool(re.search(r"是|对|嗯|好|没错|yes|yeah|yep", s, re.IGNORECASE))

        if kind == "arbitrate":
            if value == "child" or re.search(r"子类|幼体|分支|小的", v):
                self.state["pending"] = {"kind": "naming", "text": p["text"],
                                         "parent": p["candidate_id"],
                                         "modality": p.get("modality", "text"),
                                         "at": _t.time()}
                return self._say("明白，那它是新的一支。给它起个名字，再简单说说特征吧。")
            if value == "yes" or (as_yes(v) and not as_no(v)):
                self.state["pending"] = None
                self.cog.merge_into(p["candidate_id"], p["text"],
                                    note="灰色地带仲裁 MERGE")
                return self._say("好，那我就把它并到原来的记忆里，只补充、不改动旧的～")
            if value == "no" or as_no(v):
                self.state["pending"] = None
                self.state["pending"] = {"kind": "naming", "text": p["text"],
                                         "modality": p.get("modality", "text"),
                                         "at": _t.time()}
                return self._say("好，那它是另一个事物。告诉我它叫什么、长什么样吧。")
            self.state["pending"] = p  # 没听懂，再问一次
            return self._say("唔……你是说这是同一个，还是另一个呀？")

        if kind == "guess":
            if value == "no" or (as_no(v) and not as_yes(v)):
                self.state["pending"] = None
                return self._say("啊，猜错了……那先不记，等你想教我的时候再说。")
            # 闸口 1：GUESS 分支确认回环
            self.state["pending"] = {"kind": "gate1", "text": p["text"],
                                     "guess": p["guess"],
                                     "modality": p.get("modality", "text"),
                                     "at": _t.time()}
            return self._say(f"那这就是普通的{p['guess']}，对吗？还是它有特别之处？\n"
                             f"（直接回我：「普通」/ 特征描述 / 「叫<名字>」）")

        if kind == "gate1":
            text, guess = p["text"], p["guess"]
            if v in ("普通", "ordinary", "就是普通的") or \
                    (as_yes(v) and not as_no(v) and len(v) <= 4):
                self.state["pending"] = None
                return self._say("好～常识就不占用我的记忆了。")
            m = re.match(r"^(叫|named\s*)\s*(.+)$", v, re.IGNORECASE)
            if m:
                self.state["pending"] = {"kind": "named_desc", "name": m.group(2).strip(),
                                         "guess": guess,
                                         "modality": p.get("modality", "text")}
                return self._say(f"好呀，那「{m.group(2).strip()}」是什么样的呢？简单说说特征就行。")
            pid = self.cog.learn_patch(guess, {"特征": v},
                                       patch_type="feature_addition")
            self.state["pending"] = None
            return self._say(f"记下啦，作为小补丁（{pid}）挂在「{guess}」下面～")

        if kind == "named_desc":
            self.state["pending"] = None
            self.state["topic"] = None
            name = p["name"]
            desc = f"{name}，{value.strip()}"
            if p.get("parent"):
                self.cog.learn_entry(name, desc,
                                     relation={"type": "subtype_of",
                                               "parent": p["parent"]},
                                     source_modality=p.get("modality", "text"))
                return self._say("记住啦，它是挂在原来那条下面的新一支 ✓")
            return self._say(self._finish_entry(name, desc,
                                                modality=p.get("modality", "text")))

        if kind == "naming":
            self.state["pending"] = None
            self.state["topic"] = None
            desc = value.strip()
            name = re.split(r"[,，]", desc)[0].strip()
            if p.get("parent"):
                self.cog.learn_entry(name, desc,
                                     relation={"type": "subtype_of",
                                               "parent": p["parent"]},
                                     source_modality=p.get("modality", "text"))
                return self._say("记住啦，它是挂在原来那条下面的新一支 ✓")
            return self._say(self._finish_entry(name, desc,
                                                modality=p.get("modality", "text")))

        self.state["pending"] = None
        return self._say("唔，我刚才有点懵，已经理清了。我们继续～")

    # ---------- 统计 ----------
    def stats(self) -> dict:
        s = self.cog.memory.stats()
        entries = [{"id": eid, "name": e["name"],
                    "hits": e.get("health", {}).get("hit_count", 0),
                    "last_hit": e.get("health", {}).get("last_hit_at"),
                    "dormant": is_dormant(e),
                    "errors": e.get("threshold_meta", {}).get("mistake_count", 0)}
                   for eid, e in self.cog.memory.entries.items()]
        patches = [{"id": p["id"], "target": p["target"], "type": p["type"],
                    "status": p["status"], "content": p["content"]}
                   for p in self.cog.memory.patches]
        return {"stats": s, "entries": entries, "patches": patches,
                "temp": self.temp.stats(), "pending": self.state.get("pending")}

    # ---------- 模型切换（server / CLI 共用） ----------
    def reembed_all(self, embedder=None) -> tuple[int, int]:
        """按设计文档 §3.3 / §10.1 重建全部记忆向量（永久 + 临时）。"""
        emb = embedder or self.cog.embedder
        n = 0
        for e in self.cog.memory.entries.values():
            desc = e.get("features", {}).get("描述")
            if desc:
                e["features"]["embedding"] = emb.embed(desc)
                e.setdefault("embedding_meta", {})["model"] = getattr(
                    emb, "model_tag", "")
                n += 1
        self.cog.memory._persist()
        m = 0
        for r in self.temp.records.values():
            r["embedding"] = emb.embed(r["desc"])
            m += 1
        self.temp._persist()
        return n, m

    def switch_models(self, base: str | None = None,
                      embedding: str | None = None) -> str:
        """热切换基础模型 / 编码器；切编码器时重建全部向量。"""
        from .adapters.ollama import OllamaBaseModel, OllamaEmbedder
        msgs = []
        if base:
            self.cog.base = OllamaBaseModel(model=base, host=self.ollama_host)
            self.state["base_model"] = base
            msgs.append(f"基础模型 → {base}")
        if embedding:
            new_emb = OllamaEmbedder(model=embedding, host=self.ollama_host)
            self.cog.embedder = new_emb
            self.cog.memory.embedder = new_emb
            self.temp.embedder = new_emb
            self.state["embedding_model"] = embedding
            n, m = self.reembed_all(new_emb)
            msgs.append(f"embedding 模型 → {embedding}"
                        f"（已重建 {n} 条永久 + {m} 条临时记忆向量）")
        self._save()
        return "；".join(msgs) if msgs else "模型未变化"
