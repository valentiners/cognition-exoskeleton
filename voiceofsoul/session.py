"""会话编排（一个大脑，两个壳）。

HTTP 测试服务（server.py）与 CLI（voiceofsoul/exo.py）共用本模块：
- 话题 30 分钟冷却；
- 未决流程（naming / guess / gate1 / arbitrate / promote_name）的自由文本路由；
- 仲裁挂起超时 → 临时记忆降级（设计文档 §4.5）；
- 临时记忆七天法则、特征供给、识别与问答直通；
- 好奇心（设计文档 §4.6~§4.9）：T1/T2 触发、L0 先查后问、L1/L2 提问分级、
  好奇队列唤醒与静音。好奇心链路任何失败都只退化为「不问」，绝不阻断基础回复。

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
import os
import re
import sys
import time as _t
from datetime import datetime
from pathlib import Path

from .conflict import classify_conflict, pair_key
from .curiosity import CuriosityQueue
from .foresight import (ForesightState, InstinctStore, active_tags,
                        match_tags, route, warning_text)
from . import hypothesis as hypo_mod
from .doctor import _keynorm, latest_profile
from .growth import last_consolidation
from . import growth as _growth
from .knowledge import (KNOWN, UNCERTAIN, KnowledgeProbeCache,
                        judge_answers)
from .memory import is_dormant
from .tools.web import web_search

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

# ---------- 好奇心（设计 §4.6~§4.9）----------
# 占位符，待 §6.5 标定（任务书 §4）：T2 低置信猜测的触发阈值。
# 2026-11-14 主人拍板选 A：0.35 是死路径（GUESS 的置信度只会 ≥ base_th=0.51
# 或 = 记忆命中分 ≥ mem_th=0.70），抬到 GUESS 有效区间 (0.51, 0.70) 内的 0.60。
CURIOUS_TH = 0.60
AUTONOMY_MODES = ("quiet", "normal", "eager")
DEFAULT_QUIET_HOURS = "21:00-08:00"   # 产品默认，不是标定阈值
# 主人明确叫停 → 该好奇点静音 7 天（复用临时记忆的到期哲学）
MUTE_PATTERN = re.compile(r"(别问了|别问啦|以后再说|不想说|先不问了)")


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
        self.state.setdefault("autonomy", "normal")

        # 好奇队列：与临时记忆同级持久化，进程重启后恢复（§4.8）。
        self.curiosity = CuriosityQueue(self.home / "curiosity.json",
                                        getattr(cog, "embedder", None))
        self._curiosity_note = ""   # 本轮自学成果的拟人汇报，注入回复用

        # 知识边界探测（探测点 #13 / W2）：结论缓存 30 天，跨进程保留。
        self.knowledge_cache = KnowledgeProbeCache(
            self.home / "knowledge_probe_cache.json")
        self._probe_used = 0        # 本轮已用探测额度
        self._knowledge_result = None  # 本轮探测结论（供 debug）

        # L2 近似冲突（W3）：软存疑按“同对”累计，跨进程保留。
        self.state.setdefault("conflict_soft_counts", {})
        self._conflict_debug = ""
        self._hypothesis_debug = ""
        self._guess_hypothesis_qid = None

        # L3 前瞻推演（W6）：在线只做向量匹配；本能规则只由主人批准写入。
        self.foresight_state = ForesightState(self.home / "foresight_state.json")
        self.instinct = InstinctStore(self.home / "instinct_rules.json")
        self._foresight_used = 0
        self._turn_embedding = None

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
            "autonomy": self.state.get("autonomy", "normal"),
            "curiosity_last_asked": self.state.get("curiosity_last_asked"),
            "curiosity_last_l2": self.state.get("curiosity_last_l2"),
            "conflict_soft_counts": self.state.get("conflict_soft_counts", {}),
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
    def _finish_entry(self, name: str, desc: str, modality: str = "text",
                      source: str = "master",
                      source_url: str | None = None) -> str:
        """闸口 2（公共/私人自检）+ 知识边界探测（#13）+ 闸口 3（建库前去重）。

        source/source_url 只影响入库留痕（§4.9 来源纪律），闸口逻辑唯一：
        主人教的与 web 自学的事实走同一条管线、同一套标准。

        知识边界探测（W2）：只有闸口 2 判定为“公共知识”时才探
        “基础模型本来会不会”——身份/主人私有实体永不触发（它们必须走记忆通道）。
        KNOWN → 不入库（记忆卫生）；UNCERTAIN / NEW → 照常走后续流程（偏向记）。
        """
        if self.cog.is_common_knowledge(name, desc):
            probe = self._knowledge_gate(name)
            if probe.get("verdict") == KNOWN:
                return "这个我早就知道啦，不用再教我一遍，我记性好着呢～"
            if probe.get("verdict") == UNCERTAIN and source == "master":
                # W5：🟡 区可选生成假设（有证据才升级；可关）
                self._propose_knowledge_hypothesis(name, desc, probe)
            content = {"描述": desc}
            if source == "web":
                content.update({"来源": source_url or "", "source": "web"})
            self.cog.learn_patch(name, content, patch_type="feature_addition")
            return f"「{name}」听起来是大家都知道的公共知识，我就不占专属记忆了，只留个小补丁备查～"
        best = self.cog.memory.best_match(desc, include_dormant=True)  # 查重不受休眠限制
        if best and best[1] >= self.cog.mem_th:
            note = (f"web 自学补充（出处：{source_url or '未标注'}）"
                    if source == "web" else "闸口 3 查重命中 MERGE")
            self.cog.merge_into(best[0], desc, note=note)
            return "这个和我已有的记忆太像了，我就把它并进原来的条目，不重复占地方。"
        overlap = next((p for p in self.cog.memory.patches
                        if p["target"] in name or name in p["target"]), None)
        if overlap:
            self.cog.memory.update_patch(overlap["id"],
                                         {"event": "内容合并",
                                          "note": f"与新建「{name}」重合"})
            return "和已有的补丁重合了，我把它合过去了～"
        self.cog.learn_entry(name, desc, ai_proposed=name,
                             source_modality=modality,
                             source=source, source_url=source_url)
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
    def _maybe_consolidate(self) -> None:
        """睡眠进程自动触发（§10.3-#4）：安静时段、今天还没巩固过 → 跑一次。

        零打扰铁律：探测点缺失/失败/不在安静时段都静默跳过，绝不拖慢或
        打断主人对话；失败也算尝试过（不每天重试）。"""
        try:
            if not self._in_quiet_hours():
                return
            marker = _growth.last_consolidation(self.home)
            if marker and marker.get("date") == datetime.now().date().isoformat():
                return
            compose = getattr(self.cog.base, "compose_journal", None)
            if not compose:
                return
            _growth.consolidate(self.home, self.cog.base,
                                memory=self.cog.memory, temp=self.temp,
                                curiosity=self.curiosity)
        except Exception as exc:
            print(f"[睡眠进程] 自动巩固失败：{exc}", file=sys.stderr)

    def chat(self, text: str, image_b64: str | None = None) -> dict:
        self._curiosity_note = ""
        self._probe_used = 0
        self._knowledge_result = None
        self._conflict_debug = ""
        self._hypothesis_debug = ""
        self._guess_hypothesis_qid = None
        self._foresight_used = 0
        self._turn_embedding = None
        self._maybe_consolidate()
        try:
            result = self._chat(text, image_b64)
        except Exception:
            # 失败时丢弃本轮内存态，磁盘上的 session.json 保持上一轮有效状态。
            self.state = self._load()
            raise
        # 好奇心只做“追加一句”，任何失败都只意味着这句话不说（铁律 5）。
        try:
            result = self._curiosity_decorate(result)
        except Exception as exc:
            print(f"[好奇心] 本轮追加失败，已静默跳过：{exc}", file=sys.stderr)
        result = self._decorate_knowledge(result)
        result = self._decorate_conflict(result)
        result = self._decorate_hypothesis(result)
        result = self._foresight_decorate(result, text)
        self._save()
        return result

    def _chat(self, text: str, image_b64: str | None = None) -> dict:
        # 0. 好奇心静音：主人明确说「别问了 / 以后再说」→ 该点静音 7 天。
        #    必须在未决流程路由之前判定，否则会被当成对猜名的回答。
        stripped = (text or "").strip()
        if not image_b64 and stripped and len(stripped) <= 16 \
                and MUTE_PATTERN.search(stripped):
            qid = self._curiosity_mute()
            if qid:
                return self._say("好，我不问了，等你哪天想聊再说～",
                                 debug=f"action=CURIOSITY_MUTE qid={qid}")

        # 0a. 唤醒：每句话都可能是「又提到了同一个主题」（§4.8）
        if not image_b64 and stripped:
            self._curiosity_wake(stripped)

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
        try:
            changed = self.curiosity.expire()   # 好奇队列 TTL / 静音解禁
            self._expire_hypotheses(changed)     # W5：假设过期纯静默
        except Exception as exc:
            print(f"[好奇队列] 过期检查失败：{exc}", file=sys.stderr)

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
        # L2 近似冲突侦测（W3）：软存疑只进 debug，硬冲突进好奇队列。
        self._check_conflict(act, text)
        # 好奇心观察（§4.6）：T1/T2 入队并先查后问；T3/T4 仅在开关打开时扫描。
        # 在任何分支之前观察，保证「主人没理」也留下学习信号。
        cqid = self._observe_curiosity(act, text, modality, desc)
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
                                   "curiosity_qid": cqid,
                                   "hypothesis_qid": self._guess_hypothesis_qid,
                                   "at": _t.time()}
            self.state["pending"] = {"kind": "guess", "text": text,
                                     "guess": act["guess"], "modality": modality,
                                     "curiosity_qid": cqid,
                                     "hypothesis_qid": self._guess_hypothesis_qid,
                                     "at": _t.time()}
            return self._say(f"嗯……让我猜猜，这是{act['guess']}？",
                             actions=[{"label": "对", "value": "yes"},
                                      {"label": "不对", "value": "no"}],
                             debug=f"action={kind} conf={act['confidence']}")
        self.state["topic"] = {"kind": "unknown", "desc": text,
                               "modality": modality, "curiosity_qid": cqid,
                               "at": _t.time()}
        self.state["pending"] = {"kind": "naming", "text": text,
                                 "modality": modality, "curiosity_qid": cqid,
                                 "at": _t.time()}
        return self._say("唔……这个我确实不认识。你愿意教教我吗？"
                         "告诉我它叫什么、长什么样就好～",
                         debug=f"action={kind}")

    # ---------- 回答未决流程 ----------
    def respond(self, value: str) -> dict:
        self._probe_used = 0
        self._knowledge_result = None
        try:
            result = self._respond(value)
        except Exception:
            self.state = self._load()
            raise
        result = self._decorate_knowledge(result)
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

        if kind == "conflict":
            cands = [c for c in (p.get("candidates") or []) if c in self.cog.memory.entries]
            names = [self.cog.memory.entries[c].get("name", c) for c in cands]
            named = next((c for c in cands
                          if self.cog.memory.entries[c].get("name", "")
                          and self.cog.memory.entries[c]["name"] in v), None)
            if re.search(r"(记错|没有这|删掉|去掉|作废|收起来)", v):
                target = named or (cands[1] if len(cands) > 1
                                   else (cands[0] if cands else None))
                if target:
                    self.cog.memory.apply_vote(target, -1,
                                               note="主人纠错：冲突裁决")
                    nm = self.cog.memory.entries.get(target, {}).get("name", "它")
                    self.state["pending"] = None
                    self._resolve_curiosity(p)
                    return self._say(f"好，我先把「{nm}」收起来，不再拿出来用；"
                                     "哪天发现不对，你再叫它回来。")
                self.state["pending"] = None
                self._resolve_curiosity(p)
                return self._say("好，那我先不拿它当准。")
            if re.search(r"(同一个|同一个人|是一个人|就是它|一样)", v) \
                    and not re.search(r"(不一样|不是)", v) and len(cands) >= 2:
                a, b = cands[0], cands[1]
                eb = self.cog.memory.entries.get(b, {})
                self.cog.merge_into(
                    a, eb.get("features", {}).get("描述", eb.get("name", "")),
                    note="冲突裁决 MERGE")
                self.cog.memory.apply_vote(b, -1,
                                           note="冲突裁决：重复条目退休")
                nm_a = self.cog.memory.entries.get(a, {}).get("name", "A")
                nm_b = eb.get("name", "B")
                self.state["pending"] = None
                self._resolve_curiosity(p)
                return self._say(f"那我把它并回「{nm_a}」这边，「{nm_b}」"
                                 "这条先退休留个底～")
            if re.search(r"(不是|不同|两个|两个人)", v):
                self.state["pending"] = None
                self._resolve_curiosity(p)
                return self._say("好，那我把它们当两个不同的记着。")
            if named:
                self.cog.memory.touch(named)
                self.state["pending"] = None
                self._resolve_curiosity(p)
                nm = self.cog.memory.entries.get(named, {}).get("name", "它")
                return self._say(f"好，那我按「{nm}」记着。")
            joined = "、".join(f"「{n}」" for n in names) or "它们"
            return self._say(f"唔……你是说{joined}是同一个，还是两个不同的呀？")

        if kind == "hypothesis":
            item = self.curiosity.get(p.get("qid", ""))
            hypo = (item or {}).get("hypothesis")
            if not isinstance(hypo, dict):
                self.state["pending"] = None
                return self._say("唔，刚才那个猜想好像已经不在啦。")
            if as_yes(v) and not as_no(v):
                hypo_mod.confirm(hypo, self.cog.memory, source="master")
                self.curiosity.mark(p["qid"], "answered", hypothesis=hypo,
                                    solved_source="master")
                self.state["pending"] = None
                return self._say("好，那我就按这个记下啦。")
            if as_no(v):
                hypo_mod.refute(hypo, self.cog.memory, note=v[:40],
                                source="master")
                self.curiosity.mark(p["qid"], "refuted", hypothesis=hypo,
                                    solved_source="master")
                self.state["pending"] = None
                return self._say("好，看来我猜错了，这个念头我先放下。")
            return self._say("唔……你是说对，还是不对呀？")

        if kind == "foresight_confirm":
            self.state["pending"] = None
            if re.search(r"(确定|确认|继续|做吧|知道了|可以)", v) \
                    and not re.search(r"(不|别|算了)", v):
                return self._say("好，那我就不拦你了，你自己多当心。")
            return self._say("好，那先别做，咱们再想想。")

        if kind == "instinct_proposal":
            self.state["pending"] = None
            pid = p.get("pid", "")
            draft = self.foresight_state.data["proposals"].get(pid)
            if not draft:
                return self._say("唔，这条提议好像已经不在啦。")
            if as_yes(v) and not as_no(v):
                self.instinct.approve(draft)
                self.foresight_state.decide_proposal(pid, True)
                return self._say("好，我就把这条记成规矩了，以后直接提醒你。")
            self.foresight_state.decide_proposal(pid, False)
            return self._say("好，那我不擅自把它当规矩。")

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
                self._settle_guess_hypothesis(p, False)
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
                self._resolve_curiosity(p)
                self._settle_guess_hypothesis(p, True)
                return self._say("好～常识就不占用我的记忆了。")
            m = re.match(r"^(叫|named\s*)\s*(.+)$", v, re.IGNORECASE)
            if m:
                self.state["pending"] = {"kind": "named_desc", "name": m.group(2).strip(),
                                         "guess": guess,
                                         "curiosity_qid": p.get("curiosity_qid"),
                                         "hypothesis_qid": p.get("hypothesis_qid"),
                                         "modality": p.get("modality", "text")}
                self._resolve_curiosity(p)
                return self._say(f"好呀，那「{m.group(2).strip()}」是什么样的呢？简单说说特征就行。")
            pid = self.cog.learn_patch(guess, {"特征": v},
                                       patch_type="feature_addition")
            self.state["pending"] = None
            self._resolve_curiosity(p)
            self._settle_guess_hypothesis(p, True)
            return self._say(f"记下啦，作为小补丁（{pid}）挂在「{guess}」下面～")

        if kind == "named_desc":
            self.state["pending"] = None
            self.state["topic"] = None
            self._resolve_curiosity(p)
            self._settle_guess_hypothesis(p, True)
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
            self._resolve_curiosity(p)
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

    def _resolve_curiosity(self, pending: dict, status: str = "answered",
                           source: str = "master"):
        """未决流程落幕 → 对应的好奇点也结案（§4.8 闭环，不另建通道）。"""
        qid = (pending or {}).get("curiosity_qid")
        if not qid:
            return
        item = self.curiosity.get(qid)
        if not item or item.get("status") not in ("open", "muted"):
            return
        try:
            self.curiosity.mark(qid, status, solved_source=source)
        except Exception as exc:
            print(f"[好奇队列] 结案失败：{exc}", file=sys.stderr)

    def _propose_guess_hypothesis(self, act: dict, text: str) -> str | None:
        """T2 低置信 GUESS → 身份/类别假设；验证复用已有的“让我猜猜”提问。"""
        guess = act.get("guess", "")
        if not guess:
            return None
        target = next((eid for eid, e in self.cog.memory.entries.items()
                       if e.get("name") == guess
                       and self.cog.memory._retrievable(e)), None)
        evidence = [
            {"type": "observation", "ref": text[:60], "note": "新观察"},
            {"type": "threshold", "ref": str(act.get("confidence")),
             "note": "GUESS 低于好奇档"},
        ]
        return self._new_hypothesis(
            f"这次看到的应该是「{guess}」",
            float(act.get("confidence") or 0.0), evidence,
            "master", target_entry_id=target, subject=guess,
            question=f"这是「{guess}」吗？")

    def _settle_guess_hypothesis(self, pending: dict, confirmed: bool):
        """GUESS 闸口结算时同步结算关联假设（复用票箱，不新增逻辑）。"""
        qid = (pending or {}).get("hypothesis_qid")
        if not qid:
            return
        item = self.curiosity.get(qid)
        hypo = (item or {}).get("hypothesis")
        if not isinstance(hypo, dict):
            return
        if hypo.get("hypo_status") not in ("proposed", "verifying"):
            return
        try:
            if confirmed:
                hypo_mod.confirm(hypo, self.cog.memory, source="master")
                self.curiosity.mark(qid, "answered", hypothesis=hypo,
                                    solved_source="master")
            else:
                hypo_mod.refute(hypo, self.cog.memory, source="master")
                self.curiosity.mark(qid, "refuted", hypothesis=hypo,
                                    solved_source="master")
        except Exception as exc:
            print(f"[假设] GUESS 结算失败：{exc}", file=sys.stderr)

    # ---------- L3 前瞻推演（W6） ----------
    def _foresight_quota(self) -> int:
        return {"quiet": 0, "normal": 1, "eager": 2}.get(
            self.state.get("autonomy", "normal"), 1)

    def _foresight_consume(self, text: str) -> dict | None:
        """在线匹配：一次本地 embedding + 余弦，零 LLM；风险优先于机会。"""
        if not (text or "").strip():
            return None
        try:
            tags = active_tags(self.cog.memory)
        except Exception:
            return None
        if not tags:
            return None
        vec = self._turn_embedding
        if vec is None:
            try:
                vec = self.cog.embedder.embed(text)
            except Exception:
                return None
        hits = match_tags(vec, self.cog.embedder, tags)
        if not hits:
            return None
        best = route(hits)
        if not best:
            return None
        tag = best["tag"]
        if self.foresight_state.shown_today(tag.get("tag_id", "")):
            return None
        is_red = tag.get("type") == "risk" and tag.get("severity") == "red"
        if not is_red:
            if self.state.get("autonomy", "normal") == "quiet":
                return None
            if self._foresight_used >= self._foresight_quota():
                return None
            self._foresight_used += 1
        self.foresight_state.mark_shown(tag.get("tag_id", ""))
        self.foresight_state.register_hit(tag)
        proposal = None
        if tag.get("type") == "risk":
            proposal = self.foresight_state.due_proposal(tag)
        return {
            "say": warning_text(tag),
            "tag": tag,
            "needs_confirm": bool(is_red and tag.get("irreversible")),
            "proposal": proposal,
        }

    def _foresight_decorate(self, result: dict, text: str) -> dict:
        out = self._foresight_consume(text)
        if not out:
            return result
        tag = out["tag"]
        result["say"] = (result.get("say", "") + "\n" + out["say"])
        debug = (result.get("debug", "") or "").rstrip("；")
        result["debug"] = ((debug + "；") if debug else "") + \
            f"foresight={tag.get('severity') or tag.get('type')} " \
            f"{tag.get('tag_id')}"
        if out.get("needs_confirm") and not self.state.get("pending"):
            self.state["pending"] = {"kind": "foresight_confirm",
                                     "tag_id": tag.get("tag_id"),
                                     "text": text, "at": _t.time()}
        proposal = out.get("proposal")
        if proposal and not self.state.get("pending"):
            result["say"] = (result["say"] + "\n" +
                             f"有件事想跟你商量：{proposal['if']}的时候，"
                             f"我先提醒你「{proposal['then']}」，好不好？")
            self.state["pending"] = {"kind": "instinct_proposal",
                                     "pid": proposal["pid"], "at": _t.time()}
        return result

    # ---------- 假设驱动型好奇（W5） ----------
    @staticmethod
    def _hypothesis_enabled() -> bool:
        raw = os.environ.get("COGSOUL_HYPOTHESIS", "1").strip().lower()
        return raw not in ("0", "false", "off", "no")

    def _new_hypothesis(self, statement: str, confidence: float,
                        evidence: list, plan: str, *, question: str = "",
                        target_entry_id: str | None = None,
                        subject: str = "", pair: str = "") -> str | None:
        """校验并入队一个假设；无证据/陈述过泛一律不升级（出生证明铁律）。"""
        if not self._hypothesis_enabled():
            return None
        try:
            hypo = hypo_mod.build(statement, confidence, evidence, plan,
                                  target_entry_id=target_entry_id,
                                  subject=subject)
        except ValueError as exc:
            print(f"[假设] 不升级：{exc}", file=sys.stderr)
            return None
        if pair:
            hypo["pair"] = pair
        hypo["hid"] = self.curiosity.next_hypothesis_id()
        tier = "L1" if plan == "master" else "WAIT"
        try:
            qid = self.curiosity.add(question or statement, "HYPOTHESIS",
                                    statement, dedupe=False)
            self.curiosity.mark(qid, "open", hypothesis=hypo, tier_due=tier)
        except Exception as exc:
            print(f"[假设] 入队失败：{exc}", file=sys.stderr)
            return None
        self._hypothesis_debug = (f"{hypo['hid']} {plan} "
                                  f"conf={hypo['confidence']}")
        if plan == "web":
            self._verify_hypothesis_web(qid)
        return qid

    def _open_hypotheses(self) -> list[dict]:
        return [i for i in self.curiosity.items.values()
                if isinstance(i.get("hypothesis"), dict)
                and i.get("status") == "open"]

    def _propose_knowledge_hypothesis(self, name: str, desc: str,
                                      probe: dict):
        """🟡 区（W2 UNCERTAIN）→ 知识型假设，先查后问。"""
        statement = f"「{name}」的意思是：{(desc or name)[:60]}"
        if any(h["hypothesis"].get("statement") == statement
               for h in self._open_hypotheses()):
            return
        evidence = [
            {"type": "probe", "ref": probe.get("key", name),
             "note": f"知识探测 UNCERTAIN（{probe.get('consistent')} 答一致）"},
            {"type": "master", "ref": name, "note": "主人刚提到这个知识点"},
        ]
        self._new_hypothesis(statement, 0.55, evidence, "web",
                             question=f"我好像知道一点「{name}」，但不太确定，"
                                      "我去查一下好吗？",
                             subject=name)

    def _propose_conflict_hypothesis(self, a: str, b: str, stimulus: str):
        """软存疑带的分叉旁路：挂一个**静默**假设（W3 零打扰不变），3 次升级硬冲突。"""
        ea = self.cog.memory.entries.get(a) or {}
        eb = self.cog.memory.entries.get(b) or {}
        name_a, name_b = ea.get("name", a), eb.get("name", b)
        key = pair_key(a, b)
        if any(h["hypothesis"].get("pair") == key
               for h in self._open_hypotheses()):
            return
        statement = f"「{name_a}」和「{name_b}」可能是同一个人"
        evidence = [
            {"type": "memory", "ref": a, "note": name_a},
            {"type": "memory", "ref": b, "note": name_b},
            {"type": "observation", "ref": stimulus[:60], "note": "新观察"},
        ]
        # 软存疑必须零输出（W3 拍板）：先挂 wait，等未来相遇/升级硬冲突再动
        self._new_hypothesis(statement, 0.5, evidence, "wait", pair=key,
                             question=f"「{name_a}」和「{name_b}」是不是同一个呀？",
                             subject=name_a)

    def _verify_hypothesis_web(self, qid: str):
        """web 通道：复用 §4.9 先查后问；查得到且可信才确认。"""
        item = self.curiosity.get(qid)
        if not item:
            return
        hypo = item.get("hypothesis") or {}
        query_fn = getattr(self.cog.base, "curiosity_query", None)
        distill_fn = getattr(self.cog.base, "curiosity_distill", None)
        if not (query_fn and distill_fn):
            return
        try:
            query = query_fn(hypo.get("statement", ""), item.get("stimulus", ""))
            if not query:
                return
            hits = web_search(query)
            if not hits:
                return
            fact = distill_fn(hypo.get("statement", ""), hits)
            if not fact:
                return
            url = (hits[0].get("url") or "").strip()
            ok, entry_id = self._store_web_fact_ex(fact["name"],
                                                  fact["fact"], url)
            if not ok:
                return
            hypo_mod.confirm(hypo, self.cog.memory, entry_id=entry_id,
                             source="web")
            self.curiosity.mark(qid, "answered", hypothesis=hypo,
                                solved_source="web", solved_url=url)
            self._hypothesis_debug = (f"{hypo.get('hid')} confirmed via web")
        except Exception as exc:
            print(f"[假设] web 验证失败：{exc}", file=sys.stderr)

    def _expire_hypotheses(self, qids: list[str]):
        """TTL 到期：纯静默标记，不提示、不进日记。"""
        for qid in qids:
            item = self.curiosity.get(qid)
            hypo = (item or {}).get("hypothesis")
            if not isinstance(hypo, dict):
                continue
            if hypo.get("hypo_status") not in ("proposed", "verifying"):
                continue
            try:
                hypo_mod.transition(hypo, "expired", result="expired")
                self.curiosity.mark(qid, "expired", hypothesis=hypo)
            except ValueError:
                pass

    def _decorate_hypothesis(self, result: dict) -> dict:
        if not self._hypothesis_debug:
            return result
        debug = (result.get("debug", "") or "").rstrip("；")
        result["debug"] = ((debug + "；") if debug else "") + \
            f"hypothesis={self._hypothesis_debug}"
        return result

    # ---------- L2 织网与信念修正（W3） ----------
    def _check_conflict(self, act: dict, text: str):
        """检索判定后（§2.1 末段）：两级近似冲突侦测。"""
        if act.get("action") not in ("RECOGNIZED", "GUESS", "ARBITRATE"):
            return
        try:
            ranked = self.cog.memory.ranked_matches(text, layer="active", limit=2)
        except Exception as exc:
            print(f"[冲突侦测] 排名失败，已跳过：{exc}", file=sys.stderr)
            return
        if len(ranked) < 2:
            return
        (a, sa), (b, sb) = ranked[0], ranked[1]
        info = classify_conflict(sa, sb, mem_th=self.cog.mem_th)
        if not info:
            return
        key = pair_key(a, b)
        counts = self.state.setdefault("conflict_soft_counts", {})
        if info["level"] == "soft":
            counts[key] = int(counts.get(key, 0)) + 1
            if counts[key] < 3:
                self._conflict_debug = (f"soft {key} gap={info['gap']} "
                                        f"count={counts[key]}")
                # W5：软存疑分叉旁路——先挂假设验证，升级路径不变
                self._propose_conflict_hypothesis(a, b, text)
                return
            counts[key] = 0
            info = {**info, "level": "hard", "escalated": True}
        self._start_conflict(act, text, a, b, info)

    def _start_conflict(self, act: dict, text: str, a: str, b: str,
                        info: dict):
        """硬冲突：性质裁决 → 进好奇队列；知识型先查，身份型等主人定夺。"""
        ea = self.cog.memory.entries.get(a) or {}
        eb = self.cog.memory.entries.get(b) or {}
        desc_a = ea.get("features", {}).get("描述", ea.get("name", ""))
        desc_b = eb.get("features", {}).get("描述", eb.get("name", ""))
        probe = getattr(self.cog.base, "conflict_probe", None)
        kind = None
        if probe:
            try:
                verdict = probe(text, desc_a, desc_b)
                kind = verdict.get("kind") if verdict else None
            except Exception as exc:
                print(f"[冲突侦测] 性质裁决失败：{exc}", file=sys.stderr)
        if kind is None:
            # 回退：图像个体 / 私人实体 → 身份型；其余知识型
            private = any(
                e.get("provenance", {}).get("source_modality") == "image"
                or not self.cog.is_common_knowledge(
                    e.get("name", ""),
                    e.get("features", {}).get("描述", ""))
                for e in (ea, eb) if e)
            kind = "identity" if private else "knowledge"
        self._conflict_debug = (f"hard {pair_key(a, b)} kind={kind} "
                                f"gap={info['gap']}"
                                + (" escalated" if info.get("escalated") else ""))
        if kind == "none":
            return
        name_a = ea.get("name", "A")
        name_b = eb.get("name", "B")
        if kind == "identity":
            question = (f"「{name_a}」和「{name_b}」我感觉都像——刚才看到的"
                        "是哪一个呀？要是其实是同一个，或者我哪个记错了，"
                        "也直接告诉我。")
        else:
            question = (f"「{name_a}」和「{name_b}」的说法好像对不上，"
                        "哪个才是对的呀？")
        try:
            existing = next((i for i in self.curiosity.items.values()
                             if i.get("source_trigger") == "CONFLICT"
                             and set(i.get("conflict_ids") or []) == {a, b}
                             and i.get("status") == "open"), None)
            if existing:
                qid = existing["qid"]
                self.curiosity.mark(qid, "open", tier_due="L1")
            else:
                qid = self.curiosity.add(question, "CONFLICT", text,
                                         dedupe=False)
                self.curiosity.mark(qid, "open", conflict_ids=[a, b],
                                    conflict_kind=kind, tier_due="L1")
        except Exception as exc:
            print(f"[冲突侦测] 质疑任务入队失败：{exc}", file=sys.stderr)
            return
        if kind == "knowledge":
            self._curiosity_self_solve(qid, text)   # 先查后问（§4.9）

    def _decorate_conflict(self, result: dict) -> dict:
        if not self._conflict_debug:
            return result
        debug = (result.get("debug", "") or "").rstrip("；")
        result["debug"] = ((debug + "；") if debug else "") + \
            f"conflict={self._conflict_debug}"
        return result

    # ---------- 知识边界探测（探测点 #13 / W2） ----------
    def _probe_quota(self) -> int:
        """本轮探测额度（计入 §4.7 自主额度体系）。

        quiet=0（不主动探测，按“偏向记”处理）/ normal=1 / eager=2。
        缓存命中不耗额度（不产生 LLM 调用）。
        """
        mode = self.state.get("autonomy", "normal")
        return {"quiet": 0, "normal": 1, "eager": 2}.get(mode, 1)

    def _knowledge_gate(self, point: str) -> dict:
        """三问一致性探测一个知识点；任何失败都退化为 UNCERTAIN（偏向记）。"""
        key = _keynorm(point) or (point or "").strip().lower()
        cached = self.knowledge_cache.get(key)
        if cached:
            res = dict(cached)
            res.update({"cached": True, "key": key})
            self._knowledge_result = res
            return res

        if self._probe_used >= self._probe_quota():
            res = {"verdict": UNCERTAIN, "reason": "quota",
                   "cached": False, "key": key}
            self._knowledge_result = res
            return res

        probe = getattr(self.cog.base, "knowledge_probe", None)
        if not probe:
            res = {"verdict": UNCERTAIN, "reason": "unsupported",
                   "cached": False, "key": key}
            self._knowledge_result = res
            return res

        self._probe_used += 1
        try:
            answers = probe(point)   # 只给知识点名字：不把待学内容喂进提示词
        except Exception as exc:
            print(f"[知识探测] 调用失败，按偏向记处理：{exc}", file=sys.stderr)
            res = {"verdict": UNCERTAIN, "reason": "probe_failed",
                   "cached": False, "key": key}
            self._knowledge_result = res
            return res

        judged = judge_answers(answers)
        self.knowledge_cache.put(key, judged)
        res = dict(judged)
        res.update({"cached": False, "key": key})
        self._knowledge_result = res
        return res

    def _decorate_knowledge(self, result: dict) -> dict:
        """把本轮探测结论与累计分布写进 debug（机制标签绝不进主文案）。"""
        kp = self._knowledge_result
        if not kp:
            return result
        st = self.knowledge_cache.stats_snapshot()
        part = (f"knowledge={kp.get('verdict')}"
                + ("(cached)" if kp.get("cached") else "")
                + f" reason={kp.get('reason', '')}"
                + f" K/U/N={st['known']}/{st['uncertain']}/{st['new']}")
        debug = (result.get("debug", "") or "").rstrip("；")
        result["debug"] = (debug + "；" if debug else "") + part
        return result

    # ---------- 好奇心（§4.6~§4.9） ----------
    def set_autonomy(self, mode: str) -> str:
        """切换自主额度：quiet（只攒批）/ normal（L1+L2，默认）/ eager（L1 一轮两次）。"""
        m = (mode or "").strip().lower()
        if m not in AUTONOMY_MODES:
            raise ValueError(f"unknown autonomy mode: {mode!r}")
        self.state["autonomy"] = m
        self._save()
        return m

    @staticmethod
    def _t34_enabled() -> bool:
        """T3/T4 默认关闭，等 §6.5 标定（任务书 §3.3）。"""
        return os.environ.get("COGSOUL_CURIOUS_T34", "").strip().lower() \
            in ("1", "true", "yes", "on")

    def _curiosity_wake(self, text: str):
        """主人又提到同主题 → 唤醒在册好奇点（余弦 ≥ 0.62，复用标定值）。"""
        try:
            emb = self.cog.embedder.embed(text)
            self._turn_embedding = emb
        except Exception:
            return
        try:
            for qid in self.curiosity.touch(emb):
                print(f"[好奇唤醒] {qid} 又被提起了", file=sys.stderr)
        except Exception as exc:
            print(f"[好奇唤醒] 失败：{exc}", file=sys.stderr)

    def _curiosity_mute(self) -> str | None:
        """静音最近问过的那条；没有就问最新入队的一条。"""
        qid = self.state.get("curiosity_last_asked")
        item = self.curiosity.get(qid) if qid else None
        if not item or item.get("status") != "open":
            opens = [i for i in self.curiosity.items.values()
                     if i.get("status") == "open"]
            item = max(opens, key=lambda i: i["created_at"]) if opens else None
        if not item:
            return None
        try:
            self.curiosity.mark(item["qid"], "muted")
        except Exception as exc:
            print(f"[好奇队列] 静音失败：{exc}", file=sys.stderr)
            return None
        return item["qid"]

    def _observe_curiosity(self, act: dict, text: str, modality: str,
                           desc: str | None) -> str | None:
        """T1/T2 触发 + T3/T4 扫描；返回本轮入队（或命中在册）的 qid。"""
        kind = act.get("action")
        qid = None
        self._guess_hypothesis_qid = None
        try:
            created = False
            if kind == "UNKNOWN":
                qid, created = self._new_point("这个我从没见过，它是什么呀？",
                                               "T1", text)
            elif kind == "GUESS" \
                    and float(act.get("confidence") or 0.0) < CURIOUS_TH:
                qid, created = self._new_point(
                    f"我猜它是{act.get('guess', '')}，可实在不太确定……它到底是什么呀？",
                    "T2", text)
                if created:
                    # W5：GUESS 升级为假设（复用既有猜问，不额外问一遍）
                    self._guess_hypothesis_qid = self._propose_guess_hypothesis(
                        act, text)
            if qid and created:
                self._curiosity_self_solve(qid, text)
            if kind in ("DIRECT", "RECOGNIZED") and self._t34_enabled():
                self._scan_t34(act, text, desc)
        except Exception as exc:
            # 好奇心是副线：出问题就当作不好奇，绝不影响主回复。
            print(f"[好奇心] 观察失败，已跳过：{exc}", file=sys.stderr)
        return qid

    def _new_point(self, question: str, trigger: str,
                   stimulus: str) -> tuple[str | None, bool]:
        """入队（同主题已在册则只唤醒旧条目）；返回 (qid, 是否新建)。"""
        before = len(self.curiosity.items)
        try:
            qid = self.curiosity.add(question, trigger, stimulus)
        except Exception as exc:
            print(f"[好奇队列] 入队失败：{exc}", file=sys.stderr)
            return None, False
        return qid, len(self.curiosity.items) > before

    def _curiosity_self_solve(self, qid: str, stimulus: str):
        """L0 先查后问（§4.9）：中性检索 → 提炼 → 三道闸口入库 → self_solved。"""
        item = self.curiosity.get(qid)
        if not item:
            return
        query_fn = getattr(self.cog.base, "curiosity_query", None)
        distill_fn = getattr(self.cog.base, "curiosity_distill", None)
        if not (query_fn and distill_fn):
            return
        query = query_fn(item["question"], stimulus)
        if not query:
            return
        hits = web_search(query)
        if not hits:
            return          # 查不到 / 无网络 → 回落问主人（自学失败不丢人）
        fact = distill_fn(item["question"], hits)
        if not fact:
            return          # 模型认为片段不可信 → 回落问主人
        url = (hits[0].get("url") or "").strip()
        if not self._store_web_fact(fact["name"], fact["fact"], url):
            return
        self.curiosity.mark(qid, "self_solved", solved_source="web",
                            solved_url=url)
        self._curiosity_note = (
            f"对了，我自己上网查了一下：{fact['fact']}"
            + (f"（出处：{url}）" if url else ""))

    def _store_web_fact(self, name: str, fact: str, url: str) -> bool:
        """web 知识入库：复用 _finish_entry 的三道闸口，绝不自建新通道（§4.8）。"""
        return self._store_web_fact_ex(name, fact, url)[0]

    def _store_web_fact_ex(self, name: str, fact: str,
                           url: str) -> tuple[bool, str | None]:
        """同 _store_web_fact，额外返回新建条目 id（假设转正需要出生证明）。"""
        name = (name or "").strip()
        fact = (fact or "").strip()
        if not name or not fact:
            return False, None
        before = set(self.cog.memory.entries)
        self._finish_entry(name, fact, source="web", source_url=url or None)
        new = [eid for eid in self.cog.memory.entries if eid not in before]
        return True, (new[0] if new else None)

    def _scan_t34(self, act: dict, text: str, desc: str | None):
        """探测点 #10：感知矛盾（T3）/ 陌生专名（T4）。默认关闭。"""
        scan = getattr(self.cog.base, "curiosity_scan", None)
        if not scan:
            return
        context = ""
        if act.get("action") == "RECOGNIZED" and act.get("entry_id"):
            entry = self.cog.memory.entries.get(act["entry_id"])
            if entry:
                context = (f"「{entry['name']}」："
                           f"{entry.get('features', {}).get('描述', '')}")
        else:
            topic = self.state.get("topic")
            if self._fresh_topic(topic):
                context = topic.get("desc", "")
        unified = f"{text}；{desc}" if desc else text
        points = scan(unified, context)
        if not points:
            return          # [] = 不好奇；None = 拿不准 → 同样不好奇
        for point in points:
            if not isinstance(point, dict):
                continue
            what = (point.get("point") or "").strip()
            if not what:
                continue
            trigger = "T3" if point.get("kind") == "contradiction" else "T4"
            question = (f"「{what}」是什么呀？我没听过这个名字"
                        if trigger == "T4" else what)
            self._new_point(question, trigger, text)
            if trigger == "T3" and act.get("entry_id"):
                self._record_contradiction(act["entry_id"], what)

    def _record_contradiction(self, entry_id: str, point: str):
        """T3 的矛盾本身应留痕：走既有补丁管线，好奇心绝不绕闸口。"""
        entry = self.cog.memory.entries.get(entry_id)
        if not entry:
            return
        name = entry["name"]
        for patch in self.cog.memory.find_active_patches(name):
            if point in json.dumps(patch.get("content", {}), ensure_ascii=False):
                return
        self.cog.learn_patch(name, {"冲突": point}, source_entry=entry_id)

    def _curiosity_decorate(self, result: dict) -> dict:
        """把自学汇报 / L1 轻问 / L2 攒批追加到回复末尾（永不换掉正文）。"""
        extra = []
        if self._curiosity_note:
            extra.append(self._curiosity_note)
        question = self._curiosity_ask()
        if question:
            extra.append(question)
        if extra:
            result["say"] = result.get("say", "") + "\n" + "\n".join(extra)
        st = self.curiosity.stats()
        debug = (result.get("debug", "") or "").rstrip("；")
        result["debug"] = (debug + "；" if debug else "") + \
            f"curiosity total={st['total']} open={st['open']}"
        return result

    def _curiosity_ask(self) -> str:
        """提问分级（§4.7）：礼貌铁律——有未决挂起就不当场问。"""
        mode = self.state.get("autonomy", "normal")
        if mode not in AUTONOMY_MODES:
            mode = "normal"
        if self.state.get("pending"):
            item = self.curiosity.due_for_l1()
            if item:   # 主人手上还有未答的事：降为攒批，不连环问
                self.curiosity.mark(item["qid"], "open", tier_due="L2")
            return ""
        if mode == "quiet":
            # 只攒批：把所有 L1 待问项收进 L2，攒到安静时段外一次呈现
            while True:
                item = self.curiosity.due_for_l1()
                if not item:
                    break
                self.curiosity.mark(item["qid"], "open", tier_due="L2")
            return self._curiosity_l2()
        asked, limit = [], 2 if mode == "eager" else 1
        for _ in range(limit):
            item = self.curiosity.due_for_l1()
            if not item:
                break
            self.curiosity.mark(item["qid"], "open", tier_due="L2")
            self.state["curiosity_last_asked"] = item["qid"]
            if item.get("conflict_ids"):
                # 硬冲突的澄清问需要主人在下一条消息里回答 → 进入未决流程
                self.state["pending"] = {
                    "kind": "conflict",
                    "text": item.get("stimulus", ""),
                    "candidates": list(item.get("conflict_ids") or []),
                    "curiosity_qid": item["qid"],
                    "at": _t.time(),
                }
            if isinstance(item.get("hypothesis"), dict) \
                    and item["hypothesis"].get("plan") == "master":
                self.state["pending"] = {
                    "kind": "hypothesis",
                    "qid": item["qid"],
                    "text": item.get("stimulus", ""),
                    "curiosity_qid": item["qid"],
                    "at": _t.time(),
                }
            asked.append(self._curiosity_phrase(item))
        if asked:
            return "\n".join(asked)
        return self._curiosity_l2()

    @staticmethod
    def _curiosity_phrase(item: dict) -> str:
        """拟人、口语、无按钮；问题本身已由中性模板生成，不带预设答案。"""
        question = item.get("question", "")
        trigger = item.get("source_trigger")
        if trigger == "T2":
            return f"对了，我心里还惦记着一件事——{question}"
        if trigger == "T3":
            return f"有个地方我越想越不对劲：{question}"
        if trigger == "T4":
            return f"对了，{question}"
        if trigger == "CONFLICT":
            return f"对了，有件事我拿不准——{question}"
        if trigger == "HYPOTHESIS":
            return f"对了，我有个猜想想跟你确认一下——{question}"
        return f"对了，还有件事我挺好奇的——{question}"

    def _curiosity_l2(self) -> str:
        """攒批深问：安静时段不发起；同一天只呈现一次（开场白固定模板，不调模型）。"""
        if self._in_quiet_hours():
            return ""
        today = datetime.now().date().isoformat()
        if self.state.get("curiosity_last_l2") == today:
            return ""
        items = self.curiosity.due_for_l2()
        if not items:
            return ""
        lines = ["我最近攒了几个问题，你有空时教教我？"]
        for i, item in enumerate(items[:5], 1):
            lines.append(f"{i}. {item.get('question', '')}")
        self.state["curiosity_last_l2"] = today
        return "\n".join(lines)

    @staticmethod
    def _in_quiet_hours(now: datetime | None = None) -> bool:
        raw = os.environ.get("COGSOUL_QUIET_HOURS", DEFAULT_QUIET_HOURS)
        try:
            start_s, end_s = raw.split("-", 1)
            sh, sm = (int(x) for x in start_s.split(":"))
            eh, em = (int(x) for x in end_s.split(":"))
        except (ValueError, TypeError):
            sh, sm = 21, 0
            eh, em = 8, 0
        cur = now or datetime.now()
        minutes = cur.hour * 60 + cur.minute
        start, end = sh * 60 + sm, eh * 60 + em
        if start == end:
            return False
        if start < end:
            return start <= minutes < end
        return minutes >= start or minutes < end

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
        profile = latest_profile(self.home)
        return {"stats": s, "entries": entries, "patches": patches,
                "last_consolidation": last_consolidation(self.home),
                "temp": self.temp.stats(), "pending": self.state.get("pending"),
                "curiosity": self.curiosity.stats(),
                "autonomy": self.state.get("autonomy", "normal"),
                "model_profile": profile and {
                    "verdict": profile.get("verdict"),
                    "verdict_label": profile.get("verdict_label"),
                    "fitness_score": profile.get("fitness_score"),
                    "generated_at": profile.get("generated_at"),
                    "exams": {k: {"pass": v.get("pass"), "score": v.get("score")}
                              for k, v in profile.get("exams", {}).items()},
                }}

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
        # 换编码器后旧邻居清单失效，不自动解释（§3.3 / 任务书 §2.1）
        self.cog.memory.invalidate_neighbors()
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
