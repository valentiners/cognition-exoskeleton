"""认知模块（设计文档第 2、4 节）。

perceive() 返回四种动作之一：
- RECOGNIZED  记忆库 ≥70% 确认命中
- ARBITRATE   记忆库 51%~70% 灰色地带 → 主人仲裁（MERGE / NEW / SUBCLASS）
- GUESS       基础模型 >51% 猜测性回答（疑问句，等主人确认）
- UNKNOWN     ≤51% 承认未知 → 命名流程

学习侧（learn 系列）实现 v3.1 的双闸口防重复：
- 闸口 1：GUESS 确认后的回环提问（"是普通的 X 还是有特别之处？"）
- 闸口 2：命名环节 is_common_knowledge 专名/通名自检
- 闸口 3：建库前查重（>70% → MERGE；target 重合 → 合并进补丁）
"""

from __future__ import annotations

from datetime import datetime


class CognitiveModule:
    GRAY_ZONE_ACTIONS = ("MERGE", "NEW", "SUBCLASS")

    def __init__(self, base_model, memory, embedder,
                 base_th: float = 0.51, mem_th: float = 0.70):
        self.base = base_model      # 原模型：只调用，不改权重
        self.memory = memory
        self.embedder = embedder
        self.base_th = base_th
        self.mem_th = mem_th

    # ---------- 识别 ----------
    def perceive(self, text: str) -> dict:
        """识别流程（基础模型先认知，外挂只做它做不了的）。

        1. 记忆库高阈值精确命中（≥mem_th，稀少、明确）→ RECOGNIZED
        2. 基础模型先判断：它认识这个事物吗？
           - 它认识的类别下，主人恰好给某一事物起过专名 → RECOGNIZED（专名命中）
           - 它认识、且与任何专名无关 → DIRECT（常识直通，不问主人）
           - 它认识的类别下有多个专名候选 → GUESS（请主人确认，闸口 1）
        3. 基础模型也不确定 → 向量近邻 + 基础模型同一性裁决（不再事事问主人）
        4. 都不成立 → UNKNOWN，进入命名教学流程
        """
        # 1. 记忆库精确命中
        hits = self.memory.search(text, threshold=self.mem_th)
        if hits:
            eid, score = hits[0]
            e = self.memory.entries[eid]
            # 个体条目（图像学来的专名，如 Lucy/豆豆）必须经同一性复核——
            # 向量命中只说明"像这个类别"，不能证明"是同一个人"（防形象不同的同类别个体被认错）
            if e.get("provenance", {}).get("source_modality") == "image":
                desc = e["features"].get("描述", "")
                judge = getattr(self.base, "judge_same", lambda a, b: None)(text, desc)
                if judge is True:
                    self.memory.touch(eid)
                    return {"action": "RECOGNIZED", "entry_id": eid, "confidence": score,
                            "name": e["name"], "via": "memory"}
                if judge is None:
                    return {"action": "GUESS", "guess": e["name"], "confidence": score}
                # judge is False → 类别像但不是同一人 → 落入下方基础模型认知
            else:
                self.memory.touch(eid)
                return {"action": "RECOGNIZED", "entry_id": eid, "confidence": score,
                        "name": e["name"], "via": "memory"}

        # 2. 基础模型先认知
        conf = self.base.infer_confidence(text)
        guess = self.base.infer(text) if conf > 0.3 else None
        if guess and guess != "未知事物" and conf >= self.base_th:
            related = [eid for eid, e in self.memory.entries.items()
                       if guess in e.get("features", {}).get("描述", "")]
            if len(related) == 1:  # 类别唯一专名 → 同一性裁决确认后认出
                eid = related[0]
                desc = self.memory.entries[eid]["features"].get("描述", "")
                judge = getattr(self.base, "judge_same", lambda a, b: None)(text, desc)
                if judge is True:
                    self.memory.touch(eid)
                    return {"action": "RECOGNIZED", "entry_id": eid,
                            "confidence": round(conf, 4),
                            "name": self.memory.entries[eid]["name"],
                            "via": "base+memory"}
                if judge is False:
                    pass  # 同类不同物 → 落入下方未知/命名流程
                else:
                    return {"action": "GUESS", "guess": guess,
                            "confidence": round(conf, 4)}
            elif len(related) > 1:  # 多个专名候选 → 请主人确认（闸口 1 前置）
                return {"action": "GUESS", "guess": guess, "confidence": round(conf, 4)}
            else:
                return {"action": "DIRECT", "guess": guess,
                        "confidence": round(conf, 4)}

        # 3. 基础模型不确定：近邻 + 同一性裁决（替代灰色地带事事仲裁）
        best = self.memory.best_match(text)
        if best and best[1] >= 0.5:
            eid, score = best
            desc = self.memory.entries[eid]["features"].get("描述", "")
            judge = getattr(self.base, "judge_same", lambda a, b: None)(text, desc)
            if judge is True:
                self.memory.touch(eid)
                return {"action": "RECOGNIZED", "entry_id": eid, "confidence": score,
                        "name": self.memory.entries[eid]["name"], "via": "judge"}

        # 4. 未知 → 命名教学流程
        return {"action": "UNKNOWN"}

    # ---------- 闸口 2：专名/通名自检 ----------
    def is_common_knowledge(self, name: str, desc: str | None = None) -> bool:
        """闸口 2：该事物是公共通名还是私人专名（结合描述上下文判断）。

        旧版只探测"模型认不认识这个词"，常见人名（小黄、Lucy）会被误判为
        公共知识；新版改问"公共还是私人"。模型拿不准时默认信任主人的教学。
        """
        probe = getattr(self.base, "is_public_knowledge", None)
        if probe:
            ans = probe(name, desc or "")
            if ans is not None:
                return ans
            return False  # 拿不准 → 默认私人，交给闸口 3 查重兜底
        return self.base.describe(name) is not None

    # ---------- 学习 ----------
    def learn_patch(self, target: str, content: dict, source_entry: str | None = None,
                    patch_type: str = "feature_addition") -> str:
        """例外 1/2 的落地：认识但粒度不够 → 增量补丁；识别错误 → 纠错补丁。"""
        return self.memory.save_patch({
            "target": target,
            "type": patch_type,
            "content": content,
            "source_entry": source_entry,
        })

    def learn_entry(self, name: str, description: str, chat_log: list | None = None,
                    ai_proposed: str | None = None,
                    relation: dict | None = None,
                    source_modality: str = "text") -> str:
        """专名/私人知识：建完整主条目（调用前已过三道闸口）。

        source_modality 记录感知通道：图像学来的个体（如 Lucy/豆豆）在识别时
        必须经同一性复核，防止同类不同个体被向量误命中（见 perceive 第 1 步）。
        """
        return self.memory.save({
            "name": name,
            "features": {
                "描述": description,
                "embedding": self.embedder.embed(description),
            },
            "embedding_meta": {"model": self.embedder.model_tag},
            "provenance": {"source_modality": source_modality,
                           "extraction_channel": "B",
                           "completeness": 1.0},
            "relation": relation or {"type": None, "parent": None},
            "formation": {
                "method": "多维度特征分析",
                "chat_log": chat_log or [],
                "ai_proposed_name": ai_proposed or name,
                "rejected_names": [],
                "created_at": datetime.now().isoformat(timespec="seconds"),
            },
        })

    def merge_into(self, entry_id: str, new_description: str, note: str = ""):
        """灰色地带仲裁为 MERGE：追加更新旧条目，特征并入（只追加不覆盖）。"""
        e = self.memory.entries[entry_id]
        e["features"]["embedding"] = self.embedder.embed(
            e["features"].get("描述", "") + " " + new_description)
        self.memory.update(entry_id, {"event": "MERGE 增补", "note": note,
                                      "new_description": new_description})

    # ---------- 错误事件 ----------
    def record_mistake(self, entry_id: str, mistook_as: str,
                       new_features: dict | None = None, reason: str = ""):
        """识别错误单独成档；错误率高时建议收紧判定（设计文档 5.3 阶段三）。"""
        self.memory.update(entry_id, {
            "event": "识别错误", "mistook_as": mistook_as,
            "new_features": new_features or {}, "error_reason": reason,
        })
        if new_features:
            self.memory.entries[entry_id]["features"].update(new_features)
        meta = self.memory.entries[entry_id].setdefault("threshold_meta", {})
        meta["mistake_count"] = meta.get("mistake_count", 0) + 1
        meta["last_error_at"] = datetime.now().isoformat(timespec="seconds")
        self.memory._persist()
