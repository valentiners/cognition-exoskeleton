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
        hits = self.memory.search(text, threshold=self.mem_th)
        if hits:
            eid, score = hits[0]
            self.memory.touch(eid)
            return {"action": "RECOGNIZED", "entry_id": eid, "confidence": score,
                    "name": self.memory.entries[eid]["name"]}

        best = self.memory.best_match(text)
        if best and best[1] >= self.base_th:  # 51%~70% 灰色地带
            eid, score = best
            return {"action": "ARBITRATE", "candidate_id": eid, "candidate_name":
                    self.memory.entries[eid]["name"], "similarity": score}

        conf = self.base.infer_confidence(text)
        if conf > self.base_th:
            return {"action": "GUESS", "guess": self.base.infer(text),
                    "confidence": round(conf, 4)}

        return {"action": "UNKNOWN"}

    # ---------- 闸口 2：专名/通名自检 ----------
    def is_common_knowledge(self, name: str) -> bool:
        """基础模型能否零样本给出完整、稳定的描述（设计文档 4.2 闸口 2）。"""
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
                    relation: dict | None = None) -> str:
        """专名/私人知识：建完整主条目（调用前已过三道闸口）。"""
        return self.memory.save({
            "name": name,
            "features": {
                "描述": description,
                "embedding": self.embedder.embed(description),
            },
            "embedding_meta": {"model": self.embedder.model_tag},
            "provenance": {"source_modality": "text", "extraction_channel": "B",
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
