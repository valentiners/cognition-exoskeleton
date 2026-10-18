#!/usr/bin/env python3
"""自认知外挂 · 命令行交互 Demo（primitive version）

你扮演「主人」，内置一个模拟基础模型（带少量常识知识），
走完「识别 → 仲裁 → 猜测确认 → 补丁/建条目 → 再识别 → 记错误」闭环。

运行：python demo.py
"""

import re

from voiceofsoul import CognitiveModule, HashingEmbedder, MemoryModule


class DemoBaseModel:
    """模拟已训练好的基础模型：只调用，不改权重。

    infer_confidence / infer / describe 三个接口对应设计文档中
    「基础模型压缩知识」的探测点。替换成真实模型时保持接口即可。
    置信度用关键词覆盖率 + 基线偏移模拟（演示用，非真实模型行为）。
    """

    KNOWLEDGE = {
        "猫": "猫是常见的家养宠物，有毛发和尾巴，橘色黑色白色花色，体型小中型",
        "狗": "狗是常见的家养宠物，有毛发和尾巴，体型小到大型，会吠叫",
        "汽车": "汽车是常见交通工具，四个轮子，金属外壳，有车窗和车门",
    }
    KEYWORDS = {
        "猫": ["猫", "橘", "尾", "毛"],
        "狗": ["狗", "吠", "尾"],
        "汽车": ["汽车", "车", "轮", "车窗"],
    }

    def __init__(self, embedder):
        self.embedder = embedder

    def infer_confidence(self, text: str) -> float:
        best = 0.0
        for concept, kws in self.KEYWORDS.items():
            matched = sum(1 for k in kws if k in text)
            conf = 0.1 + 0.22 * matched + (0.3 if concept in text else 0)
            best = max(best, min(conf, 0.95))
        return best

    def infer(self, text: str) -> str:
        return max(self.KEYWORDS,
                   key=lambda k: sum(1 for w in self.KEYWORDS[k] if w in text))

    def describe(self, name: str) -> str | None:
        """闸口 2 自检：基础模型能否给出完整稳定描述（认识 → 通名）。"""
        return self.KNOWLEDGE.get(name)


# ---------------------------------------------------------------- 交互流程

def finish_entry(cog, name, desc):
    """闸口 2（专名/通名自检）+ 闸口 3（建库前去重）→ 建条目或转补丁。"""
    if cog.is_common_knowledge(name):
        pid = cog.learn_patch(name, {"描述": desc}, patch_type="feature_addition")
        print(f"  [通名] 「{name}」是基础模型已知概念 → 只存补丁 {pid}，不建条目")
        return
    best = cog.memory.best_match(desc)
    if best and best[1] >= cog.mem_th:
        cog.merge_into(best[0], desc, note="闸口 3 查重命中 MERGE")
        print(f"  [查重] 与 {best[0]} 高度相似（{best[1]}）→ MERGE，不建重复条目")
        return
    overlap = next((p for p in cog.memory.patches
                    if p["target"] in name or name in p["target"]), None)
    if overlap:
        cog.memory.update_patch(overlap["id"],
                                {"event": "内容合并", "note": f"与新建「{name}」重合"})
        print(f"  [查重] 与补丁 {overlap['id']} 的 target 重合 → 合并进补丁")
        return
    eid = cog.learn_entry(name, desc, ai_proposed=name)
    print(f"  [建条目] 专名「{name}」已入库：{eid}")


def handle_recognized(act):
    print(f"  [识别] 这是「{act['name']}」（记忆库命中，置信度 {act['confidence']}）")


def handle_arbitrate(act, cog, text):
    print(f"  [灰色地带] 这和之前认识的「{act['candidate_name']}」有点像"
          f"（相似度 {act['similarity']}），是同一个吗？")
    ans = input("  主人（是/不是/是TA的幼体）> ").strip()
    eid = act["candidate_id"]
    if ans == "是":
        cog.merge_into(eid, text, note="灰色地带仲裁 MERGE")
        print(f"  [MERGE] 已增补进 {eid}（只追加留痕）")
    elif ans.startswith("是") and ans.endswith("幼体"):
        name = input("  新条目叫什么名字？> ").strip()
        new_id = cog.learn_entry(name, text, relation={"type": "subtype_of", "parent": eid})
        print(f"  [SUBCLASS] 已建子类条目 {new_id}，挂在 {eid} 下")
    else:
        name = input("  那它叫什么名字？> ").strip()
        finish_entry(cog, name, text)
        print("  [NEW] 已按新条目入库")


def handle_guess(act, cog):
    print(f"  [猜测] 这是不是{act['guess']}？（基础模型置信度 {act['confidence']}）")
    if input("  主人（y/n）> ").strip().lower() != "y":
        print("  [丢弃] 猜测被否定，流程结束（不入库）")
        return
    # 闸口 1：GUESS 分支确认回环（设计文档 4.2）
    ans = input(f"  这就是普通的{act['guess']}，对吗？还是有特别之处想让我记住的？\n"
                f"  （输入「普通」丢弃 / 输入特征描述存补丁 / 输入「叫<名字>」建专名条目）> ").strip()
    if ans in ("普通", "就是普通的", "对"):
        print("  [丢弃] 常识不入库（防重复原则）")
    elif ans.startswith("叫"):
        name = ans[1:].strip()
        desc = input(f"  请补充「{name}」的特征描述> ").strip()
        finish_entry(cog, name, f"{name}，{desc}")
    else:
        pid = cog.learn_patch(act["guess"], {"特征": ans}, patch_type="feature_addition")
        print(f"  [补丁] 已存增量补丁 {pid} → target={act['guess']}（基础模型已知，只补粒度）")


def naming_flow(cog):
    """UNKNOWN 分支：承认未知 → 多维度分析 → 共同命名 → 闸口 2/3。"""
    desc = input("  请主人告诉我它是什么（名字 + 特征描述）> ").strip()
    name = re.split(r"[，,]", desc)[0].strip()
    finish_entry(cog, name, desc)


def record_mistake(cog, arg):
    """命令：错误 <条目ID> <错认为> [新特征]"""
    parts = arg.split()
    if len(parts) < 2:
        print("  用法：错误 <条目ID> <错认为> [新特征]")
        return
    eid, mistook_as = parts[0], parts[1]
    new_features = {"补充特征": parts[2]} if len(parts) > 2 else None
    if eid not in cog.memory.entries:
        print(f"  [记错] 找不到条目 {eid}")
        return
    cog.record_mistake(eid, mistook_as, new_features)
    print(f"  [记错] {eid} 的错误事件已单独成档（只追加不覆盖）")


def print_stats(cog):
    s = cog.memory.stats()
    print(f"  [统计] 主条目 {s['entries']} 条，补丁 {s['patches']} 条"
          f"（生效中 {s['patches_active']}）")
    for eid, e in cog.memory.entries.items():
        h, meta = e.get("health", {}), e.get("threshold_meta", {})
        print(f"    {eid} 「{e['name']}」 命中{h.get('hit_count', 0)}次 "
              f"错误{meta.get('mistake_count', 0)}次 更新{len(e.get('updates', []))}条")
    for p in cog.memory.patches:
        print(f"    {p['id']} → target={p['target']} [{p['type']}] "
              f"status={p['status']} 内容={p['content']}")


def main():
    print("=" * 56)
    print(" 自认知外挂 · primitive version —— 你扮演主人，按提示交互")
    print(" 命令：错误 <条目ID> <错认为> [新特征] | stats | 退出")
    print("=" * 56)
    embedder = HashingEmbedder()
    memory = MemoryModule("memory.json", embedder)
    cog = CognitiveModule(DemoBaseModel(embedder), memory, embedder)

    while True:
        try:
            text = input("\n主人 > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        if text in ("退出", "exit", "quit"):
            break
        if text.startswith("错误 "):
            record_mistake(cog, text[3:])
            continue
        if text == "stats":
            print_stats(cog)
            continue

        act = cog.perceive(text)
        if act["action"] == "RECOGNIZED":
            handle_recognized(act)
        elif act["action"] == "ARBITRATE":
            handle_arbitrate(act, cog, text)
        elif act["action"] == "GUESS":
            handle_guess(act, cog)
        else:
            print("  [未知] 我承认无法识别这个目标。")
            naming_flow(cog)

    print("\n会话结束，记忆已持久化到 memory.json")


if __name__ == "__main__":
    main()
