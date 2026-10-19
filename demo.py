#!/usr/bin/env python3
"""cognition-exoskeleton · interactive CLI demo (primitive version)

You play the MASTER (主人); a simulated base model (with a little built-in
common knowledge) drives the full loop:
guess → confirm → patch/entry → re-recognize → record mistakes.

Run: python demo.py
"""

import re

from voiceofsoul import CognitiveModule, HashingEmbedder, MemoryModule

EXIT_CMDS = ("exit", "quit", "退出")
STATS_CMDS = ("stats", "统计")
MISTAKE_PREFIXES = ("mistake ", "错误 ")


class DemoBaseModel:
    """A simulated pre-trained base model: call-only, weights untouched.

    The three methods (infer_confidence / infer / describe) are the probe
    points for "the base model's compressed knowledge" in the design doc.
    Swap in a real model behind the same interface to go to production.
    Confidence here is keyword coverage + an offset — demo simulation,
    not real model behavior.
    """

    KNOWLEDGE = {
        "cat": "A cat is a common household pet with fur and a tail, "
               "orange black white or mixed, small to medium size",
        "dog": "A dog is a common household pet with fur and a tail, "
               "small to large size, barks",
        "car": "A car is a common vehicle with four wheels, a metal body, "
               "windows and doors",
    }
    KEYWORDS = {
        "cat": ["cat", "orange", "tail", "fur"],
        "dog": ["dog", "bark", "tail"],
        "car": ["car", "wheel", "window"],
    }

    def __init__(self, embedder):
        self.embedder = embedder

    def infer_confidence(self, text: str) -> float:
        best = 0.0
        for concept, kws in self.KEYWORDS.items():
            matched = sum(1 for k in kws if k in text.lower())
            conf = 0.1 + 0.22 * matched + (0.3 if concept in text.lower() else 0)
            best = max(best, min(conf, 0.95))
        return best

    def infer(self, text: str) -> str:
        t = text.lower()
        return max(self.KEYWORDS,
                   key=lambda k: sum(1 for w in self.KEYWORDS[k] if w in t))

    def describe(self, name: str) -> str | None:
        """Gate 2 self-check: can the base model give a full, stable
        description? (Yes → common noun; None → proper name.)"""
        return self.KNOWLEDGE.get(name.lower())


# ---------------------------------------------------------------- interaction

def finish_entry(cog, name, desc):
    """Gate 2 (proper/common check) + Gate 3 (dedup) → entry or patch."""
    if cog.is_common_knowledge(name):
        pid = cog.learn_patch(name, {"description": desc},
                              patch_type="feature_addition")
        print(f"  [common] '{name}' is already known to the base model "
              f"→ patch {pid} only, no entry (anti-duplication)")
        return
    best = cog.memory.best_match(desc)
    if best and best[1] >= cog.mem_th:
        cog.merge_into(best[0], desc, note="Gate 3 dedup hit → MERGE")
        print(f"  [dedup] too similar to {best[0]} ({best[1]}) "
              f"→ MERGE, no duplicate entry")
        return
    overlap = next((p for p in cog.memory.patches
                    if p["target"] in name or name in p["target"]), None)
    if overlap:
        cog.memory.update_patch(overlap["id"],
                                {"event": "merge", "note": f"overlap with new '{name}'"})
        print(f"  [dedup] overlaps patch {overlap['id']}'s target "
              f"→ merged into the patch")
        return
    eid = cog.learn_entry(name, desc, ai_proposed=name)
    print(f"  [entry] proper name '{name}' stored: {eid}")


def handle_recognized(act):
    print(f"  [recognized] this is '{act['name']}' "
          f"(memory hit, confidence {act['confidence']})")


def handle_arbitrate(act, cog, text):
    print(f"  [gray zone] this looks a bit like '{act['candidate_name']}' "
          f"(similarity {act['similarity']}) — is it the same one?")
    ans = input("  master (yes/no/it's the child of it) > ").strip().lower()
    eid = act["candidate_id"]
    if ans == "yes":
        cog.merge_into(eid, text, note="gray-zone arbitration → MERGE")
        print(f"  [MERGE] merged into {eid} (append-only)")
    elif ans.startswith("yes") or ans.startswith("it's the child"):
        name = input("  what is the new entry called? > ").strip()
        new_id = cog.learn_entry(name, text,
                                 relation={"type": "subtype_of", "parent": eid})
        print(f"  [SUBCLASS] child entry {new_id} linked under {eid}")
    else:
        name = input("  then what is it called? > ").strip()
        finish_entry(cog, name, text)
        print("  [NEW] stored as a new entry")


def handle_guess(act, cog):
    print(f"  [guess] is this a {act['guess']}? "
          f"(base-model confidence {act['confidence']})")
    if input("  master (y/n) > ").strip().lower() != "y":
        print("  [dropped] guess rejected, nothing stored")
        return
    # Gate 1: confirmation loop on the GUESS branch (design doc 4.2)
    ans = input(f"  so it's just an ordinary {act['guess']}, right? "
                f"or is there something special to remember?\n"
                f"  (type 'ordinary' to drop / a feature to save a patch / "
                f"'named <X>' for a proper-name entry) > ").strip()
    if ans.lower() in ("ordinary", "just ordinary", "yes"):
        print("  [dropped] common knowledge is never stored (anti-duplication)")
    elif ans.lower().startswith("named "):
        name = ans[6:].strip()
        desc = input(f"  please describe '{name}' > ").strip()
        finish_entry(cog, name, f"{name}, {desc}")
    else:
        pid = cog.learn_patch(act["guess"], {"feature": ans},
                              patch_type="feature_addition")
        print(f"  [patch] incremental patch {pid} → target={act['guess']} "
              f"(base model knows the concept, we only add granularity)")


def naming_flow(cog):
    """UNKNOWN branch: admit unknown → analyze → name together → gates 2/3."""
    desc = input("  master, please tell me what it is (name + features) > ").strip()
    name = re.split(r"[,，]", desc)[0].strip()
    finish_entry(cog, name, desc)


def record_mistake(cog, arg):
    """Command: mistake <entry_id> <mistook_as> [new_feature]
    （也支持中文命令：错误 <条目ID> <错认为> [新特征]）"""
    parts = arg.split()
    if len(parts) < 2:
        print("  usage: mistake <entry_id> <mistook_as> [new_feature]")
        return
    eid, mistook_as = parts[0], parts[1]
    new_features = {"added_feature": parts[2]} if len(parts) > 2 else None
    if eid not in cog.memory.entries:
        print(f"  [mistake] no such entry: {eid}")
        return
    cog.record_mistake(eid, mistook_as, new_features)
    print(f"  [mistake] recorded for {eid} (append-only, nothing overwritten)")


def print_stats(cog):
    s = cog.memory.stats()
    print(f"  [stats] {s['entries']} entries, {s['patches']} patches "
          f"({s['patches_active']} active)")
    for eid, e in cog.memory.entries.items():
        h, meta = e.get("health", {}), e.get("threshold_meta", {})
        print(f"    {eid} '{e['name']}' hits={h.get('hit_count', 0)} "
              f"errors={meta.get('mistake_count', 0)} updates={len(e.get('updates', []))}")
    for p in cog.memory.patches:
        print(f"    {p['id']} → target={p['target']} [{p['type']}] "
              f"status={p['status']} content={p['content']}")


def main():
    print("=" * 60)
    print(" cognition-exoskeleton · primitive version")
    print(" you are the MASTER (主人) — type descriptions, or commands:")
    print(" mistake <id> <wrong_as> [feature] | 错误 <ID> <错认为> [特征]")
    print(" stats | exit / 退出")
    print("=" * 60)
    embedder = HashingEmbedder()
    memory = MemoryModule("demo_memory.json", embedder)  # 独立演示记忆，不碰真实记忆库
    cog = CognitiveModule(DemoBaseModel(embedder), memory, embedder)

    while True:
        try:
            text = input("\nmaster > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        if text.lower() in EXIT_CMDS:
            break
        low = text.lower()
        if any(low.startswith(p) for p in MISTAKE_PREFIXES):
            arg = text.split(" ", 1)[1] if " " in text else ""
            record_mistake(cog, arg)
            continue
        if low in STATS_CMDS:
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
            print("  [unknown] I admit I cannot recognize this.")
            naming_flow(cog)

    print("\nsession ended, memory persisted to demo_memory.json")


if __name__ == "__main__":
    main()
