# Sample Session

> **版本注记**：本实录记录的是离线 CLI 演示（`demo.py`）的 **v3.3 时代行为**（y/n 确认、按钮式回环）。v3.4 定型后的对话式交互（无按钮、自由文本、临时记忆七天法则）见浏览器测试服务与设计文档 §9.1/§9.2。

Recorded run of `python demo.py` (primitive version; the simulated base model knows three common concepts: cat / dog / car).

## Session: the full loop

```
master > an orange fat cat
  [guess] is this a cat? (base-model confidence 0.84)
  master (y/n) > y
  so it's just an ordinary cat, right? or is there something special to remember?
  (type 'ordinary' to drop / a feature to save a patch / 'named <X>' for a proper-name entry) > hook-shaped tail
  [patch] incremental patch P-0001 → target=cat (base model knows the concept, we only add granularity)

master > an orange cat with a hook-shaped tail
  [guess] is this a cat? (base-model confidence 0.95)
  master (y/n) > y
  so it's just an ordinary cat, right? or is there something special to remember? > named Doudou
  please describe 'Doudou' > my orange cat with a hook-shaped tail
  [entry] proper name 'Doudou' stored: M-0001

master > orange hook-shaped tail cat
  [recognized] this is 'Doudou' (memory hit, confidence 0.843)

master > orange cat
  [gray zone] this looks a bit like 'Doudou' (similarity 0.6185) — is it the same one?
  master (yes/no/it's the child of it) > yes
  [MERGE] merged into M-0001 (append-only)

master > mistake M-0001 fox notched-ear
  [mistake] recorded for M-0001 (append-only, nothing overwritten)

master > stats
  [stats] 1 entries, 1 patches (1 active)
    M-0001 'Doudou' hits=1 errors=1 updates=2
    P-0001 → target=cat [feature_addition] status=active content={'feature': 'hook-shaped tail'}
```

## What each step demonstrates

| Step | Design doc mechanism |
|---|---|
| Round 1: guess → patch | 51% guess branch + Gate 1 confirmation loop; base model already knows "cat", so only the "hook-shaped tail" granularity is patched (§4.1 exception 1, §4.2 Gate 1) 基础模型已知概念只补粒度 |
| Round 2: proper-name entry | Gate 2: `is_common_knowledge("Doudou")` = False → Gate 3 dedup passes → main entry (§4.2) 专名建条目 |
| Round 3: instant recognition | memory hit ≥70% (§2.1) |
| Round 4: gray zone | 51%~70% never auto-decides; master arbitrates MERGE / NEW / SUBCLASS (§2.1, v2.0 revision) 灰色地带主人仲裁 |
| Round 5: mistake filed | mistakes go to `updates` separately, old features untouched (§5.2 principle 2) 错误单独成档 |
| stats | everything append-only, full evolution trail inspectable 只追加留痕 |

## More things to try

- `a cat` → guess "cat" → answer "ordinary" → dropped, nothing stored (anti-duplication 防重复)
- `a red car` → guess "car" → "ordinary" → dropped
- `Mimi, a small orange cat` → base model doesn't know "Mimi" → naming flow → proper-name entry
- `orange cat` (after a "Doudou" entry exists) → gray-zone arbitration; answer "it's the child of it" → child entry linked to the parent 子类条目挂父子链接
- Commands also accept Chinese: `错误 M-0001 狐狸 缺角` works the same as `mistake` 中英文命令等效
