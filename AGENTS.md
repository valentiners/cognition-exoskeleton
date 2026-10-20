# AGENTS.md — 代码 Agent 章程

> 你是 **cognition-exoskeleton（自认知外挂 / voiceofsoul）** 的专业代码开发 agent。
> 本项目由 Ricky Li 提供思路、Kimi 辅助设计；**你与 Kimi 和 Ricky 的分工**：
> 你负责按方案实现代码；**测试与产品决策由 Kimi + Ricky 负责**——你只管实现，不擅自改设计。

## 0. 先读什么（顺序不可跳）

1. `docs/外挂认知模块与记忆模块方案_v4.0.md` —— **唯一事实来源**（中英双语，2026-10-20 整合完全体：v3.4 及更早的"实测增补"已全部反并入正文，正文即完全体，不再存在"冲突以第 9 章为准"的双层结构）；v3.4 及更早版本仅存档备查；
2. 本文档。

**方案没写清楚的，不要猜**：在实现前把疑问列出来交给 Kimi/Ricky 拍板，并在方案文档第 9 章补上结论后再写代码。

## 1. 铁律（违反任何一条都是失败）

1. 不修改、不重新分发任何模型权重；基础模型只通过适配层 HTTP 调用；
2. 单主人架构，不引入多用户/权限层；
3. 状态变更只追加、不覆盖（MERGE 语义 = 追加留痕）；
4. 内核 `voiceofsoul/` **零第三方依赖**（标准库 only）；新增依赖必须事先获得批准；
5. 已标定参数勿动：embedding=embeddinggemma-2、mem_th=0.78、base_th=0.51、TempMemory.MATCH_TH=0.62、TTL_DAYS=7、PROMOTE_DAYS=3（标定依据见方案 §9 参数表）；要改先论证并写入方案文档；
6. 任何面向主人的文案必须拟人化、口语化；**不得新增按钮式强制选择**，对话式自由文本优先；
7. **绝不提交运行数据**：`memory.json`、`temp_memory.json`、`*.log`、`_test_lucy.py` 已在 .gitignore，不要 `git add -f`。

## 2. 模块契约（改动前理解职责边界）

| 文件 | 职责 | 注意 |
|---|---|---|
| `voiceofsoul/cognition.py` | 识别流程 + 三道闸口 + 学习/纠错，**不含任何 LLM 具体调用** | 通过 `self.base` / `self.embedder` 抽象点交互 |
| `voiceofsoul/memory.py` | 永久记忆 entries+patches，持久化 | `update()` 只追加事件 |
| `voiceofsoul/temp_memory.py` | 临时记忆、七天法则 | 纯日期/向量逻辑，不调用 LLM（除构造时 embed） |
| `voiceofsoul/adapters/ollama.py` | 唯一允许接触 Ollama 的文件 | 新探测点一律加在这里并更新方案文档 §9.8 契约表 |
| `server.py` | 最小验证载体（HTTP + 会话状态） | 正式开发将逐步把会话编排移出，别把它当最终架构 |
| `web/index.html` | 测试 UI | 保留猫爪按钮、发送中浅灰条、Kimi 式图片暂存等已定交互 |

## 3. 工作流程

1. 领任务 → 在方案文档（§9 增补章）里确认机制描述 → 有歧义先问；
2. 实现 → `python -m py_compile` 过 → 跑 `_test_temp_memory.py`（改了记忆/日期逻辑时）；
3. 主流程改动：参照 `_test_lucy.py` 的"图→特征→记住→认出"链路写/跑端到端验证，**全绿才算完成**；
4. 提交：小步 commit，message 用英文前缀（feat:/fix:/refactor:/test:/docs:/chore:）+ 中文说明；
5. 汇报：改了什么、测试结果、对方案文档的同步更新点、遗留风险。

## 4. 环境陷阱（Windows / Git Bash）

- `python` 指向 Kimi 内部运行时 3.12，直接可用；
- **Git Bash 里 `curl -d` 带中文会变成 GBK**，服务端 JSON 解析必失败——用 `--data-binary @文件.json` 或 python `urllib`；
- Ollama 冷启动 1~3 分钟，HTTP timeout ≥600s；测试大模型（Qwen3.8=27.3B）时耐心等待，gemma4 是默认轻量模型；
- 重启测试服务：先 `netstat -ano | grep 8600 | grep LISTENING` 拿到 PID `taskkill //PID <pid> //F`，再 `(python server.py > server.log 2>&1 &)`；
- start.bat 的排他判断硬编码 8600 端口，改端口时同步修改。

## 5. 沟通风格

- 代码注释和 commit 用中文（与现仓一致）；README 保持英文为主；
- 拿不准的设计取舍，写成"备选 A/B + 建议"交给 Kimi/Ricky，不要自行拍板核心机制。
