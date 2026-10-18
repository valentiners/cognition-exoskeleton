# cognition-exoskeleton（自认知外挂）

> 一款大模型的认知灵魂外挂 —— 不动权重，让任何已训练模型拥有「识别 → 不确定就求助主人 → 共同命名 → 存库 → 复用与纠错」的持续学习能力。

**voiceofsoul** · primitive version (v0.1.0) · Apache-2.0

---

## 这是什么

给冻结的基础模型外接一个「认知模块 + 记忆模块」，实现外挂式持续学习：

```
新输入 ──→ 特征提取/统一描述 ──→ 查记忆模块(≥70% 确认命中)
                                  │
                    ┌─────────────┼──────────────┐
                    ▼             ▼              ▼
                 ≥70%        51%~70%         <51%
               确认命中      主人仲裁        比对基础模型
               输出结果     MERGE/NEW/      (>51% 猜测 /
                            SUBCLASS        ≤51% 承认未知→命名)
```

核心设计原则：

- **不动原模型**：只调用权重，绝不修改源文件与权重；
- **防重复记忆**：基础模型已知的常识绝不入库，只给它的知识"打补丁"（补丁区）；
- **主人仲裁不可让渡**：灰色地带、命名确认、补丁冲突，分叉路口必须主人拍板；
- **只追加、不覆盖**：所有演化轨迹（否决名、错误事件、审计记录）全程留痕；
- **单主人原则**：唯一主人 = 最高权限 + 唯一可信输入源。

## 快速开始

零依赖（纯 Python 标准库），Python 3.9+：

```bash
git clone https://github.com/<your-org>/cognition-exoskeleton.git
cd cognition-exoskeleton
python demo.py
```

`demo.py` 是一个命令行交互 Demo：你扮演"主人"，内置一个模拟基础模型（带少量常识知识），走完整个「猜测 → 确认 → 补丁/建条目 → 再识别 → 记错误」闭环。记忆库持久化为当前目录的 `memory.json`。

### 会话命令

| 输入 | 作用 |
|---|---|
| 任意描述文本 | 进入识别流程（识别/仲裁/猜测/未知） |
| `错误 <条目ID> <错认为> <新特征>` | 记录一次识别错误（只追加留痕） |
| `stats` | 查看记忆库统计 |
| `退出` / `exit` | 结束会话 |

## 示例

见 [examples/sample_session.md](examples/sample_session.md)。

## 设计文档

完整设计见 [docs/外挂认知模块与记忆模块方案_v3.3.md](docs/外挂认知模块与记忆模块方案_v3.3.md)——
涵盖两级阈值与仲裁区、双通道多模态特征提取、双闸口防重复、补丁区、换基础模型审计流程、健康度巡检遗忘机制等。

## Roadmap

primitive version 已实现：两级阈值 + 灰色地带仲裁、双闸口防重复、补丁区（feature_addition/correction）、只追加不覆盖、JSON 持久化、命令行 Demo。

| 方向 | 状态 | 设计文档章节 |
|---|---|---|
| 审计惰性化（换模型零停机） | ⏳ Roadmap | 4.5 |
| 补丁间冲突裁决（priority） | ⏳ Roadmap | 4.5.6 |
| REDUNDANT 扰动稳定性测试 | ⏳ Roadmap | 4.5.3 |
| 阈值按基础模型分桶标定 | ⏳ Roadmap | 5.3 |
| 真实 embedding 模型接入（当前为哈希基线，仅用于离线演示） | ⏳ Roadmap | 3.2 |
| 图像/多模态输入 | ⏳ Roadmap | 第 3 节 |
| 语音输入 | ⏳ Roadmap | 遗留问题 |

## 项目结构

```
cognition-exoskeleton/
├── demo.py                # 命令行交互 Demo（你扮演主人）
├── voiceofsoul/           # 核心包
│   ├── embedding.py       # 哈希基线编码器（零依赖，可替换为真实 embedding 模型）
│   ├── memory.py          # 记忆模块：主条目库 entries + 补丁区 patches
│   └── cognition.py       # 认知模块：识别/仲裁/学习/记错
├── docs/                  # 设计文档（方案 v3.3）
├── examples/              # 示例会话
└── LICENSE                # Apache-2.0
```

## English

**cognition-exoskeleton** is a plug-in "cognitive soul" for frozen base models. Without touching any weights, it gives an already-trained model a continuous-learning loop: perceive → ask the master when unsure → name together → store → reuse and self-correct.

- Two-tier thresholds (51% guess / 70% confirm) with a master-arbitrated gray zone (MERGE / NEW / SUBCLASS)
- Anti-duplicate gates: common knowledge never enters the memory; only incremental *patches* are attached to the base model's known concepts
- Append-only storage: full cognitive evolution trail (rejected names, mistakes, audits)
- Single-master design: the master is the sole authority and the only trusted input source

Quick start: `python demo.py` (zero dependencies, Python 3.9+).

## License

Apache License 2.0, see [LICENSE](LICENSE).
