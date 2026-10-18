"""voiceofsoul —— 自认知外挂的核心包。

对应设计文档《外挂认知模块与记忆模块方案 v3.3》：
- embedding.py  特征编码（primitive 版为哈希基线，可替换为任意真实编码器）
- memory.py     记忆模块：主条目库 entries + 基础模型知识补丁区 patches
- cognition.py  认知模块：识别 / 仲裁 / 学习 / 记错

单主人原则：唯一主人 = 最高权限 + 唯一可信输入源（见设计文档第 1 节）。
"""

from .embedding import HashingEmbedder
from .memory import MemoryModule
from .cognition import CognitiveModule

__version__ = "0.1.0"  # primitive version

__all__ = ["HashingEmbedder", "MemoryModule", "CognitiveModule"]
