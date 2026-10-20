"""基础模型适配层：把真实模型接到 voiceofsoul 内核后面。

设计文档要求（3.2 解耦 / 7 接口定义）：
- 基础模型适配器只需提供 infer / infer_confidence / describe 三个探测点；
- embedding 由独立编码器生成，与基础模型无关。
"""

from .ollama import OllamaBaseModel, OllamaEmbedder

__all__ = ["OllamaBaseModel", "OllamaEmbedder"]
