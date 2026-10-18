"""特征编码器（primitive 版）。

设计文档 3.2 节：embedding 由独立外部编码器生成，与基础模型解耦。
primitive 版为了零依赖离线可跑，用「字符 n-gram 哈希」做基线编码器；
生产环境应替换为 bge-m3 等真实编码器 —— 只需实现同样的 embed() 接口。
"""

from __future__ import annotations

import hashlib
import math


class HashingEmbedder:
    """把文本映射为 L2 归一化的定维向量（字符 n-gram 哈希桶 + 次线性 TF）。

    这是可替换组件：任何提供 embed(text) -> list[float]（L2 归一化）的
    编码器都可直接接入记忆模块。
    """

    def __init__(self, dim: int = 512, ngram_range: tuple = (1, 2)):
        self.dim = dim
        self.ngram_range = ngram_range
        self.model_tag = "hashing-baseline@0.1.0"  # 供 embedding_meta 版本校验（设计文档 3.3）

    def _ngrams(self, text: str) -> list[str]:
        text = text.strip()
        grams = []
        lo, hi = self.ngram_range
        for n in range(lo, hi + 1):
            if len(text) < n:
                if n == lo:  # 短文本至少保留单字
                    grams.extend(text)
                break
            grams.extend(text[i:i + n] for i in range(len(text) - n + 1))
        return grams

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for g in self._ngrams(text):
            h = int(hashlib.md5(g.encode("utf-8")).hexdigest(), 16)
            vec[h % self.dim] += 1.0
        # 次线性 TF，抑制高频词主导
        vec = [math.sqrt(x) for x in vec]
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    @staticmethod
    def cosine(a: list[float], b: list[float]) -> float:
        return sum(x * y for x, y in zip(a, b))  # 输入均为 L2 归一化向量
