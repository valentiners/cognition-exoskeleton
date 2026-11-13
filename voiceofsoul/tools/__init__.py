"""工具通道（设计文档 §8.3）。

探测点（§8.2）是"问基础模型"，工具通道是"问外部世界"。
本包只放标准库实现，当前唯一工具是 web_search。
"""

from .web import web_search

__all__ = ["web_search"]
