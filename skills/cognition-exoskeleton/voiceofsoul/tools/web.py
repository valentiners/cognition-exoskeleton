"""工具通道：web_search（设计文档 §8.3，v5.0 增补）。

契约：

    web_search(query: str, lang: str = "zh") -> list[dict]
        返回 [{title, url, snippet}, ...]；任何失败（无网络/超时/零结果）一律返回 []。

纪律（与适配层同构）：

- 本文件是**唯一接触检索后端**的文件；换检索源 = 只改这一个文件，内核零改动；
- 只用标准库（urllib / json / html.parser），不引入任何第三方依赖；
- 检索流量不携带任何凭据（不发 cookie，不带 API key）；
- 任何异常都被吞掉并返回空列表——好奇心自查失败只意味着"回落问主人"，绝不打断正常对话。

后端（环境变量）：

- ``COGSOUL_SEARCH_BACKEND``：``duckduckgo``（默认）或 ``searxng``；
- ``COGSOUL_SEARCH_BASE``：SearXNG 实例地址，例如 ``http://localhost:8888``；
- ``COGSOUL_SEARCH_TIMEOUT``：单次检索超时秒数，默认 6。

DuckDuckGo 走 ``html.duckduckgo.com`` 的无脚本 HTML 端点；SearXNG 走
``{base}/search?q=...&format=json`` 兼容 API。
"""

from __future__ import annotations

import json
import os
import re
import urllib.parse
import urllib.request
from html.parser import HTMLParser

DEFAULT_BACKEND = "duckduckgo"
DEFAULT_TIMEOUT = 6.0
MAX_RESULTS = 5

# 检索流量不带凭据，也不伪装成浏览器登录态：一个诚实的自述 UA 即可。
USER_AGENT = ("cognition-exoskeleton/0.1 "
              "(+https://github.com/valentiners/cognition-exoskeleton)")


def _timeout() -> float:
    raw = os.environ.get("COGSOUL_SEARCH_TIMEOUT", "")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return DEFAULT_TIMEOUT
    return value if value > 0 else DEFAULT_TIMEOUT


def _request(url: str, data: bytes | None = None,
             timeout: float | None = None) -> str:
    """GET/POST 一个 URL，返回 UTF-8 文本。失败抛异常，由 web_search 统一兜底。"""
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/json;q=0.9,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.6",
    }
    if data is not None:
        headers["Content-Type"] = "application/x-www-form-urlencoded"
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout or _timeout()) as resp:
        raw = resp.read()
    return raw.decode("utf-8", errors="replace")


# ------------------------------------------------------------------ DuckDuckGo

class _DDGParser(HTMLParser):
    """从 DuckDuckGo HTML 结果页里抽出 title / url / snippet。"""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.results: list[dict] = []
        self._cur: dict | None = None
        self._in_title = False
        self._in_snippet = 0

    def handle_starttag(self, tag, attrs):
        attr = {k: (v or "") for k, v in attrs}
        cls = attr.get("class", "")
        if tag == "a" and "result__a" in cls:
            self._flush()
            self._cur = {"title": "", "url": _clean_ddg_url(attr.get("href", "")),
                         "snippet": ""}
            self._in_title = True
        elif self._cur is not None and "result__snippet" in cls:
            self._in_snippet += 1
        elif self._cur is not None and "result__url" in cls and not self._in_title:
            pass  # 结果页脚里的展示域名，不进 snippet

    def handle_endtag(self, tag):
        if tag == "a" and self._in_title:
            self._in_title = False
        elif self._in_snippet and tag in ("a", "div", "span"):
            self._in_snippet = max(0, self._in_snippet - 1)

    def handle_data(self, data):
        if self._cur is None:
            return
        text = re.sub(r"\s+", " ", data)
        if self._in_title:
            self._cur["title"] += text
        elif self._in_snippet:
            self._cur["snippet"] += text

    def _flush(self):
        if self._cur:
            item = {k: re.sub(r"\s+", " ", v).strip()
                    for k, v in self._cur.items()}
            if item["url"]:
                self.results.append(item)
        self._cur = None

    def close(self):
        super().close()
        self._flush()


def _clean_ddg_url(href: str) -> str:
    """DuckDuckGo 的结果链接常是 /l/?uddg=<encoded>，解出真实地址。"""
    href = (href or "").strip()
    if href.startswith("//"):
        href = "https:" + href
    parsed = urllib.parse.urlsplit(href)
    if parsed.path.startswith("/l/"):
        target = urllib.parse.parse_qs(parsed.query).get("uddg")
        if target:
            return urllib.parse.unquote(target[0])
    return href


def _duckduckgo(query: str, lang: str) -> list[dict]:
    payload = urllib.parse.urlencode({
        "q": query,
        "kl": "cn-zh" if (lang or "zh").lower().startswith("zh") else "us-en",
    }).encode("utf-8")
    html = _request("https://html.duckduckgo.com/html/", data=payload)
    parser = _DDGParser()
    parser.feed(html)
    parser.close()
    return parser.results


# -------------------------------------------------------------------- SearXNG

def _searxng(query: str, lang: str) -> list[dict]:
    base = (os.environ.get("COGSOUL_SEARCH_BASE") or "").strip().rstrip("/")
    if not base:
        return []
    params = urllib.parse.urlencode({
        "q": query,
        "format": "json",
        "language": "zh-CN" if (lang or "zh").lower().startswith("zh") else "en",
    })
    data = json.loads(_request(f"{base}/search?{params}"))
    results = []
    for item in (data.get("results") or [])[:MAX_RESULTS]:
        url = (item.get("url") or "").strip()
        if not url:
            continue
        results.append({
            "title": (item.get("title") or "").strip(),
            "url": url,
            "snippet": (item.get("content") or item.get("snippet") or "").strip(),
        })
    return results


# ----------------------------------------------------------------- public API

def web_search(query: str, lang: str = "zh") -> list[dict]:
    """Search the web; ALWAYS returns a list, [] on any failure.

    A zero-result search and a dead network look the same to the caller by
    design: curiosity then falls back to asking the master (§4.9).
    """
    q = (query or "").strip()
    if not q:
        return []
    backend = (os.environ.get("COGSOUL_SEARCH_BACKEND") or DEFAULT_BACKEND
               ).strip().lower()
    try:
        if backend in ("searxng", "searx"):
            results = _searxng(q, lang)
        elif backend in ("duckduckgo", "ddg", "html"):
            results = _duckduckgo(q, lang)
        else:
            return []
    except Exception:
        # 无网络 / 超时 / 后端改版 / JSON 损坏：统一当作"查不到"。
        return []
    clean = []
    for item in results[:MAX_RESULTS]:
        title = (item.get("title") or "").strip()
        url = (item.get("url") or "").strip()
        snippet = (item.get("snippet") or "").strip()
        if url:
            clean.append({"title": title, "url": url, "snippet": snippet})
    return clean


__all__ = ["web_search", "DEFAULT_BACKEND", "DEFAULT_TIMEOUT"]
