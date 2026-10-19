# -*- coding: utf-8 -*-
"""TempMemory 单元测试：七天法则（mock 日期，不依赖 Ollama）。"""
import json
import sys
from datetime import date, timedelta
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent))
from voiceofsoul.temp_memory import TempMemory  # noqa: E402


class FakeEmbedder:
    """确定性假编码器：按关键词分桶，桶内相同、桶间正交。"""

    def embed(self, text):
        if "Lucy" in text:
            return [1.0, 0.0, 0.0]
        if "猫" in text:
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]

    @staticmethod
    def cosine(a, b):
        return sum(x * y for x, y in zip(a, b))


def make_tm(tmp_path, today):
    tm = TempMemory(tmp_path / "t.json", FakeEmbedder())
    d = mock.Mock(wraps=date)
    d.today.return_value = today
    return tm, d


def patch_date(tm, d):
    return mock.patch.object(sys.modules["voiceofsoul.temp_memory"], "date", d)


def test_add_and_expire(tmp_path):
    tm, d = make_tm(tmp_path, date(2026, 10, 19))
    with patch_date(tm, d):
        rid = tm.add("二次元女孩 Lucy")
        assert rid == "T-0001"
        assert tm.get(rid)["days"] == ["2026-10-19"]
        # 第 8 天没出现 → 遗忘
        d.today.return_value = date(2026, 10, 27)  # last_seen 10-19，差 8 天
        gone = tm.expire()
        assert gone == [rid] and not tm.records
    print("PASS add_and_expire")


def test_promote_three_days(tmp_path):
    tm, d = make_tm(tmp_path, date(2026, 10, 19))
    with patch_date(tm, d):
        rid = tm.add("二次元女孩 Lucy")
        # 第一天出现不算晋升
        assert tm.touch(rid) is False
        # 同一天再出现也不算新日期
        assert tm.touch(rid) is False
        # 第二天 → 2 个日期
        d.today.return_value = date(2026, 10, 20)
        assert tm.touch(rid) is False
        # 第三天 → ≥3 个不同日期 → 晋升
        d.today.return_value = date(2026, 10, 21)
        assert tm.touch(rid) is True
        # 晋升后记录还在（由调用方 remove）；第四天仍返回 True
        d.today.return_value = date(2026, 10, 22)
        assert tm.touch(rid) is True
    print("PASS promote_three_days")


def test_window_sliding(tmp_path):
    """第 1、2、7 天出现 → 晋升；若中间断太久，旧日期滑出窗口。"""
    tm, d = make_tm(tmp_path, date(2026, 10, 1))
    with patch_date(tm, d):
        rid = tm.add("二次元女孩 Lucy")
        d.today.return_value = date(2026, 10, 2)
        assert tm.touch(rid) is False
        d.today.return_value = date(2026, 10, 8)  # 7 天后，10-1 滑出 7 天窗口
        assert tm.touch(rid) is False  # 只剩 10-02、10-08 两个日期
        assert tm.get(rid)["days"] == ["2026-10-02", "2026-10-08"]
    print("PASS window_sliding")


def test_best_match(tmp_path):
    tm, d = make_tm(tmp_path, date(2026, 10, 19))
    with patch_date(tm, d):
        tm.add("二次元女孩 Lucy")
        tm.add("三条腿的流浪猫")
        hit = tm.best_match("她叫 Lucy，粉色外套")
        assert hit and hit[0] == "T-0001"
        assert tm.best_match("完全无关的量子物理") is None
    print("PASS best_match")


def test_persist_roundtrip(tmp_path):
    tm, d = make_tm(tmp_path, date(2026, 10, 19))
    with patch_date(tm, d):
        rid = tm.add("二次元女孩 Lucy", {"名字": "Lucy"})
        tm.update_features(rid, "性格", "温柔")
    tm2 = TempMemory(tmp_path / "t.json", FakeEmbedder())
    rec = tm2.get(rid)
    assert rec["features"] == {"名字": "Lucy", "性格": "温柔"}
    tm2.remove(rid)
    tm3 = TempMemory(tmp_path / "t.json", FakeEmbedder())
    assert not tm3.records
    print("PASS persist_roundtrip")


if __name__ == "__main__":
    import tempfile
    for fn in (test_add_and_expire, test_promote_three_days,
               test_window_sliding, test_best_match, test_persist_roundtrip):
        with tempfile.TemporaryDirectory() as td:
            fn(Path(td))
    print("\n全部通过 ✓")
