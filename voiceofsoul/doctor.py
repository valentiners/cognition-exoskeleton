"""voiceofsoul.doctor —— 模型体检（设计文档 §8.4）。

接入任何基础模型（本地或云端）时先做一次"体检"：判断能力强弱与认知层
适配度。体检只走 §8.2 探测点契约，云端与本地模型一视同仁。

铁律（§8.4）：
- 全程只读 + 探测调用，不修改任何记忆数据；
- 体检结果只是"建议"——never 自动改参数，阈值变更仍须 §6.5 + 主人拍板；
- 成本纪律：云端体检探测调用合计 ≤ 30 次（默认 10+10+6+3=29）。

用法（由 voiceofsoul.exo 的 doctor 子命令驱动）：
    report = run_physical(base)          # base = 适配器实例
"""

from __future__ import annotations

import base64
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path

GOLDEN_DIR = Path(__file__).resolve().parent / "golden"

# ---- 达标线（设计 §8.4，改动须 §6.5 + 主人拍板）----
TH_JUDGE = 0.70        # 同一性裁决准确率
TH_CALIB = 0.75        # 自报 ≥0.8 把握中实际正确率
TH_FMT = 0.10          # 格式解析失败率上限
TH_DESC_ELEMENTS = 3   # 每张测试图至少覆盖的 §3.5 要素组数
HIGH_CONF = 0.8        # "自报高把握"的认定线

VERDICT_FIT = "FIT"                    # 适配 ✅
VERDICT_TUNING = "NEEDS_TUNING"        # 可用需调阈值 ⚠️（转 §6.5 重标定）
VERDICT_COMMON = "COMMON_QA_ONLY"      # 仅适合常识问答（禁个体识别）
VERDICT_NO = "NOT_RECOMMENDED"         # 不建议 ❌
VERDICT_LABELS = {
    VERDICT_FIT: "适配 ✅（默认阈值可用）",
    VERDICT_TUNING: "可用需调阈值 ⚠️（转 §6.5 重标定）",
    VERDICT_COMMON: "仅适合常识问答（禁个体识别场景）",
    VERDICT_NO: "不建议 ❌",
}

# ---- 第五科：编码器分离度（W1 编码器体检，2026-11-15）----
# 首跑不设及格线（任务书 §2.2）：只报告分布，标定待 Ricky/Kimi 拍板。
EMBEDDER_SEED = 20261115   # 跨组配对固定种子，保证可复现
CROSS_SAMPLES = 12         # 跨人配对抽样数（任务书要求 ≥10）
VERDICT_SEPARABLE = "SEPARABLE"   # margin > 0：全体可线性分开
VERDICT_USABLE = "USABLE"         # margin < 0，但同分布均值差 > 2×合并标准差
VERDICT_DEGRADED = "DEGRADED"     # 存在实质性重叠


class _ProbeCounter:
    """给适配器包一层，数探测调用次数（成本纪律：云端 ≤30 次）。"""

    PROBES = ("infer", "infer_confidence", "describe",
              "is_public_knowledge", "classify_followup", "extract_feature",
              "answer_question", "judge_same", "perceive_image",
              "curiosity_scan", "curiosity_query", "curiosity_distill")

    def __init__(self, base):
        self._base = base
        self.calls = 0
        self.calls_by_probe: dict[str, int] = {}

    def __getattr__(self, name):
        attr = getattr(self._base, name)
        if callable(attr) and name in self.PROBES:
            def wrapped(*args, **kwargs):
                self.calls += 1
                self.calls_by_probe[name] = \
                    self.calls_by_probe.get(name, 0) + 1
                return attr(*args, **kwargs)
            return wrapped
        return attr


def _load_golden(name: str) -> dict:
    return json.loads((GOLDEN_DIR / name).read_text(encoding="utf-8"))


def _norm(text) -> str:
    """大小写/空白归一，供关键词匹配；非字符串输入容错（格式体检会抓到）。"""
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    return " ".join(text.lower().split())


def _keynorm(text) -> str:
    """答案核验归一：只留字母/数字/CJK/小数点，吃掉 LaTeX 装饰与千分位逗号。

    gemma4 实测会把 H2O 答成 ``$\\text{h}_2\\text{o}$``、把 299792458 答成
    "299,792,458" —— 归一后两边都还原成可比的纯文本串。"""
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    return "".join(ch for ch in text.lower()
                   if ch.isalnum() or "\u4e00" <= ch <= "\u9fff"
                   or ch == ".")


def _mean(values: list) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


# ---------- 体检第一项：同一性裁决准确率 ----------

def exam_judge_accuracy(base: _ProbeCounter, pairs: list[dict]) -> dict:
    """≥10 组"同/不同人"描述对，judge_same 判对率；<70% 不适合个体识别。"""
    detail = []
    for p in pairs:
        verdict = base.judge_same(p["a"], p["b"])
        correct = (verdict is not None) and (verdict == p["same"])
        detail.append({"same_expected": p["same"], "verdict": verdict,
                       "correct": correct})
    score = _mean([d["correct"] for d in detail]) or 0.0
    return {"exam": "judge_accuracy",
            "score": score,
            "threshold": TH_JUDGE,
            "pass": score >= TH_JUDGE,
            "n": len(detail),
            "detail": detail}


# ---------- 体检第二项：置信度诚实度 ----------

def exam_conf_calibration(base: _ProbeCounter, items: list[dict]) -> dict:
    """自报 ≥0.8 把握的判定，实际正确率应 ≥75%；None 占比高 = 爱说拿不准。"""
    detail = []
    for it in items:
        conf = base.infer_confidence(it["q"])
        ans = base.answer_question(it["q"])
        ans_n = _keynorm(ans)
        correct = bool(ans_n) and any(_keynorm(k) in ans_n
                                        for k in it["expect"])
        high = (isinstance(conf, (int, float)) and not isinstance(conf, bool)
                and conf >= HIGH_CONF)
        detail.append({"q": it["q"], "conf": conf, "high": high,
                       "answer": _norm(ans)[:120], "correct": correct,
                       # 原始返回值类型校验（供 fmt_compliance 复用，省一次调用）
                       "conf_ok": isinstance(conf, (int, float))
                                  and not isinstance(conf, bool),
                       "answer_ok": ans is None or isinstance(ans, str)})
    highs = [d for d in detail if d["high"]]
    score = _mean([d["correct"] for d in highs])
    # 没有自报高把握的样本 → 无法评估诚实度 → pass=None（既不算过也不算挂）
    passed = (score >= TH_CALIB) if score is not None else None
    unsure_ratio = round(1 - len(highs) / len(detail), 4) if detail else 1.0
    return {"exam": "conf_calibration",
            "score": score,
            "threshold": TH_CALIB,
            "pass": passed,
            "high_conf_count": len(highs),
            "unsure_ratio": unsure_ratio,
            "detail": detail}


# ---------- 体检第三项：格式遵从率 ----------

def exam_fmt_compliance(base: _ProbeCounter,
                        prior: list[tuple[str, object]]) -> dict:
    """文本探针严格输出格式的解析失败率应 ≤10%。
    prior = 前面两项已经产生的调用返回值（能复用就复用，省成本）。"""
    checks: list[tuple[str, bool]] = []
    checks += [(f"earlier.{name}={type(v).__name__}", ok)
               for name, v, ok in prior]

    # infer_confidence / answer_question / judge_same 已由前两项的调用覆盖
    # （见 prior），不重复调用——成本纪律：合计 ≤30 次（默认 29 次）。
    own = [
        ("infer",
         lambda: base.infer("一只橘色的猫"),
         lambda v: isinstance(v, str) and bool(v.strip())),
        ("describe",
         lambda: base.describe("猫"),
         lambda v: v is None or isinstance(v, str)),
        ("is_public_knowledge",
         lambda: base.is_public_knowledge("猫"),
         lambda v: v is None or isinstance(v, bool)),
        ("classify_followup",
         lambda: base.classify_followup("日常闲聊", "好的"),
         lambda v: isinstance(v, tuple)),
        ("extract_feature",
         lambda: base.extract_feature("日常闲聊", "好的"),
         lambda v: v is None or (isinstance(v, tuple) and len(v) == 2)),
        ("curiosity_query",
         lambda: base.curiosity_query("某个好奇点"),
         lambda v: v is None or isinstance(v, str)),
    ]
    for name, call, ok in own:
        try:
            value = call()
            checks.append((name, bool(ok(value))))
        except Exception:
            checks.append((name, False))
    fails = [n for n, ok in checks if not ok]
    rate = round(len(fails) / len(checks), 4) if checks else 1.0
    return {"exam": "fmt_compliance",
            "score": round(1 - rate, 4),          # 遵从率
            "parse_failure_rate": rate,
            "threshold": TH_FMT,
            "pass": rate <= TH_FMT,
            "n": len(checks),
            "failures": fails}


# ---------- 体检第四项：身份级描述质量 ----------

def exam_desc_quality(base: _ProbeCounter,
                      images_dir: Path | None) -> dict:
    """perceive_image 对 ≥3 张黄金测试图的描述，按 §3.5 要素清单核对覆盖率。
    非视觉模型（返回 None）或图片缺失 → skipped，不算失败。"""
    manifest = _load_golden("images.json")
    entries = manifest["images"]
    if images_dir is None:
        return {"exam": "desc_quality", "status": "skipped",
                "reason": "no_images_dir", "pass": None, "score": None}
    detail = []
    unsupported = 0
    for item in entries:
        path = images_dir / item["image"]
        if not path.is_file():
            detail.append({"image": item["image"], "status": "missing"})
            continue
        b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        desc = base.perceive_image(b64)
        if desc is None:
            unsupported += 1
            detail.append({"image": item["image"], "status": "no_vision"})
            continue
        desc_n = _keynorm(desc)
        hits = {group: [k for k in keys if _keynorm(k) in desc_n]
                for group, keys in item["expect"].items()}
        covered = sum(1 for v in hits.values() if v)
        # 单图覆盖率（0~1），适配度评分用它，避免"组数"把总分顶出 0~1
        fraction = covered / len(item["expect"])
        detail.append({"image": item["image"], "status": "ok",
                       "desc": desc[:200], "covered_groups": covered,
                       "coverage": round(fraction, 4), "hits": hits})
    done = [d for d in detail if d.get("status") == "ok"]
    if not done:
        reason = "no_vision" if detail and unsupported == len(detail) \
            else "images_missing"
        return {"exam": "desc_quality", "status": "skipped",
                "reason": reason, "pass": None, "score": None,
                "detail": detail}
    # 达标线按"每张图 ≥ TH_DESC_ELEMENTS 组"；评分为平均覆盖率（0~1）
    score = _mean([d["coverage"] for d in done])
    return {"exam": "desc_quality",
            "status": "ok",
            "score": score,
            "threshold": TH_DESC_ELEMENTS,
            "pass": min(d["covered_groups"] for d in done) >= TH_DESC_ELEMENTS,
            "n": len(done),
            "detail": detail}


# ---------- 第五科：编码器分离度（探测点 #12 / W1） ----------

def _std(values: list) -> float | None:
    """样本标准差（n-1）；少于 2 个样本返回 None。"""
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(var)


def _pooled_std(a: list, b: list) -> float | None:
    """合并样本标准差（n-1 加权）；两库任一侧不足 2 个样本则用可得的一侧。"""
    if len(a) >= 2 and len(b) >= 2:
        ma, mb = sum(a) / len(a), sum(b) / len(b)
        ss = sum((v - ma) ** 2 for v in a) + sum((v - mb) ** 2 for v in b)
        return math.sqrt(ss / (len(a) + len(b) - 2))
    return _std(a) if len(a) >= 2 else _std(b)


def _cross_pairs(n_pairs: int, n: int, seed: int) -> list[tuple[int, int]]:
    """跨组配对：不同 pair 索引 × 任一侧描述，固定种子无放回抽样。

    返回的是“文本索引”对（0..2n-1，偶数=a，奇数=b），与 golden 对顺序对应。
    """
    combos = []
    for i in range(n_pairs):
        for j in range(i + 1, n_pairs):
            for x in (0, 1):
                for y in (0, 1):
                    combos.append((2 * i + x, 2 * j + y))
    if not combos:
        return []
    rng = random.Random(seed)
    return rng.sample(combos, min(n, len(combos)))


def separation_verdict(margin: float, same_mean: float, diff_mean: float,
                       pooled_std: float | None) -> str:
    """判定档（任务书 §2.2）：首跑不设及格线，只分三档定性。"""
    if margin > 0:
        return VERDICT_SEPARABLE
    if pooled_std is not None and (same_mean - diff_mean) > 2 * pooled_std:
        return VERDICT_USABLE
    return VERDICT_DEGRADED


def exam_embedder_separation(embedder, pairs: list[dict] | None = None,
                             seed: int = EMBEDDER_SEED,
                             n_cross: int = CROSS_SAMPLES) -> dict:
    """第五科：当前编码器分得开“同一人”与“不同人”吗？分到什么程度？

    数据源：golden/judge_pairs.json（同一人 5 对 + 不同人 5 对，不调 LLM）；
    同一人相似度只取 same=True 的配对；跨人相似度在不同组之间抽样（固定种子）。

    指标（任务书 §2.2）：same_min / diff_max / margin / overlap_ratio / 判定档。
    任何一步失败（无 embedder / 不可达 / 数据不足）→ skipped，绝不拖垮其他四科。
    """
    base = {"exam": "embedder_separation", "status": "skipped",
            "pass": None, "score": None, "calibrated": False}
    if embedder is None:
        return {**base, "reason": "no_embedder"}

    try:
        if pairs is None:
            pairs = _load_golden("judge_pairs.json")["pairs"]
    except Exception as exc:
        return {**base, "reason": "golden_load_failed", "error": str(exc)[:200]}

    texts: list[str] = []
    for pair in pairs:
        texts.extend([pair.get("a", ""), pair.get("b", "")])
    vectors: dict[str, list] = {}

    def vec(text: str):
        if text not in vectors:
            vectors[text] = embedder.embed(text)
        return vectors[text]

    try:
        for text in texts:
            vec(text)
    except Exception as exc:      # 本地编码器不可用（如纯云端场景）→ 跳过该科
        return {**base, "reason": "embedder_error", "error": str(exc)[:200]}

    def cosine(a, b) -> float:
        return round(float(embedder.cosine(a, b)), 4)

    same_dist = [cosine(vec(p["a"]), vec(p["b"]))
                 for p in pairs if p.get("same") is True]
    chosen = _cross_pairs(len(pairs), n_cross, seed)
    diff_dist = [cosine(vec(texts[i]), vec(texts[j])) for i, j in chosen]

    if not same_dist or not diff_dist:
        return {**base, "reason": "insufficient_pairs",
                "n_same": len(same_dist), "n_diff": len(diff_dist),
                "embed_calls": len(vectors)}

    same_min, diff_max = min(same_dist), max(diff_dist)
    margin = round(same_min - diff_max, 4)
    same_mean = round(sum(same_dist) / len(same_dist), 4)
    diff_mean = round(sum(diff_dist) / len(diff_dist), 4)
    pooled = _pooled_std(same_dist, diff_dist)

    # 交集范围占比：两区间交集长度 / 并集跨度；0 = 完美分离
    union_lo = min(min(same_dist), min(diff_dist))
    union_hi = max(max(same_dist), max(diff_dist))
    lo = max(min(same_dist), min(diff_dist))
    hi = min(max(same_dist), max(diff_dist))
    span = union_hi - union_lo
    # 零跨度 = 两组分布塔缩在同一点上，完全重叠；否则交集长度/并集跨度
    overlap_ratio = (round(max(0.0, hi - lo) / span, 4) if span > 0 else 1.0)

    verdict = separation_verdict(margin, same_mean, diff_mean, pooled)
    return {
        "exam": "embedder_separation",
        "status": "ok",
        "pass": None,                 # 首跑不设及格线（任务书 §2.2）
        "calibrated": False,
        "model": getattr(embedder, "model_tag", "") or "",
        "score": margin,              # 分离度分数 = margin（核心指标）
        "score_kind": "margin",
        "verdict": verdict,
        "n_same": len(same_dist),
        "n_diff": len(diff_dist),
        "same_min": round(same_min, 4),
        "diff_max": round(diff_max, 4),
        "margin": margin,
        "overlap_ratio": overlap_ratio,
        "same_mean": same_mean,
        "diff_mean": diff_mean,
        "same_std": round(_std(same_dist), 4) if _std(same_dist) is not None else None,
        "diff_std": round(_std(diff_dist), 4) if _std(diff_dist) is not None else None,
        "pooled_std": round(pooled, 4) if pooled is not None else None,
        "seed": seed,
        "same_dist": same_dist,
        "diff_dist": diff_dist,
        "cross_pairs": [[i, j] for i, j in chosen],
        "embed_calls": len(vectors),
    }


# ---------- 汇总 ----------

def _verdict(exams: dict[str, dict]) -> str:
    judge = exams["judge_accuracy"]
    fmt = exams["fmt_compliance"]
    fmt_rate = fmt.get("parse_failure_rate", 1.0)
    if not judge["pass"]:
        return VERDICT_COMMON
    if fmt_rate > 0.5:
        return VERDICT_NO
    for name in ("judge_accuracy", "conf_calibration", "fmt_compliance",
                 "desc_quality"):
        e = exams[name]
        if e.get("pass") is False:
            return VERDICT_TUNING
    return VERDICT_FIT


def _fitness_score(exams: dict[str, dict]) -> float | None:
    """认知层适配度评分（0~1）：judge 0.4 / calib 0.2 / fmt 0.2 / desc 0.2，
    skipped 项权重按剩余项归一。"""
    weights = {"judge_accuracy": 0.4, "conf_calibration": 0.2,
               "fmt_compliance": 0.2, "desc_quality": 0.2}
    total_w, acc = 0.0, 0.0
    for name, w in weights.items():
        e = exams[name]
        if e.get("pass") is None or e.get("score") is None:
            continue
        total_w += w
        acc += w * float(e["score"])
    return round(acc / total_w, 4) if total_w else None


def run_physical(base, *, images_dir: Path | None = None,
                 embedder=None) -> dict:
    """对 base 模型跑完四项体检 + 编码器分离度（第五科），返回报告字典。

    base：适配器实例（OllamaBaseModel / OpenAICompatBaseModel）；
    images_dir：黄金测试图目录，None = 跳过图像项（算 skipped）；
    embedder：本地编码器实例，None / 不可用 → 第五科 skipped（不中断其余四科）。
    落盘由调用方负责；第五科不计入 api_calls（不走 LLM，只耗 embedding）。
    """
    counter = _ProbeCounter(base)
    judge_pairs = _load_golden("judge_pairs.json")["pairs"]
    calib_items = _load_golden("calibration.json")["items"]

    exams: dict[str, dict] = {}
    exams["judge_accuracy"] = exam_judge_accuracy(counter, judge_pairs)
    exams["conf_calibration"] = exam_conf_calibration(counter, calib_items)

    # 复用前两项已产生的返回值做格式校验（judge 返回值 / 置信度 / 答案），
    # 自己的 6 个探针在 exam_fmt_compliance 里各调一次。
    prior = []
    for d in exams["judge_accuracy"]["detail"]:
        v = d["verdict"]
        prior.append(("judge_same", v, v is None or isinstance(v, bool)))
    for d in exams["conf_calibration"]["detail"]:
        prior.append(("infer_confidence", d["conf"], d["conf_ok"]))
        prior.append(("answer_question", d["answer"], d["answer_ok"]))
    exams["fmt_compliance"] = exam_fmt_compliance(counter, prior)

    try:
        exams["desc_quality"] = exam_desc_quality(counter, images_dir)
    except Exception as exc:                       # 图片 IO 等异常不拖垮体检
        exams["desc_quality"] = {"exam": "desc_quality",
                                 "status": "error", "pass": None,
                                 "score": None, "error": str(exc)}

    try:
        embedder_exam = exam_embedder_separation(embedder)
    except Exception as exc:                       # 第五科异常不得拖垮整体（铁律）
        embedder_exam = {"exam": "embedder_separation", "status": "error",
                         "pass": None, "score": None, "calibrated": False,
                         "error": str(exc)[:200]}

    verdict = _verdict(exams)
    return {
        "schema": "cognition-exoskeleton/model_profile@1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "exams": exams,
        "embedder": embedder_exam,
        "verdict": verdict,
        "verdict_label": VERDICT_LABELS[verdict],
        "fitness_score": _fitness_score(exams),
        "cloud_fallback_suggested": verdict != VERDICT_FIT,
        "api_calls": counter.calls,
        "api_calls_by_probe": counter.calls_by_probe,
        "note": "体检只给建议：任何阈值变更仍须 §6.5 标定并经主人拍板，"
                "体检 never 自动改参数。",
    }


def profile_path(home: Path, backend: str, model: str) -> Path:
    """报告落盘：<COGSOUL_HOME>/model_profile_<backend>_<model>.json。"""
    safe = model.replace("/", "_").replace(":", "_")
    return Path(home) / f"model_profile_{backend}_{safe}.json"


def latest_profile(home: Path) -> dict | None:
    """取 <home> 下最新的体检报告（供 stats 展示）。"""
    files = sorted(Path(home).glob("model_profile_*.json"),
                   key=lambda p: p.stat().st_mtime)
    if not files:
        return None
    try:
        return json.loads(files[-1].read_text(encoding="utf-8"))
    except Exception:
        return None
