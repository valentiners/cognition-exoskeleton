# Calibrated thresholds / 已标定参数

These are the values frozen in design v4.0 §9. Do not tune them casually.
以下数值来自设计文档 v4.0 §9，不要随意改动。

| Parameter | Value | Notes |
|---|---|---|
| `mem_th` | 0.78 | permanent-memory hit threshold; embeddinggemma-2 positives 0.87 / negatives 0.60 |
| `base_th` | 0.51 | base-model guess threshold |
| temp-memory hit threshold | 0.62 | looser band on the same embedder |
| temp TTL | 7 calendar days | rolling window, cutoff = today − 6 days |
| temp promotion | ≥3 distinct days within 7 | same-day repeats count once |
| dormancy | 365 days since last hit | derived at query time; never-hit entries count from creation |
| frequent window | any hit within 90 days | no count threshold |
| topic cooldown | 30 minutes | shared by topic expiry and arbitration degradation |
| arbitration pending TTL | 30 minutes | same as topic cooldown |
| default base model | `gemma4:latest` | vision + text |
| default embedder | `embeddinggemma-2:latest` | mxbai-embed-large was abandoned for Chinese separation failure |

Recalibrate per design §6.5 when the base model or embedder changes, and update
the design table before changing code.
