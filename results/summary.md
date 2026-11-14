# TrustTrace results

Model: Qwen/Qwen2.5-1.5B-Instruct | backend: hf | test diaries: 80 (1001 symptom events, 35 needing escalation)

| Method | Precision (95% CI) | Recall (95% CI) | F1 | False events (negation / distractor / unsupported) | Summary claim precision | Escalation cost (95% CI) | Missed escalations | Unneeded reviews | Parse failures |
|---|---|---|---|---|---|---|---|---|---|
| Keyword baseline | 73.3% (71.0%-75.6%) | 95.9% (94.4%-97.3%) | 0.831 | 350 (63 / 125 / 162) | 57.0% | 0.438 (0.338-0.550) | 0 | 35 | 0.0% |
| LLM only | 33.4% (29.0%-37.8%) | 23.4% (18.5%-28.5%) | 0.275 | 466 (12 / 11 / 443) | 49.0% | 1.887 (1.387-2.388) | 30 | 1 | 0.0% |
| Agent v1 (whole diary) | 68.1% (58.5%-78.3%) | 8.1% (5.9%-10.4%) | 0.145 | 38 (0 / 2 / 36) | 75.8% | 1.725 (1.199-2.250) | 27 | 3 | 3.8% |
| TrustTrace v2 before verification | 77.3% (74.2%-80.1%) | 82.2% (79.2%-85.1%) | 0.797 | 242 (62 / 21 / 159) | 60.3% | 0.400 (0.275-0.575) | 1 | 27 | 0.0% |
| TrustTrace v2 | 79.1% (76.4%-81.6%) | 82.2% (79.3%-85.0%) | 0.806 | 218 (31 / 21 / 166) | 62.4% | 0.375 (0.237-0.550) | 1 | 25 | 0.0% |
| TrustTrace v3 (hybrid) | 98.0% (96.9%-98.8%) | 96.0% (94.3%-97.4%) | 0.970 | 20 (1 / 17 / 2) | 96.4% | 0.087 (0.000-0.225) | 1 | 2 | 0.0% |

Escalation thresholds chosen on 40 validation diaries: agent v1 0.25, TrustTrace v2 0.5, TrustTrace v3 0.75.
TrustTrace v2 evidence checker on test: rejections {"negated": 37, "unknown symptom": 24}; entries retried with tool feedback: 22; symptom events recovered by retries: 12.
TrustTrace v3 judge thresholds chosen on validation: {"both": 0.05, "kw": 0.85, "llm": 0.95} (validation F1 0.961).
TrustTrace v3 judge on test candidates: AUC 0.978, ECE 0.040, 1570 candidates (973 true). Validation AUC 0.962.
TrustTrace v3 candidates on test by source: both: 825 candidates (778 true), 790 accepted (778 true); kw: 513 candidates (182 true), 176 accepted (170 true); llm: 232 candidates (13 true), 16 accepted (13 true).
LLM-written summaries replaced by the verified-facts template: v2 2 of 80 (reasons: {"unsupported number": 2}); v3 2 of 80 (reasons: {"unsupported number": 2}).
