# Benchmark results: laya-bangla v2

**Run:** 2026-10-18 on an NVIDIA GeForce RTX 4060 (8 GB), Windows 11, Python
3.12.12, Laya 0.4.11, PyTorch 2.8.0+cu128. Both checkpoints were scored on the
same device with the same test file (`test.jsonl`, SHA-256 `785f05dd…`).

| Checkpoint | Description |
| --- | --- |
| Base | `convaiinnovations/laya-multilingual` @ `e4e9ddf21a7b1903b7acffd8814ad4307bf63a67`, as shipped (no fitted temperatures) |
| laya-bangla v2 | The base fine-tuned for 2 epochs with `configs/train.toml`, calibrated on the validation split ([method](fine-tuning.md)); published as [`algolyzer/laya-bangla`](https://huggingface.co/algolyzer/laya-bangla) tag `v2` |

Every number below is measured on held-out test data that played no part in
training, model selection, or calibration. Every question uses wording that
never appears in training: once in Bangla (`bn`) and once in English (`en`).
Each `n` counts states; the same states are asked in both languages. See
[datasets](datasets.md) for sources and the protocol.

## Summary

| Pooled over… | Decisions | Base accuracy | v2 accuracy |
| --- | ---: | ---: | ---: |
| Test splits of the six training sources | 42,576 | 53.0% | **82.0%** |
| Benchmark-only sources (SentNoB, IndicSentiment, IndicXNLI) | 11,800 | 49.6% | 52.6% |
| All of the above | 54,376 | 52.3% | **75.6%** |

- **Routing** improves most: MASSIVE end-to-end intent accuracy (scenario
  predicted, then intent) rises from **18% to 81%**; grievance routing from
  **57–65% to 93%**.
- **Banglish** (romanized) hate-speech detection improves by 11–39 points.
- **Calibration:** mean ECE over the in-domain rows falls from **0.273 to
  0.037**, so a reported probability of 0.8 now means roughly 80% correct.
- **Robustness to wording:** the base model's accuracy differs by up to 45
  points between Bangla and English questions on the same states (34 on BanTH
  `abusive`, 45 on IndicSentiment). v2's largest gap on a real benchmark is 9
  points (BanTH `abusive`), and most are under 4.
- **Regression:** yes/no questions about tasks never trained on get worse:
  IndicXNLI entailment drops to chance. See [limitations](#limitations).

## Results by benchmark

| Benchmark | Wording | n | Base acc | v2 acc | Δ acc | Base macro-F1 | v2 macro-F1 | Base ECE | v2 ECE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| banglapsg/grievance_category | bn | 487 | 0.645 | 0.934 | +0.290 | 0.656 | 0.935 | 0.122 | 0.021 |
| banglapsg/grievance_category | en | 487 | 0.569 | 0.930 | +0.361 | 0.576 | 0.927 | 0.156 | 0.015 |
| banglapsg/grievance_severity | bn | 487 | 0.339 | 0.433 | +0.094 | 0.245 | 0.389 | 0.302 | 0.031 |
| banglapsg/grievance_severity | en | 487 | 0.398 | 0.398 | +0.000 | 0.217 | 0.327 | 0.456 | 0.038 |
| banhate/hate | bn | 2720 | 0.610 | 0.736 | +0.126 | 0.576 | 0.734 | 0.202 | 0.061 |
| banhate/hate | en | 2720 | 0.555 | 0.740 | +0.185 | 0.469 | 0.737 | 0.316 | 0.091 |
| banth/abusive | bn | 3729 | 0.438 | 0.826 | +0.388 | 0.371 | 0.594 | 0.424 | 0.023 |
| banth/abusive | en | 3729 | 0.779 | 0.920 | +0.141 | 0.484 | 0.513 | 0.118 | 0.041 |
| banth/hate | bn | 3729 | 0.490 | 0.779 | +0.289 | 0.471 | 0.682 | 0.367 | 0.044 |
| banth/hate | en | 3729 | 0.663 | 0.769 | +0.106 | 0.489 | 0.703 | 0.219 | 0.047 |
| banth/personal_attack | bn | 3729 | 0.461 | 0.805 | +0.344 | 0.424 | 0.671 | 0.414 | 0.012 |
| banth/personal_attack | en | 3729 | 0.674 | 0.814 | +0.140 | 0.501 | 0.581 | 0.218 | 0.015 |
| elearning/sentiment | bn | 602 | 0.341 | 0.824 | +0.483 | 0.213 | 0.822 | 0.333 | 0.006 |
| elearning/sentiment | en | 602 | 0.473 | 0.826 | +0.352 | 0.476 | 0.824 | 0.169 | 0.027 |
| massive/intent (end-to-end) | bn | 2939 | 0.183 | 0.809 | +0.626 | — | — | — | — |
| massive/intent (end-to-end) | en | 2939 | 0.177 | 0.804 | +0.627 | — | — | — | — |
| massive/massive_intent (gold scenario) | bn | 2662 | 0.577 | 0.905 | +0.329 | 0.480 | 0.841 | 0.152 | 0.025 |
| massive/massive_intent (gold scenario) | en | 2662 | 0.530 | 0.904 | +0.375 | 0.429 | 0.837 | 0.167 | 0.014 |
| massive/massive_scenario | bn | 2939 | 0.274 | 0.875 | +0.601 | 0.284 | 0.866 | 0.398 | 0.027 |
| massive/massive_scenario | en | 2939 | 0.307 | 0.868 | +0.562 | 0.303 | 0.856 | 0.356 | 0.028 |
| sib200/sib_topic | bn | 204 | 0.574 | 0.765 | +0.191 | 0.538 | 0.757 | 0.283 | 0.076 |
| sib200/sib_topic | en | 204 | 0.578 | 0.775 | +0.196 | 0.562 | 0.762 | 0.285 | 0.094 |
| **Benchmark-only (never trained on)** | | | | | | | | | |
| sentnob/sentiment | bn | 1564 | 0.238 | 0.483 | +0.246 | 0.145 | 0.477 | 0.411 | 0.248 |
| sentnob/sentiment | en | 1564 | 0.446 | 0.494 | +0.048 | 0.400 | 0.477 | 0.226 | 0.261 |
| indicsentiment/sentiment | bn | 996 | 0.012 | 0.594 | +0.582 | 0.016 | 0.480 | 0.684 | 0.121 |
| indicsentiment/sentiment | en | 996 | 0.466 | 0.663 | +0.197 | 0.375 | 0.509 | 0.237 | 0.075 |
| indicxnli/entailment | bn | 3340 | 0.596 | 0.499 | −0.097 | 0.595 | 0.400 | 0.190 | 0.256 |
| indicxnli/entailment | en | 3340 | 0.694 | 0.529 | −0.165 | 0.687 | 0.419 | 0.134 | 0.209 |
| **Synthetic smoke set (diagnostic)** | | | | | | | | | |
| smoke/department | bn | 36 | 0.694 | 0.806 | +0.111 | 0.653 | 0.810 | 0.144 | 0.082 |
| smoke/department | en | 36 | 0.750 | 0.833 | +0.083 | 0.741 | 0.833 | 0.114 | 0.078 |
| smoke/churn_risk | bn | 36 | 0.500 | 0.417 | −0.083 | 0.333 | 0.294 | 0.481 | 0.358 |
| smoke/churn_risk | en | 36 | 0.556 | 0.500 | −0.056 | 0.518 | 0.333 | 0.422 | 0.361 |
| smoke/urgency | bn | 36 | 0.417 | 0.500 | +0.083 | 0.352 | 0.493 | 0.173 | 0.155 |
| smoke/urgency | en | 36 | 0.361 | 0.667 | +0.306 | 0.221 | 0.642 | 0.319 | 0.233 |

Ordinal mean absolute error (0–2 scale) falls for every score task: grievance
severity 0.77/0.68 → 0.63/0.62 (bn/en), eLearning sentiment 0.69/0.59 →
0.27/0.27, SentNoB 0.77/0.74 → 0.65/0.65.

The 36-message smoke set has 12 independent scenario groups; its differences
are illustrations, not evidence.

## By script

Accuracy pooled over the six training sources' test splits:

| Script | Wording | Decisions | Base | v2 |
| --- | --- | ---: | ---: | ---: |
| Bangla | bn | 9,636 | 48.0% | 82.2% |
| Bangla | en | 9,636 | 46.6% | 81.8% |
| Banglish | bn | 11,529 | 45.9% | 80.4% |
| Banglish | en | 11,529 | 69.7% | 83.4% |
| Mixed | bn | 123 | 51.2% | 87.0% |
| Mixed | en | 123 | 52.8% | 83.7% |

Mixed-script results rest on 123 decisions and are imprecise.

## English text

laya-bangla was trained only on Bangla and Banglish. To see what fine-tuning did to its
English, both checkpoints were scored on six English test sets built by
`configs/data-english.toml` ([datasets](datasets.md#english-benchmark)): 13,476 items, each
asked with the Bangla and the English test wording. These runs used the CPU
(`--device cpu`, test file SHA-256 `6249bd3c…`), so their decimals are not directly comparable
with the GPU runs above.

| Benchmark | Wording | n | Base acc | v2 acc | Δ acc | Base macro-F1 | v2 macro-F1 | Base ECE | v2 ECE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| indicsentiment_en/sentiment | bn | 996 | 0.089 | 0.737 | +0.648 | 0.108 | 0.558 | 0.597 | 0.076 |
| indicsentiment_en/sentiment | en | 996 | 0.576 | 0.743 | +0.167 | 0.449 | 0.559 | 0.219 | 0.071 |
| massive/intent (end-to-end) | bn | 2970 | 0.313 | 0.752 | +0.438 | — | — | — | — |
| massive/intent (end-to-end) | en | 2970 | 0.485 | 0.787 | +0.301 | — | — | — | — |
| massive_en/massive_intent | bn | 2690 | 0.636 | 0.894 | +0.258 | 0.562 | 0.817 | 0.162 | 0.015 |
| massive_en/massive_intent | en | 2690 | 0.717 | 0.907 | +0.190 | 0.643 | 0.839 | 0.077 | 0.014 |
| massive_en/massive_scenario | bn | 2970 | 0.452 | 0.824 | +0.372 | 0.464 | 0.811 | 0.294 | 0.032 |
| massive_en/massive_scenario | en | 2970 | 0.639 | 0.847 | +0.208 | 0.638 | 0.843 | 0.111 | 0.024 |
| sib200_en/sib_topic | bn | 204 | 0.770 | 0.858 | +0.088 | 0.741 | 0.845 | 0.140 | 0.014 |
| sib200_en/sib_topic | en | 204 | 0.809 | 0.877 | +0.069 | 0.779 | 0.871 | 0.106 | 0.037 |
| tweeteval_hate/hate | bn | 2966 | 0.528 | 0.629 | +0.102 | 0.507 | 0.622 | 0.285 | 0.102 |
| tweeteval_hate/hate | en | 2966 | 0.605 | 0.624 | +0.019 | 0.605 | 0.624 | 0.175 | 0.160 |
| tweeteval_sentiment/sentiment | bn | 3000 | 0.524 | 0.646 | +0.122 | 0.345 | 0.609 | 0.144 | 0.093 |
| tweeteval_sentiment/sentiment | en | 3000 | 0.554 | 0.648 | +0.094 | 0.508 | 0.608 | 0.151 | 0.134 |
| xnli_en/entailment | bn | 3340 | 0.725 | 0.510 | −0.215 | 0.723 | 0.363 | 0.122 | 0.283 |
| xnli_en/entailment | en | 3340 | 0.880 | 0.764 | −0.115 | 0.879 | 0.751 | 0.033 | 0.050 |

- **English improved, not just Bangla.** Pooled over all English decisions, accuracy rises
  from 61.3% to 72.7% (56.4% to 75.0% without XNLI). Routing (MASSIVE end-to-end 31–49% to
  75–79%) and sentiment (IndicSentiment English 9–58% to 74%) gain most, and calibration
  improves on every task except entailment.
- **The yes/no regression is not about language.** XNLI entailment drops from 0.725/0.880 to
  0.510/0.764, mirroring IndicXNLI in Bangla: the moderation-only `noul` training makes the
  model answer "no" readily for any new yes/no question.
- **English hate speech gains little** (0.53/0.61 to 0.63/0.62). The trained hate data is
  South Asian YouTube comments, not English tweets.

## LoRA version (sample)

[laya-bangla-lora](https://huggingface.co/algolyzer/laya-bangla-lora) was trained on the same data
with `configs/train-lora.toml` (18.3M trainable parameters instead of 125M). Its full benchmark is
pending ([next steps](next-steps.md)); this sample scores base, v2, and LoRA on the same CPU with
`--limit 100`, the first 100 test items of each source and wording. Accuracy averaged over both
question languages:

| Pooled over… | Decisions | Base | v2 | LoRA |
| --- | ---: | ---: | ---: | ---: |
| Bangla, trained sources | 1,956 | 48.3% | 80.3% | **81.0%** |
| Bangla, benchmark-only sources | 600 | 41.2% | 54.5% | **55.0%** |
| English, all six sources | 1,356 | 63.3% | 74.3% | **76.5%** |
| IndicXNLI entailment (Bangla) | 200 | **70.0%** | 53.0% | 63.5% |
| XNLI entailment (English) | 200 | **82.0%** | 65.0% | 81.0% |

LoRA matches v2 on the trained tasks and keeps most of the base model's yes/no reasoning, which
v2 loses (limitation 1). On validation it is slightly behind (NLL 0.460 vs 0.447), so v2 remains
the main release by the rule fixed before testing; LoRA is published as the alternative for
users who also ask new yes/no questions. With 100 items per source, differences of a few points
are noise.

## Limitations

1. **New yes/no questions degrade.** Every trained `noul` task is a
   moderation question, and the model learned to answer "no" readily. On
   IndicXNLI, which it never trained on, it answers "no" to 3,124 of 3,340 pairs
   with English wording; accuracy falls from 0.69 to 0.53. The synthetic churn
   question also slips. For yes/no questions unlike the trained tasks, compare
   against the base model or add training data for that question.
2. **Rare positives are under-detected.** BanTH `abusive` with English wording
   reaches 0.920 accuracy but finds only 11 of 304 abusive comments (macro-F1
   0.513). Accuracy is misleading for imbalanced yes/no tasks; read macro-F1.
3. **Grievance severity stays weak** (0.40–0.43 accuracy, three levels). Even on
   validation, score questions reach only 0.62 accuracy; the source labels may
   be inherently ambiguous.
4. **Cross-domain sentiment transfer is partial.** Training sentiment comes only
   from eLearning comments; SentNoB and IndicSentiment reach 0.48–0.66.
5. **No real customer-support data exists** for the department/churn/urgency
   workflow. Its smoke-set gains for department and urgency are indirect.
6. **The same yes/no drop appears in English** (XNLI, see [English text](#english-text)).
7. **One seed, no confidence intervals.** Differences of a few points on small
   rows (SIB-200, smoke set, mixed script) are within noise.

## Earlier run (v1) and model selection

A first run trained for 4 epochs on content-identical data. Its validation NLL
was lowest after epoch 2 (0.451) and rose afterwards, which set `epochs = 2`
for v2. v2 was chosen by the rule fixed before testing: lower validation NLL
(0.447 vs 0.451; validation accuracy 81.6% vs 82.1%).

v1 scores higher on test: 83.5% in-domain and 56.8% benchmark-only, versus
v2's 82.0% and 52.6%, mostly because its IndicXNLI drop is smaller (0.596 →
0.528 bn, 0.694 → 0.664 en). Picking v1 for that reason would be selection on
the test set, so it is reported, not released. This also shows how sensitive
general yes/no behavior is to the training run (limitation 1).

## Throughput

Batched inference over the full benchmark (60,470 decisions, batch size 64)
took 236 s for v2 and 248 s for the base, about 4 ms per decision on the RTX
4060. Fine-tuning v2 took 53 minutes at a peak of 4.6 GB of GPU memory.

## Reproduce

```bash
uv run --frozen laya-bangla prepare
uv run --frozen laya-bangla train --output models/laya-bangla-v2
uv run --frozen laya-bangla benchmark --model base --output outputs/bench-base
uv run --frozen laya-bangla benchmark --model models/laya-bangla-v2 --output outputs/bench-v2
uv run --frozen laya-bangla compare outputs/bench-base/summary.json outputs/bench-v2/summary.json
```

`train_report.json` in the checkpoint records the data and config hashes,
validation history, selected epoch, fitted temperatures, and abstention
thresholds. GPU kernels are not bit-for-bit deterministic, so a rerun can
differ slightly.
