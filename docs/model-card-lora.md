---
language:
  - bn
  - en
license: apache-2.0
base_model: convaiinnovations/laya-multilingual
base_model_relation: finetune
library_name: laya
pipeline_tag: text-classification
tags:
  - laya
  - bangla
  - bengali
  - banglish
  - lora
  - typed-decisions
  - classification
  - routing
  - moderation
  - onnx
datasets:
  - mteb/amazon_massive_intent
  - mteb/sib200
  - aplycaebous/BanTH
  - aplycaebous/BanHate
  - TanjimKIT/Bangla_Sentiments_in_eLearning
---

# laya-bangla-lora

**A lighter fine-tune of the same Bangla decision model.** It was trained on the same data as
[laya-bangla](https://huggingface.co/algolyzer/laya-bangla), but with LoRA, which changes far
fewer weights. In a sample test it matches laya-bangla on the trained tasks and **keeps much more
of the base model's yes/no reasoning**, which laya-bangla loses.

বাংলা, বাংলিশ বা ইংরেজি লেখা পড়ে নির্দিষ্ট প্রশ্নের উত্তর দেয়। এটি laya-bangla-এর হালকা (LoRA) সংস্করণ।

[**Try it live**](https://huggingface.co/spaces/algolyzer/laya-bangla-playground) (choose
"laya-bangla-lora" in the model list) · [Main model](https://huggingface.co/algolyzer/laya-bangla) ·
[All versions](https://huggingface.co/collections/algolyzer/laya-bangla-6ad51a88c6ea46c7f5cf9180) ·
[Source code](https://github.com/shamspias/laya-bangla)

## Which one should I use?

| If you… | Use |
| --- | --- |
| Only ask the kinds of questions it was trained on (routing, complaints, moderation, sentiment, topic) | Either; [laya-bangla](https://huggingface.co/algolyzer/laya-bangla) has the full benchmark |
| Also ask **new yes/no questions** (does the text say X? is this a question?) | **laya-bangla-lora** |
| Need the most thoroughly tested model | [laya-bangla](https://huggingface.co/algolyzer/laya-bangla) |

## Quick start

It works exactly like laya-bangla; only the repository name changes.

```bash
pip install "laya==0.4.11"
```

```python
import laya

agent = laya.load("algolyzer/laya-bangla-lora", revision="v1")
questions = {
    "sector": {
        "type": "choice",
        "instructions": "নাগরিকের এই সমস্যাটি কোন সেবা খাতের?",
        "criteria": {"water": "পানি সরবরাহ", "electricity": "বিদ্যুৎ", "waste": "ময়লা ও পরিচ্ছন্নতা"},
    },
    "urgent": {
        "type": "noul",
        "instructions": "এটি কি এখনই সমাধান করা দরকার?",
        "criteria": {"false": "না", "true": "হ্যাঁ"},
    },
}
result = agent.predict("আমাদের এলাকায় তিন দিন ধরে পানি নেই", questions)
print(result["answers"]["sector"]["choice"], result["answers"]["urgent"]["noul"])
```

## Versions in this repository

| Version | Best for | Size | Path |
| --- | --- | ---: | --- |
| PyTorch | Python, servers | 644 MB | repository root, `revision="v1"` |
| ONNX fp32 | Node.js, servers without PyTorch | 1.3 GB | `onnx/fp32/` |
| ONNX int8 | Web browsers | 640 MB | `onnx/int8/` |

Against PyTorch on 1,284 test decisions, the ONNX files give the same answer for 100% (`fp32`)
and 99.3% (`int8`) of them, with the same accuracy. How to run each one (Python, HTTP API,
Docker, browser, Node.js, other languages) is the same as for the main model: see
[run it on any platform](https://huggingface.co/algolyzer/laya-bangla#run-it-on-any-platform)
and use the repository name `algolyzer/laya-bangla-lora`.

## How well it works

**Validation set** (2,597 rows, used to pick the epoch and calibrate):

| | Accuracy | NLL (lower is better) |
| --- | ---: | ---: |
| laya-bangla (full fine-tune) | 81.6% | 0.447 |
| laya-bangla-lora | 81.2% | 0.460 |

**Sample test** (the first 100 test items of every dataset, each asked with Bangla and English
question wording; all three models on the same CPU). Accuracy averaged over both wordings:

| Task | Base model | laya-bangla | laya-bangla-lora |
| --- | ---: | ---: | ---: |
| **Bangla, trained tasks (1,956 decisions)** | 48.3% | 80.3% | **81.0%** |
| Complaint service sector | 55.5% | 93.0% | **96.0%** |
| Voice-assistant routing, two steps | 25.0% | **80.5%** | 77.0% |
| Hate speech (Bangla script) | 9.0% | 73.0% | **79.5%** |
| Hate speech (Banglish) | 55.0% | 78.0% | **81.5%** |
| Sentiment of course comments | 47.0% | **83.0%** | 80.5% |
| **Bangla, never-trained datasets** | 41.2% | 54.5% | **55.0%** |
| Yes/no sentence logic (IndicXNLI) | **70.0%** | 53.0% | 63.5% |
| **English, six datasets (1,356 decisions)** | 63.3% | 74.3% | **76.5%** |
| Yes/no sentence logic (XNLI English) | **82.0%** | 65.0% | 81.0% |

With 100 items per dataset, differences of a few points are within noise. The clear finding
is the yes/no row: laya-bangla loses about 17 points there, laya-bangla-lora about 1 (English)
to 7 (Bangla). The full test runs for this model are listed in the project's
[next steps](https://github.com/shamspias/laya-bangla/blob/main/docs/next-steps.md).

## How it was trained

- **Start:** [`convaiinnovations/laya-multilingual`](https://huggingface.co/convaiinnovations/laya-multilingual)
  at commit `e4e9ddf21a7b1903b7acffd8814ad4307bf63a67`, the same as laya-bangla.
- **Data:** the same 38,766 training and 2,597 validation rows of six public Bangla datasets
  (MASSIVE, SIB-200, BanglaPSG, BanTH, BanHate, Bangla Sentiments in eLearning). See the
  [main model card](https://huggingface.co/algolyzer/laya-bangla#training-data) for sources,
  licenses, and cleaning.
- **Method:** LoRA, rank 16, alpha 32, dropout 0.05, on every attention and MLP projection of
  the encoder (`Wqkv`, `Wi`, `Wo`), plus the full decision head; token embeddings frozen.
  18.3M trainable parameters (laya-bangla: 125M). After training, the adapters were merged into
  the weights, so this is an ordinary Laya checkpoint.
- **Settings:** 3 epochs, learning rate 2e-4 (adapters) and 1e-4 (head), cosine decay, batch 64,
  label smoothing 0.05, choice options shuffled each epoch, seed 20261018. The epoch with the
  lowest validation loss (epoch 2) was kept, then one temperature per question type was fitted
  on validation data (choice 0.855, score 0.999, yes/no 0.960).
- **Hardware:** one NVIDIA GeForce RTX 4060 (8 GB), shared with other jobs; peak 2.4 GB.
- **Reproduce:** `laya-bangla train --config configs/train-lora.toml --output models/laya-bangla-lora`
  in the [GitHub repository](https://github.com/shamspias/laya-bangla).

## Limitations

The main model's limitations apply here too: rare abusive comments are under-detected, complaint
severity is weak, Banglish voice commands were not trained, and there was no real
customer-support data. This model has only the sample test above so far; check it on your own
questions before relying on it. Full list:
[laya-bangla limitations](https://huggingface.co/algolyzer/laya-bangla#limitations).

## Files

| File | Contents |
| --- | --- |
| `model.safetensors`, `encoder/`, `tokenizer/`, `rl_agent_config.json` | The model, with fitted temperatures and abstention thresholds |
| `tasks.json` | Every question used in training and testing |
| `train_report.json` | Data and config checksums, LoRA settings, validation history, calibration |
| `benchmark/sample-*/summary.json` | Metrics-only results of the sample test for all three models |
| `onnx/fp32/`, `onnx/int8/` | ONNX versions for laya-ts (Node.js and browsers) |
| `LICENSE`, `NOTICE` | Apache 2.0 license and attribution |

## License and credits

Apache 2.0, like the base model and the Laya library. Laya and `laya-multilingual` are by
[Convai Innovations](https://huggingface.co/convaiinnovations/laya). Training data is credited on
the [main model card](https://huggingface.co/algolyzer/laya-bangla#license-and-credits). No
dataset text is included in this repository.
