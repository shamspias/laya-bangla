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
  - code-switching
  - typed-decisions
  - classification
  - routing
  - moderation
  - sentiment
datasets:
  - mteb/amazon_massive_intent
  - mteb/sib200
  - aplycaebous/BanTH
  - aplycaebous/BanHate
  - TanjimKIT/Bangla_Sentiments_in_eLearning
model-index:
  - name: laya-bangla
    results:
      - task:
          type: text-classification
          name: Voice-assistant intent routing (scenario, then intent)
        dataset:
          name: MASSIVE bn-BD
          type: mteb/amazon_massive_intent
          config: bn
          split: test
        metrics:
          - type: accuracy
            value: 0.809
            name: Accuracy (Bangla question)
      - task:
          type: text-classification
          name: Citizen complaint service sector
        dataset:
          name: BanglaPSG
          type: banglapsg
          split: test
        metrics:
          - type: accuracy
            value: 0.934
            name: Accuracy (Bangla question)
      - task:
          type: text-classification
          name: Hate speech in Banglish comments
        dataset:
          name: BanTH
          type: aplycaebous/BanTH
          split: test
        metrics:
          - type: accuracy
            value: 0.779
            name: Accuracy (Bangla question)
      - task:
          type: text-classification
          name: Hate speech in Bangla comments
        dataset:
          name: BanHate
          type: aplycaebous/BanHate
          split: test
        metrics:
          - type: accuracy
            value: 0.736
            name: Accuracy (Bangla question)
      - task:
          type: text-classification
          name: Sentiment of course comments
        dataset:
          name: Bangla Sentiments in eLearning
          type: TanjimKIT/Bangla_Sentiments_in_eLearning
          split: test
        metrics:
          - type: accuracy
            value: 0.824
            name: Accuracy (Bangla question)
      - task:
          type: text-classification
          name: Sentence topic
        dataset:
          name: SIB-200 ben_Beng
          type: mteb/sib200
          config: ben_Beng
          split: test
        metrics:
          - type: accuracy
            value: 0.765
            name: Accuracy (Bangla question)
      - task:
          type: text-classification
          name: Sentiment of social-media comments (never trained on)
        dataset:
          name: SentNoB
          type: khondoker/SentNoB
          split: test
        metrics:
          - type: accuracy
            value: 0.483
            name: Accuracy (Bangla question)
      - task:
          type: text-classification
          name: Sentence entailment, yes/no (never trained on)
        dataset:
          name: IndicXNLI bn
          type: mteb/IndicXnliPairClassification
          config: bn
          split: test
        metrics:
          - type: accuracy
            value: 0.499
            name: Accuracy (Bangla question)
---

# laya-bangla

**A model that makes fixed decisions about Bangla text: which option fits, yes or no, and how
much.** It reads Bangla script, Banglish (Bangla written in English letters), and Bangla-English
mixed text.

বাংলা, বাংলিশ বা মিশ্র ভাষার লেখা পড়ে নির্দিষ্ট প্রশ্নের উত্তর দেয়: কোন বিভাগে যাবে, লেখাটি কি
বিদ্বেষমূলক, অনুভূতি কেমন। প্রতিটি উত্তরের সাথে সম্ভাবনাও দেখায়।

[**Try it live**](https://huggingface.co/spaces/algolyzer/laya-bangla-playground) ·
[Source code](https://github.com/shamspias/laya-bangla) ·
[Full benchmark report](https://github.com/shamspias/laya-bangla/blob/main/docs/benchmark-results.md) ·
[Dataset guide](https://github.com/shamspias/laya-bangla/blob/main/docs/datasets.md)

## What it does

You give the model a piece of text and one or more **questions**, each with its own answer
options. It answers every question at once and tells you how likely each option is. It does not
write text; it only picks among the options you give it, so the output is always one of your
labels.

Good fits:

- **Routing:** send a message to the right team, or a voice command to the right action.
- **Moderation:** flag hate speech, personal attacks, or abusive comments.
- **Triage and feedback:** citizen-complaint sector, sentiment, severity.

laya-bangla is [`convaiinnovations/laya-multilingual`](https://huggingface.co/convaiinnovations/laya-multilingual)
fine-tuned on six published Bangla datasets. It keeps the same Laya question format and Python
API, so code written for Laya works unchanged.

## Quick start

```bash
pip install "laya==0.4.11"
```

```python
import laya

agent = laya.load("algolyzer/laya-bangla", revision="v2")

questions = {
    "sector": {
        "type": "choice",
        "instructions": "নাগরিকের এই সমস্যাটি কোন সেবা খাতের?",
        "criteria": {
            "water": "পানি সরবরাহ বা জলাবদ্ধতা",
            "electricity": "বিদ্যুৎ বা লোডশেডিং",
            "waste": "ময়লা, আবর্জনা বা পরিচ্ছন্নতা",
        },
    },
    "abusive": {
        "type": "noul",
        "instructions": "লেখাটি কি অশ্লীল গালি দেয় বা মারধরের কথা বলে?",
        "criteria": {"false": "না", "true": "হ্যাঁ"},
    },
}

result = agent.predict("আমাদের এলাকায় তিন দিন ধরে পানি নেই", questions)
print(result["answers"]["sector"]["choice"])         # water
print(result["answers"]["sector"]["probabilities"])  # {"water": 0.99, "electricity": 0.003, ...}
print(result["answers"]["abusive"]["noul"])          # 0.31: probability of "yes", so "no"
```

For many texts at once, use `agent.predict_batch(texts, questions, batch_size=32)`. The model
runs on CPU, an NVIDIA GPU, or Apple Silicon (MPS) and needs about 1.3 GB of memory.

**No code?** Open the [playground Space](https://huggingface.co/spaces/algolyzer/laya-bangla-playground).
The model runs **inside your browser** (WebGPU or WASM), so nothing you type is sent anywhere.
The first visit downloads about 640 MB, which the browser then caches.

## Choose a version

| Version | Best for | Size | Where to get it |
| --- | --- | ---: | --- |
| **laya-bangla v2** (recommended) | Python, servers; fully benchmarked | 644 MB | this repository, `revision="v2"` |
| v2 · ONNX fp32 | Node.js, servers without PyTorch | 1.3 GB | this repository, `onnx/fp32/` |
| v2 · ONNX int8 | Web browsers (WebGPU or WASM) | 640 MB | this repository, `onnx/int8/` |
| **laya-bangla-lora** | Also asking new yes/no questions: keeps more of the base model's reasoning (sample-tested) | 644 MB | [`algolyzer/laya-bangla-lora`](https://huggingface.co/algolyzer/laya-bangla-lora) |
| LoRA · ONNX fp32 / int8 | The same, in Node.js or browsers | 1.3 GB / 640 MB | `algolyzer/laya-bangla-lora`, `onnx/` |
| laya-bangla-llm (GGUF, safetensors) | Ollama, llama.cpp, LM Studio, vLLM | | Not trained yet ([why](#ollama-llamacpp-lm-studio-vllm)) |

All versions answer the same questions in the same format, and every ONNX file was checked
against PyTorch: the same answer for 99.6% (`fp32`) and 99.1% (`int8`) of 1,284 test
decisions, with the same accuracy. The [collection](https://huggingface.co/collections/algolyzer/laya-bangla-6ad51a88c6ea46c7f5cf9180)
lists every version, the demo, and the datasets.

## Run it on any platform

### Python

The [quick start](#quick-start) above: `pip install "laya==0.4.11"`, then
`laya.load("algolyzer/laya-bangla", revision="v2")`. CPU, NVIDIA GPU (CUDA), and Apple
Silicon (MPS) all work. Use `revision="v2"` so an update never changes your results.

### HTTP API (JavaScript, PHP, Java, Go, anything that can send a request)

Laya ships a production server. Register laya-bangla and start it:

```bash
pip install "laya[serve]==0.4.11"
LAYA_EXTRA_MODELS='{"bangla": "algolyzer/laya-bangla"}' LAYA_MODELS=bangla LAYA_DEFAULT_MODEL=bangla LAYA_REVISION=v2 LAYA_PORT=8000 laya-serve
```

```bash
curl -s http://localhost:8000/v1/systemone -H "Content-Type: application/json" -d '{
  "model": "bangla",
  "state": "আমাদের এলাকায় তিন দিন ধরে পানি নেই",
  "questions": {"sector": {"type": "choice", "instructions": "কোন সেবা খাতের সমস্যা?",
    "criteria": {"water": "পানি", "electricity": "বিদ্যুৎ", "waste": "ময়লা"}}}
}'
```

The reply holds the same `answers` as the Python API. `/v1/systemone/batch` answers many texts
at once, `/health` shows the loaded revision, and `LAYA_API_KEY=<secret>` requires
`Authorization: Bearer <secret>`.

### Docker

The same server in a container (this Dockerfile is a starting point; it was not built in
this project):

```dockerfile
FROM python:3.12-slim
RUN pip install --no-cache-dir "laya[serve]==0.4.11" --extra-index-url https://download.pytorch.org/whl/cpu
ENV LAYA_EXTRA_MODELS='{"bangla": "algolyzer/laya-bangla"}' LAYA_MODELS=bangla
ENV LAYA_DEFAULT_MODEL=bangla LAYA_REVISION=v2 LAYA_PORT=8000
EXPOSE 8000
CMD ["laya-serve"]
```

### Web browser, no server

The [playground](https://huggingface.co/spaces/algolyzer/laya-bangla-playground) runs entirely in
the browser with [laya-ts](https://github.com/NandhaKishorM/laya/tree/main/laya-ts), Laya's
TypeScript runtime, and ONNX Runtime Web. Copy its [`web/`](https://github.com/shamspias/laya-bangla/tree/main/web)
folder to any static host (GitHub Pages, Netlify, a Hugging Face static Space) and it loads the
model from this repository:

```js
import { Agent } from "./vendor/laya-ts/index.js"; // plus an import map for onnxruntime-web
const agent = await Agent.load("algolyzer/laya-bangla", { subfolder: "onnx/int8", revision: "7c8ad162a66c05409f9278e047e3ee80c24017a8" });
const result = await agent.predict("আমাদের এলাকায় তিন দিন ধরে পানি নেই", questions);
```

### Node.js

The same laya-ts code with `onnxruntime-node`, on the `onnx/fp32` or `onnx/int8` folder:

```bash
hf download algolyzer/laya-bangla --include "onnx/fp32/*" --local-dir laya-bangla
```

```js
const agent = await Agent.load("./laya-bangla/onnx/fp32", { device: "cpu" });
```

### Go, Rust, Java, .NET

Laya has ONNX Runtime ports for [Go](https://github.com/NandhaKishorM/laya/tree/main/laya-go),
[Rust](https://github.com/NandhaKishorM/laya/tree/main/laya-rust),
[Java](https://github.com/NandhaKishorM/laya/tree/main/laya-java), and
[.NET](https://github.com/NandhaKishorM/laya/tree/main/laya-dotnet). Point them at the `onnx/`
files (each port's README says which layout it reads). These ports were not tested with
laya-bangla.

### Ollama, llama.cpp, LM Studio, vLLM

These tools run chat language models. laya-bangla is not one: it is a text encoder with Laya's
decision head, which these tools cannot load, so a GGUF conversion would lose the part that
makes the decisions. Until the LLM version exists, use the **HTTP API** above from those apps.

A separate LLM version, **laya-bangla-llm** (Qwen3-1.7B fine-tuned with QLoRA to answer the same
questions as JSON), is fully prepared but not trained yet. Its training, benchmark, GGUF
export, and Ollama files are in the [GitHub repository](https://github.com/shamspias/laya-bangla/blob/main/docs/next-steps.md).
When it is published, it will run with:

```bash
ollama run hf.co/algolyzer/laya-bangla-llm-GGUF:Q4_K_M     # Ollama
llama-server -hf algolyzer/laya-bangla-llm-GGUF:Q4_K_M     # llama.cpp
vllm serve algolyzer/laya-bangla-llm                       # vLLM
```

Step-by-step versions of every option are in the
[deployment guide](https://github.com/shamspias/laya-bangla/blob/main/docs/deploy.md).

## Writing questions

| Type | The model returns | Example question |
| --- | --- | --- |
| `choice` | the best option, plus a probability for every option | Which department should handle this? |
| `noul` | the probability that the answer is **yes** (0 to 1) | Is this comment hate speech? |
| `score` | a level on an ordered scale, plus a probability per level | How urgent is it: low, medium, or high? |

- Write instructions and option descriptions in **Bangla or English**. Both were tested.
- `noul` questions use exactly two options, `"false"` and `"true"`. `score` options are a list
  from lowest to highest.
- Keep option descriptions short and clearly different from each other.

## Where it works best

Accuracy of laya-bangla on held-out test data, averaged over Bangla- and English-worded
questions (base model in brackets). "Not tested" means no public test data exists.

| Task | Bangla script | Banglish | English |
| --- | --- | --- | --- |
| Complaint service sector | **93%** (61%) | not tested | not tested |
| Voice-assistant routing, two steps | **81%** (18%) | often wrong (not trained on Banglish) | **77%** (40%) |
| Topic of a sentence | **77%** (57%) | not tested | **87%** (79%) |
| Hate speech | **74%** (58%) | **77%** (58%) | 63% (57%) |
| Personal attack | not tested | **81%** (57%) | not tested |
| Sentiment of course comments | **82%** (46%) | **83%** (36%) | not tested |
| Sentiment of reviews and posts | 49–63% (24–34%) | not tested | 65–74% (33–54%) |
| Yes/no sentence logic (entailment) | 51%, **worse** (64%) | not tested | 64%, **worse** (80%) |

- **Works well (75% and up):** routing requests and complaints, topics, Banglish moderation,
  course-feedback sentiment, in Bangla script and also in English.
- **Use with care (60–75%):** hate speech in Bangla script or English tweets, sentiment outside
  course comments. Check a sample first.
- **Do not use for:** general yes/no reasoning (use the base model), and Banglish voice
  commands.

## How well it works

All numbers are from **test data the model never saw**, asked with **question wording it never
saw** during training. Each question was asked twice, once in Bangla and once in English, so
each cell shows *accuracy with the Bangla question / with the English question*. The better
model on each row is in bold.

**Overall, accuracy goes from 53% to 82%** across the 42,576 test decisions of the six training
datasets.

### Tasks it was trained on

| Task | Dataset | Text | Test size | Base model | laya-bangla |
| --- | --- | --- | ---: | :---: | :---: |
| Voice-assistant routing, two steps | MASSIVE | Bangla | 2,939 | 18% / 18% | **81% / 80%** |
| Voice-assistant scenario (18 options) | MASSIVE | Bangla | 2,939 | 27% / 31% | **88% / 87%** |
| Complaint service sector (6) | BanglaPSG | Bangla | 487 | 64% / 57% | **93% / 93%** |
| Complaint severity (3 levels) | BanglaPSG | Bangla | 487 | 34% / 40% | **43% / 40%** |
| Hate speech | BanTH | Banglish | 3,729 | 49% / 66% | **78% / 77%** |
| Personal attack | BanTH | Banglish | 3,729 | 46% / 67% | **81% / 81%** |
| Abusive or violent language | BanTH | Banglish | 3,729 | 44% / 78% | **83% / 92%** |
| Hate speech | BanHate | Bangla | 2,720 | 61% / 55% | **74% / 74%** |
| Sentiment of course comments | eLearning | Bangla + Banglish | 602 | 34% / 47% | **82% / 83%** |
| Topic of a sentence (7) | SIB-200 | Bangla | 204 | 57% / 58% | **76% / 77%** |

### Datasets it never trained on

| Task | Dataset | Text | Test size | Base model | laya-bangla |
| --- | --- | --- | ---: | :---: | :---: |
| Sentiment of social-media comments | SentNoB | Bangla | 1,564 | 24% / 45% | **48% / 49%** |
| Sentiment of product reviews | IndicSentiment | Bangla | 996 | 1% / 47% | **59% / 66%** |
| Does sentence A imply B? (yes/no) | IndicXNLI | Bangla | 3,340 | **60% / 69%** | 50% / 53% |

### What else improved

- **Trustworthy probabilities.** On the trained tasks, the average gap between the reported
  probability and the real hit rate (expected calibration error) falls from 0.27 to 0.04. When
  the model says 80%, it is right about 8 times out of 10.
- **Same answer in either language.** The base model's accuracy could differ by up to 45 points
  depending on whether the question was in Bangla or English. For laya-bangla the largest gap is
  9 points, and most are under 4.
- **Banglish.** Hate-speech and abuse detection on romanized comments improves by 11 to 39 points.
- **Speed.** About 4 ms per decision in batches on an RTX 4060; 0.1 to 0.7 seconds per
  message on 2 CPU threads.

### English text

The model was trained only on Bangla and Banglish, yet it also got better on English text. On six
English test sets (13,476 items; four are the English originals of the Bangla tests above), pooled
accuracy rises from 61% to 73%, run on CPU:

| Task | Dataset | Test size | Base model | laya-bangla |
| --- | --- | ---: | :---: | :---: |
| Voice-assistant routing, two steps | MASSIVE `en` | 2,970 | 31% / 49% | **75% / 79%** |
| Voice-assistant scenario (18 options) | MASSIVE `en` | 2,970 | 45% / 64% | **82% / 85%** |
| Topic of a sentence (7) | SIB-200 `eng_Latn` | 204 | 77% / 81% | **86% / 88%** |
| Sentiment of product reviews | IndicSentiment, English originals | 996 | 9% / 58% | **74% / 74%** |
| Sentiment of tweets | TweetEval | 3,000 | 52% / 55% | **65% / 65%** |
| Hate speech in tweets | TweetEval | 2,966 | 53% / 61% | **63% / 62%** |
| Does sentence A imply B? (yes/no) | XNLI `en` | 3,340 | **72% / 88%** | 51% / 76% |

Macro-F1, calibration error, ordinal error, and results split by script (Bangla, Banglish,
mixed) are in the [full benchmark report](https://github.com/shamspias/laya-bangla/blob/main/docs/benchmark-results.md).
The metrics-only summaries are in this repository under `benchmark/` (English: `benchmark/english/`).

## Limitations

- **New kinds of yes/no questions can get worse.** All trained yes/no tasks are about
  moderation, so the model learned to answer "no" readily. On sentence entailment (IndicXNLI),
  which it never trained on, it falls below the base model to about chance. Check a sample of
  answers before relying on a new `noul` question, or compare with the base model. The same
  drop shows on English XNLI (80% to 64% averaged over both question languages).
- **Rare cases are under-detected.** Only about 8% of BanTH comments are abusive. Accuracy looks
  high (83–92%), but the model finds few of them: macro-F1 is only 0.51–0.59.
- **Complaint severity is weak** (40–43% over three levels). The source labels themselves may be
  hard to agree on.
- **Sentiment outside course comments** transfers only partly (48–66%).
- **Voice-assistant routing was trained on Bangla script only** (MASSIVE has no Banglish).
  Banglish commands such as "gaan chalao" are often routed to the wrong scenario.
- **No real customer-support data** was available. Support routing, threats to leave, and
  urgency were only checked on 36 synthetic messages, where threat-to-leave detection got
  slightly worse than the base model.
- **Mixed Bangla-English text is rare** in the training data (405 of 38,766 rows).
- One training run (seed 20261018); no confidence intervals. Small test sets (SIB-200 with 204
  sentences, mixed-script text) can move by several points by chance.
- Like its training data, the model can reflect social biases in what is labeled hateful or
  abusive. Do not use it as the only signal for decisions about people.

## Training data

Six published datasets, all real human-written Bangla text. Nothing is machine-generated.

| Dataset | What the text is | What the model learned | Training rows | License |
| --- | --- | --- | ---: | --- |
| [MASSIVE](https://huggingface.co/datasets/mteb/amazon_massive_intent) `bn-BD` (Amazon) | Voice-assistant requests | Scenario (18), intent within the scenario | 10,954 | CC BY 4.0 |
| [SIB-200](https://huggingface.co/datasets/mteb/sib200) `ben_Beng` | News-style sentences (FLORES) | Topic (7) | 701 | CC BY-SA 4.0 |
| [BanglaPSG](https://data.mendeley.com/datasets/kcr8zfmtvb) (DOI 10.17632/kcr8zfmtvb.1) | Citizen complaints from social media | Service sector (6), severity (3 levels) | 4,156 | CC BY 4.0 |
| [BanTH](https://huggingface.co/datasets/aplycaebous/BanTH) | YouTube comments in **Banglish** | Hate, personal attack, abusive | 9,000 | MIT |
| [BanHate](https://huggingface.co/datasets/aplycaebous/BanHate) | YouTube comments in Bangla script | Hate | 9,000 | MIT |
| [Bangla Sentiments in eLearning](https://huggingface.co/datasets/TanjimKIT/Bangla_Sentiments_in_eLearning) | Course comments, Bangla and romanized | Sentiment (3 levels) | 4,955 | Apache-2.0 |

That is 38,766 training rows with 70,884 question-answer pairs, plus 2,597 validation rows used
only to pick the best epoch and calibrate probabilities.

Three more datasets are used **only for testing**:
[SentNoB](https://huggingface.co/datasets/khondoker/SentNoB) (CC BY-ND 4.0, so it is never
trained on), [IndicSentiment](https://huggingface.co/datasets/mteb/IndicSentiment) (CC0), and
[IndicXNLI](https://huggingface.co/datasets/mteb/IndicXnliPairClassification) (CC BY 4.0).

How the data was cleaned:

1. Every dataset is pinned to an exact version (a Hub commit or a file checksum).
2. Text is Unicode-normalized; Bangla joiner characters are kept.
3. Duplicates are merged, and duplicates with **conflicting labels are dropped**.
4. Related texts stay in the same split: comments on the same video, or a Bangla comment and its
   romanized copy.
5. Any training text that also appears in a validation or test split, in any dataset, is removed
   (298 found).

The [dataset guide](https://github.com/shamspias/laya-bangla/blob/main/docs/datasets.md) has the
exact versions and counts. No dataset text is included in this repository.

## How it was trained

- **Starting point:** `convaiinnovations/laya-multilingual` at commit
  `e4e9ddf21a7b1903b7acffd8814ad4307bf63a67` (mmBERT-base encoder with Laya's decision head).
- **Method:** full fine-tuning of the transformer layers and decision head (125M parameters).
  The token embedding (two thirds of the model) is frozen, which keeps every language the
  tokenizer knows intact.
- **Questions:** each training row is asked with one of three wordings (two Bangla, one
  English). Test questions use a fourth wording in each language that training never sees.
  Answer options are shuffled each epoch so the model cannot learn option positions.
- **Settings:** 2 epochs, AdamW, learning rate 3e-5 (encoder) and 1e-4 (head) with cosine decay,
  batch 64, label smoothing 0.05, 512-token inputs, mixed precision, seed 20261018.
- **Model choice and calibration:** the epoch with the lowest validation loss is kept (epoch 2),
  then one temperature per question type is fitted on validation data (choice 0.910, score
  1.148, yes/no 1.178). Test data plays no part in either step.
- **Hardware:** one NVIDIA GeForce RTX 4060 (8 GB): 53 minutes, peak 4.6 GB.
- **Software:** Laya 0.4.11 (`laya.train`), PyTorch 2.8.0, Transformers 4.57.6.

An earlier 4-epoch run (v1) scored slightly higher on the test sets, but it had a worse
validation loss. Releasing it would mean choosing a model by its test score, so v2 was released
by the rule fixed in advance.

### Reproduce

```bash
git clone https://github.com/shamspias/laya-bangla.git && cd laya-bangla
uv sync --locked --all-extras
uv run --frozen laya-bangla prepare                                   # download and split the data
uv run --frozen laya-bangla train --output models/laya-bangla-v2      # about 1 hour on an 8 GB GPU
uv run --frozen laya-bangla benchmark --model base --output outputs/bench-base
uv run --frozen laya-bangla benchmark --model models/laya-bangla-v2 --output outputs/bench-v2
uv run --frozen laya-bangla compare outputs/bench-base/summary.json outputs/bench-v2/summary.json
```

GPU training is not bit-for-bit deterministic, so a rerun can differ slightly.

## Files

| File | Contents |
| --- | --- |
| `model.safetensors` | Weights (float16) |
| `rl_agent_config.json` | Model settings, fitted temperatures, and abstention thresholds |
| `encoder/`, `tokenizer/` | mmBERT-base encoder configuration and tokenizer |
| `tasks.json` | Every question used in training and testing, in Bangla and English |
| `train_report.json` | Data and config checksums, validation history, calibration report |
| `benchmark/base/summary.json`, `benchmark/laya-bangla-v2/summary.json` | Metrics-only test results for both models |
| `benchmark/english/base/summary.json`, `benchmark/english/laya-bangla-v2/summary.json` | The same for the English test sets |
| `onnx/fp32/`, `onnx/int8/` | ONNX exports for laya-ts (Node.js and browsers), checked against PyTorch |
| `LICENSE`, `NOTICE` | Apache 2.0 license and attribution |

## License and credits

Released under the **Apache 2.0** license, the same as the base model and the Laya library.

- Laya and `laya-multilingual` are by [Convai Innovations](https://huggingface.co/convaiinnovations/laya)
  ([source](https://github.com/NandhaKishorM/laya)). The encoder is
  [mmBERT-base](https://huggingface.co/jhu-clsp/mmBERT-base) by JHU CLSP.
- Training data, under its own licenses: MASSIVE (FitzGerald et al., 2022), SIB-200 (Adelani
  et al., 2024), BanglaPSG (Mendeley Data), BanTH (NAACL 2025), BanHate, and Bangla Sentiments in
  eLearning (Rahman et al., IEEE Access 2024). Test-only: SentNoB (Islam et al., 2021),
  IndicSentiment (Doddapaneni et al., 2023), and IndicXNLI (Aggarwal et al., 2022).

laya-bangla is an independent project and is not endorsed by Convai Innovations or the dataset
authors. Bug reports and Bangla failure cases are welcome as
[GitHub issues](https://github.com/shamspias/laya-bangla/issues).
