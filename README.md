# laya-bangla

**A Bangla-focused adaptation of [Laya](https://huggingface.co/convaiinnovations/laya) for typed decisions, classification, and routing.**

বাংলা ভাষার জন্য Laya-কে আরো সুন্দর এবং সঠিক করার ছোট উদ্যোগ, যেন ক্লাসিফিকেশনে বাংলা ভাষা পিছিয়ে না থাকে।

[**Model on Hugging Face**](https://huggingface.co/algolyzer/laya-bangla) ·
[**Live demo**](https://huggingface.co/spaces/algolyzer/laya-bangla-playground) ·
[All versions](https://huggingface.co/collections/algolyzer/laya-bangla-6ad51a88c6ea46c7f5cf9180) ·
[Run it anywhere](docs/deploy.md) · [Benchmark results](docs/benchmark-results.md) ·
[Datasets](docs/datasets.md) · [Fine-tuning guide](docs/fine-tuning.md) ·
[Next steps](docs/next-steps.md) · [Contributing](CONTRIBUTING.md) · [Apache 2.0 license](LICENSE)

## What is this?

laya-bangla is a model that makes **fixed decisions about Bangla text**. You give it a message
and a few questions with answer options; it picks an answer for each question and tells you how
likely every option is. It reads Bangla script, **Banglish** (Bangla written in English letters),
and Bangla-English mixed text.

Typical uses:

- **Routing:** send a message to the right team, or a voice command to the right action.
- **Moderation:** flag hate speech, personal attacks, or abusive comments.
- **Triage and feedback:** complaint sector, sentiment, severity.

It is [`convaiinnovations/laya-multilingual`](https://huggingface.co/convaiinnovations/laya-multilingual)
fine-tuned on six published Bangla datasets. Laya is a non-autoregressive decision model: it
never writes free text, so every answer is one of your options. It supports three question
types: `choice` (pick one option), `noul` (probability of yes), and `score` (a level on a scale).

## Use the model

The released checkpoint is [`algolyzer/laya-bangla`](https://huggingface.co/algolyzer/laya-bangla),
tag `v2` (commit `4c5eb48932827f83cce9514c1af5510864089c0f`).

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
print(result["answers"]["sector"]["choice"])  # water (probability 0.99)
print(result["answers"]["abusive"]["noul"])   # 0.31: probability of "yes", so "no"
```

Questions and options can be written in Bangla or English. The
[model card](https://huggingface.co/algolyzer/laya-bangla) explains how to write questions,
what the model is good and bad at, and every dataset it learned from.

### Try it without code

The [live demo](https://huggingface.co/spaces/algolyzer/laya-bangla-playground) runs the model
**inside your browser**, so nothing you type leaves the page and the demo costs nothing to host.
It answers as you type, has ready-made tasks (citizen complaints, voice-assistant routing,
moderation, sentiment, topic, support routing), a box for your own questions, and the benchmark
results. The first visit downloads about 640 MB once. Its source is in [`web/`](web/).

### Other ways to run it

| Where | How | Guide |
| --- | --- | --- |
| Python | `laya.load("algolyzer/laya-bangla", revision="v2")` | above |
| HTTP API, any language | Laya's `laya-serve` server | [deploy §2](docs/deploy.md#2-http-api-any-language) |
| Your own web page, no server | ONNX files + laya-ts | [deploy §3](docs/deploy.md#3-in-the-browser) |
| Node.js | ONNX files + laya-ts | [deploy §4](docs/deploy.md#4-nodejs) |
| Ollama, llama.cpp, vLLM | LLM version, code ready but not trained yet | [deploy §7](docs/deploy.md#7-the-llm-version-ollama-llamacpp-vllm) |

**Versions:** [`algolyzer/laya-bangla`](https://huggingface.co/algolyzer/laya-bangla) (v2, the main
release) and [`algolyzer/laya-bangla-lora`](https://huggingface.co/algolyzer/laya-bangla-lora) (a
lighter LoRA fine-tune that keeps more yes/no reasoning), each as PyTorch, ONNX fp32, and ONNX
int8. The [model card](https://huggingface.co/algolyzer/laya-bangla#choose-a-version) helps you
choose.

## Results

Accuracy on held-out test data, asked with question wording the model never saw in training.
Each cell is *Bangla-worded question / English-worded question*:

| Task | Dataset | Base model | laya-bangla v2 |
| --- | --- | :---: | :---: |
| Voice-assistant routing, two steps (2,939) | MASSIVE | 18% / 18% | **81% / 80%** |
| Complaint service sector (487) | BanglaPSG | 64% / 57% | **93% / 93%** |
| Hate speech, Banglish (3,729) | BanTH | 49% / 66% | **78% / 77%** |
| Hate speech, Bangla script (2,720) | BanHate | 61% / 55% | **74% / 74%** |
| Sentiment of course comments (602) | eLearning | 34% / 47% | **82% / 83%** |
| Sentence entailment, never trained (3,340) | IndicXNLI | **60% / 69%** | 50% / 53% |

Over all test decisions of the six training datasets, accuracy rises from **53% to 82%**, and
calibration error falls from 0.27 to 0.04, so its probabilities can be trusted on these tasks.
New kinds of yes/no questions can get worse, as the IndicXNLI row shows.

### Where it works best

Accuracy averaged over Bangla- and English-worded questions (base model in brackets):

| Task | Bangla script | Banglish | English |
| --- | --- | --- | --- |
| Complaint service sector | **93%** (61%) | not tested | not tested |
| Voice-assistant routing | **81%** (18%) | often wrong, not trained | **77%** (40%) |
| Topic of a sentence | **77%** (57%) | not tested | **87%** (79%) |
| Hate speech | **74%** (58%) | **77%** (58%) | 63% (57%) |
| Sentiment of course comments | **82%** (46%) | **83%** (36%) | not tested |
| Sentiment of reviews and posts | 49–63% (24–34%) | not tested | 65–74% (33–54%) |
| Yes/no sentence logic | 51%, worse (64%) | not tested | 64%, worse (80%) |

Although it was trained only on Bangla, it also improved on English text: 61% to 73% pooled over
six English test sets. Use it for routing, topics, moderation, and feedback; keep the base model
for general yes/no reasoning. See the [benchmark report](docs/benchmark-results.md) for every
task, script, and limitation.

## Datasets

All training data is real, published Bangla text, downloaded from the original publishers and
pinned to exact versions. Nothing is synthetic, and no dataset text is stored in this repository.

| Dataset | Text | Used for | License |
| --- | --- | --- | --- |
| [MASSIVE](https://huggingface.co/datasets/mteb/amazon_massive_intent) `bn-BD` | Voice-assistant requests | Training and test | CC BY 4.0 |
| [SIB-200](https://huggingface.co/datasets/mteb/sib200) `ben_Beng` | News-style sentences | Training and test | CC BY-SA 4.0 |
| [BanglaPSG](https://data.mendeley.com/datasets/kcr8zfmtvb) | Citizen complaints | Training and test | CC BY 4.0 |
| [BanTH](https://huggingface.co/datasets/aplycaebous/BanTH) | Banglish YouTube comments | Training and test | MIT |
| [BanHate](https://huggingface.co/datasets/aplycaebous/BanHate) | Bangla YouTube comments | Training and test | MIT |
| [Bangla Sentiments in eLearning](https://huggingface.co/datasets/TanjimKIT/Bangla_Sentiments_in_eLearning) | Course comments, Bangla and romanized | Training and test | Apache-2.0 |
| [SentNoB](https://huggingface.co/datasets/khondoker/SentNoB) | Social-media comments | Test only | CC BY-ND 4.0 |
| [IndicSentiment](https://huggingface.co/datasets/mteb/IndicSentiment) `bn` | Product reviews | Test only | CC0 1.0 |
| [IndicXNLI](https://huggingface.co/datasets/mteb/IndicXnliPairClassification) `bn` | Sentence pairs | Test only | CC BY 4.0 |

[docs/datasets.md](docs/datasets.md) lists versions, sizes, cleaning steps, and how leakage
between training and test data is removed.

## Run it yourself

### 1. Set up

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
git clone https://github.com/shamspias/laya-bangla.git
cd laya-bangla
uv sync --locked --all-extras
uv run --frozen laya-bangla doctor        # shows your Python, GPU, and package versions
```

This creates a Python 3.12 environment from the committed lock file. The commands pick an
NVIDIA GPU (CUDA), Apple Silicon (MPS), or the CPU automatically; add `--device cpu` to force
the CPU.

### 2. Try the model locally

```bash
uv run --frozen laya-bangla serve --model models/laya-bangla-v2
```

Open `http://127.0.0.1:8000` to compare laya-bangla with the base model on any text. To use the
published checkpoint instead of training your own, download it first:

```bash
uv run --frozen hf download algolyzer/laya-bangla --revision v2 --local-dir models/laya-bangla-v2
```

### 3. Test

These checks need no GPU and download no models. They are the same checks CI runs:

```bash
uv run --frozen laya-bangla validate      # checks the question schemas and smoke-set data
uv run --frozen pytest -q                 # unit tests
uv run --frozen ruff check .              # lint
uv run --frozen ruff format --check .     # formatting
```

To test the model itself, benchmark it on the held-out test data (needs step 4's `prepare`):

```bash
uv run --frozen laya-bangla benchmark --model models/laya-bangla-v2 --output outputs/bench-mine
uv run --frozen laya-bangla compare outputs/bench-base/summary.json outputs/bench-mine/summary.json
```

Add `--limit 50` for a quick check on 50 rows per dataset. For English, build the English test
set once with `laya-bangla prepare --config configs/data-english.toml` and pass
`--data data/processed-english/test.jsonl`. For the 36-message diagnostic smoke run on the base
model, see [local development](docs/local-development.md).

To test the browser and Node version, export ONNX files and score them against the PyTorch run:

```bash
uv run --frozen laya-bangla export-onnx --model models/laya-bangla-v2 --output models/laya-bangla-v2-onnx
node web/tools/predict.mjs <laya-ts>/dist/index.js models/laya-bangla-v2-onnx/int8 rows.jsonl out.jsonl
uv run --frozen laya-bangla score out.jsonl --reference outputs/bench-v2/predictions.jsonl
```

### 4. Rebuild the data, train, and benchmark

```bash
uv run --frozen laya-bangla prepare                                  # download and split the data
uv run --frozen laya-bangla train --output models/laya-bangla-v2     # about 1 hour on an 8 GB GPU
uv run --frozen laya-bangla benchmark --model base --output outputs/bench-base
uv run --frozen laya-bangla benchmark --model models/laya-bangla-v2 --output outputs/bench-v2
uv run --frozen laya-bangla compare outputs/bench-base/summary.json outputs/bench-v2/summary.json
```

Each run needs a fresh output directory. The [fine-tuning guide](docs/fine-tuning.md) explains
full fine-tuning, LoRA (`--config configs/train-lora.toml`), QLoRA for the LLM version
(`laya-bangla train-llm`), and how to train on your own data.

### 5. Publish

With a Hugging Face token that can write to the target repositories in `HF_TOKEN`:

```bash
uv run --frozen laya-bangla publish --model models/laya-bangla-v2 --repo algolyzer/laya-bangla \
  --benchmark base=outputs/bench-base --benchmark laya-bangla-v2=outputs/bench-v2 --tag v2
uv run --frozen laya-bangla publish-onnx --onnx models/laya-bangla-v2-onnx --repo algolyzer/laya-bangla
uv run --frozen laya-bangla publish-space --space web --repo algolyzer/laya-bangla-playground
```

`publish` uploads the checkpoint, [the model card](docs/model-card.md), `LICENSE`, `NOTICE`,
and metrics-only benchmark summaries, then checks that the Hub copy matches the local files
before tagging it. It refuses to move an existing tag. `publish-onnx` adds the ONNX folders and
verifies them the same way. `publish-space` reads the Space type from its README: `web/` is a
free static Space; `space/` is a Gradio Space, which Hugging Face only hosts on a paid plan.

## Project status

| Item | Status |
| --- | --- |
| Real training and held-out test data | [Available](docs/datasets.md): 9 published sources, pinned and leak-checked |
| Fine-tuning and probability calibration | [Available](docs/fine-tuning.md): `laya-bangla train` |
| Benchmark against the upstream model | [Available](docs/benchmark-results.md): `laya-bangla benchmark` |
| Fine-tuned model | **Published:** [`algolyzer/laya-bangla`](https://huggingface.co/algolyzer/laya-bangla) `v2` |
| ONNX versions for browsers and Node.js | **Published** in the model repository, checked against PyTorch |
| Live demo | **Live**, free, runs in the browser: [`algolyzer/laya-bangla-playground`](https://huggingface.co/spaces/algolyzer/laya-bangla-playground) |
| English benchmark | Available: `configs/data-english.toml` |
| LoRA version | **Published:** [`algolyzer/laya-bangla-lora`](https://huggingface.co/algolyzer/laya-bangla-lora) `v1` with ONNX; sample-tested, full test pending |
| LLM version for Ollama, llama.cpp, vLLM | Code ready, not trained yet ([next steps](docs/next-steps.md)) |
| Synthetic smoke set | 36 Bangla/Banglish/mixed messages for diagnostics ([results](docs/baseline-smoke-results.md)) |
| Speaker review and customer-support data | Open; see the [roadmap](docs/roadmap.md) |

## Upstream model

The [Laya model family](https://huggingface.co/convaiinnovations/laya) is maintained by Convai
Innovations. This project starts from
[`convaiinnovations/laya-multilingual`](https://huggingface.co/convaiinnovations/laya-multilingual)
(mmBERT-base, includes Bangla) at the immutable revision
`e4e9ddf21a7b1903b7acffd8814ad4307bf63a67`, recorded in
[`configs/baseline.toml`](configs/baseline.toml). The root checkpoint
[`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya) is English-focused.

## Contributing

Bangla failure cases, dataset suggestions, annotation help, and evaluation ideas are welcome.
Please read [CONTRIBUTING.md](CONTRIBUTING.md) before opening an
[issue](https://github.com/shamspias/laya-bangla/issues) or pull request.

## License and acknowledgments

Original code and documentation in this repository are licensed under
[Apache 2.0](LICENSE), and so is the released model. See [NOTICE](NOTICE) for attribution.

This is an independent adaptation project based on Laya by **Convai Innovations**. Datasets
keep their own licenses, listed above and in the [model card](docs/model-card.md).

- [Upstream Laya source](https://github.com/NandhaKishorM/laya)
- [Laya model family](https://huggingface.co/convaiinnovations/laya)
- [Multilingual checkpoint](https://huggingface.co/convaiinnovations/laya-multilingual)
- [Upstream fine-tuning notebook](https://github.com/NandhaKishorM/laya/blob/main/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb)
