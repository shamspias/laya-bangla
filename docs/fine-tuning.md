# Fine-tuning guide

This page explains how the laya-bangla models were trained and how to train your own: with full
fine-tuning, with LoRA, or (for the LLM version) with QLoRA. Every step writes to an ignored local
folder and records checksums of its inputs, so runs can be reproduced and compared.

## The three methods

| Method | What is trained | Trainable parameters | GPU memory | Used for |
| --- | --- | ---: | ---: | --- |
| **Full fine-tuning** | Every transformer layer and the decision head; token embeddings frozen | 125M of 307M | 4.6 GB | `laya-bangla` v2 ([`configs/train.toml`](../configs/train.toml)) |
| **LoRA** | Small low-rank "adapter" matrices on every attention and MLP layer, plus the decision head | 18M of 307M | 2.4 GB | [`laya-bangla-lora`](https://huggingface.co/algolyzer/laya-bangla-lora) ([`configs/train-lora.toml`](../configs/train-lora.toml)), published, sample-tested |
| **QLoRA** | LoRA adapters on top of a 4-bit copy of a chat LLM | 17M of 1.7B | not measured | `laya-bangla-llm` ([`configs/train-llm.toml`](../configs/train-llm.toml)), not trained yet |

In plain words:

- **Full fine-tuning** changes most of the model. It usually learns the new tasks best, but it
  can also forget more of what the base model knew.
- **LoRA** freezes the original weights and learns a small set of extra weights next to them. It
  needs less memory and changes the model less, which can help it keep its English and general
  ability. After training, the adapters are merged back into the weights, so a LoRA model is
  used exactly like any other Laya checkpoint.
- **QLoRA** is LoRA on a base model stored in 4 bits. It exists for large models that would not
  fit in GPU memory otherwise. Laya is small (307M), so QLoRA would only add rounding error;
  it is used here for the 1.7B-parameter LLM version.

## Train the models

```bash
uv sync --locked --all-extras
uv run --frozen laya-bangla prepare                                                   # data/processed/
uv run --frozen laya-bangla train --output models/laya-bangla-v2                      # full fine-tuning
uv run --frozen laya-bangla train --config configs/train-lora.toml --output models/laya-bangla-lora
uv run --frozen laya-bangla train-llm --output models/laya-bangla-llm                 # QLoRA LLM
```

`prepare` needs network access once; later runs use `data/cache/`. Set `HF_TOKEN` if anonymous
Hugging Face downloads are rate-limited. On Windows, `pyproject.toml` installs PyTorch from the
CUDA 12.8 wheel index; Linux wheels already include CUDA, and macOS wheels include MPS. QLoRA
needs an NVIDIA GPU (bitsandbytes).

All runs used one NVIDIA GeForce RTX 4060 with 8 GB of memory. [Next steps](next-steps.md)
records the LoRA validation results and how to finish the LoRA and LLM work. Each output folder holds
`train_report.json` with the data and config checksums, the validation history, and the peak
GPU memory.

## How the Laya models are trained

`laya-bangla train` reuses upstream `laya.train` (Laya 0.4.11): its item encoding, soft
cross-entropy objective, temperature fitting, abstention thresholds, and checkpoint format, so
the result loads with `laya.load(path)`. It adds:

- **A fixed validation split.** Model selection and calibration both use the prepared
  validation split, so test data never influences the checkpoint.
- **Best-epoch selection.** Validation accuracy and negative log-likelihood (NLL) are measured
  before training and after every epoch; the epoch with the lowest validation NLL is kept.
- **Frozen token embeddings.** The 256k-token embedding table is two thirds of mmBERT-base's
  parameters. Freezing it keeps every language the tokenizer knows intact and saves memory.
- **Option shuffling.** Choice options are presented in a new random order each epoch, so the
  model cannot learn option positions.
- **LoRA (optional).** A `[lora]` section in the config (rank, alpha, dropout, target modules)
  trains adapters instead of the full encoder and merges them afterwards.
- **Calibration.** One temperature per question type is fitted on validation data, so a
  probability of 0.8 means about 8 correct out of 10.

## How the LLM version is trained

*Implemented and tested without a GPU; the training run has not been done yet.*

`laya-bangla train-llm` fine-tunes [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B)
(Apache 2.0) with QLoRA on the same training rows. Each row becomes a short chat: the system
prompt explains the task, the user message is the Laya request as JSON, and the reply is the
answers as JSON.

```text
user:      {"text": "আমাদের এলাকায় তিন দিন ধরে পানি নেই", "questions": {"sector": {"type": "choice", ...}}}
assistant: {"sector": "water"}
```

Choice options are shuffled, and the loss counts only the reply. After training, the adapter is
merged into a full-precision copy of Qwen3-1.7B, which is what vLLM and the GGUF conversion use.
The chat template has no "thinking" section, and the same template is written into the
tokenizer, the GGUF files, and the Ollama files, so every runtime builds the same prompt.

## Fine-tune on your own data

1. **Write your rows.** Each line of a JSONL file holds a text (`state`), the questions in
   Laya's format, and the correct answers (`expected`):

   ```json
   {"state": "বিল দুইবার কেটেছে", "questions": {"team": {"type": "choice", "instructions": "কোন বিভাগ দেখবে?", "criteria": {"billing": "বিল ও পেমেন্ট", "technical": "অ্যাপ বা লগইন"}}}, "expected": {"team": "billing"}}
   ```

   `noul` answers are `true`/`false`; `score` answers are the level index (0 = first level).
   Keep separate training, validation, and test files, and never put the same text in two of
   them. Ask each test question with wording that training never uses.
2. **Point a config at your files.** Copy `configs/train.toml` (full) or
   `configs/train-lora.toml` (LoRA) and change `train` and `validation`.
3. **Train**, then **benchmark** on your test file with `--data` and compare with the base model:

   ```bash
   uv run --frozen laya-bangla train --config configs/my-train.toml --output models/my-model
   uv run --frozen laya-bangla benchmark --model base --data my-test.jsonl --output outputs/my-base
   uv run --frozen laya-bangla benchmark --model models/my-model --data my-test.jsonl --output outputs/my-model
   uv run --frozen laya-bangla compare outputs/my-base/summary.json outputs/my-model/summary.json
   ```

4. **Check the regressions.** Also benchmark on data you did not train on (the English set in
   `configs/data-english.toml` is a good start). Fine-tuning on narrow tasks can make other
   questions worse; keep the base model for those, or add training data for them.

Tips: start from the validation numbers in `train_report.json`, not the test numbers; choose
the release by a rule fixed in advance (here, lowest validation NLL); and keep at least a few
hundred validation rows per task so calibration is stable.

## Benchmark protocol

`laya-bangla benchmark` scores a checkpoint, the pinned upstream `base`, or an LLM checkpoint on:

- the held-out **test** split of every training source, with official test sets wherever the
  source has one;
- three **benchmark-only** sources never used for training (SentNoB, IndicSentiment, IndicXNLI);
- **end-to-end MASSIVE routing**, where the intent question uses the model's own predicted
  scenario;
- the 36-message **synthetic smoke set**, reported separately;
- optionally the **English** test sets (`--data data/processed-english/test.jsonl`).

Every test question uses wording that never appears in training, in Bangla and again in
English. The summary reports accuracy, macro-F1, Brier score, NLL, ECE, and ordinal MAE per
source, task, wording language, and script slice (`bangla`, `banglish`, `mixed`, `english`).
For the LLM version only accuracy and macro-F1 are meaningful, because it returns labels
without probabilities; `metadata.json` also counts replies that did not name a valid option.
Metric definitions are in [local development](local-development.md#outputs-and-measurement-definitions).

Models compared with each other are benchmarked on the same device: CUDA mixed precision can
move probabilities in the third decimal place relative to CPU.
