# Status and next steps

Where the work stopped on 2026-10-19, and exactly how to continue. Everything below the
"Not finished" heading is optional follow-up: the released model, the free browser demo, and
the guides are complete and tested.

## What is done and published

| Item | Where | State |
| --- | --- | --- |
| laya-bangla-lora (LoRA fine-tune) + ONNX | [`algolyzer/laya-bangla-lora`](https://huggingface.co/algolyzer/laya-bangla-lora), tag `v1` | Released; sample-tested |
| Collection of every version, the demo, and the datasets | [algolyzer/laya-bangla collection](https://huggingface.co/collections/algolyzer/laya-bangla-6ad51a88c6ea46c7f5cf9180) | Done |
| laya-bangla v2 (full fine-tune, calibrated) | [`algolyzer/laya-bangla`](https://huggingface.co/algolyzer/laya-bangla), tag `v2` | Released, benchmarked in Bangla and English |
| ONNX versions (`onnx/fp32`, `onnx/int8`) | same repository, commit `75a291d0` | Released; 99.6% / 99.1% the same answers as PyTorch |
| Free browser demo | [`algolyzer/laya-bangla-playground`](https://huggingface.co/spaces/algolyzer/laya-bangla-playground) (static Space, source `web/`) | Live; tested in Chrome |
| English benchmark | `configs/data-english.toml`, results in the model card | Done for base and v2 |
| HTTP API | Laya's `laya-serve`, see [deploy](deploy.md#2-http-api-any-language) | Tested locally |
| Guides | [deploy](deploy.md), [fine-tuning](fine-tuning.md) | Written |

Hugging Face no longer hosts Gradio or Docker Spaces for free (PRO or Team plan needed), so
the public demo is a static Space that runs the model in the visitor's browser. The Gradio
app in `space/` still works for local use or on a paid plan.

## Right after PR #7 is merged

The model cards on Hugging Face link `docs/deploy.md`, `docs/next-steps.md`, and `web/` on the
PR branch (`Ashfaq1306/laya-bangla`, `feat/run-anywhere`), because those files do not exist on
upstream `main` until the merge. Afterwards, upload both cards unchanged so they link upstream
`main` again:

```bash
hf upload algolyzer/laya-bangla docs/model-card.md README.md --commit-message "Card: link the merged docs"
hf upload algolyzer/laya-bangla-lora docs/model-card-lora.md README.md --commit-message "Card: link the merged docs"
```

## Not finished

### 1. LoRA version: published with a sample test; full test pending

[`algolyzer/laya-bangla-lora`](https://huggingface.co/algolyzer/laya-bangla-lora) (tag `v1`,
ONNX at commit `b34c71f7`) was trained with `configs/train-lora.toml`: 3 epochs, epoch 2
selected, 18.3M trainable parameters, peak 2.4 GB GPU memory, 4.6 hours on a shared GPU. By the
rule fixed in advance (lowest validation NLL: 0.460 vs 0.447), v2 stays the main release.

Without the GPU, base, v2, and LoRA were compared on a CPU sample (`--limit 100`: the first 100
test items of every source and wording). LoRA matches v2 on the trained tasks (81.0% vs 80.3%)
and keeps most of the base model's yes/no reasoning (XNLI English 81% vs v2's 65%; IndicXNLI
63.5% vs 53%). The numbers are on its model card and in its `benchmark/sample-*` files.

Still to do: the full test runs, then replace the sample table on its card.

```bash
uv run --frozen laya-bangla benchmark --model models/laya-bangla-lora --output outputs/bench-lora
uv run --frozen laya-bangla benchmark --model models/laya-bangla-lora --data data/processed-english/test.jsonl --output outputs/bench-en-lora --device cpu
uv run --frozen laya-bangla compare outputs/bench-v2/summary.json outputs/bench-lora/summary.json
uv run --frozen laya-bangla compare outputs/bench-en-v2/summary.json outputs/bench-en-lora/summary.json
```

Benchmark the Bangla test on the GPU (as v2 was) and the English test on the CPU (as base and
v2 were). Download the checkpoint first if `models/` is empty:
`hf download algolyzer/laya-bangla-lora --revision v1 --local-dir models/laya-bangla-lora`.

### 2. The LLM version for Ollama, llama.cpp, and vLLM: ready, not trained

Laya's decision head cannot run in Ollama, llama.cpp, or vLLM, so `src/laya_bangla/llm.py`
fine-tunes Qwen3-1.7B with QLoRA to answer the same JSON questions. The code, config, and
tests are in place; the training run has not been started.

- Config: `configs/train-llm.toml` (QLoRA r=16 on all projections, 1 epoch, effective
  batch 32, micro-batch 1 so it fits beside other GPU jobs on 8 GB). Qwen3-1.7B revision
  `70d244cc` is already in the local Hugging Face cache.
- 10 of 38,766 training rows exceed 1,280 tokens and are skipped (recorded in the report).
- A CPU check showed the untuned base answers with "thinking" text, which the parser counts
  as invalid. Fine-tuning teaches the JSON-only reply; the invalid count is in the benchmark
  metadata.

```bash
uv sync --locked --all-extras                     # includes bitsandbytes (the llm extra)
uv run --frozen laya-bangla train-llm --output models/laya-bangla-llm           # several hours
uv run --frozen laya-bangla benchmark --model models/laya-bangla-llm --output outputs/bench-llm --batch-size 16
uv run --frozen laya-bangla benchmark --model models/laya-bangla-llm --data data/processed-english/test.jsonl --output outputs/bench-en-llm --batch-size 16
```

Then convert to GGUF with llama.cpp at the commit Ollama 0.40.5 bundles (`620d2225`), whose
`llama-quantize` ships with Ollama:

```python
from pathlib import Path
from laya_bangla.llm import export_gguf
export_gguf(
    Path("models/laya-bangla-llm"), Path("models/laya-bangla-llm-gguf"),
    convert_script=Path("<llama.cpp checkout>/convert_hf_to_gguf.py"),
    quantize_bin=Path(r"%LOCALAPPDATA%\Programs\Ollama\lib\ollama\llama-quantize.exe"),
    python="<python with llama.cpp's gguf-py, transformers 4.57.6, protobuf<5>",
)
```

This writes F16, Q8_0, and Q4_K_M files plus Ollama's `template`, `system`, `params`, and a
`Modelfile`; `tests/test_llm.py` checks that Ollama's template builds the same prompt as
training. Before publishing, check the GGUF answers against the benchmark on a sample
(`ollama create laya-bangla-llm -f Modelfile`, or `llama-server -m <file>.gguf`), then upload
the merged model to `algolyzer/laya-bangla-llm` (vLLM) and the GGUF folder to
`algolyzer/laya-bangla-llm-GGUF` (Ollama: `ollama run hf.co/algolyzer/laya-bangla-llm-GGUF:Q4_K_M`).
vLLM does not run on Windows; test it on Linux or WSL with
`vllm serve algolyzer/laya-bangla-llm`.

### 3. Still open from the roadmap

- Review the Bangla question wording and a sample of labels with fluent speakers.
- Real customer-support data (none is public), more code-mixed text, and Banglish voice
  commands.
- Training data for general yes/no questions, to fix the entailment regression.
- Repeated seeds and confidence intervals.

## Practical notes

- The GPU on the development machine is shared with other projects. Keep micro-batches small
  (`configs/train-lora.toml` and `configs/train-llm.toml` already do) and stop GPU jobs when
  other work needs the card; a crowded 8 GB GPU spills to system memory and slows everything.
- On Windows, upstream `laya` reads `tokenizer_config.json` with the system code page and
  prints a harmless warning; set `PYTHONUTF8=1` to silence it.
- `uv run` from inside `web/` creates a `.cache/` there; the Space upload skips hidden
  folders, but delete it before committing.
- On Windows without Developer Mode, a brand-new `HF_HOME` folder can fail with
  `WinError 1314` (symlink privilege) during `laya.load`; the default Hugging Face cache works.
