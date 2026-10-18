# Roadmap

The goal is a reproducible Bangla adaptation of Laya with a fine-tuned checkpoint
published on Hugging Face. Phases 1–5 have a working first iteration built on
published datasets, and v2 is released as
[`algolyzer/laya-bangla`](https://huggingface.co/algolyzer/laya-bangla). Speaker
review and target-workflow data remain open.

## 0. Repository foundation

- [x] Establish the `laya-bangla` identity and project scope.
- [x] Document the upstream family and multilingual starting point.
- [x] Add ignore rules, formatting conventions, license, and attribution.
- [x] Record contribution guidance and a future model-card template.

## 1. Establish the Bangla baseline

- [x] Choose initial diagnostic workflows and typed question schemas: support
  routing, explicit churn threats, and urgency.
- [x] Pin the upstream multilingual model revision and runtime dependencies.
- [x] Implement local hardware inspection, schema/data validation, typed-decision
  metrics, and a model-backed runner with reproducibility manifests.
- [x] Add a 36-message synthetic smoke set across three input forms and two
  instruction languages; this is not the representative evaluation dataset.
- [x] Collect representative Bangla-script, Banglish, and code-switched examples
  from published datasets ([datasets](datasets.md)). Code-mixed text remains
  scarce (88 test states).
- [x] Compare Bangla and English question instructions on the synthetic smoke set.
- [x] Complete the first MPS run and document [diagnostic results](baseline-smoke-results.md),
  including prompt-sensitive churn answers and collapsed urgency predictions.
- [x] Record baseline task metrics and confidence calibration on held-out data
  with `laya-bangla benchmark --model base`.
- [ ] Define measurable acceptance criteria for the first fine-tuned release.

**Deliverable:** a versioned evaluation set and baseline report. Explicitly load
the multilingual checkpoint for these comparisons so routing does not change
the model under evaluation. The diagnostic run is complete; the representative,
speaker-reviewed dataset and benchmark remain open work.

## 2. Prepare data

- [x] Select sources and record their versions, licenses, and language/domain
  coverage (`configs/data.toml`, [datasets](datasets.md)).
- [x] Define typed-decision schemas for every source (`configs/tasks.json`).
- [ ] Review the Bangla question wording and sample labels with fluent speakers.
- [x] Add normalization, validation, deduplication, and dataset statistics.
- [x] Create train, validation/calibration, and held-out test splits; keep related
  examples, source documents, and their translations in the same split, and
  remove cross-source leakage.
- [ ] Cover negation, spelling variations, numerals, names, informal phrasing,
  and mixed-language inputs relevant to the chosen workflows. Public data covers
  informal social-media text and Banglish well; no open Bangla customer-support
  dataset exists, so the support workflow still lacks real training data.

**Deliverable:** documented, reproducible dataset preparation and split manifests.

## 3. Fine-tune

- [x] Review the upstream training implementation (`laya.train`, Laya 0.4.11)
  for compatibility with the multilingual encoder and decision head.
- [x] Select a training approach and compute budget: full fine-tuning with frozen
  token embeddings on one 8 GB GPU ([fine-tuning](fine-tuning.md)).
- [x] Implement the training entry point with versioned configuration, dependency
  pins, seeds, checkpointing, and experiment metadata.
- [x] Run an initial experiment and compare it with the fixed baseline
  ([results](benchmark-results.md)).

**Deliverable:** a reproducible training run and candidate checkpoint. Full
fine-tuning versus parameter-efficient adaptation remains an implementation
decision, not an assumed capability of the upstream trainer.

## 4. Evaluate and calibrate

- [x] Fit calibration parameters on the designated held-out validation/calibration
  data and evaluate once on the untouched test set.
- [x] Report accuracy and macro-F1 for categorical decisions, ordinal error for
  score questions, and appropriate probability metrics such as Brier score and
  expected calibration error.
- [x] Break down results by script, code-switching, domain, and question type.
- [x] Compare the base and fine-tuned checkpoints under identical schemas,
  sequence budgets, and hardware; record sample counts.
- [ ] Add confidence intervals and repeated-seed runs.
- [x] Document failure cases and limitations alongside any improvements.

**Deliverable:** an auditable comparison report and calibrated release candidate.

## 5. Publish to Hugging Face

- [x] Choose the Hugging Face owner and model repository ID:
  `algolyzer/laya-bangla`.
- [x] Verify the checkpoint can be loaded with the documented inference path:
  `laya.load("algolyzer/laya-bangla", revision="v2")` from an empty cache gives
  the same answers as the local checkpoint.
- [x] Package weights, tokenizer, model configuration, calibration artifacts,
  license, and attribution (`laya-bangla publish`).
- [x] Write the [model card](model-card.md) with actual provenance, training
  details, evaluation results, limitations, and usage.
- [x] Upload a versioned release (tag `v2`, commit
  `4c5eb48932827f83cce9514c1af5510864089c0f`) and link it from the README.
- [x] Add a live demo. Hugging Face hosts Gradio Spaces only on paid plans, so the public
  [Space](https://huggingface.co/spaces/algolyzer/laya-bangla-playground) is static and runs the
  model in the visitor's browser (`web/`); the Gradio app in `space/` is kept for local use.

## 6. More ways to run it

- [x] ONNX export checked against PyTorch, with fp32 (Node, servers) and int8-embedding
  (browser) variants published in the model repository ([deploy](deploy.md)).
- [x] HTTP API through Laya's `laya-serve`, tested locally.
- [x] English benchmark for base and v2 ([results](benchmark-results.md#english-text)).
- [x] LoRA training option; a LoRA run is trained and validated.
- [x] Publish the LoRA run as [`algolyzer/laya-bangla-lora`](https://huggingface.co/algolyzer/laya-bangla-lora)
  with ONNX versions; a CPU sample test shows it keeps more yes/no reasoning than v2.
- [ ] Run the full Bangla and English benchmarks for the LoRA version.
- [ ] Train, benchmark, and publish the QLoRA LLM version for Ollama, llama.cpp, and vLLM
  (code ready). [Next steps](next-steps.md) has the exact commands.

**Deliverable:** a downloadable fine-tuned checkpoint with reproducible usage.

## Decisions to make during implementation

- Initial domains, datasets, annotation budget, and target Bangla varieties.
- Final base checkpoint revision, training method, and available hardware.
- Dataset schema, split sizes, calibration protocol, and release thresholds.
- Whether to add an ONNX export for CPU-only or in-browser use.

## Upstream references

Reviewed during repository setup on **2026-09-25**:

- [Laya family model card](https://huggingface.co/convaiinnovations/laya)
- [Multilingual model card](https://huggingface.co/convaiinnovations/laya-multilingual)
- [Source repository](https://github.com/NandhaKishorM/laya)
- [Fine-tuning notebook](https://github.com/NandhaKishorM/laya/blob/main/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb)
