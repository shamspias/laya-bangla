# Roadmap

The goal is a reproducible Bangla adaptation of Laya with a fine-tuned checkpoint
published on Hugging Face. The current milestone is repository setup. Later
phases are planned work, and their details will be refined using baseline results.

## 0. Repository foundation

- [x] Establish the `laya-bangla` identity and project scope.
- [x] Document the upstream family and multilingual starting point.
- [x] Add ignore rules, formatting conventions, license, and attribution.
- [x] Record contribution guidance and a future model-card template.

## 1. Establish the Bangla baseline

- [ ] Choose the first target workflows and typed question schemas.
- [ ] Pin the upstream multilingual model revision and runtime dependencies.
- [ ] Collect representative Bangla-script, Banglish, and code-switched examples.
- [ ] Compare Bangla and English question instructions where relevant.
- [ ] Record baseline task metrics, confidence calibration, latency, hardware,
  and error categories using a reproducible evaluation command.
- [ ] Define measurable acceptance criteria for the first fine-tuned release.

**Deliverable:** a versioned evaluation set and baseline report. Explicitly load
the multilingual checkpoint for these comparisons so routing does not change
the model under evaluation.

## 2. Prepare data

- [ ] Select sources and record their versions, licenses, and language/domain
  coverage.
- [ ] Define an annotation schema compatible with Laya's typed decisions and
  review examples with fluent Bangla speakers.
- [ ] Add normalization, validation, deduplication, and dataset statistics.
- [ ] Create train, validation/calibration, and held-out test splits; keep related
  examples, source documents, and their translations in the same split.
- [ ] Cover negation, spelling variations, numerals, names, informal phrasing,
  and mixed-language inputs relevant to the chosen workflows.

**Deliverable:** documented, reproducible dataset preparation and split manifests.

## 3. Fine-tune

- [ ] Review the upstream training implementation and fine-tuning notebook for
  compatibility with the multilingual encoder and decision head.
- [ ] Select a training approach and compute budget based on that implementation
  and the measured failure cases.
- [ ] Implement the training entry point with versioned configuration, dependency
  pins, seeds, checkpointing, and experiment metadata.
- [ ] Run an initial experiment and compare it with the fixed baseline.

**Deliverable:** a reproducible training run and candidate checkpoint. Full
fine-tuning versus parameter-efficient adaptation remains an implementation
decision, not an assumed capability of the upstream trainer.

## 4. Evaluate and calibrate

- [ ] Fit calibration parameters on the designated held-out validation/calibration
  data and evaluate once on the untouched test set.
- [ ] Report accuracy and macro-F1 for categorical decisions, ordinal error for
  score questions, and appropriate probability metrics such as Brier score and
  expected calibration error.
- [ ] Break down results by script, code-switching, domain, and question type.
- [ ] Compare the base and fine-tuned checkpoints under identical schemas,
  sequence budgets, and hardware; record sample counts and uncertainty.
- [ ] Document failure cases and limitations alongside any improvements.

**Deliverable:** an auditable comparison report and calibrated release candidate.

## 5. Publish to Hugging Face

- [ ] Choose the Hugging Face owner and model repository ID.
- [ ] Verify the checkpoint can be loaded with the documented inference path.
- [ ] Package weights, tokenizer, model configuration, calibration artifacts,
  license, and attribution required for the released model.
- [ ] Complete the [model card template](model-card-template.md) with actual
  provenance, training details, evaluation results, limitations, and usage.
- [ ] Upload a versioned release and add its exact link and revision to the README.

**Deliverable:** a downloadable fine-tuned checkpoint with reproducible usage.

## Decisions to make during implementation

- Initial domains, datasets, annotation budget, and target Bangla varieties.
- Final base checkpoint revision, training method, and available hardware.
- Dataset schema, split sizes, calibration protocol, and release thresholds.
- Hugging Face namespace and distribution format.

## Upstream references

Reviewed during repository setup on **2026-09-25**:

- [Laya family model card](https://huggingface.co/convaiinnovations/laya)
- [Multilingual model card](https://huggingface.co/convaiinnovations/laya-multilingual)
- [Source repository](https://github.com/NandhaKishorM/laya)
- [Fine-tuning notebook](https://github.com/NandhaKishorM/laya/blob/main/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb)
