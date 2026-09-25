# Hugging Face model card template

Use this file to prepare the future model repository's `README.md`. This is a
template, not a published model card. Replace every `TODO`, update the metadata
to match the actual release, and put the YAML block below at the very beginning
of the final card without the surrounding Markdown code fence.

`base_model`, `license`, and `library_name` below reflect the planned upstream
starting point. Confirm them against the model actually trained and its loading
implementation before publishing. Add dataset IDs and structured evaluation
metadata once those artifacts exist.

```yaml
---
language:
  - bn
license: apache-2.0
base_model: convaiinnovations/laya-multilingual
base_model_relation: finetune
library_name: laya
pipeline_tag: text-classification
tags:
  - laya
  - bangla
  - bengali
  - typed-decisions
  - classification
  - routing
---
```

## Model details

- **Name:** laya-bangla
- **Model repository and release revision:** TODO
- **Maintainer:** TODO
- **Source code:** https://github.com/shamspias/laya-bangla
- **Base checkpoint and exact revision:** TODO
- **Architecture and parameter count:** TODO
- **Question types supported and evaluated:** TODO
- **Languages, scripts, and domains evaluated:** TODO

Describe the actual adaptation, the intended workflows, and how it differs from
the base checkpoint. Attribute Laya to Convai Innovations and retain applicable
upstream notices.

## Intended use and limitations

TODO: Document evaluated uses, known failure cases, and boundaries of language
coverage. Distinguish Bangla-script, romanized Bangla, and mixed-language results.
Explain how to interpret the returned probabilities based on measured calibration.

## Training data

TODO: List dataset sources, versions, licenses, sizes, language/domain coverage,
annotation methods, and preprocessing. Describe synthetic data, if used, and
document train/validation/calibration/test splits and deduplication boundaries.
Link dataset cards or preparation instructions where available.

## Training procedure

TODO: Record the training-code revision, configuration, method, hyperparameters,
seeds, dependency versions, hardware, precision, sequence budgets, duration, and
checkpoint selection procedure. Include a reproducible command or notebook link.

## Evaluation

TODO: Report measured results against the pinned upstream baseline on the same
held-out data and question schemas. Include sample counts, metric definitions,
uncertainty, calibration methodology, and per-language/script/domain breakdowns.

| Metric and evaluation slice | Upstream baseline | laya-bangla | Sample count |
| --- | --- | --- | --- |
| TODO | TODO | TODO | TODO |

Use task-appropriate metrics, such as accuracy and macro-F1 for classification,
ordinal error for scores, and Brier score and expected calibration error for
probabilities. Describe latency measurements with the hardware and batch settings.

## Usage

TODO: Add a tested installation command and a minimal Bangla inference example
using the actual published model ID and supported loader. List required runtime
versions, model files, sequence limits, and hardware requirements. Identify
illustrative output separately from measured evaluation results.

## Artifacts and reproducibility

TODO: List the released weights, tokenizer, configuration, calibration files,
checksums or revisions, and their relationship to the training and evaluation
artifacts in the source repository.

## License and attribution

TODO: Confirm the checkpoint's release license and document base-model and data
provenance. Include the applicable license and attribution files with the release.
Link the upstream [Laya family](https://huggingface.co/convaiinnovations/laya) and
[source repository](https://github.com/NandhaKishorM/laya).
