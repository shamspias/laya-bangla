# Project context

`laya-bangla` is a Bangla-focused adaptation of the Laya decision-model family.
The long-term goal is to prepare suitable data, fine-tune and evaluate a model
for Bangla, and publish the resulting checkpoint on Hugging Face.

## Current scope

- The CLI in `src/laya_bangla/` evaluates the upstream model on 36 synthetic
  diagnostic messages (`baseline`), builds real training/test data from pinned
  public datasets (`prepare`, see `docs/datasets.md`), fine-tunes and calibrates
  (`train`), benchmarks checkpoints on held-out data (`benchmark`, `compare`),
  serves a comparison playground (`serve`), exports ONNX (`export-onnx`), scores other
  runtimes (`score`), and uploads releases (`publish`, `publish-onnx`, `publish-space`).
  See `docs/fine-tuning.md` and `docs/deploy.md`. v2 is published as
  `algolyzer/laya-bangla` (tag `v2`); `docs/model-card.md` is its Hub README. The public
  Space is the static browser app in `web/` (free hosting); `space/` is the Gradio app.
  `docs/next-steps.md` records unfinished work (LoRA benchmark, LLM version).
- Never train on the smoke set or on benchmark-only sources in `configs/data.toml`.
- `configs/baseline.toml` pins `convaiinnovations/laya-multilingual` to revision
  `e4e9ddf21a7b1903b7acffd8814ad4307bf63a67`. The root checkpoint at
  `convaiinnovations/laya` is English-focused.
- Laya produces typed decisions (`choice`, `noul`, and `score`); preserve that
  model interface when planning the adaptation.
- Use `docs/roadmap.md` for milestones and unresolved implementation choices.
  See `docs/local-development.md` for the local evaluation workflow.

## Local commands

- `uv sync --locked` sets up Python 3.12 and the pinned environment.
- `uv run --frozen laya-bangla doctor` inspects hardware; the initial development
  machine is an Apple M2 Pro with 16 GB RAM and an available MPS device.
- `uv run --frozen laya-bangla validate` checks the data and paired schemas.
- `uv run --frozen laya-bangla baseline --device mps --output outputs/new-run`
  runs inference. Every run needs a fresh output directory.
- `uv sync --locked --all-extras` adds the playground dependencies. Then
  `prepare`, `train --output models/<name>`, `benchmark --model base|models/<name>
  --output outputs/<name>`, and `serve --model models/<name>`.
- Checks: `uv run --frozen pytest -q`, `uv run --frozen ruff check .`, and
  `uv run --frozen ruff format --check .`.

## Working guidance

- Keep the README, roadmap, and future model card aligned with actual progress.
  Clearly distinguish upstream examples from fine-tuned model results.
- Preserve Bangla Unicode text. Consider Bangla script, Banglish, and
  Bangla-English code-switching in future datasets and evaluations.
- Keep dataset versions, splits, base revisions, training configuration, and
  evaluation/calibration procedures reproducible.
- Keep large data, weights, caches, credentials, and generated runs out of Git;
  follow `.gitignore` and `CONTRIBUTING.md` for artifact locations.
- Add dependency pins, setup commands, and meaningful checks with the code that
  needs them. Keep `uv.lock` aligned with `pyproject.toml`.
- Preserve upstream license and attribution notices when incorporating code or
  publishing model artifacts.
- When asked to commit, use `shamspias <shamsuddinpias0@gmail.com>` for both
  author and committer, with no co-author or assistant attribution trailers.
  Supply identity per command rather than changing Git configuration.
