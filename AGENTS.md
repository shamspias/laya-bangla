# Project context

`laya-bangla` is a Bangla-focused adaptation of the Laya decision-model family.
The long-term goal is to prepare suitable data, fine-tune and evaluate a model
for Bangla, and publish the resulting checkpoint on Hugging Face.

## Current scope

- This repository is at the documentation/scaffolding stage. There is no local
  training pipeline, released checkpoint, or project benchmark yet.
- The planned base is `convaiinnovations/laya-multilingual`. The root checkpoint
  at `convaiinnovations/laya` is English-focused. Verify and pin the chosen base
  revision when implementing the baseline.
- Laya produces typed decisions (`choice`, `noul`, and `score`); preserve that
  model interface when planning the adaptation.
- Use `docs/roadmap.md` for milestones and unresolved implementation choices.
  The initial setup task defers training to a later request.

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
  needs them. No project-specific build or test command exists at this stage.
- Preserve upstream license and attribution notices when incorporating code or
  publishing model artifacts.
