# Contributing to laya-bangla

Thanks for helping improve typed decisions for Bangla. The project is currently
in the repository-setup stage; the [roadmap](docs/roadmap.md) records the next
steps toward fine-tuning and publishing a model.

## Useful contributions now

- Small Bangla, Banglish, or mixed-language examples that reveal a model failure.
- Dataset suggestions with source links, language coverage, and license details.
- Annotation guidelines and evaluation cases for concrete workflows.
- Documentation corrections and reproducibility improvements.

## Reporting a model failure

Open a [GitHub issue](https://github.com/shamspias/laya-bangla/issues) with:

1. The model ID and revision, runtime version, and relevant configuration.
2. A minimal input state and the complete question schema.
3. The actual structured output, including probabilities where available.
4. The expected decision and a short explanation in Bangla or English.
5. Whether the input is Bangla script, Banglish, or mixed-language text.

Use a synthetic or anonymized example that can be shared publicly. State whether
the result came from the upstream model or a future `laya-bangla` checkpoint.

## Working conventions

- Keep changes focused and describe how they help the project.
- Use UTF-8 and preserve meaningful Bangla characters and punctuation.
- Follow `.editorconfig` and keep commands and links in documentation current.
- Document the source, license, and preparation steps for any dataset proposal.
- Keep training, validation/calibration, and test data separate when adding the
  pipeline. Record model revisions, dataset versions, seeds, and configurations.
- Include actual verification steps and results in pull requests. Add runnable
  setup and test commands alongside future code; this scaffold has no project
  test suite yet.

## Local files and published artifacts

The following paths are reserved for local artifacts and ignored by Git. They
will be created as the relevant tools are implemented:

| Path | Intended contents |
| --- | --- |
| `data/raw/` | Original downloaded or collected data |
| `data/interim/`, `data/processed/`, `data/splits/` | Prepared datasets and split files |
| `data/cache/`, `datasets/` | Dataset caches and local dataset copies |
| `models/`, `checkpoints/`, `adapters/`, `exports/` | Model weights and training/export artifacts |
| `outputs/`, `runs/`, `results/`, `logs/` | Generated experiment outputs |

Small, shareable examples and test fixtures can be tracked under `examples/` or
`tests/fixtures/` when needed. Put curated evaluation summaries in `docs/`.
Publish release weights to Hugging Face and link the exact revision from the
release documentation. Keep local credentials in ignored environment files.

## License

Contributions to the original code and documentation are accepted under the
repository's [Apache 2.0 license](LICENSE). Preserve upstream attribution when
adapting code; document third-party data licensing separately.
