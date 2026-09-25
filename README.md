# laya-bangla

**A Bangla-focused adaptation of [Laya](https://huggingface.co/convaiinnovations/laya) for typed decisions, classification, and routing.**

বাংলা ভাষার জন্য Laya-কে আরো সুন্দর এবং সঠিক করার ছোট উদ্যক যেন ক্লাসিফিকেসন এ বাংলা ভাষা পিছিয়ে না থাকে।

[Roadmap](docs/roadmap.md) · [Contributing](CONTRIBUTING.md) · [Hugging Face model card template](docs/model-card-template.md) · [Apache 2.0 license](LICENSE)

## Overview

`laya-bangla` aims to improve Laya's performance on real-world Bangla (Bengali)
text through targeted data preparation, fine-tuning, and evaluation. The goal is
to make its decisions more useful for Bangla workflows and publish the resulting
fine-tuned checkpoint on Hugging Face.

Laya is a non-autoregressive decision model: it takes an input **state** and
**typed questions**, then returns structured answers and probabilities. Its
question types include `choice` (classification), `noul` (yes/no probability),
and `score` (ordinal scoring).

## Project status

**Repository setup / planning.** Bangla fine-tuning and evaluation have not started.
There is no released `laya-bangla` checkpoint or measured improvement yet.

| Item | Status |
| --- | --- |
| Project documentation and repository conventions | Available |
| Bangla dataset preparation and baseline evaluation | Planned |
| Fine-tuning and probability calibration | Planned |
| Bangla evaluation results and usage examples | Planned |
| Fine-tuned Hugging Face model | Planned; repository ID to be selected |

Implementation milestones and open decisions are recorded in the
[roadmap](docs/roadmap.md).

## Why a Bangla adaptation?

Multilingual coverage alone does not establish accuracy on a particular language
or workflow. This project is motivated by weak Bangla results in the intended
use cases and will measure progress against a reproducible upstream baseline.

The intended coverage includes:

- **Bangla script:** everyday and formal writing, spelling variations, and
  colloquial expressions.
- **Banglish:** romanized Bangla with varied transliterations.
- **Code-switching:** Bangla and English mixed within the same input.
- **Local context:** names, places, Bangla numerals, dates, and domain vocabulary.
- **Typed decisions:** intent classification, request routing, yes/no decisions,
  and ordinal scoring.

The exact domains, datasets, and success criteria will be chosen during baseline
work. These are development goals, not a claim of supported performance.

## Upstream model and planned starting point

The [Laya model family](https://huggingface.co/convaiinnovations/laya) is maintained
by Convai Innovations. Its model cards distinguish these checkpoints:

| Checkpoint | Upstream description | Role in this project |
| --- | --- | --- |
| [`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya) | English-focused root checkpoint, ModernBERT-large | Original project reference |
| [`convaiinnovations/laya-multilingual`](https://huggingface.co/convaiinnovations/laya-multilingual) | Multilingual checkpoint, mmBERT-base; includes Bangla | Planned baseline and fine-tuning starting point |

The multilingual checkpoint is also bundled under the `multilingual` subfolder
of the upstream family repository. The exact base revision will be pinned when
the baseline is established. Model selection and any later changes will be
documented with the experiments.

## Getting started

### Explore this repository

```bash
git clone https://github.com/shamspias/laya-bangla.git
cd laya-bangla
```

Start with the [roadmap](docs/roadmap.md) for the development plan and
[contribution guide](CONTRIBUTING.md) for reporting Bangla failure cases.

### Try the upstream Bangla baseline (optional)

This example uses the **original multilingual checkpoint**, not a fine-tuned
`laya-bangla` model. It follows the upstream SDK interface; predictions need to
be evaluated on your own examples.

Use Python 3.10 or newer. Create an isolated environment and install the upstream
runtime version referenced by this scaffold:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install "laya==0.3.20"
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1` instead.
The first model load downloads the upstream checkpoint.

```python
import laya

agent = laya.load("convaiinnovations/laya-multilingual")

state = "আমার অ্যাকাউন্ট থেকে দুইবার টাকা কাটা হয়েছে। অতিরিক্ত টাকা ফেরত চাই।"
questions = {
    "department": {
        "type": "choice",
        "instructions": "এই অনুরোধটি কোন বিভাগে পাঠানো উচিত?",
        "criteria": {
            "billing": "বিল, পেমেন্ট ও টাকা ফেরত",
            "technical": "অ্যাপ বা সেবার প্রযুক্তিগত সমস্যা",
            "other": "অন্যান্য অনুরোধ",
        },
    }
}

result = agent.predict(state, questions)
print(result["answers"]["department"])
```

This is an illustrative SDK example, not a benchmark result. See the
[upstream documentation](https://nandhakishorm.github.io/laya/) for runtime and
hardware setup details. Project-specific training dependencies and commands
will be added with the implementation.

## Repository layout

```text
laya-bangla/
├── README.md                    # Project overview and upstream baseline example
├── CONTRIBUTING.md              # Contribution and failure-reporting guidance
├── AGENTS.md                    # Persistent project context for coding assistants
├── LICENSE                      # Apache License 2.0
├── NOTICE                       # Project and upstream attribution
├── .editorconfig                # Shared text formatting conventions
├── .gitattributes               # Consistent text line endings
├── .gitignore                   # Local environments, data, weights, and runs
└── docs/
    ├── roadmap.md               # Fine-tuning and release milestones
    └── model-card-template.md   # Starting point for the future Hugging Face card
```

Local datasets, model weights, and experiment outputs belong in the ignored
directories documented in [CONTRIBUTING.md](CONTRIBUTING.md). Code, small
reviewable examples, configurations, and curated evaluation summaries belong
in Git; the planned model release belongs on Hugging Face.

## Contributing

Useful early contributions include Bangla failure cases, dataset suggestions,
annotation guidance, and evaluation ideas. Please read
[CONTRIBUTING.md](CONTRIBUTING.md) before opening an
[issue](https://github.com/shamspias/laya-bangla/issues) or pull request.

## License and acknowledgments

Original code and documentation in this repository are licensed under
[Apache 2.0](LICENSE). See [NOTICE](NOTICE) for attribution.

This is an independent adaptation project based on Laya by **Convai Innovations**.
The upstream Laya model cards and source repository declare Apache 2.0 licensing.
Datasets and other third-party materials retain their own licenses; the eventual
model card will document the released checkpoint's provenance and license.

- [Upstream Laya source](https://github.com/NandhaKishorM/laya)
- [Laya model family](https://huggingface.co/convaiinnovations/laya)
- [Multilingual checkpoint](https://huggingface.co/convaiinnovations/laya-multilingual)
- [Upstream fine-tuning notebook](https://github.com/NandhaKishorM/laya/blob/main/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb)
