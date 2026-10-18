# Datasets

`laya-bangla prepare` builds the training, validation, and test splits from
real, published Bangla datasets. Nothing in these splits is synthetic. The
36-message smoke set in `examples/` remains a separate diagnostic and is never
trained on.

Every source is pinned in [`configs/data.toml`](../configs/data.toml) to an
immutable Hugging Face revision or a SHA-256 content hash. `data/processed/manifest.json`
records the hash of every downloaded file and every generated split.

```bash
uv run --frozen laya-bangla prepare      # writes data/processed/{train,validation,test}.jsonl
```

## Sources

| Source | Text | Size used | Typed decisions | Splits | License | Use |
| --- | --- | --- | --- | --- | --- | --- |
| [MASSIVE](https://huggingface.co/datasets/mteb/amazon_massive_intent) `bn-BD` | Voice-assistant requests, Bangla script | 16.5k | `massive_scenario` (18-way choice), `massive_intent` (2–9-way choice within the scenario) | Official | CC BY 4.0 | Train + test |
| [SIB-200](https://huggingface.co/datasets/mteb/sib200) `ben_Beng` | FLORES sentences | 1k | `sib_topic` (7-way choice) | Official | CC BY-SA 4.0 | Train + test |
| [BanglaPSG](https://data.mendeley.com/datasets/kcr8zfmtvb) | Citizen grievances from social media | 5.3k | `grievance_category` (6-way choice), `grievance_severity` (3-level score) | Hash of text | CC BY 4.0 | Train + test |
| [BanTH](https://huggingface.co/datasets/aplycaebous/BanTH) | YouTube comments, **transliterated (Banglish)** | 37k | `hate`, `personal_attack`, `abusive` (yes/no) | Official | MIT | Train + test |
| [BanHate](https://huggingface.co/datasets/aplycaebous/BanHate) | YouTube comments, Bangla script (2024–25) | 19k | `hate` (yes/no) | Hash of video URL | MIT | Train + test |
| [Bangla Sentiments in eLearning](https://huggingface.co/datasets/TanjimKIT/Bangla_Sentiments_in_eLearning) | Course comments, **Bangla and romanized pairs** | 6.3k | `sentiment` (3-level score) | Hash of the Bangla text | Apache-2.0 | Train + test |
| [SentNoB](https://huggingface.co/datasets/khondoker/SentNoB) | Noisy social-media comments | 1.6k test | `sentiment` | Official test | CC BY-ND 4.0 | **Benchmark only** |
| [IndicSentiment](https://huggingface.co/datasets/mteb/IndicSentiment) `bn` | Product reviews | 1k test | `sentiment` | Official test | CC0 1.0 | **Benchmark only** |
| [IndicXNLI](https://huggingface.co/datasets/mteb/IndicXnliPairClassification) `bn` | Translated NLI pairs | 3.3k test | `entailment` (yes/no) | Official test | CC BY 4.0 | **Benchmark only** |

The benchmark-only sources test transfer to domains and a task the model never
trains on. SentNoB's no-derivatives license also rules out training on it.

## Questions and wording

[`configs/tasks.json`](../configs/tasks.json) defines each task's typed question,
with Bangla and English criteria. Training and validation rows draw one of three
training wordings per question (two Bangla, one English). **Test rows are asked
twice, once with a Bangla and once with an English wording that never appears
in training**, so benchmark scores include robustness to rephrased questions,
the failure the baseline smoke run exposed.

MASSIVE intents are routed in two stages, as a deployed assistant would be:
first the scenario, then the intent among that scenario's options. Test rows ask
the intent question within the gold scenario, and the benchmark also reports
end-to-end intent accuracy using the model's own predicted scenario.

## Cleaning and splits

`laya-bangla prepare` applies these steps, counted in the manifest:

1. Unicode NFC normalization and whitespace collapsing (ZWJ/ZWNJ preserved).
2. Rows without a usable label are skipped: 4 BanglaPSG rows and 1 blank
   eLearning row. Malformed BanTH flags fail loudly instead of reading as `false`.
3. Within each source and split, exact duplicates (ignoring case, spacing, and
   punctuation) are merged; duplicates with **conflicting labels are dropped**.
4. Sources without official splits are split 80/10/10 by a stable hash of a group
   key: the text for BanglaPSG, the **video** for BanHate (comments on one video
   share context), and the **Bangla original** for eLearning (a romanized comment
   and its Bangla source stay together).
5. **Leakage removal across all sources:** a training row whose text appears in
   any validation or test split is removed, as is a validation row whose text
   appears in any test split.
6. Training caps: 9,000 rows each for BanTH and BanHate, 12,000 otherwise; 500
   validation rows per source. The test splits are not capped.

Each row records its `slice`: `bangla`, `banglish` (romanized), or `mixed`, by
counting words in each script. A single English loanword does not make a
sentence mixed.

## Prepared data (seed 20261018)

| Split | Rows | Bangla script | Banglish | Mixed |
| --- | ---: | ---: | ---: | ---: |
| train | 38,766 | 26,880 | 11,481 | 405 |
| validation | 2,597 | 1,805 | 769 | 23 |
| test | 33,162 (16,581 states × 2 wordings) | 24,864 | 8,122 | 176 |

Test counts are rows: each state appears once per wording language.

Cleaning removed 825 duplicates, 352 rows with conflicting duplicate labels,
298 training rows that leaked into a validation or test split, and 44
validation rows that leaked into a test split.

Code-mixed text is under-represented in every public source above; the smoke
set and the playground remain the main checks for it.

## English benchmark

Fine-tuning on Bangla can cost a model some of its English and general ability.
[`configs/data-english.toml`](../configs/data-english.toml) builds English test sets for the
same typed decisions, used only for benchmarking. They are written to
`data/processed-english/`, so the Bangla splits above stay byte-identical.

```bash
uv run --frozen laya-bangla prepare --config configs/data-english.toml
uv run --frozen laya-bangla benchmark --model base --data data/processed-english/test.jsonl --output outputs/bench-en-base
```

| Source | Text | Size | Typed decisions | License |
| --- | --- | ---: | --- | --- |
| [MASSIVE](https://huggingface.co/datasets/mteb/amazon_massive_intent) `en` | Voice-assistant requests, the English side of the Bangla test | 2,970 | scenario, intent | CC BY 4.0 |
| [SIB-200](https://huggingface.co/datasets/mteb/sib200) `eng_Latn` | The English sentences behind the Bangla SIB-200 test | 204 | topic | CC BY-SA 4.0 |
| [IndicSentiment](https://huggingface.co/datasets/mteb/IndicSentiment) English reviews | The English originals of the Bangla test reviews | 996 | sentiment | CC0 1.0 |
| [XNLI](https://huggingface.co/datasets/facebook/xnli) `en` | The English test IndicXNLI was translated from (neutral pairs dropped) | 3,340 | entailment | CC BY-NC 4.0 |
| [TweetEval](https://huggingface.co/datasets/cardiffnlp/tweet_eval) `hate` | English tweets (HatEval 2019) | 2,966 | hate | See TweetEval |
| [TweetEval](https://huggingface.co/datasets/cardiffnlp/tweet_eval) `sentiment` | English tweets (SemEval 2017), stable 3,000-tweet sample | 3,000 | sentiment | See TweetEval |

Four of these are parallel to Bangla tests (MASSIVE, SIB-200, IndicSentiment, and XNLI pairs
are the same items in English), so the gap between the two languages is directly comparable.
Every item is asked with the Bangla and the English test wording, as in the Bangla benchmark:
13,476 items, 26,952 rows.

## Licensing notes

The fine-tuned checkpoint is trained on CC BY 4.0, CC BY-SA 4.0, MIT, and
Apache-2.0 data. The model card credits every source. The ShareAlike
condition of SIB-200 applies to adapted versions of the dataset itself; this
repository does not redistribute any dataset text: `prepare` downloads it from
the original publishers.
