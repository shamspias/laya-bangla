"""Download pinned Bangla sources and build leak-free typed-decision splits."""

import csv
import gzip
import hashlib
import io
import json
import re
import tomllib
import unicodedata
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

from .data import parse_json

SPLITS = ("train", "validation", "test")
LANGUAGES = ("bn", "en")
BENGALI = re.compile(r"[ঀ-৿]")
LATIN = re.compile(r"[A-Za-z]")
# Grievance labels in BanglaPSG carry stray spaces and mixed case.
GRIEVANCE_CATEGORIES = {
    "food": "food",
    "road & transport": "road_transport",
    "water": "water",
    "waste": "waste",
    "electricity": "electricity",
    "healthcare": "healthcare",
}
LEVELS = {"low": 0, "medium": 1, "high": 2}
SENTIMENT = {"negative": 0, "neutral": 1, "positive": 2}
# SentNoB: 0 neutral, 1 positive, 2 negative (dataset card).
SENTNOB = {0: 1, 1: 2, 2: 0}
# TweetEval sentiment: 0 negative, 1 neutral, 2 positive (dataset card).
TWEETEVAL_SENTIMENT = {0: 0, 1: 1, 2: 2}


def normalize_text(text: str) -> str:
    """NFC with collapsed whitespace; keeps ZWJ/ZWNJ, which Bangla spelling uses."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", str(text))).strip()


def leak_key(text: str) -> str:
    """A loose identity for duplicate and leakage checks: letters and digits only."""
    return "".join(ch for ch in normalize_text(text).casefold() if ch.isalnum())


def script_slice(text: str) -> str:
    """Bangla script, romanized Banglish, or code-mixed, by counting words in each script.

    A lone loanword such as "OTP" in a Bangla sentence does not make it mixed; at least
    two words and 15% of the words must be in the minority script.
    """
    words = text.split()
    latin = sum(bool(LATIN.search(w)) for w in words)
    bengali = sum(bool(BENGALI.search(w)) for w in words)
    if not latin:
        return "bangla"
    if not bengali:
        return "banglish"
    minority = min(latin, bengali)
    if minority >= 2 and minority / (latin + bengali) >= 0.15:
        return "mixed"
    return "bangla" if bengali > latin else "banglish"


def stable_unit(*parts) -> float:
    digest = hashlib.sha256("\x1f".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def hash_split(group: str, seed: int, fractions: dict) -> str:
    value, edge = stable_unit("split", seed, group), 0.0
    for name in SPLITS:
        edge += fractions[name]
        if value < edge:
            return name
    return SPLITS[-1]


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fetch(source: dict, filename: str, cache: Path) -> Path:
    if "repo" in source:
        from huggingface_hub import hf_hub_download

        return Path(
            hf_hub_download(
                source["repo"],
                filename,
                repo_type="dataset",
                revision=source["revision"],
                cache_dir=str(cache / "hub"),
            )
        )
    path = cache / "files" / source["filename"]
    if not path.is_file() or sha256_file(path) != source["sha256"]:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Mendeley rejects urllib's default User-Agent with 403.
        request = urllib.request.Request(source["url"], headers={"User-Agent": "laya-bangla"})
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = response.read()
        if hashlib.sha256(payload).hexdigest() != source["sha256"]:
            raise ValueError(f"{source['filename']}: downloaded file does not match sha256")
        path.write_bytes(payload)
    return path


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def read_json_lines(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def read_parquet(path: Path) -> list[dict]:
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pylist()


def read_xlsx(path: Path) -> list[dict]:
    """Rows of the first sheet keyed by header, via the stdlib (no openpyxl)."""
    import zipfile
    from xml.etree import ElementTree

    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as book:
        shared = []
        if "xl/sharedStrings.xml" in book.namelist():
            root = ElementTree.fromstring(book.read("xl/sharedStrings.xml"))
            for item in root.findall("m:si", ns):
                shared.append("".join(node.text or "" for node in item.iter(f"{{{ns['m']}}}t")))
        sheet = ElementTree.fromstring(book.read("xl/worksheets/sheet1.xml"))
    table = []
    for row in sheet.find("m:sheetData", ns).findall("m:row", ns):
        values = {}
        for cell in row.findall("m:c", ns):
            column = re.match(r"[A-Z]+", cell.get("r")).group()
            value = cell.find("m:v", ns)
            if cell.get("t") == "s" and value is not None:
                values[column] = shared[int(value.text)]
            elif cell.get("t") == "inlineStr":
                values[column] = "".join(t.text or "" for t in cell.iter(f"{{{ns['m']}}}t"))
            else:
                values[column] = value.text if value is not None else ""
        table.append(values)
    header = table[0]
    return [{header[col]: row.get(col, "") for col in header} for row in table[1:]]


def record(source, split, rid, group, state, labels) -> dict:
    state = normalize_text(state)
    return {
        "id": f"{source}-{rid}",
        "group": f"{source}:{group}",
        "source": source,
        "split": split,
        "slice": script_slice(state),
        "state": state,
        "labels": labels,
    }


def load_massive(paths, seed, fractions):
    for split, path in paths.items():
        for row in read_json_lines(path):
            intent = row["label"]
            labels = {"massive_scenario": intent.split("_")[0], "massive_intent": intent}
            yield record("massive", split, f"{split}-{row['id']}", row["id"], row["text"], labels)


def load_sib200(paths, seed, fractions):
    for split, path in paths.items():
        for row in read_parquet(path):
            labels = {"sib_topic": row["category"]}
            yield record("sib200", split, row["index_id"], row["index_id"], row["text"], labels)


def load_banglapsg(paths, seed, fractions):
    for row in read_xlsx(paths["all"]):
        text = row["Grievance Text"]
        # A handful of rows lack a category or severity; they cannot be labelled.
        if not normalize_text(text) or not row["Category"].strip() or not row["Severity"].strip():
            continue
        category = GRIEVANCE_CATEGORIES[" ".join(row["Category"].split()).lower()]
        severity = LEVELS[row["Severity"].strip().lower()]
        group = leak_key(text)
        labels = {"grievance_category": category, "grievance_severity": severity}
        split = hash_split(group, seed, fractions)
        yield record("banglapsg", split, row["No."], group, text, labels)


def flag(value: str) -> bool:
    if value.strip() not in ("0", "1"):
        raise ValueError(f"expected a 0/1 label, got {value!r}")
    return value.strip() == "1"


def load_banth(paths, seed, fractions):
    for split, path in paths.items():
        for index, row in enumerate(read_csv(path)):
            labels = {
                "hate": flag(row["Label"]),
                "personal_attack": flag(row["Personal Offense"]),
                "abusive": flag(row["Abusive/Violence"]),
            }
            yield record(
                "banth", split, f"{split}-{index}", f"{split}-{index}", row["Text"], labels
            )


def load_banhate(paths, seed, fractions):
    rows = parse_json(paths["all"].read_text(encoding="utf-8"), str(paths["all"]))
    for row in rows:
        # Comments under one video share context, so the video decides the split.
        group = row["Video URL"]
        split = hash_split(group, seed, fractions)
        labels = {"hate": row["H/NH"].strip() == "H"}
        yield record("banhate", split, row["sample"], group, row["Comment"], labels)


def load_elearning(paths, seed, fractions):
    for name, path in paths.items():
        for index, row in enumerate(read_csv(path)):
            if not row["label"].strip():
                continue  # One romanized row has neither text nor label.
            # Romanized rows carry their Bangla original: keep each pair in one split.
            original = row.get("bengali_text") or row["text"]
            group = leak_key(original)
            split = hash_split(group, seed, fractions)
            labels = {"sentiment": SENTIMENT[row["label"].strip().lower()]}
            yield record("elearning", split, f"{name}-{index}", group, row["text"], labels)


def load_sentnob(paths, seed, fractions):
    for index, row in enumerate(read_csv(paths["test"])):
        labels = {"sentiment": SENTNOB[int(row["Label"])]}
        yield record("sentnob", "test", index, index, row["Data"], labels)


def load_indicsentiment(paths, seed, fractions):
    for index, row in enumerate(read_json_lines(paths["test"])):
        label = (row.get("LABEL") or "").strip().lower()
        if label not in ("positive", "negative") or not (row.get("INDIC REVIEW") or "").strip():
            continue
        labels = {"sentiment": SENTIMENT[label]}
        yield record("indicsentiment", "test", index, index, row["INDIC REVIEW"], labels)


def load_indicsentiment_en(paths, seed, fractions):
    """The English originals of the IndicSentiment reviews: a parallel English test."""
    for index, row in enumerate(read_json_lines(paths["test"])):
        label = (row.get("LABEL") or "").strip().lower()
        if label not in ("positive", "negative") or not (row.get("ENGLISH REVIEW") or "").strip():
            continue
        labels = {"sentiment": SENTIMENT[label]}
        yield record("indicsentiment_en", "test", index, index, row["ENGLISH REVIEW"], labels)


def load_xnli_en(paths, seed, fractions):
    """English XNLI, the source IndicXNLI was translated from, as entailed/contradicted pairs."""
    for index, row in enumerate(read_parquet(paths["test"])):
        if row["label"] == 1:
            continue  # IndicXNLI's pair classification drops neutral pairs too.
        state = f"Sentence 1: {row['premise']}\nSentence 2: {row['hypothesis']}"
        labels = {"entailment": row["label"] == 0}
        yield record("xnli_en", "test", index, leak_key(row["premise"]), state, labels)


def load_tweeteval_hate(paths, seed, fractions):
    for index, row in enumerate(read_parquet(paths["test"])):
        yield record(
            "tweeteval_hate", "test", index, index, row["text"], {"hate": row["label"] == 1}
        )


def load_tweeteval_sentiment(paths, seed, fractions):
    for index, row in enumerate(read_parquet(paths["test"])):
        labels = {"sentiment": TWEETEVAL_SENTIMENT[row["label"]]}
        yield record("tweeteval_sentiment", "test", index, index, row["text"], labels)


def load_indicxnli(paths, seed, fractions):
    for index, row in enumerate(read_parquet(paths["test"])):
        state = f"বাক্য ১: {row['sentence1']}\nবাক্য ২: {row['sentence2']}"
        labels = {"entailment": row["labels"] == 1}
        # Each premise appears with an entailed and a contradicting hypothesis.
        yield record("indicxnli", "test", index, leak_key(row["sentence1"]), state, labels)


LOADERS = {
    "massive": load_massive,
    "sib200": load_sib200,
    "banglapsg": load_banglapsg,
    "banth": load_banth,
    "banhate": load_banhate,
    "elearning": load_elearning,
    "sentnob": load_sentnob,
    "indicsentiment": load_indicsentiment,
    "indicxnli": load_indicxnli,
    "indicsentiment_en": load_indicsentiment_en,
    "xnli_en": load_xnli_en,
    "tweeteval_hate": load_tweeteval_hate,
    "tweeteval_sentiment": load_tweeteval_sentiment,
}


def renamed(item: dict, old: str, new: str) -> dict:
    return dict(
        item,
        source=new,
        id=new + item["id"].removeprefix(old),
        group=new + item["group"].removeprefix(old),
    )


def load_config(path: Path) -> dict:
    config = tomllib.loads(path.read_text(encoding="utf-8"))
    for key in ("tasks", "output", "cache"):
        config[key] = (path.parent / config[key]).resolve()
    if abs(sum(config["hash_split"].values()) - 1) > 1e-9 or set(config["hash_split"]) != set(
        SPLITS
    ):
        raise ValueError(
            "hash_split must assign train, validation, and test fractions summing to 1"
        )
    unknown = {source.get("loader", name) for name, source in config["sources"].items()}
    unknown -= set(LOADERS)
    if unknown:
        raise ValueError(f"No loader for sources: {sorted(unknown)}")
    return config


def load_tasks(path: Path) -> dict:
    tasks = parse_json(path.read_text(encoding="utf-8"), str(path))
    for name, task in tasks.items():
        if task.get("type") not in ("choice", "noul", "score"):
            raise ValueError(f"{name}: unsupported task type")
        for lang in LANGUAGES:
            if not task["instructions"]["train"][lang] or not task["instructions"]["eval"][lang]:
                raise ValueError(f"{name}: missing {lang} instructions")
            if task["instructions"]["eval"][lang] in task["instructions"]["train"][lang]:
                raise ValueError(f"{name}: evaluation wording must not appear in training")
    return tasks


def label_options(task: dict, labels: dict) -> dict | list | None:
    """The option set for one record: a scenario's intents, or the task's own criteria."""
    if "conditioned_on" in task:
        options = task["groups"].get(labels[task["conditioned_on"]])
        return options if options and len(options) >= 2 else None
    return task["criteria"]


def render_question(task: dict, instructions: str, lang: str, options) -> dict:
    if task["type"] == "score":
        criteria = [option[lang] for option in options]
    else:
        criteria = {label: option[lang] for label, option in options.items()}
    return {"type": task["type"], "instructions": instructions, "criteria": criteria}


def build_rows(records: list[dict], tasks: dict, source_tasks: dict, seed: int) -> list[dict]:
    """Training/validation rows draw one known wording per question; test rows use unseen wording."""
    rows = []
    for item in records:
        names = [name for name in source_tasks[item["source"]] if name in item["labels"]]
        for lang in LANGUAGES if item["split"] == "test" else (None,):
            questions, expected, languages = {}, {}, set()
            for name in names:
                task = tasks[name]
                options = label_options(task, item["labels"])
                if options is None:
                    continue
                if lang is None:
                    choices = [
                        (code, text)
                        for code in LANGUAGES
                        for text in task["instructions"]["train"][code]
                    ]
                    pick = int(stable_unit("wording", seed, item["id"], name) * len(choices))
                    code, instructions = choices[pick]
                else:
                    code, instructions = lang, task["instructions"]["eval"][lang]
                questions[name] = render_question(task, instructions, code, options)
                expected[name] = item["labels"][name]
                languages.add(code)
            if questions:
                rows.append(
                    {
                        "id": item["id"] if lang is None else f"{item['id']}-{lang}",
                        "record_id": item["id"],
                        "group": item["group"],
                        "source": item["source"],
                        "split": item["split"],
                        "slice": item["slice"],
                        "instruction_language": lang
                        or (languages.pop() if len(languages) == 1 else "mixed"),
                        "state": item["state"],
                        "questions": questions,
                        "expected": expected,
                    }
                )
    return rows


def clean(records: list[dict], seed: int, caps: dict) -> tuple[list[dict], dict]:
    """Drop empty, duplicate, conflicting, and leaked states; apply per-source caps."""
    stats = Counter()
    by_key = defaultdict(list)
    for item in records:
        key = leak_key(item["state"])
        if not key:
            stats["empty"] += 1
            continue
        by_key[(item["source"], item["split"], key)].append(item)
    kept = []
    for items in by_key.values():
        if len({json.dumps(i["labels"], sort_keys=True) for i in items}) > 1:
            stats["conflicting_labels"] += len(items)
            continue
        stats["duplicates"] += len(items) - 1
        kept.append(items[0])
    # A state seen in any evaluation split must not be trained or calibrated on.
    held = {
        split: {leak_key(i["state"]) for i in kept if i["split"] == split}
        for split in ("validation", "test")
    }
    result = []
    for item in kept:
        key = leak_key(item["state"])
        if item["split"] == "train" and (key in held["test"] or key in held["validation"]):
            stats["train_leak_removed"] += 1
        elif item["split"] == "validation" and key in held["test"]:
            stats["validation_leak_removed"] += 1
        else:
            result.append(item)
    groups = defaultdict(list)
    for item in result:
        groups[(item["source"], item["split"])].append(item)
    capped = []
    for key, items in sorted(groups.items()):
        limit = caps.get(key)
        if limit and len(items) > limit:
            stats[f"capped_{key[1]}"] += len(items) - limit
            items = sorted(items, key=lambda i: stable_unit("cap", seed, i["id"]))[:limit]
        capped.extend(items)
    return capped, dict(stats)


def prepare(config_path: Path) -> dict:
    config = load_config(config_path)
    tasks = load_tasks(config["tasks"])
    records, provenance, source_tasks, caps = [], {}, {}, {}
    for name, source in config["sources"].items():
        missing = set(source["tasks"]) - set(tasks)
        if missing:
            raise ValueError(f"{name}: unknown tasks {sorted(missing)}")
        files = source.get("files") or {"all": source["filename"]}
        paths = {
            split: fetch(source, filename, config["cache"]) for split, filename in files.items()
        }
        loader = source.get("loader", name)
        loaded = list(LOADERS[loader](paths, config["seed"], config["hash_split"]))
        if loader != name:  # One loader can read another language of the same dataset.
            loaded = [renamed(item, loader, name) for item in loaded]
        if source["use"] == "benchmark":
            loaded = [dict(item, split="test") for item in loaded]
        if "slice" in source:  # e.g. English text, which the script heuristic calls Banglish
            loaded = [dict(item, slice=source["slice"]) for item in loaded]
        records.extend(loaded)
        source_tasks[name] = source["tasks"]
        caps[(name, "train")] = source.get("max_train_rows", config["max_train_rows_per_task"])
        caps[(name, "validation")] = config["max_validation_rows_per_task"]
        caps[(name, "test")] = source.get("max_test_rows")
        provenance[name] = {
            key: source[key]
            for key in (
                "repo",
                "revision",
                "url",
                "doi",
                "sha256",
                "license",
                "use",
                "tasks",
                "loader",
            )
            if key in source
        }
        provenance[name]["files_sha256"] = {
            files[split]: sha256_file(path) for split, path in paths.items()
        }
    records, stats = clean(records, config["seed"], caps)
    rows = build_rows(records, tasks, source_tasks, config["seed"])
    output = config["output"]
    output.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split in SPLITS:
        selected = [row for row in rows if row["split"] == split]
        path = output / f"{split}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as stream:
            for row in selected:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        counts[split] = {
            "rows": len(selected),
            "sha256": sha256_file(path),
            "by_source": dict(Counter(row["source"] for row in selected)),
            "by_slice": dict(Counter(row["slice"] for row in selected)),
            "questions_by_task": dict(Counter(q for row in selected for q in row["questions"])),
        }
    manifest = {
        "config_sha256": sha256_file(config_path),
        "tasks_sha256": sha256_file(config["tasks"]),
        "seed": config["seed"],
        "sources": provenance,
        "cleaning": stats,
        "splits": counts,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def describe(manifest: dict) -> str:
    out = io.StringIO()
    for split, info in manifest["splits"].items():
        print(f"{split}: {info['rows']} rows {info['by_slice']}", file=out)
        for task, n in sorted(info["questions_by_task"].items()):
            print(f"  {task}: {n}", file=out)
    print(f"cleaning: {manifest['cleaning']}", file=out)
    return out.getvalue()
