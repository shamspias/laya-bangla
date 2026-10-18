"""Data preparation: normalization, splits, leakage, and row rendering without downloads."""

import json
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

import pytest

from laya_bangla import prepare as p

ROOT = Path(__file__).resolve().parents[1]


def test_normalization_and_leak_key_ignore_spacing_case_and_punctuation():
    assert p.normalize_text("  আমার  বিল \n ভুল  ") == "আমার বিল ভুল"
    assert p.leak_key("Bill bhul, ASAP!!") == p.leak_key("bill  bhul asap")
    # Decomposed and precomposed forms of the same Bangla letter normalize together.
    assert p.leak_key("ড়") == p.leak_key("ড়")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("আজই টাকা ফেরত দিন", "bangla"),
        ("Ajkei taka ferot din", "banglish"),
        ("Invoice-এ ভুল আছে, tomorrow ঠিক করুন", "mixed"),
        ("আমার OTP আসছে না, সাহায্য করুন", "bangla"),
        ("Sir class ta khub helpful chilo ধন্যবাদ", "banglish"),
        ("গত মাসে double charge হয়েছে। Extra টাকাটা refund দিন।", "mixed"),
    ],
)
def test_script_slice(text, expected):
    assert p.script_slice(text) == expected


def test_hash_split_is_stable_and_roughly_proportional():
    fractions = {"train": 0.8, "validation": 0.1, "test": 0.1}
    first = [p.hash_split(f"group-{i}", 7, fractions) for i in range(5000)]
    assert first == [p.hash_split(f"group-{i}", 7, fractions) for i in range(5000)]
    assert 0.77 < first.count("train") / 5000 < 0.83
    assert first != [p.hash_split(f"group-{i}", 8, fractions) for i in range(5000)]


def write_xlsx(path, rows):
    """A minimal workbook with shared strings, as Excel writes them."""
    strings = sorted({cell for row in rows for cell in row})
    index = {s: i for i, s in enumerate(strings)}
    shared = "".join(f"<si><t>{escape(s)}</t></si>" for s in strings)
    sheet_rows = "".join(
        f'<row r="{r}">'
        + "".join(
            f'<c r="{chr(65 + c)}{r}" t="s"><v>{index[value]}</v></c>'
            for c, value in enumerate(row)
        )
        + "</row>"
        for r, row in enumerate(rows, 1)
    )
    ns = 'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'
    with zipfile.ZipFile(path, "w") as book:
        book.writestr("xl/sharedStrings.xml", f"<sst {ns}>{shared}</sst>")
        book.writestr(
            "xl/worksheets/sheet1.xml",
            f"<worksheet {ns}><sheetData>{sheet_rows}</sheetData></worksheet>",
        )


def test_banglapsg_loader_cleans_labels_and_skips_unlabelled(tmp_path):
    path = tmp_path / "psg.xlsx"
    write_xlsx(
        path,
        [
            ["No.", "Grievance Text", "Category", "Severity"],
            ["1", "রাস্তায় বড় গর্ত", " Road &  Transport", "High "],
            ["2", "ময়লা জমে আছে", "Waste  ", "medium "],
            ["3", "পানি নেই", "Water", " "],
        ],
    )
    rows = list(p.load_banglapsg({"all": path}, 1, {"train": 1.0, "validation": 0, "test": 0}))
    assert [r["labels"] for r in rows] == [
        {"grievance_category": "road_transport", "grievance_severity": 2},
        {"grievance_category": "waste", "grievance_severity": 1},
    ]
    assert {r["split"] for r in rows} == {"train"}


def test_banth_loader_rejects_unexpected_flags(tmp_path):
    path = tmp_path / "train.csv"
    path.write_text(
        "Text,Label,Personal Offense,Abusive/Violence\nvalo laglo,0,0,0\nbaje,maybe,0,0\n",
        encoding="utf-8",
    )
    loader = p.load_banth({"train": path}, 1, {})
    assert next(loader)["labels"] == {"hate": False, "personal_attack": False, "abusive": False}
    with pytest.raises(ValueError, match="0/1"):
        next(loader)


def test_sentnob_labels_map_to_negative_neutral_positive(tmp_path):
    path = tmp_path / "Test.csv"
    path.write_text("Data,Label\nভালো,1\nখারাপ,2\nজানি না,0\n", encoding="utf-8")
    rows = list(p.load_sentnob({"test": path}, 1, {}))
    assert [r["labels"]["sentiment"] for r in rows] == [2, 0, 1]


def item(rid, split, state, labels, source="s"):
    return p.record(source, split, rid, rid, state, labels)


def test_clean_drops_duplicates_conflicts_and_leakage_then_caps():
    records = [
        item("a", "train", "একই কথা", {"hate": True}),
        item("b", "train", "একই কথা!", {"hate": True}),  # duplicate of a
        item("c", "train", "দ্বন্দ্ব", {"hate": True}),
        item("d", "train", "দ্বন্দ্ব", {"hate": False}),  # conflicting label
        item("e", "train", "পরীক্ষার লেখা", {"hate": False}),  # leaks into test
        item("f", "test", "পরীক্ষার লেখা", {"hate": False}, source="other"),
        item("g", "validation", "যাচাই", {"hate": True}),
        item("h", "test", "যাচাই", {"hate": True}),  # validation leaks into test
        *(item(f"n{i}", "train", f"নতুন {i}", {"hate": False}) for i in range(5)),
    ]
    kept, stats = p.clean(records, 3, {("s", "train"): 4})
    ids = {r["id"] for r in kept}
    assert "s-c" not in ids and "s-d" not in ids
    assert "s-e" not in ids and "s-g" not in ids
    assert stats["duplicates"] == 1
    assert stats["conflicting_labels"] == 2
    assert stats["train_leak_removed"] == 1
    assert stats["validation_leak_removed"] == 1
    assert sum(r["split"] == "train" and r["source"] == "s" for r in kept) == 4
    assert {"other-f", "s-h"} <= ids


@pytest.fixture(scope="module")
def tasks():
    return p.load_tasks(ROOT / "configs/tasks.json")


def test_task_config_keeps_evaluation_wording_out_of_training(tasks):
    for task in tasks.values():
        for lang in p.LANGUAGES:
            assert task["instructions"]["eval"][lang] not in task["instructions"]["train"][lang]
    groups = tasks["massive_intent"]["groups"]
    intents = [intent for options in groups.values() for intent in options]
    assert len(intents) == len(set(intents)) == 58  # 60 MASSIVE intents minus news and weather
    assert set(groups) | {"news", "weather"} == set(tasks["massive_scenario"]["criteria"])
    assert all(
        intent.startswith(f"{scenario}_") for scenario in groups for intent in groups[scenario]
    )


def test_every_rendered_question_is_valid_for_laya(tasks):
    from laya.agent import Agent

    for name, task in tasks.items():
        option_sets = task["groups"].values() if "groups" in task else [task["criteria"]]
        for options in option_sets:
            for lang in p.LANGUAGES:
                question = p.render_question(
                    task, task["instructions"]["eval"][lang], lang, options
                )
                Agent._check_question(name, question)


def test_build_rows_uses_training_wording_for_train_and_both_eval_wordings_for_test(tasks):
    labels = {"massive_scenario": "alarm", "massive_intent": "alarm_set"}
    train = p.build_rows(
        [item("t", "train", "অ্যালার্ম দাও", labels)],
        tasks,
        {"s": ["massive_scenario", "massive_intent"]},
        5,
    )
    test = p.build_rows(
        [item("x", "test", "অ্যালার্ম দাও", labels)],
        tasks,
        {"s": ["massive_scenario", "massive_intent"]},
        5,
    )
    assert len(train) == 1 and len(test) == 2
    for row in train:
        for name, question in row["questions"].items():
            wordings = [
                w for lang in p.LANGUAGES for w in tasks[name]["instructions"]["train"][lang]
            ]
            assert question["instructions"] in wordings
    assert {row["instruction_language"] for row in test} == {"bn", "en"}
    for row in test:
        lang = row["instruction_language"]
        assert (
            row["questions"]["massive_scenario"]["instructions"]
            == tasks["massive_scenario"]["instructions"]["eval"][lang]
        )
        # Intent options are restricted to the record's scenario.
        assert set(row["questions"]["massive_intent"]["criteria"]) == {
            "alarm_query",
            "alarm_remove",
            "alarm_set",
        }
        assert row["expected"] == labels


def test_build_rows_skips_intent_for_single_intent_scenarios(tasks):
    labels = {"massive_scenario": "weather", "massive_intent": "weather_query"}
    rows = p.build_rows(
        [item("w", "train", "আজ বৃষ্টি হবে?", labels)],
        tasks,
        {"s": ["massive_scenario", "massive_intent"]},
        5,
    )
    assert list(rows[0]["questions"]) == ["massive_scenario"]


@pytest.mark.parametrize("name", ["data.toml", "data-english.toml"])
def test_data_config_sources_have_loaders_tasks_and_pins(tasks, name):
    config = p.load_config(ROOT / "configs" / name)
    for name, source in config["sources"].items():
        assert set(source["tasks"]) <= set(tasks), name
        assert source["use"] in ("train", "benchmark"), name
        if "repo" in source:
            assert len(source["revision"]) == 40, name
        else:
            assert len(source["sha256"]) == 64, name
    # License terms: no-derivatives data must never be trained on.
    assert all(s["use"] == "benchmark" for s in config["sources"].values() if "ND" in s["license"])


def test_prepare_end_to_end_with_local_files(tmp_path, monkeypatch, tasks):
    """Run the full pipeline on tiny local sources in place of downloads."""
    massive = tmp_path / "massive.json"
    massive.write_text(
        "\n".join(
            json.dumps({"id": i, "label": label, "text": text}, ensure_ascii=False)
            for i, (label, text) in enumerate(
                [
                    ("alarm_set", "সাতটায় জাগাও"),
                    ("weather_query", "বৃষ্টি হবে?"),
                    ("alarm_set", "সাতটায় জাগাও"),
                ]
            )
        ),
        encoding="utf-8",
    )
    config = tmp_path / "data.toml"
    config.write_text(
        f"""
tasks = "{(ROOT / "configs/tasks.json").as_posix()}"
output = "out"
cache = "cache"
seed = 1
hash_split = {{ train = 0.8, validation = 0.1, test = 0.1 }}
max_train_rows_per_task = 100
max_validation_rows_per_task = 100
[sources.massive]
repo = "example/massive"
revision = "{"a" * 40}"
files = {{ train = "train.json", test = "test.json" }}
license = "CC-BY-4.0"
use = "train"
tasks = ["massive_scenario", "massive_intent"]
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(p, "fetch", lambda source, filename, cache: massive)
    manifest = p.prepare(config)
    train = p.read_rows(tmp_path / "out/train.jsonl")
    test = p.read_rows(tmp_path / "out/test.jsonl")
    # The test copy of every state removes its training twin.
    assert train == []
    assert len(test) == 4
    assert manifest["cleaning"]["train_leak_removed"] == 2
    assert manifest["sources"]["massive"]["license"] == "CC-BY-4.0"
    assert manifest["splits"]["test"]["sha256"] == p.sha256_file(tmp_path / "out/test.jsonl")


def test_english_benchmark_is_test_only_and_marked_english():
    config = p.load_config(ROOT / "configs/data-english.toml")
    assert config["output"] != p.load_config(ROOT / "configs/data.toml")["output"]
    for name, source in config["sources"].items():
        assert source["use"] == "benchmark" and source["slice"] == "english", name


def test_shared_loader_renames_source_and_caps_test_rows(tmp_path, monkeypatch, tasks):
    massive = tmp_path / "en.json"
    massive.write_text(
        "\n".join(
            json.dumps({"id": i, "label": "alarm_set", "text": f"wake me at {i}"}) for i in range(5)
        ),
        encoding="utf-8",
    )
    config = tmp_path / "data.toml"
    config.write_text(
        f"""
tasks = "{(ROOT / "configs/tasks.json").as_posix()}"
output = "out"
cache = "cache"
seed = 1
hash_split = {{ train = 0.8, validation = 0.1, test = 0.1 }}
max_train_rows_per_task = 0
max_validation_rows_per_task = 0
[sources.massive_en]
loader = "massive"
repo = "example/massive"
revision = "{"a" * 40}"
files = {{ test = "test/en.json" }}
license = "CC-BY-4.0"
use = "benchmark"
slice = "english"
max_test_rows = 3
tasks = ["massive_scenario", "massive_intent"]
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(p, "fetch", lambda source, filename, cache: massive)
    manifest = p.prepare(config)
    test = p.read_rows(tmp_path / "out/test.jsonl")
    assert len(test) == 6  # 3 capped states, each asked with Bangla and English wording
    assert {row["source"] for row in test} == {"massive_en"}
    assert {row["slice"] for row in test} == {"english"}
    assert all(row["id"].startswith("massive_en-") for row in test)
    assert all(row["group"].startswith("massive_en:") for row in test)
    assert manifest["cleaning"]["capped_test"] == 2
    assert manifest["sources"]["massive_en"]["loader"] == "massive"


def test_xnli_en_drops_neutral_pairs(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = tmp_path / "xnli.parquet"
    rows = {"premise": ["A", "B", "C"], "hypothesis": ["a", "b", "c"], "label": [0, 1, 2]}
    pq.write_table(pa.table(rows), path)
    records = list(p.load_xnli_en({"test": path}, 1, {}))
    assert [r["labels"]["entailment"] for r in records] == [True, False]
    # Like IndicXNLI, the pair is one line after whitespace normalization.
    assert records[0]["state"] == "Sentence 1: A Sentence 2: a"
