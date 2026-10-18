"""Benchmark batching, two-stage routing, and reports with a fake agent."""

import json
from pathlib import Path

import pytest

from laya_bangla import benchmark as b
from laya_bangla.prepare import load_tasks

ROOT = Path(__file__).resolve().parents[1]


class FakeAgent:
    """Answers every question with its first option (or false/level 0), recording calls."""

    def __init__(self, scenario="alarm"):
        self.calls = []
        self.scenario = scenario

    def answer(self, qid, question):
        kind = question["type"]
        if kind == "noul":
            return {"type": "noul", "noul": 0.25}
        if kind == "score":
            k = len(question["criteria"])
            probs = {str(i): (1.0 if i == 0 else 0.0) for i in range(k)}
            return {"type": "score", "score": 0.0, "probabilities": probs}
        labels = list(question["criteria"])
        choice = self.scenario if qid == "massive_scenario" else labels[0]
        probs = {label: (1.0 if label == choice else 0.0) for label in labels}
        return {"type": "choice", "choice": choice, "probabilities": probs}

    def predict_batch(self, states, questions, batch_size=None):
        self.calls.append((len(states), sorted(questions)))
        return [
            {"answers": {qid: self.answer(qid, q) for qid, q in questions.items()}} for _ in states
        ]

    def predict(self, state, questions):
        return self.predict_batch([state], questions)[0]


@pytest.fixture(scope="module")
def tasks():
    return load_tasks(ROOT / "configs/tasks.json")


def massive_row(rid, scenario, intent, lang, tasks):
    from laya_bangla.prepare import render_question

    eval_words = {name: tasks[name]["instructions"]["eval"][lang] for name in tasks}
    return {
        "id": rid,
        "source": "massive",
        "instruction_language": lang,
        "slice": "bangla",
        "state": f"state {rid}",
        "questions": {
            "massive_scenario": render_question(
                tasks["massive_scenario"],
                eval_words["massive_scenario"],
                lang,
                tasks["massive_scenario"]["criteria"],
            ),
            "massive_intent": render_question(
                tasks["massive_intent"],
                eval_words["massive_intent"],
                lang,
                tasks["massive_intent"]["groups"][scenario],
            ),
        },
        "expected": {"massive_scenario": scenario, "massive_intent": intent},
    }


def test_predict_rows_batches_identical_schemas_and_keeps_order(tasks):
    rows = [
        massive_row("a", "alarm", "alarm_query", "bn", tasks),
        massive_row("b", "audio", "audio_volume_up", "bn", tasks),
        massive_row("c", "alarm", "alarm_set", "bn", tasks),
    ]
    agent = FakeAgent()
    answers = b.predict_rows(agent, rows, batch_size=8)
    assert len(agent.calls) == 2  # alarm rows share a schema; audio differs.
    assert answers[1]["massive_intent"]["choice"] == "audio_volume_down"
    assert answers[2]["massive_intent"]["choice"] == "alarm_query"


def test_end_to_end_routing_uses_the_predicted_scenario(tasks):
    rows = [
        massive_row("a", "alarm", "alarm_query", "bn", tasks),
        massive_row("b", "audio", "audio_volume_down", "en", tasks),
    ]
    agent = FakeAgent(scenario="alarm")
    answers = b.predict_rows(agent, rows, 8)
    records = b.end_to_end_intents(agent, rows, answers, tasks, 8)
    by_id = {r["id"]: r for r in records}
    # Correct scenario and first alarm intent: right. Wrong scenario: wrong intent.
    assert by_id["a"]["correct"] is True
    assert by_id["b"]["predicted"] == "alarm_query" and by_id["b"]["correct"] is False


def test_end_to_end_routing_handles_single_intent_scenarios(tasks):
    rows = [massive_row("w", "alarm", "weather_query", "bn", tasks)]
    agent = FakeAgent(scenario="weather")
    answers = b.predict_rows(agent, rows, 8)
    records = b.end_to_end_intents(agent, rows, answers, tasks, 8)
    assert records == [
        {
            "id": "w",
            "lang": "bn",
            "slice": "bangla",
            "predicted": "weather_query",
            "expected": "weather_query",
            "correct": True,
        }
    ]


def test_end_to_end_routing_scores_rows_whose_gold_scenario_has_one_intent(tasks):
    # Test rows for news/weather carry no intent question (see prepare.label_options).
    row = massive_row("n", "alarm", "alarm_set", "en", tasks)
    del row["questions"]["massive_intent"], row["expected"]["massive_intent"]
    row["expected"]["massive_scenario"] = "news"
    for scenario, correct in (("news", True), ("alarm", False)):
        agent = FakeAgent(scenario=scenario)
        answers = b.predict_rows(agent, [row], 8)
        (record,) = b.end_to_end_intents(agent, [row], answers, tasks, 8)
        assert record["expected"] == "news_query"
        assert record["correct"] is correct


def test_first_per_group_limits_each_source_and_wording():
    rows = [
        {"source": s, "instruction_language": lang, "n": i}
        for s in ("x", "y")
        for lang in ("bn", "en")
        for i in range(3)
    ]
    kept = b.first_per_group(rows, 2)
    assert len(kept) == 8
    assert all(row["n"] < 2 for row in kept)


def test_run_writes_predictions_summary_and_metadata(tmp_path, monkeypatch, tasks):
    rows = [
        massive_row(f"r{i}", "alarm", "alarm_query", lang, tasks)
        for i in range(2)
        for lang in ("bn", "en")
    ]
    data = tmp_path / "test.jsonl"
    data.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    checkpoint = tmp_path / "ckpt"
    checkpoint.mkdir()
    (checkpoint / "rl_agent_config.json").write_text("{}", encoding="utf-8")

    class Loaded(FakeAgent):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr("laya.load", lambda *a, **k: Loaded())
    monkeypatch.setattr("laya_bangla.cli.hardware", lambda: {"test": True})
    summary = b.run(
        str(checkpoint),
        tmp_path / "out",
        data,
        ROOT / "configs/tasks.json",
        ROOT / "configs/baseline.toml",
        "cpu",
    )
    assert summary["tasks"]["massive/massive_scenario"]["bn"]["all"]["accuracy"] == 1.0
    assert summary["massive_end_to_end_intent"]["en"] == {"n": 2, "accuracy": 1.0}
    assert summary["smoke"]["smoke/department"]["bn"]["all"]["n"] == 36
    metadata = json.loads((tmp_path / "out/metadata.json").read_text(encoding="utf-8"))
    assert metadata["rows"] == 4 and metadata["model"]["path"].endswith("ckpt")
    assert len((tmp_path / "out/predictions.jsonl").read_text(encoding="utf-8").splitlines()) == 8
    report = b.compare_report(summary, summary)
    assert "| massive/massive_scenario | bn | 2 | 1.000 | 1.000 | +0.000 |" in report


def test_checkpoint_path_rejects_directories_without_a_laya_config(tmp_path):
    with pytest.raises(FileNotFoundError, match="not a Laya or laya-bangla-llm checkpoint"):
        b.checkpoint_path(str(tmp_path), ROOT / "configs/baseline.toml")
