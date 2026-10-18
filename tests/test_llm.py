"""laya-bangla-llm prompt and answer format, without loading a model."""

import json
import random
from pathlib import Path

import pytest

from laya_bangla import llm
from laya_bangla.benchmark import checkpoint_path
from laya_bangla.data import label_space
from laya_bangla.metrics import decision_record

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = {
    "sector": {"type": "choice", "instructions": "x", "criteria": {"water": "w", "power": "p"}},
    "abusive": {"type": "noul", "instructions": "y", "criteria": {"false": "no", "true": "yes"}},
    "severity": {"type": "score", "instructions": "z", "criteria": ["low", "mid", "high"]},
}


def test_messages_carry_the_laya_schema_and_typed_answers():
    chat = llm.messages("পানি নেই", QUESTIONS, {"sector": "water", "abusive": False, "severity": 2})
    assert [m["role"] for m in chat] == ["system", "user", "assistant"]
    assert json.loads(chat[1]["content"]) == {"text": "পানি নেই", "questions": QUESTIONS}
    assert json.loads(chat[2]["content"]) == {"sector": "water", "abusive": False, "severity": 2}
    assert "পানি" in chat[1]["content"]  # Bangla is kept as text, not \\u escapes


def test_valid_reply_parses_to_laya_answers_that_score():
    answers, invalid = llm.parse_answers(
        'Sure: {"sector": "power", "abusive": true, "severity": 1}', QUESTIONS
    )
    assert invalid == []
    assert answers["sector"]["choice"] == "power"
    assert answers["abusive"]["noul"] == 1.0
    assert answers["severity"]["probabilities"] == {"0": 0.0, "1": 1.0, "2": 0.0}
    expected = {"sector": "power", "abusive": True, "severity": 1}
    for qid, question in QUESTIONS.items():
        record = decision_record(answers[qid], question, expected[qid])
        assert record["correct"], qid
        assert record["predicted"] in label_space(question)


@pytest.mark.parametrize(
    "reply",
    ["not json", '{"sector": "fire", "abusive": "yes", "severity": 7}', "[1, 2]", ""],
)
def test_invalid_replies_become_uniform_guesses_and_are_counted(reply):
    answers, invalid = llm.parse_answers(reply, QUESTIONS)
    assert set(invalid) == set(QUESTIONS)
    assert answers["abusive"]["noul"] == 0.5
    assert answers["sector"]["probabilities"] == {"water": 0.5, "power": 0.5}
    for qid, question in QUESTIONS.items():  # still valid input for the benchmark scorer
        decision_record(
            answers[qid], question, {"sector": "water", "abusive": False, "severity": 0}[qid]
        )


def test_shuffled_reorders_choice_options_only():
    many = {f"o{i}": str(i) for i in range(12)}
    questions = dict(QUESTIONS, sector=dict(QUESTIONS["sector"], criteria=many))
    out = llm.shuffled(questions, random.Random(1))
    assert out["sector"]["criteria"] == many and list(out["sector"]["criteria"]) != list(many)
    assert out["severity"] == QUESTIONS["severity"] and out["abusive"] == QUESTIONS["abusive"]


def test_llm_config_and_checkpoint_detection(tmp_path):
    config = llm.load_llm_config(ROOT / "configs/train-llm.toml")
    assert len(config["revision"]) == 40 and config["train"].name == "train.jsonl"
    assert config["trainer"]["max_len"] == 1280
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    assert llm.is_llm_checkpoint(tmp_path)
    assert checkpoint_path(str(tmp_path), ROOT / "configs/baseline.toml")[0] == str(tmp_path)
    (tmp_path / "rl_agent_config.json").write_text("{}", encoding="utf-8")
    assert not llm.is_llm_checkpoint(tmp_path)


def test_chat_template_renders_chatml_without_thinking():
    jinja2 = pytest.importorskip("jinja2")
    template = jinja2.Environment().from_string(llm.CHAT_TEMPLATE)
    chat = llm.messages("t", {"a": QUESTIONS["abusive"]}, {"a": True})
    text = template.render(messages=chat[:2], add_generation_prompt=True)
    assert text.startswith("<|im_start|>system\n") and text.endswith("<|im_start|>assistant\n")
    assert "<think>" not in text and text.count("<|im_end|>") == 2


def test_ollama_files_build_the_same_prompt_as_training(tmp_path):
    jinja2 = pytest.importorskip("jinja2")
    files = llm.write_ollama_files(tmp_path, "model.gguf")
    assert set(files) == {"template", "system", "params", "Modelfile"}
    assert (tmp_path / "system").read_text(encoding="utf-8") == llm.SYSTEM_PROMPT
    assert json.loads((tmp_path / "params").read_text(encoding="utf-8"))["temperature"] == 0
    chat = llm.messages("পানি নেই", QUESTIONS)
    trained = (
        jinja2.Environment()
        .from_string(llm.CHAT_TEMPLATE)
        .render(messages=chat, add_generation_prompt=True)
    )
    # Go's text/template over .Messages, written out: one block per message, then the reply.
    template = (tmp_path / "template").read_text(encoding="utf-8")
    block = template.split("{{- range .Messages }}")[1].split("{{ end }}")[0]
    tail = template.split("{{ end }}")[1]
    ollama = (
        "".join(
            block.replace("{{ .Role }}", m["role"]).replace("{{ .Content }}", m["content"])
            for m in chat
        )
        + tail
    )
    assert ollama == trained
    modelfile = (tmp_path / "Modelfile").read_text(encoding="utf-8")
    assert modelfile.startswith("FROM ./model.gguf\n") and "PARAMETER temperature 0" in modelfile
