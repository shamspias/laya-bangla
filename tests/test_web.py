"""The browser playground (web/) asks the same questions as the benchmark; runs web/presets.js."""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from laya_bangla import app
from laya_bangla.data import load_questions
from laya_bangla.prepare import load_tasks, render_question

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
node = shutil.which("node")
needs_node = pytest.mark.skipif(node is None, reason="Node.js is not installed")


def run_node(script: str):
    """Evaluate an ES module snippet that imports web/presets.js and prints JSON."""
    result = subprocess.run(
        [node, "--input-type=module", "-e", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    return json.loads(result.stdout)


def test_support_questions_match_the_diagnostic_schemas():
    support = json.loads((WEB / "support_questions.json").read_text(encoding="utf-8"))
    for lang in ("bn", "en"):
        assert support[lang] == load_questions(ROOT / f"configs/questions.{lang}.json")


@needs_node
def test_presets_and_intent_routing_match_python():
    tasks = load_tasks(ROOT / "configs/tasks.json")
    support = json.loads((WEB / "support_questions.json").read_text(encoding="utf-8"))
    got = run_node(
        """
        import { readFileSync } from "node:fs";
        import { PRESETS, presetQuestions, intentQuestion } from "./web/presets.js";
        const tasks = JSON.parse(readFileSync("configs/tasks.json", "utf8"));
        const support = JSON.parse(readFileSync("web/support_questions.json", "utf8"));
        const out = { presets: PRESETS, questions: {}, intent: intentQuestion(tasks, "alarm", "en") };
        for (const preset of Object.keys(PRESETS))
          for (const lang of ["bn", "en"])
            out.questions[`${preset}/${lang}`] = presetQuestions(tasks, support, preset, lang);
        console.log(JSON.stringify(out));
        """
    )
    assert {k: tuple(v) for k, v in got["presets"].items()} == app.PRESETS
    for preset in app.PRESETS:
        for lang in ("bn", "en"):
            expected = app.preset_questions(tasks, support, preset, lang)
            assert got["questions"][f"{preset}/{lang}"] == expected
    task = tasks["massive_intent"]
    expected = render_question(
        task, task["instructions"]["eval"]["en"], "en", task["groups"]["alarm"]
    )
    assert got["intent"] == {"massive_intent": expected}


@needs_node
def test_question_validation_and_answer_rows():
    got = run_node(
        """
        import { parseQuestions, answerRows, CUSTOM_TEMPLATE } from "./web/presets.js";
        const errors = ['{', '[]', '{"q": {"type": "noul", "instructions": "x", "criteria": {"no": 1, "yes": 2}}}']
          .map((text) => { try { parseQuestions(text); return null; } catch (e) { return e.message; } });
        const ok = parseQuestions(JSON.stringify(CUSTOM_TEMPLATE));
        const noul = answerRows({ type: "noul", noul: 0.2 }, { criteria: { false: "no", true: "yes" } });
        const score = answerRows({ type: "score", probabilities: { 0: 0.1, 1: 0.6, 2: 0.3 } },
          { criteria: ["low", "mid", "high"] });
        console.log(JSON.stringify({ errors, ok: Object.keys(ok), noul: noul[0], score: score[0] }));
        """
    )
    assert all(got["errors"]) and "false and true" in got["errors"][2]
    assert got["ok"] == ["is_question", "product"]
    assert got["noul"] == "No / না" and got["score"] == "Level 1: mid"


def test_static_space_card_and_pins():
    readme = (WEB / "README.md").read_text(encoding="utf-8")
    assert re.search(r"^sdk: static$", readme, re.M)
    # Hugging Face also isolates the Space page that frames the app, and an isolated page only
    # loads a cross-origin frame that opts in with CORP; without it the app shows as broken.
    for header in (
        "cross-origin-embedder-policy: credentialless",
        "cross-origin-opener-policy: same-origin",
        "cross-origin-resource-policy: cross-origin",
    ):
        assert f"  {header}\n" in readme, header
    short = re.search(r"^short_description: (.+)$", readme, re.M).group(1)
    assert len(short) <= 60
    html = (WEB / "index.html").read_text(encoding="utf-8")
    assert "onnxruntime-web@1.30.0/" in html  # exact version, matching laya-ts's lockfile
    script = (WEB / "app.js").read_text(encoding="utf-8")
    for name in ("REVISION", "LORA_REVISION"):
        revision = re.search(rf'const {name} = "([^"]+)"', script).group(1)
        assert re.fullmatch(r"[0-9a-f]{40}", revision), f"pin {name} to an immutable commit"
    assert script.count("revision: REVISION") == 2 and script.count("revision: LORA_REVISION") == 2
    assert (WEB / "vendor/laya-ts/LICENSE").is_file()
