"""Benchmark a checkpoint on held-out Bangla test sets with question wording unseen in training."""

import json
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from .data import load_inputs
from .metrics import decision_record, summarize
from .prepare import load_tasks, read_rows, render_question, sha256_file


def checkpoint_path(model: str, baseline_config: Path) -> tuple[str, dict]:
    """A local checkpoint directory, or the pinned upstream baseline for `base`."""
    if model == "base":
        from .cli import download_checkpoint

        config, _schemas, _examples = load_inputs(baseline_config)
        snapshot = download_checkpoint(config)
        return str(snapshot), {"model_id": config["model_id"], "revision": config["revision"]}
    from .llm import is_llm_checkpoint

    path = Path(model)
    if not (path / "rl_agent_config.json").is_file() and not is_llm_checkpoint(path):
        raise FileNotFoundError(f"{path} is not a Laya or laya-bangla-llm checkpoint directory")
    return str(path), {"path": str(path.resolve())}


def load_agent(path: str, device: str):
    """A Laya agent, or the LLM variant behind the same predict/predict_batch interface."""
    from .llm import LlmAgent, is_llm_checkpoint

    if is_llm_checkpoint(Path(path)):
        return LlmAgent(path, device=device)
    import laya

    return laya.load(path, device=device)


def predict_rows(agent, rows: list[dict], batch_size: int) -> list[dict]:
    """Answers for each row, batching rows that share an identical question schema."""
    by_schema = defaultdict(list)
    for index, row in enumerate(rows):
        by_schema[json.dumps(row["questions"], ensure_ascii=False, sort_keys=True)].append(index)
    answers = [None] * len(rows)
    for indices in by_schema.values():
        questions = rows[indices[0]]["questions"]
        results = agent.predict_batch(
            [rows[i]["state"] for i in indices], questions, batch_size=batch_size
        )
        for i, result in zip(indices, results, strict=True):
            answers[i] = result["answers"]
    return answers


def end_to_end_intents(agent, rows, answers, tasks, batch_size) -> list[dict]:
    """Route each MASSIVE request by its predicted scenario, then pick an intent there.

    The `massive_intent` question in the test rows is conditioned on the gold scenario;
    this two-stage pass measures routing as deployed, where the scenario is predicted.
    """
    task = tasks["massive_intent"]
    pending, records = [], []
    for row, answer in zip(rows, answers, strict=True):
        if "massive_scenario" not in row["questions"]:
            continue
        scenario = answer["massive_scenario"]["choice"]
        options = task["groups"].get(scenario)
        # Scenarios with a single intent (news, weather) have no intent question:
        # their only intent is `<scenario>_query`, for gold labels and predictions alike.
        gold = row["expected"].get("massive_intent", f"{row['expected']['massive_scenario']}_query")
        base = {"id": row["id"], "lang": row["instruction_language"], "slice": row["slice"]}
        if not options:
            records.append(dict(base, predicted=f"{scenario}_query", expected=gold))
        else:
            question = render_question(
                task,
                task["instructions"]["eval"][row["instruction_language"]],
                row["instruction_language"],
                options,
            )
            pending.append((base, gold, {"state": row["state"], "questions": {"intent": question}}))
    staged = predict_rows(agent, [item for _b, _g, item in pending], batch_size)
    for (base, gold, _item), answer in zip(pending, staged, strict=True):
        records.append(dict(base, predicted=answer["intent"]["choice"], expected=gold))
    for record in records:
        record["correct"] = record["predicted"] == record["expected"]
    return records


def smoke_records(agent, baseline_config: Path, batch_size: int) -> list[dict]:
    """The original 36-message diagnostic set, in both instruction languages."""
    _config, schemas, examples = load_inputs(baseline_config)
    records = []
    for lang, questions in schemas.items():
        results = agent.predict_batch(
            [e["state"] for e in examples], questions, batch_size=batch_size
        )
        for example, result in zip(examples, results, strict=True):
            for qid, question in questions.items():
                record = decision_record(result["answers"][qid], question, example["expected"][qid])
                records.append(
                    dict(record, source="smoke", task=qid, lang=lang, slice=example["slice"])
                )
    return records


def group_summary(records: list[dict], bins: int) -> dict:
    labels = sorted({r["expected"] for r in records} | {r["predicted"] for r in records})
    return summarize(records, labels, bins=bins)


def summarize_records(records: list[dict], bins: int) -> dict:
    groups = defaultdict(list)
    for r in records:
        for scope in (r["slice"], "all"):
            groups[(r["source"], r["task"], r["lang"], scope)].append(r)
    summary = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for (source, task, lang, scope), selected in sorted(groups.items()):
        summary[f"{source}/{task}"][lang][scope] = group_summary(selected, bins)
    return json.loads(json.dumps(summary))


def run(
    model: str,
    output: Path,
    data: Path,
    tasks_path: Path,
    baseline_config: Path,
    device: str,
    batch_size: int = 64,
    limit: int | None = None,
    bins: int = 10,
) -> dict:
    from .cli import choose_device, hardware, write_json

    output.mkdir(parents=True, exist_ok=False)
    rows = read_rows(data)
    if limit:
        rows = first_per_group(rows, limit)
    tasks = load_tasks(tasks_path)
    device = choose_device(device)
    path, source = checkpoint_path(model, baseline_config)
    metadata = {
        "started_at": datetime.now(UTC).isoformat(),
        "model": source,
        "device": device,
        "data": str(data),
        "data_sha256": sha256_file(data),
        "tasks_sha256": sha256_file(tasks_path),
        "rows": len(rows),
        "hardware": hardware(),
    }
    print(f"Benchmarking {model} on {len(rows)} rows ({device})...", flush=True)
    start = time.perf_counter()
    with load_agent(path, device) as agent:
        answers = predict_rows(agent, rows, batch_size)
        records = []
        with (output / "predictions.jsonl").open("x", encoding="utf-8") as stream:
            for row, answer in zip(rows, answers, strict=True):
                for qid, question in row["questions"].items():
                    record = decision_record(answer[qid], question, row["expected"][qid])
                    record.update(
                        id=row["id"],
                        source=row["source"],
                        task=qid,
                        lang=row["instruction_language"],
                        slice=row["slice"],
                    )
                    records.append(record)
                    stream.write(json.dumps(dict(record, answer=answer[qid]), ensure_ascii=False))
                    stream.write("\n")
        routing = end_to_end_intents(agent, rows, answers, tasks, batch_size)
        smoke = smoke_records(agent, baseline_config, batch_size)
    metadata["seconds"] = round(time.perf_counter() - start, 1)
    if hasattr(agent, "invalid"):  # LLM replies that did not name a valid option
        metadata["invalid_answers"] = {"count": agent.invalid, "of": agent.questions_answered}
    metadata["decisions"] = len(records) + len(routing) + len(smoke)
    summary = {
        "tasks": summarize_records(records, bins),
        "smoke": summarize_records(smoke, bins),
        "massive_end_to_end_intent": {
            lang: {
                "n": len(sel),
                "accuracy": sum(r["correct"] for r in sel) / len(sel),
            }
            for lang in ("bn", "en")
            if (sel := [r for r in routing if r["lang"] == lang])
        },
    }
    write_json(output / "summary.json", summary)
    write_json(output / "metadata.json", metadata)
    print(f"Saved {metadata['decisions']} decisions in {metadata['seconds']} s to {output}")
    return summary


def score_predictions(
    rows: list[dict], answers: dict, reference: list[dict] | None = None, bins: int = 10
) -> dict:
    """Score answers produced by another runtime (ONNX, laya-ts) on benchmark rows.

    `answers` maps a row id to its answers object. With `reference`, the predictions of a
    PyTorch benchmark run, the report also gives how often the two agree on the label and
    the largest gap in the probability of that label.
    """
    records = []
    for row in rows:
        if row["id"] not in answers:
            continue
        for qid, question in row["questions"].items():
            answer = answers[row["id"]][qid]
            record = decision_record(answer, question, row["expected"][qid])
            record.update(
                id=row["id"],
                source=row["source"],
                task=qid,
                lang=row["instruction_language"],
                slice=row["slice"],
            )
            records.append(record)
    if not records:
        raise ValueError("No prediction matches a benchmark row id")
    report = {"decisions": len(records), "tasks": summarize_records(records, bins)}
    if reference is not None:
        ref = {(r["id"], r["task"]): r for r in reference}
        pairs = [(r, ref[(r["id"], r["task"])]) for r in records if (r["id"], r["task"]) in ref]
        if not pairs:
            raise ValueError("No prediction matches a reference decision")
        report["reference"] = {
            "decisions": len(pairs),
            "label_agreement": sum(a["predicted"] == b["predicted"] for a, b in pairs) / len(pairs),
            "accuracy": sum(b["correct"] for _a, b in pairs) / len(pairs),
            "max_confidence_gap": max(abs(a["confidence"] - b["confidence"]) for a, b in pairs),
        }
    report["accuracy"] = sum(r["correct"] for r in records) / len(records)
    return report


def first_per_group(rows: list[dict], limit: int) -> list[dict]:
    """The first `limit` rows of each source and instruction language, for quick checks."""
    seen, kept = defaultdict(int), []
    for row in rows:
        key = (row["source"], row["instruction_language"])
        seen[key] += 1
        if seen[key] <= limit:
            kept.append(row)
    return kept


def headline(summary: dict) -> dict:
    """Accuracy, macro-F1 and ECE per benchmark task and instruction language."""
    table = {}
    for name, langs in summary["tasks"].items():
        for lang, scopes in langs.items():
            m = scopes["all"]
            table[(name, lang)] = (m["n"], m["accuracy"], m["macro_f1"], m["ece"])
    for lang, m in summary["massive_end_to_end_intent"].items():
        table[("massive/intent (end-to-end)", lang)] = (m["n"], m["accuracy"], None, None)
    for name, langs in summary["smoke"].items():
        for lang, scopes in langs.items():
            m = scopes["all"]
            table[(f"{name} (synthetic)", lang)] = (m["n"], m["accuracy"], m["macro_f1"], m["ece"])
    return table


def compare_report(base: dict, tuned: dict, labels=("Base", "Fine-tuned")) -> str:
    """A Markdown table of base and fine-tuned metrics for every shared row."""
    a, b = headline(base), headline(tuned)
    lines = [
        f"| Benchmark | Wording | n | {labels[0]} acc | {labels[1]} acc | Δ acc "
        f"| {labels[0]} macro-F1 | {labels[1]} macro-F1 | {labels[0]} ECE | {labels[1]} ECE |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    def fmt(value):
        return "—" if value is None else f"{value:.3f}"

    for key in sorted(set(a) & set(b)):
        (n, acc_a, f1_a, ece_a), (_n, acc_b, f1_b, ece_b) = a[key], b[key]
        lines.append(
            f"| {key[0]} | {key[1]} | {n} | {acc_a:.3f} | {acc_b:.3f} | {acc_b - acc_a:+.3f} "
            f"| {fmt(f1_a)} | {fmt(f1_b)} | {fmt(ece_a)} | {fmt(ece_b)} |"
        )
    return "\n".join(lines) + "\n"
