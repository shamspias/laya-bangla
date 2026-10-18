"""Local hardware inspection, data validation, and pinned baseline inference."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import shutil
import statistics
import subprocess
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from .data import SLICES, label_space, load_inputs
from .metrics import decision_record, summarize

CHECKPOINT_FILES = (
    "model.safetensors",
    "rl_agent_config.json",
    "encoder/config.json",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
)


def download_checkpoint(config: dict) -> Path:
    from huggingface_hub import snapshot_download

    # PyPI laya 0.3.20 loads local directories but does not accept revision=.
    snapshot = Path(
        snapshot_download(
            repo_id=config["model_id"],
            revision=config["revision"],
            allow_patterns=[*CHECKPOINT_FILES, "README.md"],
        )
    )
    if snapshot.name != config["revision"]:
        raise ValueError(f"Resolved snapshot {snapshot.name} does not match the pinned revision")
    for name in CHECKPOINT_FILES:
        if not (snapshot / name).is_file():
            raise FileNotFoundError(f"Pinned checkpoint is missing {name}")
    return snapshot


def prepare_runtime() -> None:
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HOME", str(Path(".cache/huggingface").resolve()))


def command_output(*command: str) -> str | None:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def hardware() -> dict:
    import torch

    return {
        "os": platform.platform(),
        "machine": platform.machine(),
        "processor": command_output("sysctl", "-n", "machdep.cpu.brand_string")
        if platform.system() == "Darwin"
        else platform.processor(),
        "memory_bytes": command_output("sysctl", "-n", "hw.memsize")
        if platform.system() == "Darwin"
        else None,
        "logical_cpus": os.cpu_count(),
        "free_disk_gib": round(shutil.disk_usage(Path.cwd()).free / 2**30, 2),
        "python": platform.python_version(),
        "mps_available": torch.backends.mps.is_available(),
        "cuda_available": torch.cuda.is_available(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("laya", "torch", "transformers", "huggingface-hub", "numpy", "safetensors")
        },
    }


def choose_device(requested: str) -> str:
    import torch

    available = {
        "cpu": True,
        "mps": torch.backends.mps.is_available(),
        "cuda": torch.cuda.is_available(),
    }
    if requested == "auto":
        return next(device for device in ("cuda", "mps", "cpu") if available[device])
    if not available[requested]:
        raise ValueError(f"Requested device {requested} is unavailable; use --device cpu")
    return requested


def synchronize(device: str) -> None:
    import torch

    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def fingerprint(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def evaluate(args: argparse.Namespace, config: dict, schemas: dict, examples: list[dict]) -> None:
    import laya
    import numpy as np
    import torch

    device = choose_device(args.device or config["device"])
    examples = examples[: args.limit] if args.limit else examples
    languages = (
        ("bn", "en") if args.instruction_language == "both" else (args.instruction_language,)
    )
    torch.set_num_threads(config["cpu_threads"])
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    # Never overwrite a prior experiment, including an interrupted one.
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    paths = {key: Path(config[key]) for key in ("dataset", "questions_bn", "questions_en")}
    paths["config"] = args.config
    for path in sorted(Path(__file__).parent.glob("*.py")):
        paths[f"source/{path.name}"] = path
    for name in ("pyproject.toml", "uv.lock", ".python-version"):
        if Path(name).is_file():
            paths[name] = Path(name)
    git_status = command_output("git", "status", "--porcelain")
    metadata = {
        "status": "running",
        "started_at": datetime.now(UTC).isoformat(),
        "config": config,
        "requested_device": args.device or config["device"],
        "device": device,
        "instruction_languages": languages,
        "sample_count": len(examples),
        "group_count": len({item["group_id"] for item in examples}),
        "example_ids": [item["id"] for item in examples],
        "git_commit": command_output("git", "rev-parse", "HEAD"),
        "git_dirty": None if git_status is None else bool(git_status),
        "sha256": {name: fingerprint(path) for name, path in paths.items()},
        "hardware": hardware(),
        "runtime_environment": {
            key: os.environ.get(key)
            for key in (
                "USE_TF",
                "TOKENIZERS_PARALLELISM",
                "HF_HUB_DISABLE_XET",
                "HF_HUB_OFFLINE",
                "PYTORCH_ENABLE_MPS_FALLBACK",
                "LAYA_MPS_AMP_MIN_ROWS",
                "LAYA_CPU_AMP",
                "LAYA_CUDA_AMP",
            )
        },
        "calibration": "upstream checkpoint parameters; no temperatures fitted on this set",
        "dataset_role": "synthetic diagnostic smoke set, not a held-out benchmark",
    }
    write_json(output / "metadata.json", metadata)
    try:
        print(f"Loading {config['model_id']}@{config['revision']} on {device}...", flush=True)
        start = time.perf_counter()
        snapshot = download_checkpoint(config)
        with laya.load(str(snapshot), device=device) as agent:
            if str(agent.device) != device:
                raise RuntimeError(
                    f"Model fell back to {agent.device}; rerun explicitly on that device"
                )
            synchronize(device)
            metadata["load_seconds"] = time.perf_counter() - start
            metadata["resolved_revision"] = snapshot.name
            metadata["checkpoint_sha256_after_sdk_load"] = {
                name: fingerprint(snapshot / name) for name in CHECKPOINT_FILES
            }
            metadata["model_config"] = agent.cfg
            metadata["precision"] = {
                "parameter_dtype": str(next(agent.model.parameters()).dtype),
                "autocast_dtype": str(agent.dtype),
                "amp_enabled": agent.amp_enabled,
                "mps_amp_min_rows": agent.mps_amp_min_rows,
                "question_rows_per_call": len(schemas[languages[0]]),
            }
            # Conservative check: these short smoke inputs must not lose state tokens.
            for item in examples:
                tokens = len(agent.tok(item["state"], add_special_tokens=False)["input_ids"])
                if tokens + config["head_max_len"] + 4 > config["max_len"]:
                    raise ValueError(f"{item['id']}: state may truncate; increase max_len")
            kwargs = {key: config[key] for key in ("max_len", "head_max_len")}
            rows, latencies = [], {lang: [] for lang in languages}
            with (output / "predictions.jsonl").open("x", encoding="utf-8") as stream:
                for lang in languages:
                    questions = schemas[lang]
                    for _ in range(config["warmup_calls"]):
                        agent.predict(examples[0]["state"], questions, **kwargs)
                    synchronize(device)
                    for index, item in enumerate(examples, 1):
                        synchronize(device)
                        start = time.perf_counter()
                        result = agent.predict(item["state"], questions, **kwargs)
                        synchronize(device)
                        elapsed_ms = (time.perf_counter() - start) * 1000
                        latencies[lang].append(elapsed_ms)
                        for qid, question in questions.items():
                            answer = result["answers"][qid]
                            row = {
                                "id": item["id"],
                                "group_id": item["group_id"],
                                "slice": item["slice"],
                                "instruction_language": lang,
                                "state": item["state"],
                                "question_id": qid,
                                "question_type": question["type"],
                                "answer": answer,
                                "request_latency_ms": elapsed_ms,
                                "usage": result["usage"],
                                **decision_record(answer, question, item["expected"][qid]),
                            }
                            stream.write(
                                json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                            )
                            rows.append(row)
                        stream.flush()
                        print(
                            f"[{lang}] {index}/{len(examples)} {item['id']} {elapsed_ms:.0f} ms",
                            flush=True,
                        )
            summary = {}
            for lang in languages:
                summary[lang] = {}
                for slice_name in ("all", *SLICES):
                    summary[lang][slice_name] = {}
                    for qid, question in schemas[lang].items():
                        selected = [
                            row
                            for row in rows
                            if row["instruction_language"] == lang
                            and row["question_id"] == qid
                            and (slice_name == "all" or row["slice"] == slice_name)
                        ]
                        if selected:
                            summary[lang][slice_name][qid] = summarize(
                                selected, label_space(question), bins=config["ece_bins"]
                            )
            write_json(output / "summary.json", summary)
            metadata["latency_ms"] = {
                lang: {
                    "requests": len(times),
                    "median": statistics.median(times),
                    "p95_nearest_rank": sorted(times)[int(np.ceil(0.95 * len(times))) - 1],
                }
                for lang, times in latencies.items()
            }
            metadata["decision_count"] = len(rows)
        metadata["status"] = "complete"
        metadata["finished_at"] = datetime.now(UTC).isoformat()
        write_json(output / "metadata.json", metadata)
        print(f"Saved {len(rows)} decisions to {output}")
        for lang in languages:
            for qid, metrics in summary[lang]["all"].items():
                print(
                    f"  {lang}/{qid}: accuracy={metrics['accuracy']:.3f} "
                    f"macro-F1={metrics['macro_f1']:.3f} ECE={metrics['ece']:.3f}"
                )
    except BaseException as exc:
        metadata["status"] = "failed"
        metadata["error"] = f"{type(exc).__name__}: {exc}"
        write_json(output / "metadata.json", metadata)
        raise


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def benchmark_run(value: str) -> tuple[str, Path]:
    name, sep, path = value.partition("=")
    if not sep or not name or not path:
        raise argparse.ArgumentTypeError("expected NAME=DIR")
    return name, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Inspect this machine without downloading a model")
    for name in ("validate", "baseline"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, default=Path("configs/baseline.toml"))
        if name == "baseline":
            command.add_argument("--output", type=Path, required=True)
            command.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"))
            command.add_argument(
                "--instruction-language", choices=("both", "bn", "en"), default="both"
            )
            command.add_argument("--limit", type=positive_int, help="Run only the first N states")
    command = commands.add_parser("prepare", help="Download pinned sources and build splits")
    command.add_argument("--config", type=Path, default=Path("configs/data.toml"))
    command = commands.add_parser("train", help="Fine-tune on the prepared splits")
    command.add_argument("--config", type=Path, default=Path("configs/train.toml"))
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    command = commands.add_parser("train-llm", help="QLoRA fine-tune the LLM variant")
    command.add_argument("--config", type=Path, default=Path("configs/train-llm.toml"))
    command.add_argument("--output", type=Path, required=True)
    command = commands.add_parser("benchmark", help="Score a checkpoint on held-out test sets")
    command.add_argument("--model", required=True, help="'base' or a checkpoint directory")
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--data", type=Path, default=Path("data/processed/test.jsonl"))
    command.add_argument("--tasks", type=Path, default=Path("configs/tasks.json"))
    command.add_argument("--baseline", type=Path, default=Path("configs/baseline.toml"))
    command.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    command.add_argument("--batch-size", type=positive_int, default=64)
    command.add_argument("--limit", type=positive_int, help="Rows per source and wording")
    command = commands.add_parser("score", help="Score predictions from another runtime")
    command.add_argument("predictions", type=Path, help='JSONL lines of {"id", "answers"}')
    command.add_argument("--data", type=Path, default=Path("data/processed/test.jsonl"))
    command.add_argument(
        "--reference", type=Path, help="A PyTorch benchmark's predictions.jsonl to compare with"
    )
    command.add_argument("--output", type=Path, help="Write the report as JSON")
    command = commands.add_parser("export-onnx", help="Export a checkpoint for laya-ts")
    command.add_argument("--model", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True, help="Writes fp32/ and int8/")
    command = commands.add_parser("compare", help="Markdown table of two benchmark summaries")
    command.add_argument("base", type=Path)
    command.add_argument("tuned", type=Path)
    command = commands.add_parser("serve", help="Web playground (requires the app extra)")
    command.add_argument("--model", type=Path, required=True)
    command.add_argument("--tasks", type=Path, default=Path("configs/tasks.json"))
    command.add_argument("--baseline", type=Path, default=Path("configs/baseline.toml"))
    command.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    command.add_argument("--host", default="127.0.0.1")
    command.add_argument("--port", type=int, default=8000)
    command.add_argument("--no-base", action="store_true", help="Skip the baseline comparison")
    command = commands.add_parser("publish", help="Upload a checkpoint and its card to the Hub")
    command.add_argument("--model", type=Path, required=True)
    command.add_argument("--repo", required=True, help="Hub model repository, e.g. org/name")
    command.add_argument("--card", type=Path, default=Path("docs/model-card.md"))
    command.add_argument(
        "--benchmark",
        type=benchmark_run,
        action="append",
        default=[],
        metavar="NAME=DIR",
        help="Publish DIR/summary.json as benchmark/NAME/summary.json (repeatable)",
    )
    command.add_argument("--tag", help="Tag the uploaded commit, e.g. v2")
    command.add_argument("--message", default="Upload laya-bangla checkpoint")
    command = commands.add_parser("publish-onnx", help="Add ONNX variants to a model repo")
    command.add_argument("--onnx", type=Path, required=True, help="Folder with fp32/ and int8/")
    command.add_argument("--repo", required=True, help="Hub model repository, e.g. org/name")
    command.add_argument("--message", default="Add ONNX variants for laya-ts (Node and browser)")
    command = commands.add_parser("publish-space", help="Upload the playground Space")
    command.add_argument("--space", type=Path, default=Path("space"))
    command.add_argument("--repo", required=True, help="Hub Space repository, e.g. org/name")
    command.add_argument("--message", default="Update the laya-bangla playground")
    args = parser.parse_args()
    prepare_runtime()
    if args.command == "doctor":
        print(json.dumps(hardware(), indent=2))
        return
    if args.command == "prepare":
        from .prepare import describe, prepare

        print(describe(prepare(args.config)), end="")
        return
    if args.command == "train":
        from .finetune import finetune

        report = finetune(args.config, args.output, args.device)
        print(f"Saved checkpoint to {args.output} (epoch {report['selected_epoch']})")
        return
    if args.command == "train-llm":
        from .llm import train_llm

        report = train_llm(args.config, args.output)
        print(f"Saved laya-bangla-llm to {args.output} ({report['optimizer_steps']} steps)")
        return
    if args.command == "benchmark":
        from .benchmark import run

        run(
            args.model,
            args.output,
            args.data,
            args.tasks,
            args.baseline,
            args.device,
            args.batch_size,
            args.limit,
        )
        return
    if args.command == "score":
        from .benchmark import score_predictions
        from .prepare import read_rows

        answers = {row["id"]: row["answers"] for row in read_rows(args.predictions)}
        reference = read_rows(args.reference) if args.reference else None
        report = score_predictions(read_rows(args.data), answers, reference)
        if args.output:
            write_json(args.output, report)
        print(json.dumps({k: v for k, v in report.items() if k != "tasks"}, indent=2))
        return
    if args.command == "export-onnx":
        from .export import export_onnx, quantize_int8

        report = export_onnx(args.model, args.output / "fp32")
        sizes = quantize_int8(args.output / "fp32", args.output / "int8")
        print(json.dumps({"verification": report, "int8_bytes": sizes}, indent=2))
        return
    if args.command == "compare":
        from .benchmark import compare_report

        summaries = [json.loads(p.read_text(encoding="utf-8")) for p in (args.base, args.tuned)]
        # Windows consoles default to a legacy code page that lacks "Δ".
        sys.stdout.reconfigure(encoding="utf-8")
        print(compare_report(*summaries), end="")
        return
    if args.command == "serve":
        from .app import serve

        serve(
            args.model,
            args.tasks,
            args.baseline,
            args.device,
            args.host,
            args.port,
            not args.no_base,
        )
        return
    if args.command == "publish":
        from .publish import publish_model

        commit = publish_model(
            args.model,
            args.repo,
            args.card,
            dict(args.benchmark),
            [Path("LICENSE"), Path("NOTICE")],
            args.tag,
            args.message,
        )
        print(f"Published https://huggingface.co/{args.repo} at {commit}")
        return
    if args.command == "publish-onnx":
        from .publish import publish_onnx

        commit = publish_onnx(args.onnx, args.repo, args.message)
        print(f"Published ONNX to https://huggingface.co/{args.repo} at {commit}")
        return
    if args.command == "publish-space":
        from .publish import publish_space

        commit = publish_space(args.space, args.repo, args.message)
        print(f"Published https://huggingface.co/spaces/{args.repo} at {commit}")
        return
    try:
        config, schemas, examples = load_inputs(args.config)
        if args.command == "validate":
            print(
                json.dumps(
                    {
                        "examples": len(examples),
                        "groups": len({row["group_id"] for row in examples}),
                        "slices": dict(Counter(row["slice"] for row in examples)),
                        "questions": list(schemas["bn"]),
                        "revision": config["revision"],
                    },
                    indent=2,
                )
            )
        else:
            evaluate(args, config, schemas, examples)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f"Error: {exc}\n")
