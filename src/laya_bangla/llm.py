"""laya-bangla-llm: the same typed decisions from a small generative model.

Laya's decision head cannot run in Ollama, llama.cpp, or vLLM, so this module fine-tunes a
small chat LLM with QLoRA on the same prepared rows. The model reads the Laya question
schema as JSON and answers with JSON, so one question definition works for every variant:

    user:      {"text": "...", "questions": {"sector": {"type": "choice", ...}}}
    assistant: {"sector": "water"}

`choice` answers are option labels, `noul` answers are true/false, and `score` answers are
the level index. The model returns labels, not calibrated probabilities.
"""

import json
import math
import random
import time
import tomllib
from datetime import UTC, datetime
from pathlib import Path

from .prepare import read_rows, sha256_file

SYSTEM_PROMPT = (
    "You are laya-bangla-llm. Read the text and answer every question with exactly one of "
    "its options. Reply with one JSON object that maps each question id to its answer: the "
    "option label for choice questions, true or false for noul (yes/no) questions, and the "
    "level number (0 = first level) for score questions."
)
# ChatML without a thinking block; the same template is written into the tokenizer, the
# GGUF metadata, and the Ollama Modelfile, so every runtime builds identical prompts.
CHAT_TEMPLATE = (
    "{%- for message in messages %}"
    "{{- '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' }}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- endif %}"
)


def user_message(state: str, questions: dict) -> str:
    return json.dumps({"text": state, "questions": questions}, ensure_ascii=False)


def answer_value(question: dict, expected):
    if question["type"] == "noul":
        return bool(expected)
    if question["type"] == "score":
        return int(expected)
    return str(expected)


def assistant_message(questions: dict, expected: dict) -> str:
    answers = {qid: answer_value(q, expected[qid]) for qid, q in questions.items()}
    return json.dumps(answers, ensure_ascii=False)


def messages(state: str, questions: dict, expected: dict | None = None) -> list[dict]:
    chat = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message(state, questions)},
    ]
    if expected is not None:
        chat.append({"role": "assistant", "content": assistant_message(questions, expected)})
    return chat


def shuffled(questions: dict, rng: random.Random) -> dict:
    """Choice options in a random order, so the model cannot learn option positions."""
    out = {}
    for qid, question in questions.items():
        if question["type"] == "choice":
            items = list(question["criteria"].items())
            rng.shuffle(items)
            question = dict(question, criteria=dict(items))
        out[qid] = question
    return out


def parse_answers(text: str, questions: dict) -> tuple[dict, list[str]]:
    """Laya-format answers from the model's JSON reply, and the ids it got wrong in form.

    A missing, unparsable, or out-of-schema answer becomes a uniform guess over the options:
    it still scores (usually as wrong) and is counted as invalid in the benchmark metadata.
    """
    reply = {}
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            reply = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            reply = {}
    if not isinstance(reply, dict):
        reply = {}
    answers, invalid = {}, []
    for qid, question in questions.items():
        value, kind = reply.get(qid), question["type"]
        if kind == "noul":
            ok = isinstance(value, bool)
            p = float(value) if ok else 0.5
            answers[qid] = {"type": "noul", "noul": p}
        elif kind == "score":
            k = len(question["criteria"])
            ok = isinstance(value, int) and not isinstance(value, bool) and 0 <= value < k
            probs = {str(i): (float(i == value) if ok else 1 / k) for i in range(k)}
            answers[qid] = {
                "type": "score",
                "score": float(value) if ok else (k - 1) / 2,
                "probabilities": probs,
            }
        else:
            labels = list(question["criteria"])
            ok = isinstance(value, str) and value in labels
            probs = {label: (float(label == value) if ok else 1 / len(labels)) for label in labels}
            answers[qid] = {
                "type": "choice",
                "choice": value if ok else labels[0],
                "probabilities": probs,
            }
        if not ok:
            invalid.append(qid)
    return answers, invalid


def load_llm_config(path: Path) -> dict:
    config = tomllib.loads(path.read_text(encoding="utf-8"))
    for key in ("train", "validation"):
        config[key] = (path.parent / config[key]).resolve()
    return config


def is_llm_checkpoint(path: Path) -> bool:
    return (path / "config.json").is_file() and not (path / "rl_agent_config.json").is_file()


def encode(tokenizer, chat: list[dict], max_len: int) -> dict | None:
    """Token ids with the loss on the assistant reply only; None if the row is too long."""
    prompt = tokenizer.apply_chat_template(chat[:-1], tokenize=False, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(chat, tokenize=False)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    ids = tokenizer(full, add_special_tokens=False)["input_ids"]
    if len(ids) > max_len or ids[: len(prompt_ids)] != prompt_ids:
        return None
    return {"input_ids": ids, "labels": [-100] * len(prompt_ids) + ids[len(prompt_ids) :]}


def train_llm(config_path: Path, output: Path) -> dict:
    """QLoRA fine-tuning, then merge the adapter into a bfloat16 copy of the base model."""
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    from .cli import write_json

    config = load_llm_config(config_path)
    base, revision = config["base_model"], config["revision"]
    train_cfg, lora_cfg = config["trainer"], config["lora"]
    output.mkdir(parents=True, exist_ok=False)
    rng = random.Random(config["seed"])
    torch.manual_seed(config["seed"])
    tokenizer = AutoTokenizer.from_pretrained(base, revision=revision)
    tokenizer.chat_template = CHAT_TEMPLATE
    report = {
        "config_sha256": sha256_file(config_path),
        "data_sha256": {s: sha256_file(config[s]) for s in ("train", "validation")},
        "base_model": base,
        "revision": revision,
    }
    data = {}
    for split in ("train", "validation"):
        rows = read_rows(config[split])
        if split == "validation":
            rows = rows[: train_cfg["validation_rows"]]
        encoded = [
            encode(
                tokenizer,
                messages(r["state"], shuffled(r["questions"], rng), r["expected"]),
                train_cfg["max_len"],
            )
            for r in rows
        ]
        data[split] = [e for e in encoded if e is not None]
        report[f"{split}_rows"] = len(data[split])
        report[f"{split}_too_long"] = len(encoded) - len(data[split])
        print(
            f"{split}: {len(data[split])} rows, {report[f'{split}_too_long']} too long", flush=True
        )
    rng.shuffle(data["train"])

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base, revision=revision, quantization_config=quant, dtype=torch.bfloat16, device_map={"": 0}
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(
        model,
        LoraConfig(
            r=lora_cfg["r"],
            lora_alpha=lora_cfg["alpha"],
            lora_dropout=lora_cfg["dropout"],
            target_modules=list(lora_cfg["target_modules"]),
            task_type="CAUSAL_LM",
        ),
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable parameters: {trainable:,}", flush=True)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params, lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"]
    )
    micro, accum = train_cfg["micro_batch"], train_cfg["grad_accum"]
    steps = math.ceil(len(data["train"]) / (micro * accum)) * train_cfg["epochs"]
    warmup = max(1, int(steps * train_cfg["warmup_ratio"]))

    def lr_at(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_at)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def batches(items, size):
        for i in range(0, len(items), size):
            chunk = items[i : i + size]
            width = max(len(x["input_ids"]) for x in chunk)
            ids = torch.full((len(chunk), width), pad)
            labels = torch.full((len(chunk), width), -100)
            mask = torch.zeros((len(chunk), width), dtype=torch.long)
            for j, x in enumerate(chunk):
                n = len(x["input_ids"])
                ids[j, :n] = torch.tensor(x["input_ids"])
                labels[j, :n] = torch.tensor(x["labels"])
                mask[j, :n] = 1
            yield ids.cuda(), labels.cuda(), mask.cuda()

    def validation_loss():
        model.eval()
        total = count = 0
        with torch.no_grad():
            for ids, labels, mask in batches(data["validation"], micro):
                out = model(input_ids=ids, attention_mask=mask, labels=labels)
                n = int((labels[:, 1:] != -100).sum())
                total, count = total + float(out.loss) * n, count + n
        model.train()
        return total / count

    history = [{"step": 0, "validation_loss": validation_loss()}]
    print(f"validation loss before training: {history[0]['validation_loss']:.4f}", flush=True)
    model.train()
    start, step, running = time.perf_counter(), 0, []
    for epoch in range(train_cfg["epochs"]):
        order = data["train"][:]
        if epoch:
            rng.shuffle(order)
        for i, (ids, labels, mask) in enumerate(batches(order, micro)):
            loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss / accum
            loss.backward()
            running.append(float(loss) * accum)
            if (i + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if step % train_cfg["log_every"] == 0:
                    rate = (time.perf_counter() - start) / step
                    print(
                        f"step {step}/{steps} loss {sum(running) / len(running):.4f} "
                        f"({rate:.1f} s/step, {(steps - step) * rate / 60:.0f} min left)",
                        flush=True,
                    )
                    running = []
        history.append({"epoch": epoch + 1, "step": step, "validation_loss": validation_loss()})
        print(f"validation after epoch {epoch + 1}: {history[-1]}", flush=True)
    train_seconds = time.perf_counter() - start
    adapter_dir = output / "adapter"
    model.save_pretrained(adapter_dir)
    peak = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    model = optimizer = None  # release the 4-bit model before loading the merge base
    torch.cuda.empty_cache()

    # Merge into an unquantized base so every runtime (vLLM, GGUF, ONNX) gets plain weights.
    from peft import PeftModel

    base_model = AutoModelForCausalLM.from_pretrained(base, revision=revision, dtype=torch.bfloat16)
    merged = PeftModel.from_pretrained(base_model, adapter_dir).merge_and_unload()
    merged.generation_config.do_sample = False
    for key in ("temperature", "top_p", "top_k"):
        setattr(merged.generation_config, key, None)
    merged.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)
    report.update(
        finished_at=datetime.now(UTC).isoformat(),
        gpu=torch.cuda.get_device_name(0),
        peak_gpu_memory_gib=peak,
        train_seconds=round(train_seconds, 1),
        trainable_parameters=trainable,
        optimizer_steps=step,
        lora=lora_cfg,
        trainer=train_cfg,
        seed=config["seed"],
        validation_history=history,
        system_prompt=SYSTEM_PROMPT,
        chat_template=CHAT_TEMPLATE,
    )
    write_json(output / "train_report.json", report)
    return report


class LlmAgent:
    """Answers Laya question schemas with laya-bangla-llm; the benchmark's agent interface."""

    def __init__(self, path: str, device: str = "cuda", max_new_tokens: int = 24):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(path, padding_side="left")
        dtype = torch.bfloat16 if device != "cpu" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype).to(device).eval()
        self.device, self.per_question = device, max_new_tokens
        self.invalid = 0
        self.questions_answered = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def predict_batch(self, states, questions, batch_size: int = 16):
        import torch

        results = []
        budget = 16 + self.per_question * len(questions)
        for i in range(0, len(states), batch_size):
            prompts = [
                self.tokenizer.apply_chat_template(
                    messages(s, questions), tokenize=False, add_generation_prompt=True
                )
                for s in states[i : i + batch_size]
            ]
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                out = self.model.generate(**inputs, max_new_tokens=budget, do_sample=False)
            replies = self.tokenizer.batch_decode(
                out[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )
            for reply in replies:
                answers, invalid = parse_answers(reply, questions)
                self.invalid += len(invalid)
                self.questions_answered += len(questions)
                results.append({"answers": answers, "reply": reply})
        return results

    def predict(self, state, questions):
        return self.predict_batch([state], questions, batch_size=1)[0]


# Ollama's Go-template equivalent of CHAT_TEMPLATE. Shipped as `template` (with `system` and
# `params`) in the GGUF repository, so `ollama run hf.co/<repo>:<quant>` needs no Modelfile.
OLLAMA_TEMPLATE = (
    "{{- range .Messages }}<|im_start|>{{ .Role }}\n{{ .Content }}<|im_end|>\n"
    "{{ end }}<|im_start|>assistant\n"
)
OLLAMA_PARAMS = {"temperature": 0, "stop": ["<|im_end|>"], "num_ctx": 4096}


def write_ollama_files(out_dir: Path, gguf_name: str) -> list[str]:
    """`template`, `system`, `params` for Hugging Face pulls and a Modelfile for local files."""
    (out_dir / "template").write_text(OLLAMA_TEMPLATE, encoding="utf-8", newline="\n")
    (out_dir / "system").write_text(SYSTEM_PROMPT, encoding="utf-8", newline="\n")
    params = json.dumps(OLLAMA_PARAMS, indent=2) + "\n"
    (out_dir / "params").write_text(params, encoding="utf-8", newline="\n")
    lines = [
        f"FROM ./{gguf_name}",
        f'TEMPLATE """{OLLAMA_TEMPLATE}"""',
        f'SYSTEM """{SYSTEM_PROMPT}"""',
        "PARAMETER temperature 0",
        'PARAMETER stop "<|im_end|>"',
        "PARAMETER num_ctx 4096",
    ]
    (out_dir / "Modelfile").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return ["template", "system", "params", "Modelfile"]


def export_gguf(
    model_dir: Path,
    out_dir: Path,
    convert_script: Path,
    quantize_bin: Path,
    python: str,
    quants=("Q8_0", "Q4_K_M"),
) -> dict:
    """Convert a merged checkpoint to F16 GGUF with llama.cpp, then quantize it.

    `convert_script` is llama.cpp's convert_hf_to_gguf.py and `quantize_bin` its
    llama-quantize, from the same llama.cpp commit; `python` runs the converter in an
    environment with that commit's gguf-py.
    """
    import subprocess

    out_dir.mkdir(parents=True, exist_ok=False)
    f16 = out_dir / "laya-bangla-llm-F16.gguf"
    subprocess.run(
        [python, str(convert_script), str(model_dir), "--outfile", str(f16), "--outtype", "f16"],
        check=True,
    )
    files = {f16.name: f16.stat().st_size}
    for quant in quants:
        target = out_dir / f"laya-bangla-llm-{quant}.gguf"
        subprocess.run([str(quantize_bin), str(f16), str(target), quant], check=True)
        files[target.name] = target.stat().st_size
    for name in write_ollama_files(out_dir, f"laya-bangla-llm-{quants[-1]}.gguf"):
        files[name] = (out_dir / name).stat().st_size
    return files
