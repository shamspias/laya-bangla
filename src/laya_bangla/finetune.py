"""Fine-tune the pinned multilingual checkpoint on prepared Bangla typed decisions.

The loop, losses, calibration fitter, and checkpoint format are upstream `laya.train`;
this module adds what the prepared data needs: a fixed validation split for model
selection and calibration instead of a random slice of training items, and frozen
token embeddings so full fine-tuning fits a consumer GPU.
"""

import copy
import math
import time
import tomllib
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from .prepare import read_rows, sha256_file


def load_train_config(path: Path) -> dict:
    config = tomllib.loads(path.read_text(encoding="utf-8"))
    for key in ("baseline", "train", "validation"):
        config[key] = (path.parent / config[key]).resolve()
    return config


def validation_metrics(records) -> dict:
    """Accuracy and mean NLL of the argmax option against each item's target."""
    correct, nll = 0, 0.0
    for _qtype, row, target, k in records:
        logits = [float(v) for v in row[:k]]
        top = max(range(k), key=logits.__getitem__)
        best = max(range(k), key=lambda i: target[i])
        correct += top == best
        peak = max(logits)
        nll -= logits[best] - peak - math.log(sum(math.exp(v - peak) for v in logits))
    return {"accuracy": correct / len(records), "nll": nll / len(records)}


def add_lora(encoder, settings: dict):
    """Freeze the encoder and train low-rank adapters on its attention and MLP projections.

    The decision head stays fully trainable; `laya.train` puts the adapter parameters in
    the encoder learning-rate group because their names start with `encoder.`.
    """
    from peft import LoraConfig, get_peft_model

    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    lora = LoraConfig(
        r=settings["r"],
        lora_alpha=settings["alpha"],
        lora_dropout=settings["dropout"],
        target_modules=list(settings["target_modules"]),
        bias="none",
    )
    return get_peft_model(encoder, lora)


def finetune(config_path: Path, output: Path, device: str = "auto") -> dict:
    import torch
    from laya.calibrate import fit_abstention_thresholds, fit_temperature_map
    from laya.train import (
        TrainConfig,
        calibration_records,
        calibration_report,
        items_from_rows,
        load_checkpoint,
        resolve_device,
        save_checkpoint,
        train_model,
    )

    from .cli import download_checkpoint, write_json
    from .data import load_inputs

    config = load_train_config(config_path)
    baseline, _schemas, _examples = load_inputs(config["baseline"])
    trainer = TrainConfig(**dict(config["trainer"], seed=config["seed"]))
    trainer.shuffle_options = tuple(trainer.shuffle_options)
    trainer.validate()
    output.mkdir(parents=True, exist_ok=False)
    dev = resolve_device(device)
    torch.manual_seed(config["seed"])

    snapshot = download_checkpoint(baseline)
    model, tok, cfg = load_checkpoint(str(snapshot))
    max_len, head_max_len = trainer.max_len, trainer.head_max_len
    # Hash inputs as they are read, so the report describes exactly what was trained on.
    items, report = {}, {"config_sha256": sha256_file(config_path)}
    report["data_sha256"] = {split: sha256_file(config[split]) for split in ("train", "validation")}
    for split in ("train", "validation"):
        items[split], skipped = items_from_rows(
            tok,
            read_rows(config[split]),
            max_len,
            head_max_len,
            label_smoothing=trainer.label_smoothing if split == "train" else 0.0,
        )
        report[f"{split}_items"], report[f"{split}_skipped"] = len(items[split]), skipped
        print(f"{split}: {len(items[split])} items, skipped {skipped}", flush=True)
    if config.get("freeze_embeddings"):
        model.encoder.get_input_embeddings().weight.requires_grad_(False)
    if config.get("lora"):
        model.encoder = add_lora(model.encoder, config["lora"])
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable parameters: {trainable:,}", flush=True)
    model.to(dev)

    history, best = [], {"nll": math.inf}

    def evaluate(epoch, loss=None):
        records = calibration_records(model, tok, items["validation"], dev, max_len, head_max_len)
        metrics = dict(validation_metrics(records), epoch=epoch, train_loss=loss)
        history.append(metrics)
        print(f"validation after epoch {epoch}: {metrics}", flush=True)
        if epoch > 0 and metrics["nll"] < best["nll"]:
            best.update(metrics, state={k: v.detach().cpu() for k, v in model.state_dict().items()})
        model.train()
        return records

    base_records = evaluate(0)
    start = time.perf_counter()
    train_model(
        model,
        tok,
        items["train"],
        trainer,
        dev,
        max_len,
        head_max_len,
        on_epoch_end=lambda epoch, loss: evaluate(epoch + 1, loss),
    )
    train_seconds = time.perf_counter() - start
    model.load_state_dict(best.pop("state"))
    if config.get("lora"):
        # Fold the adapter into the encoder weights: the result is a plain Laya checkpoint.
        model.encoder = model.encoder.merge_and_unload()
    model.to(dev).eval()

    # Temperatures and abstention thresholds come from validation, never from test data.
    records = calibration_records(model, tok, items["validation"], dev, max_len, head_max_len)
    fitted = fit_temperature_map(records)
    calibration = calibration_report(records, fitted["temperature"])
    thresholds = fit_abstention_thresholds(
        records,
        temperature=fitted["temperature"],
        temperature_by_options=fitted["temperature_by_options"],
        target_error=trainer.target_error,
        min_bucket_n=trainer.min_abstain_n,
    )
    report.update(
        started_from={"model_id": baseline["model_id"], "revision": baseline["revision"]},
        finished_at=datetime.now(UTC).isoformat(),
        device=str(dev),
        gpu=torch.cuda.get_device_name(dev) if dev.type == "cuda" else None,
        peak_gpu_memory_gib=round(torch.cuda.max_memory_allocated(dev) / 2**30, 2)
        if dev.type == "cuda"
        else None,
        train_seconds=round(train_seconds, 1),
        trainable_parameters=trainable,
        freeze_embeddings=bool(config.get("freeze_embeddings")),
        lora=config.get("lora"),
        trainer=asdict(trainer),
        validation_history=history,
        selected_epoch=best["epoch"],
        base_validation=validation_metrics(base_records),
        calibration=calibration,
        temperature=fitted["temperature"],
        temperature_by_options=fitted["temperature_by_options"],
        abstention_thresholds=thresholds,
    )
    out_cfg = dict(
        cfg,
        max_len=max_len,
        head_max_len=head_max_len,
        fine_tuned=True,
        temperature=fitted["temperature"],
        temperature_by_options=fitted["temperature_by_options"],
    )
    out_cfg["training"] = dict(
        out_cfg.get("training") or {},
        laya_bangla={k: report[k] for k in ("started_from", "selected_epoch", "data_sha256")},
        abstention_thresholds=thresholds,
    )
    save_checkpoint(model, tok, out_cfg, str(output))
    write_json(output / "train_report.json", report)
    (output / "tasks.json").write_bytes((config_path.parent / "tasks.json").read_bytes())
    return copy.deepcopy(report)
