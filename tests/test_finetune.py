import math
from pathlib import Path

from laya_bangla.finetune import load_train_config, validation_metrics

ROOT = Path(__file__).resolve().parents[1]


def test_validation_metrics_uses_each_items_own_options():
    records = [
        (0, [2.0, 0.0, 9.9], [1.0, 0.0], 2),  # logit beyond k is padding and ignored
        (1, [0.0, 1.0], [1.0, 0.0], 2),
    ]
    metrics = validation_metrics(records)
    assert metrics["accuracy"] == 0.5
    expected = (math.log(1 + math.exp(-2)) + math.log(1 + math.exp(1))) / 2
    assert math.isclose(metrics["nll"], expected)


def test_train_config_resolves_paths_and_is_valid_for_laya():
    from laya.train import TrainConfig

    config = load_train_config(ROOT / "configs/train.toml")
    assert config["baseline"] == (ROOT / "configs/baseline.toml").resolve()
    assert config["train"].name == "train.jsonl"
    trainer = TrainConfig(**dict(config["trainer"], seed=config["seed"]))
    trainer.shuffle_options = tuple(trainer.shuffle_options)
    trainer.validate()
    assert trainer.head_max_len < trainer.max_len


def test_lora_trains_only_adapters_and_merges_to_a_plain_encoder():
    import torch
    from transformers import ModernBertConfig, ModernBertModel

    from laya_bangla.finetune import add_lora

    torch.manual_seed(0)
    config = ModernBertConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=64,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        cls_token_id=1,
        sep_token_id=2,
    )
    encoder = ModernBertModel(config).eval()
    settings = load_train_config(ROOT / "configs/train-lora.toml")["lora"]
    wrapped = add_lora(encoder, settings)
    trainable = {n for n, p in wrapped.named_parameters() if p.requires_grad}
    assert trainable and all("lora_" in n for n in trainable)
    assert {n.split(".")[-4] for n in trainable} == {"Wqkv", "Wi", "Wo"}
    # Give the adapters a non-zero update, as training would.
    with torch.no_grad():
        for name, parameter in wrapped.named_parameters():
            if "lora_B" in name:
                parameter.normal_(0, 0.1)
    wrapped.eval()  # get_peft_model switches to training mode, where LoRA dropout is active.
    ids = torch.randint(0, 64, (2, 8))
    mask = torch.ones_like(ids)
    adapted = wrapped(input_ids=ids, attention_mask=mask).last_hidden_state
    merged = wrapped.merge_and_unload()
    assert type(merged) is ModernBertModel
    assert not any("lora_" in n for n, _ in merged.named_parameters())
    plain = merged(input_ids=ids, attention_mask=mask).last_hidden_state
    assert torch.allclose(adapted, plain, atol=1e-5)
