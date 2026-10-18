"""ONNX export and the browser int8 variant, on a tiny randomly initialized Laya model."""

import json

import pytest

pytest.importorskip("onnxruntime")
pytest.importorskip("onnxscript")


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory):
    import torch
    from laya.common import build_model
    from safetensors.torch import save_file
    from transformers import ModernBertConfig

    root = tmp_path_factory.mktemp("tiny")
    encoder = ModernBertConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=1024,
        local_attention=8,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        cls_token_id=1,
        sep_token_id=2,
    )
    encoder.save_pretrained(root / "encoder")
    cfg = {"encoder": "tiny", "head_layers": 1, "act_costs": {"escalate": 0.5}}
    torch.manual_seed(0)
    model = build_model(cfg, encoder_dir=str(root / "encoder"), pretrained=False)
    for parameter in model.parameters():  # build_model skips initialisation by design
        torch.nn.init.normal_(parameter, std=0.05)
    state = {k: v.contiguous() for k, v in model.state_dict().items()}
    save_file(state, root / "model.safetensors")
    (root / "rl_agent_config.json").write_text(json.dumps(cfg), encoding="utf-8")
    (root / "tokenizer").mkdir()
    (root / "tokenizer/tokenizer.json").write_text("{}", encoding="utf-8")
    return root


def test_export_matches_torch_and_int8_keeps_the_head_exact(tiny_checkpoint, tmp_path):
    from laya_bangla.export import export_onnx, quantize_int8

    report = export_onnx(tiny_checkpoint, tmp_path / "fp32")
    assert set(report) == {"len16_batch1", "len16_batch2", "len512_batch2"}
    assert all(r["logits"] <= 1e-4 and r["act_probs"] <= 1e-4 for r in report.values())
    names = {p.name for p in (tmp_path / "fp32").iterdir()}
    assert {"encoder.onnx", "head.onnx", "rl_agent_config.json", "tokenizer.json"} <= names

    sizes = quantize_int8(tmp_path / "fp32", tmp_path / "int8")
    assert {"encoder.onnx", "head.onnx", "tokenizer.json"} <= set(sizes)
    for name in names - {"encoder.onnx", "encoder.onnx.data"}:
        assert (tmp_path / "int8" / name).read_bytes() == (tmp_path / "fp32" / name).read_bytes()


def test_export_refuses_an_existing_output(tiny_checkpoint, tmp_path):
    from laya_bangla.export import export_onnx

    (tmp_path / "out").mkdir()
    with pytest.raises(FileExistsError):
        export_onnx(tiny_checkpoint, tmp_path / "out")
