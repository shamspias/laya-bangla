"""Export a Laya checkpoint to the split ONNX layout that `laya-ts` loads (Node and browser).

Adapted from upstream `laya-ts/scripts/export_onnx.py` (Apache-2.0, Convai Innovations):
the same `encoder.onnx` (input_ids, attention_mask -> last_hidden_state) and `head.onnx`
(hidden_states, marker_pos, marker_mask, qtype, attention_mask -> logits, act_logits)
graphs and the same 1e-4 torch-vs-ONNX check. Differences: the dynamo exporter is requested
explicitly, which PyTorch 2.8 needs for symbolic shapes, and `quantize_int8` derives a
smaller variant for browsers.
"""

import json
import shutil
import tempfile
from pathlib import Path

# The config and tokenizer files laya-ts reads next to the two graphs.
SIDE_FILES = {
    "rl_agent_config.json": "rl_agent_config.json",
    "tokenizer.json": "tokenizer/tokenizer.json",
}


def _modules(model):
    import torch

    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = model.encoder

        def forward(self, input_ids, attention_mask):
            return self.encoder(
                input_ids=input_ids, attention_mask=attention_mask
            ).last_hidden_state

    class Head(torch.nn.Module):
        """DecisionModel's tail: type embedding, head transformer, scorer, and act head."""

        def __init__(self):
            super().__init__()
            self.head, self.type_emb = model.head, model.type_emb
            self.scorer, self.act_head = model.scorer, model.act_head

        def forward(self, hidden_states, marker_pos, marker_mask, qtype, attention_mask):
            h = hidden_states + self.type_emb(qtype.squeeze(-1))[:, None, :]
            if self.head is not None:
                pad = attention_mask == 0  # padding must not be attended, as in DecisionModel
                for layer in self.head.layers:
                    h = layer(h, src_key_padding_mask=pad)
            idx = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
            logits = self.scorer(torch.gather(h, 1, idx)).squeeze(-1).float()
            logits = logits.masked_fill(~marker_mask, -1e4)
            p = torch.softmax(logits.detach(), -1)
            k = marker_mask.sum(-1).clamp(min=2).float()
            entropy = -(p * torch.log(p.clamp_min(1e-9))).sum(-1) / torch.log(k)
            top2 = p.topk(2, -1).values
            feats = torch.stack([top2[:, 0], top2[:, 0] - top2[:, 1], entropy, k / 255.0], -1)
            return logits, self.act_head(torch.cat([h[:, 0].float(), feats], -1))

    return Encoder().eval(), Head().eval()


def _reference(encoder, head, seq: int, batch: int):
    import torch

    ids = torch.ones(batch, seq, dtype=torch.long)
    mask = torch.ones(batch, seq, dtype=torch.long)
    with torch.no_grad():  # inference-mode tensors cannot be traced by torch.export
        hidden = encoder(ids, mask)
        pos = torch.tensor([[1, 2]] * batch, dtype=torch.long)
        markers = torch.tensor([[True, True]] * batch)
        qtype = torch.tensor([[0]] * batch, dtype=torch.long)
        logits, act = head(hidden, pos, markers, qtype, mask)
    return (ids, mask), (hidden, pos, markers, qtype, mask), (logits.numpy(), act.numpy())


def run_onnx(directory: Path, encoder_inputs, head_inputs):
    """Logits and act logits from the two graphs in `directory` on CPU."""
    import onnxruntime as ort

    options = {"providers": ["CPUExecutionProvider"]}
    encoder = ort.InferenceSession(str(directory / "encoder.onnx"), **options)
    head = ort.InferenceSession(str(directory / "head.onnx"), **options)
    ids, mask = (t.numpy() for t in encoder_inputs)
    (hidden,) = encoder.run(None, {"input_ids": ids, "attention_mask": mask})
    names = ("hidden_states", "marker_pos", "marker_mask", "qtype", "attention_mask")
    feed = dict(zip(names, (hidden, *(t.numpy() for t in head_inputs[1:])), strict=True))
    return head.run(None, feed)


def _softmax(x):
    import numpy as np

    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def export_onnx(model_dir: Path, out_dir: Path, opset: int = 18) -> dict:
    """Write encoder.onnx, head.onnx, and the side files; refuse unless ONNX matches torch."""
    import numpy as np
    import torch
    from laya.common import build_model
    from safetensors.torch import load_file

    cfg = json.loads((model_dir / "rl_agent_config.json").read_text(encoding="utf-8"))
    model = build_model(cfg, encoder_dir=str(model_dir / "encoder"), pretrained=False)
    model.load_state_dict(load_file(model_dir / "model.safetensors"), strict=True)
    model.eval().float()
    encoder, head = _modules(model)
    batch = torch.export.Dim("batch", min=1, max=64)
    seq = torch.export.Dim("seq", min=1, max=8192)
    markers = torch.export.Dim("markers", min=1, max=256)
    # Trace with batch 2: batch 1 lets the exporter bake a fixed batch into a head reshape.
    enc_in, head_in, _ = _reference(encoder, head, 16, 2)
    report = {}
    with tempfile.TemporaryDirectory() as tmp:
        staged = Path(tmp)
        torch.onnx.export(
            encoder,
            enc_in,
            str(staged / "encoder.onnx"),
            input_names=["input_ids", "attention_mask"],
            output_names=["last_hidden_state"],
            dynamic_shapes=({0: batch, 1: seq}, {0: batch, 1: seq}),
            opset_version=opset,
            dynamo=True,
        )
        torch.onnx.export(
            head,
            head_in,
            str(staged / "head.onnx"),
            input_names=["hidden_states", "marker_pos", "marker_mask", "qtype", "attention_mask"],
            output_names=["logits", "act_logits"],
            dynamic_shapes=(
                {0: batch, 1: seq},
                {0: batch, 1: markers},
                {0: batch, 1: markers},
                {0: batch},
                {0: batch, 1: seq},
            ),
            opset_version=opset,
            dynamo=True,
        )
        # Past 128 tokens ModernBERT's sliding-window layers take a different path.
        for length, size in ((16, 1), (16, 2), (512, 2)):
            enc_in, head_in, (ref_logits, ref_act) = _reference(encoder, head, length, size)
            logits, act = run_onnx(staged, enc_in, head_in)
            diff = float(np.abs(logits - ref_logits).max())
            act_diff = float(np.abs(_softmax(act) - _softmax(ref_act)).max())
            if diff > 1e-4 or act_diff > 1e-4:
                raise RuntimeError(f"ONNX differs from torch at length {length}: {diff:.1e}")
            report[f"len{length}_batch{size}"] = {"logits": diff, "act_probs": act_diff}
        out_dir.mkdir(parents=True, exist_ok=False)
        for path in staged.iterdir():  # includes any *.onnx.data external-weight sidecars
            shutil.move(str(path), out_dir / path.name)
    for target, source in SIDE_FILES.items():
        shutil.copyfile(model_dir / source, out_dir / target)
    return report


def quantize_int8(source: Path, out_dir: Path) -> dict:
    """Int8 token embedding; everything else stays float32. Halves the browser download.

    The 256k-row embedding table is two thirds of the weights, and a table lookup loses
    little from int8. Quantizing the transformer's matrix multiplies as well (dynamic int8,
    per tensor or per channel) or to 4-bit blocks changed 7-13% of answers on a benchmark
    sample, so they stay float32. `laya-bangla score` measures each variant.
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic

    out_dir.mkdir(parents=True, exist_ok=False)
    quantize_dynamic(
        source / "encoder.onnx",
        out_dir / "encoder.onnx",
        op_types_to_quantize=["Gather"],
        weight_type=QuantType.QInt8,
    )
    for path in source.iterdir():
        if not path.name.startswith("encoder.onnx"):
            shutil.copyfile(path, out_dir / path.name)
    return {path.name: path.stat().st_size for path in sorted(out_dir.iterdir())}
