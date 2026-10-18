# Run laya-bangla anywhere

laya-bangla comes in several versions so it can run wherever you need it: in Python, behind
an HTTP API, inside a web browser, in Node.js, and (as the LLM version) in Ollama,
llama.cpp, LM Studio, and vLLM. This page shows how to run each one, from the simplest.

## Which version should I use?

| You want to… | Use | Files | Checked here |
| --- | --- | --- | --- |
| Call it from Python | **laya-bangla** with `laya` | [`algolyzer/laya-bangla`](https://huggingface.co/algolyzer/laya-bangla) (tag `v2`) | Yes |
| Call it over HTTP from any language | `laya-serve` (Laya's server) | same | Yes |
| Run it in a web page, no server | **ONNX** + `laya-ts` | `onnx/int8/` (640 MB) in the model repo | Yes, [live demo](https://huggingface.co/spaces/algolyzer/laya-bangla-playground) |
| Run it in Node.js | **ONNX** + `laya-ts` | `onnx/fp32/` or `onnx/int8/` | Yes |
| Run it in Go, Rust, Java, or .NET | **ONNX** with Laya's runtimes for those languages | `onnx/` | No |
| Use the lighter LoRA fine-tune | **laya-bangla-lora**: same API, every option above | [`algolyzer/laya-bangla-lora`](https://huggingface.co/algolyzer/laya-bangla-lora) (tag `v1`, with `onnx/`) | Yes |
| Run it in Ollama, llama.cpp, or LM Studio | **laya-bangla-llm** GGUF | [planned](#7-the-llm-version-ollama-llamacpp-vllm) | Not trained yet |
| Serve it with vLLM | **laya-bangla-llm** | [planned](#7-the-llm-version-ollama-llamacpp-vllm) | Not trained yet |

**laya-bangla** (the Laya model) gives a calibrated probability for every option and runs
everywhere in the first five rows today. The **LLM version** is for runtimes that only run
chat models; its code is ready but it has not been trained yet ([status](next-steps.md)).

## 1. Python

```bash
pip install "laya==0.4.11"
```

```python
import laya

agent = laya.load("algolyzer/laya-bangla", revision="v2")
questions = {
    "sector": {
        "type": "choice",
        "instructions": "নাগরিকের এই সমস্যাটি কোন সেবা খাতের?",
        "criteria": {"water": "পানি সরবরাহ", "electricity": "বিদ্যুৎ", "waste": "ময়লা ও পরিচ্ছন্নতা"},
    },
}
result = agent.predict("আমাদের এলাকায় তিন দিন ধরে পানি নেই", questions)
print(result["answers"]["sector"]["choice"])  # water
```

Runs on CPU, NVIDIA GPUs (CUDA), and Apple Silicon (MPS). Use
`agent.predict_batch(texts, questions, batch_size=32)` for many texts.

## 2. HTTP API (any language)

Laya ships a production server, `laya-serve`. Register laya-bangla as an extra model and pin
its revision:

```bash
pip install "laya[serve]==0.4.11"
LAYA_EXTRA_MODELS='{"bangla": "algolyzer/laya-bangla"}' LAYA_MODELS=bangla \
LAYA_DEFAULT_MODEL=bangla LAYA_REVISION=v2 LAYA_PORT=8000 laya-serve
```

```bash
curl -s http://localhost:8000/v1/systemone -H "Content-Type: application/json" -d '{
  "model": "bangla",
  "state": "আমাদের এলাকায় তিন দিন ধরে পানি নেই",
  "questions": {"sector": {"type": "choice", "instructions": "কোন সেবা খাতের সমস্যা?",
    "criteria": {"water": "পানি", "electricity": "বিদ্যুৎ", "waste": "ময়লা"}}}
}'
```

The reply has the same `answers` as the Python API. `/v1/systemone/batch` answers many texts,
`/health` reports the loaded revision, and `LAYA_API_KEY` turns on bearer-token
authentication. Laya's own [Dockerfile](https://github.com/NandhaKishorM/laya/blob/main/Dockerfile)
runs the same server in a container with the same environment variables (not tested here).

## 3. In the browser

The [live demo](https://huggingface.co/spaces/algolyzer/laya-bangla-playground) runs the model
inside the visitor's browser with [laya-ts](https://github.com/NandhaKishorM/laya/tree/main/laya-ts)
and ONNX Runtime Web: WebGPU in Chrome and Edge, WASM (CPU) elsewhere. Nothing is sent to a
server, so it can be hosted for free as a static site. The first visit downloads about 640 MB,
which the browser then caches.

To put it on your own site, copy the [`web/`](../web) folder to any static host (GitHub Pages,
Netlify, a Hugging Face static Space). The page loads the model straight from Hugging Face:

```js
import { Agent } from "./vendor/laya-ts/index.js";  // with an import map for onnxruntime-web
const agent = await Agent.load("algolyzer/laya-bangla", {
  subfolder: "onnx/int8",
  revision: "75a291d09a5f61e8e3c37d018130cc734cca98ad",
});
const result = await agent.predict("আমাদের এলাকায় তিন দিন ধরে পানি নেই", questions);
```

For multi-threaded WASM, serve the page with the headers
`Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: credentialless`
([`web/README.md`](../web/README.md) sets them for Hugging Face).

### ONNX variants

| Folder | Size | Same answer as PyTorch | Use for |
| --- | ---: | ---: | --- |
| `onnx/fp32/` | 1.3 GB | 99.6% of 1,284 test decisions | Servers and Node.js |
| `onnx/int8/` | 640 MB | 99.1% of 1,284 test decisions | Browsers (default) |

`int8` stores only the token-embedding table (two thirds of the weights) in 8 bits; the
transformer stays in full precision. Smaller variants were tried and rejected: 8-bit weights
everywhere (310 MB) changed 13% of answers and cost 5 accuracy points, and 4-bit weights
(270 MB) changed 7% and cost 1.3 points. The PyTorch reference differs slightly too,
because it ran on a GPU in mixed precision.

## 4. Node.js

The same `laya-ts` code runs in Node with `onnxruntime-node`. Build `laya-ts` from the
[Laya repository](https://github.com/NandhaKishorM/laya/tree/main/laya-ts) (`npm ci && npm run build`),
download an ONNX folder (`hf download algolyzer/laya-bangla --include "onnx/fp32/*" --local-dir laya-bangla`),
and load it:

```js
import { Agent } from "laya-ts/dist/index.js";
const agent = await Agent.load("./laya-bangla/onnx/fp32", { device: "cpu" });
```

[`web/tools/predict.mjs`](../web/tools/predict.mjs) answers a file of benchmark rows this way,
and `laya-bangla score` compares the result with the PyTorch benchmark.

## 5. Go, Rust, Java, and .NET

Laya has ONNX Runtime ports for [Go](https://github.com/NandhaKishorM/laya/tree/main/laya-go),
[Rust](https://github.com/NandhaKishorM/laya/tree/main/laya-rust),
[Java](https://github.com/NandhaKishorM/laya/tree/main/laya-java), and
[.NET](https://github.com/NandhaKishorM/laya/tree/main/laya-dotnet). Each README says which
ONNX layout it reads; the split `encoder.onnx` + `head.onnx` files in `onnx/fp32/` are what
the Java and TypeScript runtimes load. These ports have not been tested with laya-bangla here.

## 6. Export your own ONNX files

```bash
uv sync --locked --all-extras
uv run --frozen laya-bangla export-onnx --model models/laya-bangla-v2 --output models/laya-bangla-v2-onnx
uv run --frozen laya-bangla publish-onnx --onnx models/laya-bangla-v2-onnx --repo <org>/<model>
```

`export-onnx` refuses to write anything unless the ONNX graphs match PyTorch within 1e-4 at
16 and 512 tokens, and batch sizes 1 and 2.

## 7. The LLM version (Ollama, llama.cpp, vLLM)

Ollama, llama.cpp, LM Studio, and vLLM run chat language models. They cannot run Laya's
decision head, so converting laya-bangla to GGUF would not work. Instead, `laya-bangla
train-llm` teaches a small chat model, Qwen3-1.7B, to answer the same questions: it reads the
Laya request as JSON and replies with JSON.

```text
user:      {"text": "আমাদের এলাকায় তিন দিন ধরে পানি নেই", "questions": {"sector": {"type": "choice", ...}}}
assistant: {"sector": "water"}
```

**Status: ready to train, not trained yet.** The training, benchmark, GGUF export, and Ollama
files are implemented and tested without a GPU; [next steps](next-steps.md#2-the-llm-version-for-ollama-llamacpp-and-vllm-ready-not-trained)
has the exact commands. Once it is trained and published, it will run like this:

```bash
ollama run hf.co/algolyzer/laya-bangla-llm-GGUF:Q4_K_M         # Ollama
llama-server -hf algolyzer/laya-bangla-llm-GGUF:Q4_K_M         # llama.cpp, OpenAI-compatible API
vllm serve algolyzer/laya-bangla-llm                           # vLLM on Linux
```

Expect labels without calibrated probabilities, and check its benchmark before choosing it
over the Laya model.
