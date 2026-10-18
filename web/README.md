---
title: laya-bangla browser playground
emoji: 🧭
colorFrom: green
colorTo: gray
sdk: static
pinned: true
license: apache-2.0
short_description: Bangla & Banglish decisions, run in your browser
models:
  - algolyzer/laya-bangla
  - algolyzer/laya-bangla-lora
tags:
  - bangla
  - bengali
  - banglish
  - laya
  - onnx
  - webgpu
custom_headers:
  cross-origin-embedder-policy: credentialless
  cross-origin-opener-policy: same-origin
  cross-origin-resource-policy: cross-origin
---

# laya-bangla browser playground

Try [laya-bangla](https://huggingface.co/algolyzer/laya-bangla) without installing anything. The
model runs **inside your browser** with [laya-ts](https://github.com/NandhaKishorM/laya/tree/main/laya-ts)
and ONNX Runtime Web (WebGPU in Chrome and Edge, WASM elsewhere), so this Space is free to host and
your text never leaves the page. The first visit downloads about 640 MB once; the browser caches it.

Choose between four versions: [laya-bangla](https://huggingface.co/algolyzer/laya-bangla) v2 (the
main release) or [laya-bangla-lora](https://huggingface.co/algolyzer/laya-bangla-lora) (keeps more
yes/no reasoning), each as int8 (640 MB) or fp32 (1.3 GB).

Source: [github.com/shamspias/laya-bangla](https://github.com/shamspias/laya-bangla) (`web/`).
