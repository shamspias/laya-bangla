---
title: laya-bangla playground
emoji: 🧭
colorFrom: green
colorTo: red
sdk: gradio
sdk_version: 6.17.3
python_version: "3.12"
app_file: app.py
pinned: true
license: apache-2.0
short_description: Real-time Bangla & Banglish typed decisions
models:
  - algolyzer/laya-bangla
  - convaiinnovations/laya-multilingual
tags:
  - bangla
  - bengali
  - banglish
  - laya
---

# laya-bangla playground

Try [laya-bangla](https://huggingface.co/algolyzer/laya-bangla) in your browser. Type a message in
Bangla, Banglish, or a mix, pick what you want decided, and see the fine-tuned model's answers and
probabilities next to the base model's.

- **Try it:** support routing, citizen complaints, moderation, sentiment, voice-assistant routing,
  topic, or your own questions. Answers update as you type.
- **Benchmark results:** accuracy on held-out test data, read from the model repository's
  benchmark summaries.
- **How to use** and **Data & limits:** the Python API, question types, datasets, and known
  weaknesses.

This Gradio version runs the model on a server, which Hugging Face only hosts on a paid plan.
The public demo is the free browser version in `web/`. To run this one locally:
`cd space && pip install gradio==6.17.3 -r requirements.txt && python app.py`.

Source: [github.com/shamspias/laya-bangla](https://github.com/shamspias/laya-bangla) (`space/`).
