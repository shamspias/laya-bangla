// Browser playground: laya-bangla answered locally with laya-ts and ONNX Runtime Web.
import { Agent } from "./vendor/laya-ts/index.js";
import {
  BENCHMARKS, CUSTOM_TEMPLATE, ENGLISH, EXAMPLES, MODEL_REPO, PRESET_LABELS, UNSEEN,
  accuracy, answerRows, intentQuestion, parseQuestions, presetQuestions,
} from "./presets.js";

// Immutable commit of the model repository holding the ONNX files and benchmark summaries.
const REVISION = "7c8ad162a66c05409f9278e047e3ee80c24017a8";
// Every version a visitor can load, each pinned to an immutable commit of its repository.
const LORA_REPO = "algolyzer/laya-bangla-lora";
const LORA_REVISION = "b34c71f7dd4d5bc71bed9797ec0a5af13635dabe";
const VARIANTS = [
  { label: "laya-bangla v2 · int8 · 640 MB (recommended)", repo: MODEL_REPO, revision: REVISION, subfolder: "onnx/int8" },
  { label: "laya-bangla v2 · fp32 · 1.3 GB (exact PyTorch match)", repo: MODEL_REPO, revision: REVISION, subfolder: "onnx/fp32" },
  { label: "laya-bangla-lora · int8 · 640 MB (lighter fine-tune)", repo: LORA_REPO, revision: LORA_REVISION, subfolder: "onnx/int8" },
  { label: "laya-bangla-lora · fp32 · 1.3 GB", repo: LORA_REPO, revision: LORA_REVISION, subfolder: "onnx/fp32" },
];
const BENCH_RUNS = {
  base: "benchmark/base/summary.json",
  tuned: "benchmark/laya-bangla-v2/summary.json",
  englishBase: "benchmark/english/base/summary.json",
  englishTuned: "benchmark/english/laya-bangla-v2/summary.json",
};
const hub = (path) => `https://huggingface.co/${MODEL_REPO}/resolve/${REVISION}/${path}`;

const $ = (id) => document.getElementById(id);
const el = (tag, props = {}, ...children) => {
  const node = Object.assign(document.createElement(tag), props);
  node.append(...children);
  return node;
};

let agent = null;
let tasks = null;
let support = null;
let lang = "bn";
let running = false;
let pending = false;

// ---- Tabs ----
for (const tab of document.querySelectorAll("[data-tab]")) {
  tab.onclick = () => {
    for (const other of document.querySelectorAll("[data-tab]")) {
      const selected = other === tab;
      other.setAttribute("aria-selected", String(selected));
      $(`panel-${other.dataset.tab}`).hidden = !selected;
    }
  };
}

// ---- Controls ----
VARIANTS.forEach((v, i) => $("variant").append(el("option", { value: String(i), textContent: v.label })));
for (const [key, label] of Object.entries(PRESET_LABELS)) {
  $("preset").append(el("option", { value: key, textContent: label }));
}
for (const [text, preset] of EXAMPLES) {
  const chip = el("button", { className: "chip", type: "button", textContent: text, title: text });
  chip.onclick = () => {
    $("text").value = text;
    $("preset").value = preset;
    refreshQuestions();
    schedule(0);
  };
  $("examples").append(chip);
}
for (const button of document.querySelectorAll("[data-lang]")) {
  button.onclick = () => {
    lang = button.dataset.lang;
    for (const b of document.querySelectorAll("[data-lang]")) b.setAttribute("aria-pressed", String(b === button));
    refreshQuestions();
    schedule(0);
  };
}
$("preset").onchange = () => { refreshQuestions(); schedule(0); };
$("run").onclick = () => schedule(0);
$("text").oninput = () => { if ($("live").checked) schedule(400); };
$("text").onkeydown = (event) => { if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) schedule(0); };

function refreshQuestions() {
  const preset = $("preset").value;
  if (preset === "custom") {
    $("questions").value = JSON.stringify(CUSTOM_TEMPLATE, null, 2);
    $("qbox").open = true;
  } else if (tasks && support) {
    $("questions").value = JSON.stringify(presetQuestions(tasks, support, preset, lang), null, 2);
  }
}

// ---- Loading ----
async function loadQuestionData() {
  const [taskResponse, supportResponse] = await Promise.all([fetch(hub("tasks.json")), fetch("./support_questions.json")]);
  if (!taskResponse.ok || !supportResponse.ok) throw new Error("could not download the question definitions");
  tasks = await taskResponse.json();
  support = await supportResponse.json();
  refreshQuestions();
}

$("load").onclick = async () => {
  const variant = VARIANTS[Number($("variant").value)];
  const status = $("load-status");
  const progress = $("progress");
  $("load").disabled = true;
  $("variant").disabled = true;
  status.className = "status";
  progress.hidden = false;
  // laya-ts reports progress per file, and the small files finish first, so the bar shows
  // activity instead of a percentage until the model is ready.
  progress.classList.add("busy");
  const start = performance.now();
  const clock = setInterval(() => {
    status.textContent = `Downloading and starting the model… ${((performance.now() - start) / 1000).toFixed(0)} s`;
  }, 500);
  try {
    if (!tasks) await loadQuestionData();
    agent = await Agent.load(variant.repo, { subfolder: variant.subfolder, revision: variant.revision });
    clearInterval(clock);
    progress.classList.remove("busy");
    progress.firstElementChild.style.width = "100%";
    const backend = navigator.gpu ? "WebGPU where supported, otherwise WASM" : "WASM (CPU)";
    status.textContent = `Ready in ${((performance.now() - start) / 1000).toFixed(0)} s · ${backend}.`;
    $("run").disabled = false;
    $("run-meta").textContent = "Type or pick a message.";
    schedule(0);
  } catch (error) {
    clearInterval(clock);
    progress.classList.remove("busy");
    progress.hidden = true;
    status.className = "status err";
    status.textContent = `Could not load the model: ${error.message}`;
    $("load").disabled = false;
    $("variant").disabled = false;
  }
};

// ---- Predictions ----
let timer = null;
function schedule(delay) {
  clearTimeout(timer);
  timer = setTimeout(run, delay);
}

async function run() {
  if (!agent) return;
  if (running) { pending = true; return; }
  const text = $("text").value.trim();
  if (!text) {
    $("results").replaceChildren(el("p", { className: "empty", textContent: "Type or pick a message." }));
    return;
  }
  let questions;
  try {
    questions = parseQuestions($("questions").value);
  } catch (error) {
    $("run-meta").textContent = error.message;
    return;
  }
  running = true;
  $("run-meta").textContent = "Thinking…";
  const start = performance.now();
  try {
    const answers = { ...(await agent.predict(text, questions)).answers };
    const shown = { ...questions };
    if ($("preset").value === "assistant" && answers.massive_scenario) {
      const followUp = intentQuestion(tasks, answers.massive_scenario.choice, lang);
      if (followUp) {
        Object.assign(answers, (await agent.predict(text, followUp)).answers);
        Object.assign(shown, followUp);
      }
    }
    renderAnswers(answers, shown);
    $("run-meta").textContent = `${(performance.now() - start).toFixed(0)} ms in your browser`;
  } catch (error) {
    $("run-meta").textContent = `The model could not answer: ${error.message}`;
  } finally {
    running = false;
    if (pending) { pending = false; schedule(0); }
  }
}

function renderAnswers(answers, questions) {
  const blocks = [];
  for (const [name, question] of Object.entries(questions)) {
    if (!answers[name]) continue;
    const [decision, rows] = answerRows(answers[name], question);
    const block = el("div", { className: "q" },
      el("div", { className: "qname", textContent: `${name} · ${question.type}` }),
      el("div", { className: "instr", textContent: question.instructions }),
      el("div", { className: "answer", textContent: decision }));
    for (const [label, value] of rows) {
      const fill = el("i");
      fill.style.width = `${(value * 100).toFixed(1)}%`;
      block.append(el("div", { className: "opt" },
        el("span", { textContent: label, title: label }),
        el("span", { className: "bar" }, fill),
        el("span", { className: "pct", textContent: `${(value * 100).toFixed(1)}%` })));
    }
    blocks.push(block);
  }
  $("results").replaceChildren(...blocks);
}

// ---- Benchmarks ----
function pct(value) { return value == null ? "—" : `${Math.round(value * 100)}%`; }

function benchTable(base, tuned, rows) {
  const head = el("tr", {}, ...["Task", "Dataset", "Text", "Test size", "Base model", "laya-bangla"]
    .map((h, i) => el("th", { className: i >= 3 ? "num" : "", textContent: h })));
  const table = el("table", {}, head);
  for (const [key, task, dataset, script] of rows) {
    const cells = [base, tuned].map((summary) => ["bn", "en"].map((l) => accuracy(summary, key, l)?.accuracy));
    const means = cells.map(([a, b]) => (a ?? 0) + (b ?? 0));
    const best = means[0] > means[1] ? 0 : 1;
    const size = accuracy(tuned, key, "bn")?.n ?? 0;
    const tds = cells.map((pair, i) => {
      const td = el("td", { className: "num" });
      const text = `${pct(pair[0])} / ${pct(pair[1])}`;
      td.append(i === best ? el("b", { textContent: text }) : text);
      return td;
    });
    table.append(el("tr", {},
      el("td", { textContent: task }), el("td", { textContent: dataset }), el("td", { textContent: script }),
      el("td", { className: "num", textContent: size.toLocaleString("en") }), ...tds));
  }
  return el("div", { className: "table-wrap" }, table);
}

async function loadBenchmarks() {
  try {
    const [base, tuned, englishBase, englishTuned] = await Promise.all(Object.values(BENCH_RUNS).map(async (path) => {
      const response = await fetch(hub(path));
      if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`);
      return response.json();
    }));
    $("bench").replaceChildren(
      el("h3", { textContent: "Tasks it was trained on" }), benchTable(base, tuned, BENCHMARKS),
      el("h3", { textContent: "Datasets it never trained on" }), benchTable(base, tuned, UNSEEN),
      el("h3", { textContent: "English text (never trained on English)" }),
      el("p", { textContent: "Four of these are the English originals of the Bangla tests above. Fine-tuning on Bangla also helped English, except yes/no sentence logic." }),
      benchTable(englishBase, englishTuned, ENGLISH));
  } catch (error) {
    $("bench").replaceChildren(el("p", { className: "status err", textContent: `Could not load results: ${error.message}` }));
  }
}

loadBenchmarks();
loadQuestionData().catch((error) => { $("run-meta").textContent = `Could not load questions: ${error.message}`; });
