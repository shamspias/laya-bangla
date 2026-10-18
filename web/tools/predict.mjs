// Answer benchmark rows with laya-ts and an exported ONNX model: the same JavaScript
// code path the browser playground runs, here on onnxruntime-node (CPU).
//
//   node web/tools/predict.mjs <laya-ts dist/index.js> <onnx model dir> <rows.jsonl> <out.jsonl>
//
// Each input line is {"id", "state", "questions"}; each output line is {"id", "answers"}.
// `laya-bangla score` compares the output with a PyTorch benchmark run.
import { createReadStream, createWriteStream } from "node:fs";
import { resolve } from "node:path";
import { createInterface } from "node:readline";
import { pathToFileURL } from "node:url";

const [dist, modelDir, rowsPath, outPath] = process.argv.slice(2);
if (!outPath) {
  console.error("usage: predict.mjs <laya-ts dist/index.js> <model dir> <rows.jsonl> <out.jsonl>");
  process.exit(2);
}
const { Agent } = await import(pathToFileURL(resolve(dist)).href);
const agent = await Agent.load(resolve(modelDir), { device: "cpu" });
const out = createWriteStream(outPath);
const lines = createInterface({ input: createReadStream(rowsPath), crlfDelay: Infinity });
const start = performance.now();
let count = 0;
for await (const line of lines) {
  if (!line.trim()) continue;
  const row = JSON.parse(line);
  const result = await agent.predict(row.state, row.questions);
  out.write(`${JSON.stringify({ id: row.id, answers: result.answers })}\n`);
  if (++count % 100 === 0) console.log(`${count} rows, ${((performance.now() - start) / count).toFixed(0)} ms/row`);
}
out.end();
console.log(`done: ${count} rows in ${((performance.now() - start) / 1000).toFixed(0)} s`);
