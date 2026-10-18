import { checkMinConfidence, flagLowConfidence } from "./common.js";
export const MAX_PROPERTIES = 32;
export const MAX_OPTIONS = 32;
export const MAX_SCORE_LEVELS = 10;
/** A schema cannot be expressed as Laya questions; the message names the path. */
export class SchemaError extends Error {
    constructor(message) {
        super(message);
        this.name = "SchemaError";
    }
}
const r4 = (v) => Math.round(v * 1e4) / 1e4;
function pyType(v) {
    if (v === null || v === undefined)
        return "NoneType";
    if (Array.isArray(v))
        return "list";
    switch (typeof v) {
        case "boolean": return "bool";
        case "number": return Number.isInteger(v) ? "int" : "float";
        case "string": return "str";
        case "object": return "dict";
        default: return typeof v;
    }
}
/** Accept a JSON schema object or a model exposing toJSONSchema()/model_json_schema()/schema(). */
function schemaOf(model) {
    const m = model;
    for (const key of ["toJSONSchema", "model_json_schema", "schema"]) {
        if (typeof m?.[key] === "function")
            return m[key]();
    }
    if (m !== null && typeof m === "object" && !Array.isArray(m))
        return m;
    throw new SchemaError(`expected a JSON schema object or a model with toJSONSchema(), got ${pyType(model)}`);
}
function noulField(name, description) {
    return {
        name,
        kind: "noul",
        question: { type: "noul", instructions: description || `Is \`${name}\` true?` },
    };
}
function enumField(path, name, values, description) {
    if (values.length > MAX_OPTIONS) {
        throw new SchemaError(`${path}: ${values.length} options exceeds MAX_OPTIONS=${MAX_OPTIONS}`);
    }
    if (values.length === 0)
        throw new SchemaError(`${path}: 'enum' must not be empty`);
    if (values.every((v) => typeof v === "boolean"))
        return noulField(name, description);
    const options = values.map((v) => [v == null ? "null" : String(v), v]);
    if (new Set(options.map(([label]) => label)).size !== options.length) {
        throw new SchemaError(`${path}: enum values produce duplicate choice labels`);
    }
    return {
        name,
        kind: "choice",
        question: {
            type: "choice",
            instructions: description || `What is \`${name}\`?`,
            criteria: Object.fromEntries(options.map(([label]) => [label, null])),
        },
        options,
    };
}
function scoreField(path, name, prop, description) {
    const lo = prop.minimum;
    const hi = prop.maximum;
    if (!Number.isInteger(lo) || !Number.isInteger(hi)) {
        throw new SchemaError(`${path}: a numeric field needs integer 'minimum' and 'maximum' to become a score`);
    }
    const loN = lo;
    const hiN = hi;
    if (hiN < loN)
        throw new SchemaError(`${path}: 'maximum' ${hiN} is below 'minimum' ${loN}`);
    const span = hiN - loN + 1;
    if (span > MAX_SCORE_LEVELS) {
        throw new SchemaError(`${path}: ${span} levels exceeds MAX_SCORE_LEVELS=${MAX_SCORE_LEVELS}; narrow the range or use an enum`);
    }
    return {
        name,
        kind: "score",
        question: {
            type: "score",
            instructions: description || `Score \`${name}\` from ${loN} to ${hiN}`,
            criteria: Array.from({ length: span }, (_, i) => String(loN + i)),
        },
        minimum: loN,
    };
}
function fieldFor(path, name, prop) {
    if (prop === null || typeof prop !== "object" || Array.isArray(prop)) {
        throw new SchemaError(`${path}: property must be an object, got ${pyType(prop)}`);
    }
    const p = prop;
    const description = p.description;
    if (!("const" in p || "enum" in p || "type" in p)) {
        const union = (p.anyOf ?? p.oneOf);
        if (union !== undefined && Array.isArray(union)) {
            const branches = union.filter((b) => b !== null && typeof b === "object" && !Array.isArray(b) && b.type !== "null");
            if (branches.length !== 1) {
                throw new SchemaError(`${path}: only 'Optional[...]' unions (one non-null branch) are supported, got ${branches.length}`);
            }
            const branch = { ...branches[0] };
            if (branch.description === undefined && description !== undefined) {
                branch.description = description;
            }
            return fieldFor(path, name, branch);
        }
    }
    if ("const" in p)
        return enumField(path, name, [p.const], description);
    if ("enum" in p) {
        if (!Array.isArray(p.enum))
            throw new SchemaError(`${path}: 'enum' must be a list, got ${pyType(p.enum)}`);
        return enumField(path, name, p.enum, description);
    }
    let jtype = p.type;
    if (Array.isArray(jtype)) {
        const nonNullTypes = jtype.filter((t) => t !== "null"); // nullable: ["string", "null"]
        if (nonNullTypes.length > 1) {
            throw new SchemaError(`${path}: 'type' has multiple non-null types; unions are not supported`);
        }
        jtype = nonNullTypes[0];
    }
    if (jtype === "boolean")
        return noulField(name, description);
    if (jtype === "string") {
        throw new SchemaError(`${path}: a free string cannot be a fixed option set; use 'enum' or a boolean`);
    }
    if (jtype === "integer" || jtype === "number")
        return scoreField(path, name, p, description);
    if (jtype === "array")
        throw new SchemaError(`${path}: arrays are not supported; ask one field per element`);
    if (jtype === "object")
        throw new SchemaError(`${path}: nested objects are not supported; flatten the schema`);
    if ("$ref" in p)
        throw new SchemaError(`${path}: $ref/recursion is not supported; flatten the schema`);
    throw new SchemaError(`${path}: unsupported schema ${JSON.stringify(prop)}`);
}
/** Validate a JSON schema and return one planned field per property. */
export function planFromJsonSchema(schema) {
    if (schema === null || typeof schema !== "object" || Array.isArray(schema)) {
        throw new SchemaError(`expected a JSON schema object, got ${pyType(schema)}`);
    }
    const s = schema;
    if (!(s.type === undefined || s.type === null || s.type === "object") || !("properties" in s)) {
        throw new SchemaError("the top level must be an object with 'properties'");
    }
    const properties = s.properties;
    if (properties === null || typeof properties !== "object" || Array.isArray(properties)
        || Object.keys(properties).length === 0) {
        throw new SchemaError("'properties' must be a non-empty object");
    }
    const entries = Object.entries(properties);
    if (entries.length > MAX_PROPERTIES) {
        throw new SchemaError(`${entries.length} properties exceeds MAX_PROPERTIES=${MAX_PROPERTIES}`);
    }
    return entries.map(([name, prop]) => fieldFor(`properties.${name}`, name, prop));
}
/** Turn a JSON schema into Laya questions (a documented subset; see the module docstring). */
export function questionsFromJsonSchema(schema) {
    return Object.fromEntries(planFromJsonSchema(schema).map((f) => [f.name, f.question]));
}
function project(answers, fields) {
    const values = {};
    for (const f of fields) {
        const answer = answers[f.name];
        if (answer == null)
            continue;
        if (answer.low_confidence) {
            values[f.name] = null;
            continue;
        }
        if (f.kind === "noul") {
            values[f.name] = Number(answer.noul ?? 0.0) >= 0.5;
        }
        else if (f.kind === "score") {
            const probs = answer.probabilities ?? {};
            const n = Object.keys(probs).length;
            let idx;
            if (n > 0) {
                idx = 0;
                let best = -Infinity;
                for (let i = 0; i < n; i++) {
                    const v = Number(probs[String(i)] ?? 0.0);
                    if (v > best) {
                        best = v;
                        idx = i;
                    }
                }
            }
            else {
                // `score` is always the probability-weighted 0-based level index,
                // the same space `idx` is in above -- not an absolute field value --
                // so it is rounded on its own, with no `minimum` subtracted first.
                idx = Math.round(Number(answer.score ?? 0.0));
            }
            values[f.name] = (f.minimum ?? 0) + idx;
        }
        else {
            const label = String(answer.choice);
            const hit = (f.options ?? []).find(([l]) => l === label);
            values[f.name] = hit ? hit[1] : label;
        }
    }
    return values;
}
/** Project Laya answers onto the schema values (choice value, integer level, boolean). */
export function answersToJson(answers, schema) {
    return project(answers, planFromJsonSchema(schema));
}
function answerConfidenceValue(answer) {
    const conf = answer.answer_confidence;
    if (typeof conf === "number" && typeof conf !== "boolean" && Number.isFinite(conf)) {
        return conf;
    }
    return null;
}
function detailsOf(values, answers, result) {
    const confidence = {};
    const answerConfidence = {};
    const probabilities = {};
    for (const [name, answer] of Object.entries(answers)) {
        if (answer == null)
            continue;
        confidence[name] = Number(answer.confidence ?? 0.0);
        answerConfidence[name] = answerConfidenceValue(answer);
        if (answer.type === "noul") {
            const p = Number(answer.noul ?? 0.0);
            probabilities[name] = { false: r4(1.0 - p), true: r4(p) };
        }
        else {
            probabilities[name] = { ...(answer.probabilities ?? {}) };
        }
    }
    return {
        values,
        confidence,
        answerConfidence,
        probabilities,
        answers: { ...answers },
        usage: result.usage,
        routing: result.routing,
    };
}
export async function decide(runner, state, schema, opts = {}) {
    const { questions, returnDetails = false, minConfidence, min_confidence, ...predictOpts } = opts;
    if ((schema == null) === (questions == null)) {
        throw new Error("pass exactly one of schema= or questions=");
    }
    const mcOpt = minConfidence ?? min_confidence;
    const mc = mcOpt !== undefined && mcOpt !== null ? checkMinConfidence(mcOpt) : null;
    if (mc !== null) {
        predictOpts.minConfidence = mc;
    }
    let fields = null;
    let qs = questions;
    if (schema != null) {
        fields = planFromJsonSchema(schemaOf(schema));
        qs = Object.fromEntries(fields.map((f) => [f.name, f.question]));
    }
    const result = await runner.predict(state, qs, predictOpts);
    if (mc !== null && result && typeof result === "object") {
        flagLowConfidence(result, mc);
    }
    const answers = result.answers ?? {};
    const values = fields ? project(answers, fields) : { ...answers };
    if (returnDetails)
        return detailsOf(values, answers, result);
    return values;
}
export async function decideBatch(runner, states, schema, opts = {}) {
    const { questions, returnDetails = false, minConfidence, min_confidence, ...predictOpts } = opts;
    if ((schema == null) === (questions == null)) {
        throw new Error("pass exactly one of schema= or questions=");
    }
    if (typeof states === "string" || !Array.isArray(states)) {
        throw new TypeError(`states must be an array of states, not ${pyType(states)}`);
    }
    const mcOpt = minConfidence ?? min_confidence;
    const mc = mcOpt !== undefined && mcOpt !== null ? checkMinConfidence(mcOpt) : null;
    let fields = null;
    let qs = questions;
    if (schema != null) {
        fields = planFromJsonSchema(schemaOf(schema));
        qs = Object.fromEntries(fields.map((f) => [f.name, f.question]));
    }
    const r = runner;
    const predictBatchFn = typeof r.predictBatch === "function" ? r.predictBatch : null;
    if (!predictBatchFn) {
        const runnerName = r?.constructor?.name ?? typeof runner;
        throw new TypeError(`${runnerName} has no predictBatch; loop decide() over the states instead`);
    }
    const isRouter = typeof r.routeBatch === "function";
    let results;
    if (isRouter) {
        const requests = states.map((s) => ({ state: s, questions: qs, ...predictOpts }));
        const batchSize = (predictOpts.batchSize ?? null);
        results = await predictBatchFn.call(runner, requests, batchSize);
    }
    else {
        results = await predictBatchFn.call(runner, states, qs, predictOpts);
    }
    if (mc !== null && Array.isArray(results)) {
        flagLowConfidence(results, mc);
    }
    const one = (res) => {
        const answers = (res && typeof res === "object" ? res.answers : null) || {};
        const values = fields !== null ? project(answers, fields) : { ...answers };
        if (returnDetails) {
            return detailsOf(values, answers, res || {});
        }
        return values;
    };
    return (results || []).map(one);
}
