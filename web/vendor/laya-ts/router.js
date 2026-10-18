import { analyse } from "./lang.js";
import { checkQuestion } from "./agent.js";
import { checkMinConfidence, flagLowConfidence } from "./common.js";
import { decide, decideBatch } from "./structured.js";
import { HookRegistry, PredictContext, aggregateUsage, composeHooks, dispatch, markDefaultsRan, dispatchAsync, normaliseHooks, } from "./hooks.js";
export const BUNDLE_REPO = "convaiinnovations/laya";
export const DEFAULT_MODELS = {
    english: { repo: BUNDLE_REPO, subfolder: null },
    multilingual: { repo: BUNDLE_REPO, subfolder: "multilingual" },
    "typed-decisions": { repo: BUNDLE_REPO, subfolder: "typed-decisions" },
};
export const STANDALONE_MODELS = {
    english: "convaiinnovations/laya",
    multilingual: "convaiinnovations/laya-multilingual",
    "typed-decisions": "convaiinnovations/laya-typed-decisions",
};
const ALIASES = {
    en: "english",
    laya: "english",
    default: "english",
    multi: "multilingual",
    ml: "multilingual",
    "laya-multilingual": "multilingual",
    typed: "typed-decisions",
    typed_decisions: "typed-decisions",
    "laya-typed-decisions": "typed-decisions",
    decisions: "typed-decisions",
};
export function normaliseName(name) {
    const raw = String(name).trim().toLowerCase();
    const key = ALIASES[raw] ?? raw;
    if (!(key in DEFAULT_MODELS)) {
        throw new Error(`unknown model ${JSON.stringify(name)}; choose one of ${JSON.stringify(Object.keys(DEFAULT_MODELS).sort())} (or an alias: ${JSON.stringify(Object.keys(ALIASES).sort())})`);
    }
    return key;
}
const TYPED_DECISION_WORKFLOWS = {
    agent_trace_observability: new Set(["action", "needs_review", "outcome", "risk", "urgency"]),
    customer_service: new Set(["action", "category", "churn_risk", "needs_human", "urgency"]),
    invoice_processing: new Set(["discrepancy_severity", "disposition", "duplicate", "matches_order", "urgency"]),
    security_incidents: new Set(["credential_compromise", "disposition", "severity", "true_positive", "urgency"]),
};
export function matchTypedDecisionsWorkflow(questions) {
    const ids = new Set(Object.keys(questions ?? {}));
    for (const [wf, sig] of Object.entries(TYPED_DECISION_WORKFLOWS)) {
        if (sig.size === ids.size && [...sig].every((id) => ids.has(id)))
            return wf;
    }
    return null;
}
const ENGLISH_SUBTAGS = new Set(["en", "eng", "english"]);
// Valid `$LANG` values that name no language, so they answer nothing about the state: `C`,
// `POSIX` and `C.UTF-8` (the official Python image's default), plus the ISO 639-2 special codes
// `und` (undetermined), `zxx` (no linguistic content) and `mul` (multiple). They abstain like a
// blank code instead of forcing the multilingual checkpoint on English text (Python parity).
const LANGUAGE_AGNOSTIC_CODES = new Set(["c", "posix", "und", "zxx", "mul"]);
export function englishFromCode(value) {
    if (value === null || value === undefined)
        return null;
    let code = String(value).trim().toLowerCase();
    if (!code)
        return null;
    code = code.split(".", 1)[0]; // en_US.UTF-8 -> en_US
    const primary = code.replace(/_/g, "-").split("-", 1)[0]; // en_US -> en
    if (!primary || LANGUAGE_AGNOSTIC_CODES.has(primary))
        return null;
    return ENGLISH_SUBTAGS.has(primary);
}
/** Parity alias for the Python `_english_from_code` name. */
export const _englishFromCode = englishFromCode;
function questionSchema(questions) {
    // Python groups on json.dumps(sort_keys=False, default=str): insertion order is significant
    // at every nesting level because options are positional in the rendered sequence, so two
    // equal schemas with different key orders must not share a forward pass. JSON.stringify
    // preserves insertion order for string keys; the replacer stands in for default=str on
    // the few value types JSON cannot represent.
    return JSON.stringify(questions, (_key, value) => typeof value === "function" || typeof value === "symbol" || typeof value === "bigint"
        ? String(value)
        : value);
}
/** Turn `LAYA_SHA256_DIGESTS` into per-checkpoint digest maps when it names models. */
export function digestsFromEnv(models) {
    if (typeof process === "undefined" || !process.env)
        return {};
    const raw = (process.env["LAYA_SHA256_DIGESTS"] ?? "").trim();
    if (!raw)
        return {};
    let data;
    try {
        data = JSON.parse(raw);
    }
    catch {
        return {};
    }
    if (typeof data !== "object" || data === null || Array.isArray(data))
        return {};
    const entries = Object.entries(data);
    if (entries.length === 0)
        return {};
    const values = entries.map(([, v]) => v);
    if (values.every((v) => typeof v === "string")) {
        return {}; // flat: providers already applies it
    }
    if (!values.every((v) => typeof v === "object" && v !== null && !Array.isArray(v))) {
        throw new Error(`LAYA_SHA256_DIGESTS must be either {artifact: digest} for every checkpoint or {model: {artifact: digest}} per checkpoint; ${JSON.stringify(Object.keys(data).sort())} mixes the two or holds a value that is neither`);
    }
    const perModel = {};
    for (const [k, v] of entries) {
        perModel[normaliseName(k)] = v;
    }
    for (const name of Object.keys(models)) {
        const norm = normaliseName(name);
        if (!(norm in perModel)) {
            perModel[norm] = {};
        }
    }
    return perModel;
}
export const _digestsFromEnv = digestsFromEnv;
export const _digests_from_env = digestsFromEnv;
function toSpec(spec) {
    if (typeof spec === "string")
        return { repo: spec, subfolder: null };
    if (Array.isArray(spec)) {
        const [repo, sub] = [...spec, null].slice(0, 2);
        return { repo, subfolder: sub ?? null };
    }
    return { repo: spec.repo, subfolder: spec.subfolder ?? null };
}
function repoStr(spec) {
    return spec.subfolder ? `${spec.repo}/${spec.subfolder}` : spec.repo;
}
export class Router extends HookRegistry {
    hooksRaise;
    models;
    device;
    token;
    revision;
    revisions;
    sha256Digests;
    get sha256_digests() {
        return this.sha256Digests;
    }
    maxLoaded;
    default;
    autoTaskDetection;
    langGuess;
    loader;
    _agents = new Map();
    _order = []; // least-recently-used first
    _loading = new Map();
    constructor(opts = {}) {
        super();
        // Hooks are opt-in; an unset hook list is a no-op. Router-level onPredictStart /
        // onPredictEnd hooks wrap the whole route+infer call and see ctx.decision; see hooks.ts.
        this.hooks = normaliseHooks(opts.hooks, opts.onPredictStart, opts.onPredictEnd);
        this.hooksRaise = opts.hooksRaise ?? true;
        const base = opts.standaloneRepos ?? opts.standalone_repos
            ? { ...STANDALONE_MODELS }
            : Object.fromEntries(Object.entries(DEFAULT_MODELS).map(([k, v]) => [k, { ...v }]));
        this.models = Object.fromEntries(Object.entries(base).map(([k, v]) => [k, toSpec(v)]));
        if (opts.models) {
            for (const [k, v] of Object.entries(opts.models)) {
                this.models[normaliseName(k)] = toSpec(v);
            }
        }
        this.device = opts.device ?? null;
        this.token = opts.token ?? (typeof process !== "undefined" ? process.env?.["HF_TOKEN"] : undefined);
        // Optional hub revision applied to every checkpoint load. Per-model overrides support
        // standalone repositories whose reviewed commits differ.
        this.revision = opts.revision ?? null;
        this.revisions = Object.fromEntries(Object.entries(opts.revisions ?? {}).map(([name, value]) => [normaliseName(name), value]));
        // Per checkpoint SHA-256 map: seeded from a model-named LAYA_SHA256_DIGESTS, then
        // overridden checkpoint by checkpoint by the argument. Keyed and normalised exactly like
        // revisions, so a misspelled model name fails here rather than leaving that checkpoint
        // unverified.
        this.sha256Digests = digestsFromEnv(this.models);
        const rawDigests = opts.sha256Digests ?? opts.sha256_digests;
        if (rawDigests) {
            for (const [name, val] of Object.entries(rawDigests)) {
                this.sha256Digests[normaliseName(name)] = val;
            }
        }
        this.maxLoaded = Math.max(1, Math.trunc(Number(opts.maxLoaded ?? opts.max_loaded ?? 2)));
        this.default = normaliseName(opts.default ?? "multilingual");
        this.autoTaskDetection = Boolean(opts.autoTaskDetection ?? opts.auto_task_detection ?? false);
        this.langGuess = opts.langGuess ?? opts.lang_guess ?? null;
        this.loader = opts.loader ?? null;
        if (opts.preload === true) {
            void this.preload();
        }
        else if (Array.isArray(opts.preload)) {
            void this.preload(opts.preload);
        }
    }
    async load(name) {
        const key = normaliseName(name);
        if (this._agents.has(key)) {
            this._touch(key);
            return this._agents.get(key);
        }
        const loading = this._loading.get(key);
        if (loading)
            return loading;
        // Start in a microtask so even a synchronous loader sees its in-flight entry.
        const pending = Promise.resolve().then(async () => {
            let agent;
            if (this.loader) {
                agent = await this.loader(key, this.models[key]);
            }
            else {
                const { Agent } = await import("./agent.js");
                const spec = this.models[key];
                const revision = Object.prototype.hasOwnProperty.call(this.revisions, key)
                    ? this.revisions[key]
                    : this.revision;
                const opts = {
                    subfolder: spec.subfolder,
                    device: this.device ?? undefined,
                    token: this.token ?? undefined,
                };
                if (revision)
                    opts.revision = revision;
                if (Object.prototype.hasOwnProperty.call(this.sha256Digests, key)) {
                    opts.expectedSha256 = this.sha256Digests[key] ?? {};
                }
                agent = await Agent.load(spec.repo, opts);
            }
            this._agents.set(key, agent);
            this._order.push(key);
            const evicted = this._evict();
            // Lifecycle hooks fire after the maps settle, so a hook can safely call the Router.
            for (const victim of evicted) {
                await dispatchAsync(composeHooks(this.hooks), "onEvict", new PredictContext({ states: [], questions: {}, model: victim, router: this }), { raiseErrors: this.hooksRaise });
            }
            await dispatchAsync(composeHooks(this.hooks), "onLoad", new PredictContext({ states: [], questions: {}, model: key, agent, router: this }), { raiseErrors: this.hooksRaise });
            return agent;
        });
        this._loading.set(key, pending);
        try {
            return await pending;
        }
        finally {
            this._loading.delete(key);
        }
    }
    _touch(key) {
        const i = this._order.indexOf(key);
        if (i !== -1)
            this._order.splice(i, 1);
        this._order.push(key);
    }
    /** Drop least-recently-used agents until `maxLoaded` holds. Returns evicted names. */
    _evict() {
        const evicted = [];
        while (this._order.length > this.maxLoaded) {
            const victim = this._order.shift();
            if (this._agents.delete(victim))
                evicted.push(victim);
        }
        // Keep the two views consistent.
        if (this._order.length < this._agents.size) {
            for (const k of [...this._agents.keys()]) {
                if (!this._order.includes(k)) {
                    this._agents.delete(k);
                    evicted.push(k);
                }
            }
        }
        return evicted;
    }
    attach(name, agent) {
        const key = normaliseName(name);
        this._agents.set(key, agent);
        this._touch(key);
        this.maxLoaded = Math.max(this.maxLoaded, this._agents.size);
        return agent;
    }
    async preload(names) {
        const keys = (names ?? Object.keys(this.models)).map((n) => normaliseName(n));
        this.maxLoaded = Math.max(this.maxLoaded, new Set([...keys, ...this._agents.keys()]).size);
        for (const n of keys) {
            if (!this._agents.has(n))
                await this.load(n);
        }
        return this;
    }
    unload(name) {
        if (name === null || name === undefined) {
            this._agents.clear();
            this._order = [];
        }
        else {
            const key = normaliseName(name);
            this._agents.delete(key);
            const i = this._order.indexOf(key);
            if (i !== -1)
                this._order.splice(i, 1);
        }
    }
    get loaded() {
        return [...this._order];
    }
    _resolveHint(hint, state) {
        if (hint === null || hint === undefined)
            return null;
        const value = typeof hint === "function" ? hint(state) : hint;
        return englishFromCode(value);
    }
    /**
     * Decide which checkpoint to use, then let `onRoute` hooks observe or replace the decision.
     *
     * `ctx.decision` is the RouteDecision; a hook may replace it (for example to pin a
     * checkpoint) and the replacement is what gets returned and used. `opts.hooks` are
     * per-call hooks, appended after any installed on the Router.
     */
    route(state, questions = null, opts = {}) {
        if (questions !== null && questions !== undefined) {
            if (typeof questions !== "object" || Array.isArray(questions)) {
                throw new TypeError(`questions must be a dict of question id -> definition, got ${Array.isArray(questions) ? "Array" : typeof questions}`);
            }
        }
        const decision = this._route(state, questions, opts);
        const raiseErrors = opts.hooksRaise ?? this.hooksRaise;
        const active = composeHooks(this.hooks, opts.hooks);
        const ctx = new PredictContext({
            states: [state],
            questions: (questions ?? {}),
            decision: decision,
            router: this,
        });
        dispatch(active, "onRoute", ctx, { raiseErrors });
        return ctx.decision;
    }
    /** Decide which checkpoint to use, without loading, running, or hooking anything. */
    _route(state, questions = null, opts = {}) {
        const { model = null, task = null, lang = null } = opts;
        const langGuessOpt = opts.langGuess ?? opts.lang_guess ?? null;
        if (model !== null && model !== undefined) {
            const key = normaliseName(model);
            return {
                model: key,
                repo: repoStr(this.models[key]),
                reason: `explicit model=${JSON.stringify(model)}`,
                detection: null,
                workflow: null,
            };
        }
        if (task !== null && task !== undefined) {
            const key = normaliseName(task);
            return {
                model: key,
                repo: repoStr(this.models[key]),
                reason: `explicit task=${JSON.stringify(task)}`,
                detection: null,
                workflow: null,
            };
        }
        const workflow = matchTypedDecisionsWorkflow(questions ?? {});
        if (workflow && this.autoTaskDetection) {
            return {
                model: "typed-decisions",
                repo: repoStr(this.models["typed-decisions"]),
                reason: `question ids match the ${JSON.stringify(workflow)} typed-decisions workflow`,
                detection: null,
                workflow,
            };
        }
        // An explicit `lang` is decisive only when the code names a language. Blank or whitespace
        // resolves to no usable hint, so it falls through to langGuess/detection exactly as an
        // abstaining hint does (Python parity); real English/non-English codes still route now.
        const resolvedLang = englishFromCode(lang);
        if (resolvedLang !== null) {
            const key = resolvedLang ? "english" : "multilingual";
            return {
                model: key,
                repo: repoStr(this.models[key]),
                reason: `explicit lang=${JSON.stringify(lang)}`,
                detection: null,
                workflow,
            };
        }
        const hints = [
            ["lang_guess", langGuessOpt],
            ["Router(lang_guess=...)", this.langGuess],
        ];
        for (const [source, hint] of hints) {
            const resolved = this._resolveHint(hint, state);
            if (resolved !== null && resolved !== undefined) {
                const key = resolved ? "english" : "multilingual";
                return {
                    model: key,
                    repo: repoStr(this.models[key]),
                    reason: `${source}: the caller identified this as ${resolved ? "English" : "non-English"} text`,
                    detection: null,
                    workflow,
                };
            }
        }
        const det = analyse(state);
        let key;
        let reason;
        if (det.script === "unknown") {
            key = this.default;
            reason = `no letters detected in state; using default (${key})`;
        }
        else if (det.script !== "latin") {
            key = "multilingual";
            reason =
                `non-Latin script (${det.script}, ${Math.round(100 * det.nonLatinFraction)}% of letters); ` +
                    "the English checkpoint cannot read it";
        }
        else if (!det.isEnglish) {
            key = "multilingual";
            if (det.mixedSegment) {
                reason =
                    `Latin script, mostly English, but a line or field reads as ${JSON.stringify(det.language)} ` +
                        `(${JSON.stringify(det.mixedSegment.slice(0, 60))}); the English checkpoint cannot read it`;
            }
            else if (det.language) {
                reason = `Latin script but language looks like ${JSON.stringify(det.language)}, not English`;
            }
            else {
                reason =
                    `Latin script, language not identified but ${Math.round(100 * det.diacriticRate)}% ` +
                        "non-English letters; not safe for the English checkpoint";
            }
        }
        else if (det.languageUndecided) {
            key = this.default;
            reason = `Latin script, language not identified and no non-English letters; using default (${key})`;
        }
        else {
            key = "english";
            reason = "English Latin text";
        }
        return { model: key, repo: repoStr(this.models[key]), reason, detection: det, workflow };
    }
    /**
     * Route, then answer every question in one forward pass on the chosen checkpoint.
     *
     * The result is the usual systemOne payload plus a `routing` key recording the decision.
     * Router-level `onPredictStart` / `onPredictEnd` hooks wrap the whole route+infer call and
     * see `ctx.decision`; see hooks.ts.
     */
    async predict(state, questions, opts = {}) {
        if (state === null || state === undefined) {
            throw new TypeError("state must not be None; pass a string, dict, or list");
        }
        if (typeof questions !== "object" || questions === null || Array.isArray(questions)) {
            throw new TypeError(`questions must be a dict of question id -> definition, got ${questions === null ? "null" : Array.isArray(questions) ? "Array" : typeof questions}`);
        }
        const mcOpt = opts.minConfidence ?? opts.min_confidence;
        const mc = mcOpt !== undefined && mcOpt !== null ? checkMinConfidence(mcOpt) : null;
        const active = composeHooks(this.hooks, opts.hooks, opts.onPredictStart, opts.onPredictEnd);
        const raiseErrors = opts.hooksRaise ?? this.hooksRaise;
        const hasStart = active.some((h) => typeof h.onPredictStart === "function");
        const ctx = new PredictContext({
            states: [state],
            questions: questions,
            decision: null,
            model: null,
            agent: null,
            router: this,
        });
        try {
            // Per-call hooks apply to the whole call, including onRoute inside route().
            const decision = this.route(state, questions, opts);
            ctx.decision = { ...decision };
            ctx.model = decision.model;
            const agent = (await this.load(decision.model));
            ctx.agent = agent;
            await dispatchAsync(active, "onPredictStart", ctx, { raiseErrors });
            if (ctx.results === null) {
                if (hasStart) {
                    if (typeof ctx.questions !== "object" || ctx.questions === null || Array.isArray(ctx.questions)) {
                        throw new TypeError(`questions must be a dict of question id -> definition, got ${ctx.questions === null ? "null" : Array.isArray(ctx.questions) ? "Array" : typeof ctx.questions}`);
                    }
                }
                for (const [qid, qdef] of Object.entries(ctx.questions)) {
                    checkQuestion(qid, qdef);
                }
                // Python parity (router.py predict): the request's language also shapes the answer
                // distribution through the agent's lang_temperatures. An explicit lang wins;
                // otherwise forward the language the router detected for the routing decision.
                // TS analyse() names English "en" where Python's analyse returns None (it only
                // ever names non-English), so a detected "en" forwards as null — in Python only an
                // explicit lang="en" can select an "en" override.
                const detected = decision.detection?.language;
                const effectiveLang = opts.lang ?? (detected && detected !== "en" ? detected : null);
                const agentOpts = { lang: effectiveLang };
                if (mc !== null)
                    agentOpts.minConfidence = mc;
                markDefaultsRan(agentOpts);
                const result = (await agent.systemOne(ctx.states[0], ctx.questions, agentOpts));
                result["routing"] = { ...decision };
                ctx.results = [result];
            }
            else {
                // A cache hit short-circuits inference, but predict still promises a `routing` key.
                // Add it without overwriting a routing the cached payload already has.
                for (const result of ctx.results) {
                    if (result && typeof result === "object" && !("routing" in result)) {
                        result.routing = { ...decision };
                    }
                }
            }
        }
        catch (err) {
            ctx.error = err;
            try {
                await dispatchAsync(active, "onError", ctx, { raiseErrors });
            }
            catch {
                // A failing onError hook must not hide the failure that triggered it.
            }
            throw err;
        }
        finally {
            ctx.markElapsed();
            if (ctx.results !== null) {
                ctx.usage = aggregateUsage(ctx.results);
                if (mc !== null) {
                    flagLowConfidence(ctx.results, mc);
                }
            }
            try {
                await dispatchAsync(active, "onPredictEnd", ctx, { raiseErrors });
            }
            catch (hookErr) {
                // End hooks run on the failure path too; do not let one mask the real error.
                if (ctx.error === null)
                    throw hookErr;
            }
        }
        return ctx.results[0];
    }
    async decide(state, schema, opts = {}) {
        return decide(this, state, schema, opts);
    }
    async decideBatch(states, schema, opts = {}) {
        return decideBatch(this, states, schema, opts);
    }
    async systemOne(state, questions, opts = {}) {
        return this.predict(state, questions, opts);
    }
    /**
     * Route many requests in one call (Python `Router.route_batch` parity). Every request is
     * validated before anything loads, so a malformed batch fails fast. Entries keep input
     * order; routing is deterministic, so repeated batches route identically.
     */
    routeBatch(requests, opts = {}) {
        if (!Array.isArray(requests)) {
            throw new TypeError("requests must be an array of request objects");
        }
        const decisions = [];
        for (let i = 0; i < requests.length; i++) {
            const request = requests[i];
            if (typeof request !== "object" || request === null || Array.isArray(request)) {
                const got = Array.isArray(request) ? "list" : request === null ? "null" : typeof request;
                throw new TypeError(`request ${i} must be an object, got ${got}`);
            }
            if (!("state" in request)) {
                throw new Error(`request ${i} is missing required key 'state'`);
            }
            if (request.state === null || request.state === undefined) {
                throw new TypeError(`request ${i} 'state' must not be None; pass a string, dict, or list`);
            }
            if (!("questions" in request)) {
                throw new Error(`request ${i} is missing required key 'questions'`);
            }
            const questions = request.questions;
            if (typeof questions !== "object" || questions === null || Array.isArray(questions)) {
                const got = Array.isArray(questions) ? "list" : questions === null ? "null" : typeof questions;
                throw new TypeError(`request ${i} 'questions' must be an object, got ${got}`);
            }
            decisions.push(this.route(request.state, questions, {
                model: request.model ?? null,
                task: request.task ?? null,
                lang: request.lang ?? null,
                langGuess: request.langGuess ?? request.lang_guess ?? null,
                hooks: opts.hooks,
                hooksRaise: opts.hooksRaise,
            }));
        }
        return decisions;
    }
    /**
     * Route and run many requests in one call (Python `Router.predict_batch` parity).
     * Requests are routed first, then grouped by checkpoint so each loaded Agent scores its
     * requests in as few forward passes as possible; results are restored to input order.
     * Requests routed to the same checkpoint still split into separate `predictBatch` calls
     * when their question schemas differ (order-sensitively), when per-request start hooks
     * set different maxLen/headMaxLen overrides, or — for an agent carrying
     * `lang_temperatures` — when their languages differ: each request's effective language
     * (an explicit `lang`, otherwise the detected non-English one) is forwarded so the
     * batched path scores exactly like `predict`.
     *
     * Router-level predict hooks run per request: each request gets its own PredictContext
     * carrying `decision`; `onPredictStart` may rewrite a request or `ctx.skip()` it, and
     * `onPredictEnd` runs once per started request even when the batch fails.
     * `opts.hooks`, `opts.onPredictStart` and `opts.onPredictEnd` are composed the way `predict`
     * composes them: `setDefaultHooks` defaults, then installed hooks, then this call's hooks.
     * `null` and `[]` add nothing. `hooks`
     * also run `onRoute`. `opts.hooksRaise` overrides the Router's policy for this call, the way
     * `predict` uses `opts.hooksRaise ?? this.hooksRaise`.
     */
    async predictBatch(requests, batchSize = null, opts = {}) {
        const decisions = this.routeBatch(requests, opts);
        if (decisions.length === 0)
            return [];
        const groups = new Map();
        for (let i = 0; i < decisions.length; i++) {
            const model = decisions[i].model;
            const bucket = groups.get(model);
            if (bucket)
                bucket.push(i);
            else
                groups.set(model, [i]);
        }
        const results = new Array(requests.length).fill(null);
        // `composeHooks`, not `this.hooks`: `setDefaultHooks` defaults, then installed hooks,
        // then per-call `opts.hooks`, then `opts.onPredictStart` / `opts.onPredictEnd`. The instance
        // list alone would drop every default `predict` keeps. `routeBatch` has already validated
        // `opts.hooks`; start and end are validated here, after routing and before a checkpoint loads.
        const active = composeHooks(this.hooks, opts.hooks, opts.onPredictStart, opts.onPredictEnd);
        const raiseErrors = opts.hooksRaise ?? this.hooksRaise;
        for (const [modelName, indices] of groups) {
            const agent = (await this.load(modelName));
            const started = [];
            try {
                for (const i of indices) {
                    const ctx = new PredictContext({
                        states: [requests[i].state],
                        questions: requests[i].questions,
                        decision: { ...decisions[i] },
                        model: modelName,
                        agent,
                        router: this,
                    });
                    started.push(ctx);
                    dispatch(active, "onPredictStart", ctx, { raiseErrors });
                }
                // Order-sensitive at every nesting level (#166): options are positional, so two
                // equal schemas with different key orders must not share a group.
                //
                // Python parity (router.py predict_batch): `predict` forwards the request's language so
                // the agent can apply its per-language temperatures; the batched path forwarded only the
                // token budgets, so the same request scored differently depending on the entry point.
                // Only computed for an agent that actually carries overrides: `lang` is otherwise
                // unused, and adding it to the group key would split a group that shares one forward
                // pass today.
                const hasLangOverrides = Object.keys(agent.langTemperatures ?? {}).length > 0;
                const questionGroups = [];
                for (let s = 0; s < indices.length; s++) {
                    const i = indices[s];
                    const ctx = started[s];
                    if (ctx.results !== null) {
                        // A cache hit short-circuits inference, but predict still promises a `routing`
                        // key. Add it without overwriting a routing the cached payload already has.
                        for (const result of ctx.results) {
                            if (result && typeof result === "object" && !("routing" in result)) {
                                result.routing = { ...decisions[i] };
                            }
                        }
                        continue;
                    }
                    // An explicit `lang` wins; otherwise forward the language the router detected for the
                    // routing decision. TS analyse() names English "en" where Python's returns None (it
                    // only ever names non-English), so a detected "en" forwards as null — in Python only
                    // an explicit lang="en" can select an "en" override. Same rule as `predict`.
                    const detectedLang = decisions[i].detection?.language ?? null;
                    const effectiveLang = requests[i].lang ?? (detectedLang && detectedLang !== "en" ? detectedLang : null);
                    const langKey = hasLangOverrides ? effectiveLang : null;
                    const schema = questionSchema(ctx.questions);
                    const found = questionGroups.find((g) => g.schema === schema &&
                        g.maxLen === ctx.maxLen &&
                        g.headMaxLen === ctx.headMaxLen &&
                        g.lang === langKey);
                    if (found)
                        found.items.push([i, ctx]);
                    else {
                        questionGroups.push({
                            questions: ctx.questions,
                            schema,
                            maxLen: ctx.maxLen,
                            headMaxLen: ctx.headMaxLen,
                            lang: langKey,
                            items: [[i, ctx]],
                        });
                    }
                }
                for (const group of questionGroups) {
                    const agentOpts = { batchSize };
                    // Only pass overrides when set, so an Agent-like object that does not accept
                    // them still works.
                    if (group.maxLen !== null)
                        agentOpts.maxLen = group.maxLen;
                    if (group.headMaxLen !== null)
                        agentOpts.headMaxLen = group.headMaxLen;
                    if (group.lang !== null)
                        agentOpts.lang = group.lang;
                    let batchResults;
                    if (typeof agent.predictBatch === "function") {
                        batchResults = await agent.predictBatch(group.items.map(([, ctx]) => ctx.states[0]), group.questions, agentOpts);
                    }
                    else {
                        // Agent-like objects that only implement systemOne (light wrappers, tests)
                        // still work; their states simply share no forward pass.
                        batchResults = [];
                        for (const [, ctx] of group.items) {
                            batchResults.push(await agent.systemOne(ctx.states[0], group.questions, group.lang === null ? undefined : { lang: group.lang }));
                        }
                    }
                    if (batchResults.length !== group.items.length) {
                        throw new Error(`internal error: Agent.predictBatch returned ${batchResults.length} results for ${group.items.length} states`);
                    }
                    for (let s = 0; s < group.items.length; s++) {
                        const [i, ctx] = group.items[s];
                        const result = batchResults[s];
                        result.routing = { ...decisions[i] };
                        ctx.results = [result];
                    }
                }
            }
            catch (err) {
                for (const ctx of started) {
                    if (ctx.results === null) {
                        ctx.error = err;
                        try {
                            dispatch(active, "onError", ctx, { raiseErrors });
                        }
                        catch {
                            // Chained onto the batch failure, which is the one that propagates.
                        }
                    }
                }
                try {
                    this._endContexts(active, started, raiseErrors);
                }
                catch {
                    // A failing end hook is chained onto the batch failure, not raised instead.
                }
                throw err;
            }
            this._endContexts(active, started, raiseErrors);
            for (let s = 0; s < indices.length; s++) {
                results[indices[s]] = started[s].results[0];
            }
        }
        if (results.some((r) => r === null)) {
            throw new Error("internal error: batch execution did not produce every result");
        }
        return results;
    }
    /** Alias of predictBatch, mirroring Python's `Router.predict_many`. */
    async predictMany(requests, batchSize = null, opts = {}) {
        return this.predictBatch(requests, batchSize, opts);
    }
    /**
     * Run onPredictEnd for every started per-request context and raise the first hook
     * failure, if any. Elapsed and usage are stamped on every context before any end hook
     * runs, so one request's end hooks never inflate another request's elapsed time.
     */
    _endContexts(active, contexts, raiseErrors) {
        for (const ctx of contexts) {
            ctx.markElapsed();
            if (ctx.results !== null)
                ctx.usage = aggregateUsage(ctx.results);
        }
        let firstError = null;
        for (const ctx of contexts) {
            try {
                dispatch(active, "onPredictEnd", ctx, { raiseErrors });
            }
            catch (hookErr) {
                if (ctx.error === null && firstError === null)
                    firstError = hookErr;
            }
        }
        if (firstError !== null)
            throw firstError;
    }
}
