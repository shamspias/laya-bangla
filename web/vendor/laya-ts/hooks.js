/**
 * Opt-in prediction hooks: observe or shape every decision without forking.
 *
 * A hook is either a plain `(ctx) => void` or an object implementing any subset of the
 * lifecycle methods on `Hook`. Hooks are configured on `Agent` / `Router` and can be
 * overridden per call. Port of `laya/hooks.py`.
 */
/** Mutable state passed to every hook for one call. */
export class PredictContext {
    /** `states` / `questions` may be rewritten by `onPredictStart`. */
    states;
    questions;
    /** Shared by every hook of one call. */
    runId;
    /** Set by `skip()` or after inference; `onPredictEnd` may rewrite it. */
    results = null;
    /** Router: the RouteDecision. */
    decision = null;
    /** Resolved checkpoint name. */
    model = null;
    agent = null;
    router = null;
    /** Per-call token-budget overrides; null = agent config. */
    maxLen = null;
    headMaxLen = null;
    /** Aggregated input/output tokens across results. */
    usage = null;
    startedAt;
    elapsedMs = null;
    error = null;
    constructor(init) {
        this.states = init.states;
        this.questions = init.questions;
        this.runId = newRunId();
        this.startedAt = now();
        for (const [k, v] of Object.entries(init)) {
            if (k !== "states" && k !== "questions" && v !== undefined) {
                this[k] = v;
            }
        }
    }
    /** Set cached results from a start hook; inference is skipped, end hooks still run. */
    skip(results) {
        this.results = results;
    }
    /** Set elapsedMs from startedAt. Called by the dispatching call, not by hooks. */
    markElapsed() {
        this.elapsedMs = now() - this.startedAt;
    }
}
function newRunId() {
    const c = globalThis.crypto;
    if (c && typeof c.randomUUID === "function")
        return c.randomUUID().replaceAll("-", "");
    let hex = "";
    for (let i = 0; i < 32; i++)
        hex += Math.floor(Math.random() * 16).toString(16);
    return hex;
}
function now() {
    const p = globalThis.performance;
    return p && typeof p.now === "function" ? p.now() : Date.now();
}
export const HOOK_EVENTS = [
    "onPredictStart",
    "onPredictEnd",
    "onRoute",
    "onLoad",
    "onEvict",
    "onError",
];
/**
 * No-op base class: subclass it and override only the events you need.
 *
 * `Hook` is the structural interface; `BaseHook` is the concrete convenience when you would
 * rather subclass than implement methods by shape. Every method does nothing by default.
 */
export class BaseHook {
    onPredictStart(_ctx) { }
    onPredictEnd(_ctx) { }
    onRoute(_ctx) { }
    onLoad(_ctx) { }
    onEvict(_ctx) { }
    onError(_ctx) { }
}
function isHookLike(item) {
    return (typeof item === "object" &&
        item !== null &&
        HOOK_EVENTS.some((e) => typeof item[e] === "function"));
}
function typeName(v) {
    if (v === null)
        return "null";
    if (Array.isArray(v))
        return "Array";
    if (typeof v === "object")
        return v.constructor?.name ?? "object";
    return typeof v;
}
/**
 * Normalise the hook arguments shared by Agent, Router and every predict call: an optional
 * hook (object or list) plus optional plain start/end callables. A plain callable is not a
 * hook on its own -- which lifecycle event would it bind to? Passing one as `hooks` is a
 * TypeError; pass it as `onPredictStart` / `onPredictEnd` instead. The same callable as both
 * start and end is a TypeError too: the registry dedupes by identity, so one of the two would
 * silently never fire.
 */
export function normaliseHooks(hooks, onPredictStart, onPredictEnd) {
    const out = [];
    const append = (item) => {
        if (item === null || item === undefined)
            return;
        if (typeof item === "function") {
            // A JS function can carry properties; one that also defines hook methods is ambiguous.
            if (HOOK_EVENTS.some((e) => typeof item[e] === "function")) {
                throw new TypeError("a hook must be a plain callable or implement hook methods, not both");
            }
            throw new TypeError("a plain callable is ambiguous; pass it as onPredictStart or onPredictEnd");
        }
        if (Array.isArray(item)) {
            for (const h of item)
                append(h);
            return;
        }
        if (typeof item !== "object") {
            throw new TypeError(`hooks must be Hook objects, got ${typeName(item)}`);
        }
        if (!isHookLike(item)) {
            throw new TypeError(`a hook must implement at least one hook method (${HOOK_EVENTS.join(", ")})`);
        }
        out.push(item);
    };
    append(hooks);
    if (onPredictStart !== null && onPredictStart !== undefined) {
        if (typeof onPredictStart !== "function") {
            throw new TypeError(`onPredictStart must be callable, got ${typeName(onPredictStart)}`);
        }
        out.push({ onPredictStart });
    }
    if (onPredictEnd !== null && onPredictEnd !== undefined) {
        if (typeof onPredictEnd !== "function") {
            throw new TypeError(`onPredictEnd must be callable, got ${typeName(onPredictEnd)}`);
        }
        out.push({ onPredictEnd });
    }
    if (onPredictStart !== null &&
        onPredictStart !== undefined &&
        onPredictStart === onPredictEnd) {
        throw new TypeError("onPredictStart and onPredictEnd must be different callables; one callable cannot serve both events");
    }
    return out;
}
const defaultHooksList = [];
/** The process-wide hooks, a copy, in order. Empty unless set via `setDefaultHooks`. */
export function defaultHooks() {
    return [...defaultHooksList];
}
/**
 * Replace the process-wide default hooks.
 *
 * Defaults run before installed and per-call hooks for every `Agent` and `Router` in the
 * process, so a tracer or metrics hook does not have to be threaded through every
 * construction. Accepts the same arguments as the `hooks` option.
 */
export function setDefaultHooks(hooks, onPredictStart, onPredictEnd) {
    const normalised = normaliseHooks(hooks, onPredictStart, onPredictEnd);
    defaultHooksList.length = 0;
    defaultHooksList.push(...normalised);
}
/** Append one hook or a list of hooks to the process-wide defaults. */
export function addDefaultHook(hook) {
    defaultHooksList.push(...normaliseHooks(hook));
}
/** Remove every process-wide default hook. */
export function clearDefaultHooks() {
    defaultHooksList.length = 0;
}
const defaultsRan = new WeakSet();
/** `Router.predict` marks the options of its Agent call: the Router already ran the defaults for this request. */
export function markDefaultsRan(opts) {
    defaultsRan.add(opts);
}
/** Whether `opts` was marked by `markDefaultsRan`, so the Agent must not run the defaults a second time. */
export function defaultsAlreadyRan(opts) {
    return defaultsRan.has(opts);
}
/**
 * Effective hook list for one call: defaults, then installed, then per-call hooks.
 *
 * Reads the process-wide defaults at call time, so hooks set after construction still apply.
 */
export function composeHooks(installed, hooks, onPredictStart, onPredictEnd) {
    if (defaultHooksList.length === 0 && installed.length === 0 && (hooks === null || hooks === undefined) &&
        (onPredictStart === null || onPredictStart === undefined) &&
        (onPredictEnd === null || onPredictEnd === undefined)) {
        return EMPTY_HOOKS;
    }
    return [...defaultHooksList, ...installed, ...normaliseHooks(hooks, onPredictStart, onPredictEnd)];
}
const EMPTY_HOOKS = [];
Object.freeze(EMPTY_HOOKS);
/**
 * Base class giving a runtime-mutable hook list.
 *
 * `Agent` and `Router` extend it so hooks can be added, removed or scoped after construction.
 * Mutation replaces the array, so a call that already snapshotted its active hooks is never
 * disturbed by an add or remove that lands mid-flight.
 */
export class HookRegistry {
    hooks = [];
    /** Install one hook or a list of them. Returns this for chaining. */
    addHook(hook) {
        const added = normaliseHooks(hook);
        if (added.length > 0)
            this.hooks = [...this.hooks, ...added];
        return this;
    }
    /** Remove a hook by identity. Returns true if it was installed. */
    removeHook(hook) {
        const before = this.hooks.length;
        this.hooks = this.hooks.filter((installed) => installed !== hook);
        return this.hooks.length !== before;
    }
    /**
     * Run `fn` with hooks installed, then remove them -- works for sync and async `fn`:
     *
     *     await router.withHooks([tracer], () => router.predict(state, questions));
     */
    withHooks(hookArgs, fn) {
        const added = normaliseHooks(hookArgs);
        this.hooks = [...this.hooks, ...added];
        const release = () => {
            this.hooks = this.hooks.filter((installed) => !added.includes(installed));
        };
        let out;
        try {
            out = fn();
        }
        catch (err) {
            release();
            throw err;
        }
        if (out instanceof Promise) {
            return out.finally(release);
        }
        release();
        return out;
    }
}
/** Sum the per-state usage blocks so a hook sees one total for the call. */
export function aggregateUsage(results) {
    const total = (key) => results.reduce((acc, r) => {
        const usage = r?.["usage"];
        const v = usage?.[key];
        return acc + (typeof v === "number" && Number.isFinite(v) ? Math.trunc(v) : 0);
    }, 0);
    return { input_tokens: total("input_tokens"), output_tokens: total("output_tokens") };
}
function reportHookFailure(hook, event, err) {
    const name = hook?.constructor?.name ?? "hook";
    console.warn(`laya: hook ${name}.${event} failed: ${String(err)}`);
}
/**
 * Call `event` on every hook that implements it.
 *
 * `raiseErrors=false` warns and continues, for hooks (telemetry) that must not fail a
 * request. Python also takes a `lock` to serialise dispatch for hooks that are not safe to
 * run concurrently; JS hooks run on one thread, so there is nothing to lock.
 *
 * It does not wait for a hook that returns a promise, so it is for callers that cannot wait,
 * such as the synchronous `Router.route`; a rejection can then only be reported as a warning.
 * `dispatchAsync` waits and applies `raiseErrors` to it.
 */
function isThenable(value) {
    return typeof value?.then === "function";
}
function trimTrailingZeros(decimal) {
    const dot = decimal.indexOf(".");
    if (dot < 0)
        return decimal;
    let end = decimal.length;
    while (end > 0 && decimal.charAt(end - 1) === "0")
        end--;
    if (end > 0 && decimal.charAt(end - 1) === ".")
        end--;
    return decimal.slice(0, end);
}
/** Seconds the way CPython prints `"%g"`: `exceeded 0.05s` and `exceeded 1s`, not `1.0s`. */
function formatTimeoutSeconds(timeoutMs) {
    const value = timeoutMs / 1000;
    if (!Number.isFinite(value))
        return String(value);
    const scientific = value.toExponential(5);
    const marker = scientific.indexOf("e");
    if (marker < 0)
        return scientific;
    const exponent = Number(scientific.slice(marker + 1));
    if (exponent < -4 || exponent >= 6) {
        const sign = exponent < 0 ? "-" : "+";
        return (trimTrailingZeros(scientific.slice(0, marker)) +
            `e${sign}${String(Math.abs(exponent)).padStart(2, "0")}`);
    }
    return trimTrailingZeros(value.toFixed(5 - exponent));
}
/** JS has no `TimeoutError` class; Python raises that builtin, so the name is set to match. */
function hookTimeoutError(operationName, timeoutMs) {
    const err = new Error(`laya: hook ${operationName} exceeded ${formatTimeoutSeconds(timeoutMs)}s`);
    err.name = "TimeoutError";
    return err;
}
/**
 * Await one hook result. The return value is discarded: nothing reads it.
 *
 * A non-thenable has already finished, so a configured timeout starts no timer. A thenable is
 * raced against `timeoutMs`, and the timer is cleared whether the hook settles or the deadline
 * wins. The deadline matches Python: `TimeoutError`, `laya: hook <Class.event> exceeded <N>s`.
 */
async function runWithTimeout(result, timeoutMs, operationName) {
    if (timeoutMs === null || timeoutMs === undefined) {
        await result;
        return;
    }
    if (!isThenable(result))
        return;
    let timer;
    try {
        await Promise.race([
            result,
            new Promise((_, reject) => {
                timer = setTimeout(() => {
                    reject(hookTimeoutError(operationName, timeoutMs));
                }, timeoutMs);
            }),
        ]);
    }
    finally {
        if (timer !== undefined)
            clearTimeout(timer);
    }
}
export function dispatch(hooks, event, ctx, opts = {}) {
    const raiseErrors = opts.raiseErrors ?? true;
    for (const hook of hooks) {
        const method = hook?.[event];
        if (typeof method !== "function")
            continue;
        try {
            const result = method.call(hook, ctx);
            if (isThenable(result)) {
                Promise.resolve(result).catch((err) => reportHookFailure(hook, event, err));
            }
        }
        catch (err) {
            if (raiseErrors)
                throw err;
            reportHookFailure(hook, event, err);
        }
    }
}
/** Like `dispatch`, but waits for each hook that returns a promise before the next runs, and a
 *  rejection follows `raiseErrors` the way a thrown error does. */
export async function dispatchAsync(hooks, event, ctx, opts = {}) {
    const timeoutMs = opts.timeoutMs ?? null;
    const raiseErrors = opts.raiseErrors ?? true;
    for (const hook of hooks) {
        const method = hook?.[event];
        if (typeof method !== "function")
            continue;
        const operationName = `${hook?.constructor?.name ?? "hook"}.${event}`;
        try {
            await runWithTimeout(method.call(hook, ctx), timeoutMs, operationName);
        }
        catch (err) {
            if (raiseErrors)
                throw err;
            reportHookFailure(hook, event, err);
        }
    }
}
