/** One-line question factories; output always passes checkQuestion. */
export const q = {
    choice(instructions, criteria) {
        return { type: "choice", instructions, criteria };
    },
    score(instructions, criteria) {
        return { type: "score", instructions, criteria };
    },
    noul(instructions, criteria, labels) {
        const out = { type: "noul", instructions };
        if (criteria !== undefined)
            out["criteria"] = criteria;
        if (labels !== undefined)
            out["labels"] = labels;
        return out;
    },
};
/** Argmax label: choice string, score key, noul boolean. */
export function best(answer) {
    if (answer.type === "choice")
        return answer.choice;
    if (answer.type === "score") {
        let bk = "";
        let bv = -Infinity;
        for (const [k, v] of Object.entries(answer.probabilities)) {
            if (v > bv) {
                bv = v;
                bk = k;
            }
        }
        return bk;
    }
    return answer.noul >= 0.5;
}
/** Top-k probability keys, descending. */
export function topK(probs, k) {
    return Object.entries(probs)
        .sort((a, b) => b[1] - a[1])
        .slice(0, Math.max(0, k))
        .map(([key]) => key);
}
/** Confidence gate (default 0.8) on answer_confidence (max(p)) alone, so the
 * gate uses one scale across question types instead of mixing in entropy. */
export function isConfident(answer, thresh = 0.8) {
    return (answer.answer_confidence ?? 0) >= thresh;
}
