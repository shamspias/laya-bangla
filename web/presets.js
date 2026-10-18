// Questions, validation, and answer formatting for the browser playground (no DOM access).
// Mirrors space/playground.py; web/tests/presets.test.mjs checks both ask the same questions.

export const MODEL_REPO = "algolyzer/laya-bangla";
export const GITHUB = "https://github.com/shamspias/laya-bangla";

export const PRESETS = {
  support: ["department", "churn_risk", "urgency"],
  grievance: ["grievance_category", "grievance_severity"],
  moderation: ["hate", "personal_attack", "abusive"],
  sentiment: ["sentiment"],
  assistant: ["massive_scenario"],
  topic: ["sib_topic"],
};

export const PRESET_LABELS = {
  grievance: "Citizen complaint: service sector and severity",
  assistant: "Voice assistant: what is the user asking for?",
  moderation: "Moderation: hate speech, personal attack, abuse",
  sentiment: "Sentiment: negative, neutral, or positive",
  topic: "Topic of a sentence",
  support: "Customer support: department, threat to leave, urgency",
  custom: "Write your own questions",
};

export const EXAMPLES = [
  ["আমাদের এলাকায় তিন দিন ধরে পানি নেই, বাচ্চারা অসুস্থ হয়ে পড়ছে।", "grievance"],
  ["রাস্তার পাশে ময়লার স্তূপ, দুর্গন্ধে হাঁটা যায় না।", "grievance"],
  ["কাল সকাল সাতটায় আমাকে জাগিয়ে দিও", "assistant"],
  ["আজকের আবহাওয়া কেমন? বৃষ্টি হবে?", "assistant"],
  ["Sir apnar class gulo onek helpful, onek dhonnobad!", "sentiment"],
  ["ডেলিভারি তিন দিন দেরি, খাবারও ঠান্ডা ছিল। খুবই হতাশ।", "sentiment"],
  ["Tor moto beyadob manush ke deshe rakha uchit na", "moderation"],
  ["বিশ্বকাপ ফাইনালে আর্জেন্টিনা টাইব্রেকারে জয় পেয়েছে।", "topic"],
  ["বিলে দুইবার টাকা কেটেছে, আজকের মধ্যে ফেরত না দিলে অন্য কোম্পানিতে চলে যাব।", "support"],
  ["App e login korte parchi na, kal theke kaj atke ache. Please ajkei fix korun.", "support"],
];

export const CUSTOM_TEMPLATE = {
  is_question: {
    type: "noul",
    instructions: "লেখাটি কি কোনো প্রশ্ন?",
    criteria: { false: "না, এটি প্রশ্ন নয়", true: "হ্যাঁ, এটি একটি প্রশ্ন" },
  },
  product: {
    type: "choice",
    instructions: "লেখাটি কোন পণ্য বা সেবা নিয়ে?",
    criteria: {
      mobile: "মোবাইল ফোন বা সিম",
      internet: "ইন্টারনেট বা ওয়াইফাই",
      bank: "ব্যাংক বা মোবাইল ব্যাংকিং",
      other: "অন্য কিছু",
    },
  },
};

export function renderQuestion(task, instructions, lang, options) {
  const criteria = task.type === "score"
    ? options.map((option) => option[lang])
    : Object.fromEntries(Object.entries(options).map(([label, option]) => [label, option[lang]]));
  return { type: task.type, instructions, criteria };
}

/** Questions for a preset, using the evaluation wording (never seen in training). */
export function presetQuestions(tasks, support, preset, lang) {
  if (preset === "support") return support[lang];
  if (!(preset in PRESETS)) throw new Error(`unknown preset ${preset}`);
  const questions = {};
  for (const name of PRESETS[preset]) {
    const task = tasks[name];
    questions[name] = renderQuestion(task, task.instructions.eval[lang], lang, task.criteria);
  }
  return questions;
}

/** Second routing stage: the intent among the predicted MASSIVE scenario's options. */
export function intentQuestion(tasks, scenario, lang) {
  const task = tasks.massive_intent;
  const options = task.groups[scenario];
  if (!options) return null;
  return { massive_intent: renderQuestion(task, task.instructions.eval[lang], lang, options) };
}

/** Validate a user-written question schema before it reaches the model. */
export function parseQuestions(text) {
  let questions;
  try {
    questions = JSON.parse(text);
  } catch (error) {
    throw new Error(`Questions are not valid JSON: ${error.message}`);
  }
  if (!questions || typeof questions !== "object" || Array.isArray(questions)) {
    throw new Error("Questions must be a JSON object with at least one question");
  }
  const entries = Object.entries(questions);
  if (entries.length === 0) throw new Error("Questions must be a JSON object with at least one question");
  if (entries.length > 8) throw new Error("Ask at most 8 questions at once");
  for (const [name, q] of entries) {
    if (!q || typeof q !== "object" || Array.isArray(q)) throw new Error(`${name}: each question must be an object`);
    if (!["choice", "noul", "score"].includes(q.type)) throw new Error(`${name}: type must be choice, noul, or score`);
    if (!String(q.instructions ?? "").trim()) throw new Error(`${name}: instructions are required`);
    const c = q.criteria;
    if (q.type === "score") {
      if (!Array.isArray(c) || c.length < 2) throw new Error(`${name}: a score question needs a list of at least 2 levels`);
    } else {
      if (!c || typeof c !== "object" || Array.isArray(c) || Object.keys(c).length < 2) {
        throw new Error(`${name}: criteria must map at least 2 option names to descriptions`);
      }
      const keys = Object.keys(c).sort().join(",");
      if (q.type === "noul" && keys !== "false,true") {
        throw new Error(`${name}: a noul question needs exactly the options false and true`);
      }
    }
  }
  return questions;
}

/** The predicted label and every option's probability, most likely first. */
export function answerRows(answer, question) {
  if (answer.type === "noul") {
    const yes = Number(answer.noul);
    const rows = [
      [`true: ${question.criteria.true}`, yes],
      [`false: ${question.criteria.false}`, 1 - yes],
    ].sort((a, b) => b[1] - a[1]);
    return [yes >= 0.5 ? "Yes / হ্যাঁ" : "No / না", rows];
  }
  const probabilities = answer.probabilities;
  if (answer.type === "score") {
    const levels = question.criteria;
    const rows = levels.map((level, i) => [`${i}: ${level}`, Number(probabilities[String(i)])]);
    let top = 0;
    rows.forEach((row, i) => { if (row[1] > rows[top][1]) top = i; });
    return [`Level ${top}: ${levels[top]}`, rows];
  }
  const criteria = question.criteria;
  const rows = Object.entries(probabilities)
    .map(([key, value]) => [`${key}: ${criteria[key] ?? key}`, Number(value)])
    .sort((a, b) => b[1] - a[1]);
  return [`${answer.choice}: ${criteria[answer.choice] ?? answer.choice}`, rows];
}

// [summary key, plain-language task, dataset, text]
export const BENCHMARKS = [
  ["massive/intent (end-to-end)", "Voice-assistant routing, two steps", "MASSIVE", "Bangla"],
  ["massive/massive_scenario", "Voice-assistant scenario (18 options)", "MASSIVE", "Bangla"],
  ["banglapsg/grievance_category", "Complaint service sector (6)", "BanglaPSG", "Bangla"],
  ["banglapsg/grievance_severity", "Complaint severity (3 levels)", "BanglaPSG", "Bangla"],
  ["banth/hate", "Hate speech", "BanTH", "Banglish"],
  ["banth/personal_attack", "Personal attack", "BanTH", "Banglish"],
  ["banth/abusive", "Abusive or violent language", "BanTH", "Banglish"],
  ["banhate/hate", "Hate speech", "BanHate", "Bangla"],
  ["elearning/sentiment", "Sentiment of course comments", "eLearning", "Bangla + Banglish"],
  ["sib200/sib_topic", "Topic of a sentence (7)", "SIB-200", "Bangla"],
];
export const UNSEEN = [
  ["sentnob/sentiment", "Sentiment of social-media comments", "SentNoB", "Bangla"],
  ["indicsentiment/sentiment", "Sentiment of product reviews", "IndicSentiment", "Bangla"],
  ["indicxnli/entailment", "Does sentence A imply B? (yes/no)", "IndicXNLI", "Bangla"],
];

export const ENGLISH = [
  ["massive/intent (end-to-end)", "Voice-assistant routing, two steps", "MASSIVE", "English"],
  ["massive_en/massive_scenario", "Voice-assistant scenario (18 options)", "MASSIVE", "English"],
  ["sib200_en/sib_topic", "Topic of a sentence (7)", "SIB-200", "English"],
  ["indicsentiment_en/sentiment", "Sentiment of product reviews", "IndicSentiment", "English"],
  ["tweeteval_sentiment/sentiment", "Sentiment of tweets", "TweetEval", "English"],
  ["tweeteval_hate/hate", "Hate speech in tweets", "TweetEval", "English"],
  ["xnli_en/entailment", "Does sentence A imply B? (yes/no)", "XNLI", "English"],
];

/** Accuracy for one benchmark row and question language, or null when it was not run. */
export function accuracy(summary, key, lang) {
  if (key === "massive/intent (end-to-end)") return summary.massive_end_to_end_intent?.[lang] ?? null;
  return summary.tasks?.[key]?.[lang]?.all ?? null;
}
