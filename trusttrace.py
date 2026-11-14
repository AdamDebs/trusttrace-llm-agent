# TrustTrace: an evidence-grounded LLM agent for messy patient symptom diaries
# Script version of TrustTrace.ipynb. Install first: pip install -r requirements.txt


# %%
# # TrustTrace: an evidence-grounded LLM agent for messy patient symptom diaries
#
# Patients often record symptoms as messy daily notes: slang, typos, skipped days, entries out of order, and phrases that only *sound* like symptoms ("sick of this weather", "had red wine"). A language model that summarises these notes for a clinician has two jobs that pull against each other: it must not invent symptoms, and it must not miss a red flag.
#
# **TrustTrace** is a small LLM agent that reads a 14-day diary, extracts each symptom with a quote from the entry it came from, checks that evidence with deterministic tools, writes a clinician summary that may only use verified facts, and decides whether to escalate the case, with the escalation threshold set from the cost of each error (a missed red flag costs 5x an unnecessary review).
#
# It is evaluated on synthetic diaries whose ground truth is known exactly, so every hallucination can be counted:
#
# 1. **Keyword baseline:** a symptom lexicon with simple negation handling.
# 2. **LLM only:** the same model asked for the symptoms, a summary and an escalation decision in one step, with no tools.
# 3. **Agent v1 (whole diary):** one extraction prompt for the whole diary, then the same verification, summary and escalation steps as v2.
# 4. **TrustTrace v2:** splits each entry into numbered phrases and asks the model, one entry at a time, which phrases show a symptom; verifies every answer with deterministic tools, sends rejected answers back to the model with the reason so it can retry, computes trends, summarises only verified facts (checking every symptom and number in the summary), and escalates with a cost-aware threshold chosen on validation data.
# 5. **TrustTrace v3 (hybrid):** a keyword tool and the v2 extraction both propose candidate (phrase, symptom) pairs; the model then judges each candidate with a yes/no question, giving a probability from its own token scores; candidates are accepted above thresholds chosen on validation data, and the same summary and escalation steps follow.
#
# Comparing v1, v2 and v3 shows how much the agent's design matters for a small model.
#
# **Model:** Qwen2.5-1.5B-Instruct (Apache 2.0, runs on Colab's free T4 GPU). Without a GPU, the notebook switches to Qwen2.5-0.5B-Instruct and a smaller sample.
#
# **Run:** `Runtime > Change runtime type > T4 GPU`, then `Runtime > Run all` (about 30 to 35 minutes on a T4).

# %%
# ## 1. Setup and configuration

# %%
import os, re, json, math, time, random, csv
import numpy as np
import matplotlib.pyplot as plt
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ON_GPU = DEVICE.type == "cuda"
BACKEND = os.environ.get("TRUSTTRACE_BACKEND", "hf")   # "hf" = real model; "mock" = test double used only for software tests

CFG = {
    "model_name": os.environ.get("TRUSTTRACE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct" if ON_GPU else "Qwen/Qwen2.5-0.5B-Instruct"),
    "n_val": 40 if ON_GPU else 12,
    "n_test": 80 if ON_GPU else 24,
    "days": 14,
    "batch_size": 8 if ON_GPU else 4,
    "batch_size_entry": 24 if ON_GPU else 8,
    "max_new_tokens_entry": 96,
    "batch_size_judge": 16 if ON_GPU else 4,
    "max_new_tokens_extract": 384,
    "max_new_tokens_direct": 384,
    "max_new_tokens_summary": 160,
    "cost_missed_escalation": 5.0,
    "cost_unneeded_review": 1.0,
    "n_bootstrap": 1000,
    "n_human_check": 30,
    "seed": 0,
    "out_dir": "./results",
}
random.seed(CFG["seed"]); np.random.seed(CFG["seed"]); torch.manual_seed(CFG["seed"])
os.makedirs(CFG["out_dir"], exist_ok=True)
print(f"Device: {DEVICE} | backend: {BACKEND} | model: {CFG['model_name']} | "
      f"val {CFG['n_val']}, test {CFG['n_test']} diaries")

# %%
# ## 2. Synthetic diaries with exact ground truth
# Each diary covers 14 days. Symptoms follow a worsening, improving or stable pattern; entries include slang, negations, distractor phrases, filler, one-word days, skipped days and some out-of-order entries. Red flags are simplified for this study (blood in stool, fever, weight loss, vomiting) and are not clinical guidance.

# %%
SYMPTOMS = {
    "abdominal pain": ["tummy pain", "stomach cramps", "belly ache", "stomach killing me", "cramping again"],
    "diarrhoea":      ["the runs", "loose stools", "diarrhea", "on the loo all morning", "bathroom 6 times"],
    "fatigue":        ["knackered", "exhausted", "wiped out", "running on empty", "tired all day"],
    "nausea":         ["felt queasy", "nauseous", "felt like throwing up"],
    "vomiting":       ["threw up", "vomited", "was sick twice"],
    "fever":          ["temp 38.5", "feverish", "hot and shivery"],
    "blood in stool": ["blood in my poo", "bit of blood when i went", "blood in the toilet"],
    "weight loss":    ["lost 2kg", "weight down again", "jeans are loose"],
    "joint pain":     ["knees aching", "achy joints", "wrists sore"],
    "poor sleep":     ["barely slept", "couldn't sleep", "up all night"],
}
SYMPTOM_NAMES = list(SYMPTOMS)
RED_FLAGS = {"blood in stool", "fever", "weight loss", "vomiting"}
NEGATIONS = {
    "abdominal pain": ["no tummy pain", "stomach fine today"], "diarrhoea": ["no runs today"],
    "fatigue": ["not tired"], "nausea": ["no nausea"], "vomiting": ["didn't throw up", "no vomiting"],
    "fever": ["no fever", "temp normal"], "blood in stool": ["no blood today", "no blood this time"],
    "weight loss": ["weight stable"], "joint pain": ["joints fine"], "poor sleep": ["slept well"],
}
DISTRACTORS = [("had red wine last night", "blood in stool"), ("sick of this weather", "nausea"),
               ("felt a bit warm on the bus", "fever"), ("tired of waiting for my results", "fatigue"),
               ("lost my keys again", "weight loss"), ("work is a pain in the neck", "abdominal pain")]
FILLER = ["work was busy", "ate pasta", "walked the dog", "rainy day", "saw friends", "long shift", "watched a film"]
ONE_WORD = ["fine", "ok", "meh", "good day", "same as yesterday"]
SEVERITY = ["", "", "bit of ", "really bad ", "mild "]

def make_diary(rng, idx):
    days = CFG["days"]; half = days // 2
    active = rng.sample([s for s in SYMPTOM_NAMES if s not in RED_FLAGS], rng.randint(2, 3))
    if rng.random() < 0.45:
        active.append(rng.choice(sorted(RED_FLAGS)))
    pattern = {s: rng.choice(["worsening", "improving", "stable"]) for s in active}
    entries, truth, negated, distract = [], set(), set(), set()
    for d in range(1, days + 1):
        if rng.random() > 0.75:
            continue                                          # skipped day
        frags = []
        for s in active:
            pos = (d - 1) / (days - 1)
            p = {"worsening": 0.1 + 0.6 * pos, "improving": 0.7 - 0.6 * pos, "stable": 0.4}[pattern[s]]
            if rng.random() < p:
                sev = rng.choice(SEVERITY) if s in ("abdominal pain", "joint pain") else ""
                frags.append(sev + rng.choice(SYMPTOMS[s])); truth.add((s, d))
        if rng.random() < 0.18:
            s = rng.choice([x for x in SYMPTOM_NAMES if (x, d) not in truth])
            frags.append(rng.choice(NEGATIONS[s])); negated.add((s, d))
        if rng.random() < 0.18:
            phrase, s = rng.choice(DISTRACTORS)
            if (s, d) not in truth:
                frags.append(phrase); distract.add((s, d))
        if rng.random() < 0.5:
            frags.append(rng.choice(FILLER))
        if not frags:
            frags = [rng.choice(ONE_WORD)]
        rng.shuffle(frags)
        entries.append({"day": d, "text": ", ".join(frags)})
    for i in range(len(entries) - 1):                          # some entries written out of order
        if rng.random() < 0.15:
            entries[i], entries[i + 1] = entries[i + 1], entries[i]
    for i, e in enumerate(entries):
        e["id"] = f"E{i + 1}"
    present = sorted({s for s, _ in truth})
    trends = {}
    for s in present:
        a = sum(1 for (x, d) in truth if x == s and d <= half); b = sum(1 for (x, d) in truth if x == s and d > half)
        trends[s] = "worsening" if b - a >= 2 else "improving" if a - b >= 2 else "stable"
    escalate = any(s in RED_FLAGS and d > half for s, d in truth)
    return {"id": f"D{idx:03d}", "entries": entries, "truth": truth, "negated": negated, "distract": distract,
            "present": present, "trends": trends, "escalate": escalate}

def diary_text(diary):
    return "\n".join(f"{e['id']} (Day {e['day']}): {e['text']}" for e in diary["entries"])

rng_data = random.Random(CFG["seed"])
VAL = [make_diary(rng_data, i) for i in range(CFG["n_val"])]
TEST = [make_diary(rng_data, i + 1000) for i in range(CFG["n_test"])]
print("Example diary:\n" + diary_text(TEST[0]))
print("\nGround truth symptoms:", TEST[0]["present"], "| escalate:", TEST[0]["escalate"])
print(f"\nTest set: {sum(len(d['truth']) for d in TEST)} symptom events, "
      f"{sum(d['escalate'] for d in TEST)} of {len(TEST)} diaries need escalation")

# %%
# ## 3. Language model backend
# The real backend runs Qwen locally with batched generation, and reads yes/no probabilities straight from its next-token scores for the v3 judge. A deterministic mock backend exists only so the software can be tested without a GPU; it is never used for reported results.

# %%
SYSTEM_PROMPT = "You are a careful clinical information assistant. You only report what the text supports, and you answer in the exact format requested."

class HFBackend:
    def __init__(self, model_name, device, batch_size):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name)
        if device.type == "cuda":
            model = model.half()
        self.model = model.to(device).eval()
        self.device, self.batch_size = device, batch_size

    @torch.no_grad()
    def generate(self, prompts, max_new_tokens, label=""):
        outputs, t0 = [], time.time()
        for i in range(0, len(prompts), self.batch_size):
            chunk = prompts[i:i + self.batch_size]
            texts = [self.tok.apply_chat_template([{"role": "system", "content": SYSTEM_PROMPT}] +
                                                  (p if isinstance(p, list) else [{"role": "user", "content": p}]),
                                                  tokenize=False, add_generation_prompt=True) for p in chunk]
            enc = self.tok(texts, return_tensors="pt", padding=True).to(self.device)
            gen = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                      pad_token_id=self.tok.pad_token_id)
            outputs += self.tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            print(f"  {label}: {min(i + self.batch_size, len(prompts))}/{len(prompts)} ({time.time() - t0:.0f}s)")
        return outputs

    def _first_ids(self, words):
        """Token ids of the answer words that are a single token for this tokenizer."""
        ids = [self.tok.encode(w, add_special_tokens=False) for w in words]
        single = sorted({x[0] for x in ids if len(x) == 1})
        return single or sorted({x[0] for x in ids if x})   # fallback for unusual tokenizers

    @torch.no_grad()
    def yes_prob(self, prompts, label=""):
        """P(yes) for yes/no questions, read from the model's next-token scores (no text generation, so nothing to parse)."""
        yes_ids = self._first_ids(["yes", "Yes", " yes", " Yes", "YES"])
        no_ids = self._first_ids(["no", "No", " no", " No", "NO"])
        base, head = getattr(self.model, "model", None), self.model.get_output_embeddings()
        bs, out, t0 = CFG["batch_size_judge"], [], time.time()
        for i in range(0, len(prompts), bs):
            chunk = prompts[i:i + bs]
            texts = [self.tok.apply_chat_template([{"role": "system", "content": SYSTEM_PROMPT}] + p,
                                                  tokenize=False, add_generation_prompt=True) for p in chunk]
            enc = self.tok(texts, return_tensors="pt", padding=True).to(self.device)
            pos = (enc["attention_mask"].cumsum(-1) - 1).clamp(min=0)
            if base is not None and head is not None:
                hidden = base(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], position_ids=pos).last_hidden_state
                logits = head(hidden[:, -1, :]).float()
            else:
                logits = self.model(**enc).logits[:, -1, :].float()
            lp = torch.log_softmax(logits, dim=-1)
            p = torch.sigmoid(torch.logsumexp(lp[:, yes_ids], -1) - torch.logsumexp(lp[:, no_ids], -1))
            out += [float(x) if math.isfinite(float(x)) else 0.5 for x in p.cpu()]
            print(f"  {label}: {min(i + bs, len(prompts))}/{len(prompts)} ({time.time() - t0:.0f}s)")
        return out

class MockBackend:
    """Test double for software testing only: imitates a small, error-prone model (wrong formats, invented
    symptoms, negation mistakes) so every parsing and checking path is exercised."""
    def __init__(self, seed=0):
        self.rng = random.Random(seed)
    def _entries(self, prompt):
        return [(m.group(1), int(m.group(2)), m.group(3)) for m in re.finditer(r"^(E\d+) \(Day (\d+)\): (.*)$", prompt, re.M)]
    def _find(self, text):
        hits = []
        for s, forms in SYMPTOMS.items():
            for f in forms:
                if f in text:
                    hits.append((s, f))
        for s, forms in NEGATIONS.items():
            for f in forms:
                if f in text and self.rng.random() < 0.3:
                    hits.append((s, f))                       # negation mistake
        for phrase, s in DISTRACTORS:
            if phrase in text and self.rng.random() < 0.3:
                hits.append((s, phrase))                      # distractor mistake
        return hits
    def _entry_json(self, text, careful=False):
        items = [{"symptom": s, "quote": q} for s, q in self._find(text) if careful or self.rng.random() < 0.95]
        if not careful and self.rng.random() < 0.12:
            items.append({"symptom": "fatigue", "quote": "feeling rough"})          # paraphrased quote
        if not careful and self.rng.random() < 0.05:
            items.append({"symptom": "headache", "quote": text.split(",")[0]})       # symptom outside the list
        return json.dumps({"symptoms": items})
    def _one(self, prompt):
        r = self.rng.random()
        if isinstance(prompt, list):
            prompt = prompt[-1]["content"]
            careful = "Your previous answer" in prompt
            frags = re.findall(r"^(\d+)\. (.*)$", prompt, re.M)
            items = []
            for num, text in frags:
                for sname, _ in self._find(text):
                    if careful or self.rng.random() < 0.95:
                        items.append({self.rng.choice(["phrase", "phrase_number"]): int(num), "symptom": sname})
            if not careful and self.rng.random() < 0.10:
                items.append({"phrase": len(frags) + 3, "symptom": "fatigue"})       # phrase that does not exist
            if not careful and self.rng.random() < 0.05:
                items.append({"phrase": 1, "symptom": "headache"})                    # symptom outside the list
            out = json.dumps({"symptoms": items})
            return out[:-8] if r < 0.04 else ("```json\n" + out + "\n```" if r < 0.2 else out)
        if "TASK: EXTRACT" in prompt:
            ev = []
            for eid, day, text in self._entries(prompt):
                for s, q in self._find(text):
                    if self.rng.random() < 0.9:
                        ev.append({"entry": eid, "symptom": s, "quote": q})
                if self.rng.random() < 0.08:
                    ev.append({"entry": eid, "symptom": self.rng.choice(SYMPTOM_NAMES), "quote": "felt awful"})
            out = json.dumps({"events": ev})
            return out[:-15] if r < 0.05 else ("```json\n" + out + "\n```" if r < 0.25 else out)
        if "TASK: DIRECT" in prompt:
            days = {}
            for eid, day, text in self._entries(prompt):
                for s, q in self._find(text):
                    if self.rng.random() < 0.8:
                        days.setdefault(s, set()).add(day)
            if self.rng.random() < 0.25:
                days.setdefault(self.rng.choice(SYMPTOM_NAMES), set()).add(self.rng.randint(1, 14))
            esc = any(s in RED_FLAGS and any(d > 7 for d in v) for s, v in days.items())
            out = json.dumps({"symptoms": [{"name": s, "days": sorted(v)} for s, v in days.items()],
                              "escalate": esc if self.rng.random() > 0.1 else (not esc),
                              "summary": "Patient reports " + ", ".join(days) + "."})
            return out[:-10] if r < 0.05 else out
        facts = re.search(r"Facts \(JSON\): (.*)$", prompt, re.M)
        names = [f["symptom"] for f in json.loads(facts.group(1))["symptoms"]] if facts else []
        extra = [self.rng.choice(SYMPTOM_NAMES)] if self.rng.random() < 0.15 else []
        return "Over 14 days the patient reported " + (", ".join(names + extra) or "no symptoms") + "."
    def generate(self, prompts, max_new_tokens, label=""):
        print(f"  {label}: {len(prompts)} prompts (mock)")
        return [self._one(p) for p in prompts]
    def yes_prob(self, prompts, label=""):
        print(f"  {label}: {len(prompts)} questions (mock)")
        out = []
        for p in prompts:
            m = re.search(r'Diary phrase: "(.*)"\nDoes this phrase say the patient HAD (.+?) \(', p[-1]["content"])
            truth = candidate_is_true(m.group(1), m.group(2)) if m else False
            right = self.rng.random() < (0.9 if truth else 0.8)
            hi, lo = self.rng.uniform(0.55, 0.99), self.rng.uniform(0.01, 0.45)
            out.append(hi if truth == right else lo)
        return out

if BACKEND == "mock":
    print("MOCK BACKEND: for software testing only. These results are NOT model results.")
    LLM = MockBackend(CFG["seed"])
else:
    print("Loading", CFG["model_name"], "...")
    LLM = HFBackend(CFG["model_name"], DEVICE, CFG["batch_size"])
    print("Model loaded.")
LLM_ENTRY = LLM
if BACKEND != "mock":
    class _EntryView:                      # same model, bigger batches for the short per-entry prompts
        def generate(self, prompts, max_new_tokens, label=""):
            old, LLM.batch_size = LLM.batch_size, CFG["batch_size_entry"]
            try:
                return LLM.generate(prompts, max_new_tokens, label)
            finally:
                LLM.batch_size = old
    LLM_ENTRY = _EntryView()

# %%
# ## 4. Parsing and evidence-checking tools

# %%
NEG_CUES = {"no", "not", "didn't", "didnt", "never", "without", "wasn't", "isn't", "dont", "don't"}
ALIASES = {
    "abdominal pain": ["abdominal", "stomach pain", "tummy", "belly", "cramp"], "diarrhoea": ["diarrhoea", "diarrhea", "loose stool"],
    "fatigue": ["fatigue", "tired", "exhaust"], "nausea": ["nausea", "nauseous", "queasy"], "vomiting": ["vomit"],
    "fever": ["fever", "temperature"], "blood in stool": ["blood"], "weight loss": ["weight loss", "lost weight", "losing weight"],
    "joint pain": ["joint"], "poor sleep": ["sleep", "insomnia"],
}

def extract_json(text):
    """Return the first JSON object in a model output, repairing common small-model mistakes; None if impossible."""
    if not isinstance(text, str):
        return None
    t = re.sub(r"```(?:json)?", "", text).strip()
    start, end = t.find("{"), t.rfind("}")
    if start == -1 or end <= start:
        return None
    cand = t[start:end + 1]
    for attempt in (cand, re.sub(r",\s*([}\]])", r"\1", cand), cand.replace("'", '"')):
        try:
            obj = json.loads(attempt)
            return obj if isinstance(obj, dict) else None
        except Exception:
            continue
    return None

def canon_symptom(name):
    if not isinstance(name, str):
        return None
    n = name.strip().lower()
    if n in SYMPTOMS:
        return n
    for s, al in ALIASES.items():
        if any(a in n for a in al):
            return s
    return None

def words_before(low, pos, n=3):
    """The last n words before position pos, within the same phrase (stops at punctuation)."""
    return re.findall(r"[a-z']+", re.split(r"[,.;:!?\n]", low[:pos])[-1])[-n:]

def is_negated(entry_text, quote):
    """True if the quote is itself a negation or a negation cue appears in the three words before it."""
    low, q = entry_text.lower(), quote.lower().strip()
    pos = low.find(q)
    if pos == -1:
        return False
    first = re.findall(r"[a-z']+", q)[:1]
    return bool((first and first[0] in NEG_CUES) or any(w in NEG_CUES for w in words_before(low, pos)))

def norm_entry(x):
    m = re.search(r"E\d+", str(x or ""), re.I)
    return m.group(0).upper() if m else ""

def clean_quote(q):
    return str(q or "").strip().strip(" '\"\u201c\u201d\u2018\u2019.,;:!?").strip().lower()

def to_day(x):
    m = re.search(r"\d+", str(x))
    return int(m.group(0)) if m else None

def verify_event(ev, entries_by_id):
    """Deterministic evidence check. Returns (accepted, reason)."""
    sym = canon_symptom(ev.get("symptom")) if isinstance(ev, dict) else None
    if sym is None:
        return False, "unknown symptom"
    e = entries_by_id.get(norm_entry(ev.get("entry")))
    if e is None:
        return False, "entry not found"
    quote = clean_quote(ev.get("quote"))
    if len(quote) < 3 or quote not in e["text"].lower():
        return False, "quote not in entry"
    if is_negated(e["text"], quote):
        return False, "negated"
    return True, "ok"

def trends_from_events(events):
    half = CFG["days"] // 2
    out = {}
    for s in sorted({s for s, _ in events}):
        a = sum(1 for (x, d) in events if x == s and d <= half); b = sum(1 for (x, d) in events if x == s and d > half)
        out[s] = "worsening" if b - a >= 2 else "improving" if a - b >= 2 else "stable"
    return out

def claimed_symptoms(text):
    """Symptoms a free-text summary asserts (mentions preceded by a negation cue are ignored)."""
    low, found = (text or "").lower(), set()
    for s in SYMPTOM_NAMES:
        for a in [s] + ALIASES[s]:
            for m in re.finditer(re.escape(a), low):
                if not any(w in NEG_CUES for w in words_before(low, m.start())):
                    found.add(s)
    return found

# %%
# ## 5. Methods: keyword baseline, LLM only, and TrustTrace agents v1, v2 and v3

# %%
LEXICON = {   # keyword baseline: realistic trigger words, including ambiguous ones
    "abdominal pain": ["tummy", "stomach", "belly", "cramp", "pain"], "diarrhoea": ["runs", "loose", "diarr", "loo", "bathroom"],
    "fatigue": ["tired", "exhausted", "knackered", "wiped", "energy"], "nausea": ["queasy", "nause", "sick", "throwing up"],
    "vomiting": ["threw up", "vomit", "sick twice"], "fever": ["fever", "temp", "warm", "shivery"],
    "blood in stool": ["blood", "red"], "weight loss": ["lost", "weight", "loose"],
    "joint pain": ["knee", "joint", "wrist"], "poor sleep": ["slept", "sleep", "up all night"],
}
SYMPTOM_LIST = ", ".join(SYMPTOM_NAMES)

def keyword_baseline(diary):
    events = set()
    for e in diary["entries"]:
        low = e["text"].lower()
        for s, words in LEXICON.items():
            for w in words:
                for m in re.finditer(re.escape(w), low):
                    if not any(x in NEG_CUES for x in words_before(low, m.start())):
                        events.add((s, e["day"]))
    present = {s for s, _ in events}
    return {"events": events, "claimed": present, "escalate": any(s in RED_FLAGS and d > CFG["days"] // 2 for s, d in events),
            "summary": "Keyword matches: " + ", ".join(sorted(present)), "parse_failed": False}

def prompt_direct(diary):
    return (f"TASK: DIRECT\nAllowed symptoms: {SYMPTOM_LIST}.\n"
            "Read the patient's diary. Return only JSON in this format:\n"
            '{"symptoms": [{"name": "fatigue", "days": [2, 5]}], "escalate": true, "summary": "two or three sentences for a clinician"}\n'
            "List only symptoms the patient says they had. Set escalate to true if blood in stool, fever, weight loss or vomiting "
            "appears on days 8 to 14.\n\nDiary:\n" + diary_text(diary))

def llm_only(diaries):
    raws = LLM.generate([prompt_direct(d) for d in diaries], CFG["max_new_tokens_direct"], label="LLM only")
    RAW_SAMPLES.extend({"method": "LLM only", "diary": d["id"], "output": r} for d, r in list(zip(diaries, raws))[:5])
    out = []
    for d, raw in zip(diaries, raws):
        obj = extract_json(raw)
        events, claimed = set(), set()
        if obj:
            for item in obj.get("symptoms", []) if isinstance(obj.get("symptoms"), list) else []:
                s = canon_symptom(item.get("name")) if isinstance(item, dict) else None
                if s is None:
                    continue
                claimed.add(s)
                for day in item.get("days", []) if isinstance(item.get("days"), list) else []:
                    dd = to_day(day)
                    if dd is not None:
                        events.add((s, dd))
        summary = str(obj.get("summary", "")) if obj else ""
        claimed |= claimed_symptoms(summary)
        out.append({"events": events, "claimed": claimed, "escalate": (str(obj.get("escalate")).strip().lower() in ("true", "yes", "1")) if obj else False,
                    "summary": summary, "parse_failed": obj is None, "raw": raw})
    return out

def prompt_extract(diary):
    return (f"TASK: EXTRACT\nAllowed symptoms: {SYMPTOM_LIST}.\n"
            "Read the patient's diary and list every symptom the patient says they HAD.\n"
            "Rules: use only the allowed symptom names; skip symptoms the patient says they did not have (for example 'no fever'); "
            "skip phrases that only sound like symptoms (for example 'sick of the rain'); for each mention give the entry id "
            "and copy the exact words from that entry as the quote.\n"
            'Return only JSON in this format: {"events": [{"entry": "E1", "symptom": "fatigue", "quote": "knackered"}]}\n\n'
            "Diary:\n" + diary_text(diary))

def prompt_summary(facts):
    return ("TASK: SUMMARISE\nWrite two or three sentences for a clinician summarising this patient's last 14 days. "
            "Use only these verified facts and mention no other symptoms.\n"
            f"Facts (JSON): {json.dumps(facts)}\nReturn only the summary text.")

def template_summary(facts):
    if not facts["symptoms"]:
        return "No symptoms were verified in the diary."
    parts = [f"{f['symptom']} on {len(f['days'])} day(s), {f['trend']}" for f in facts["symptoms"]]
    flags = f" Red flags: {', '.join(facts['red_flags'])}." if facts["red_flags"] else ""
    return "Verified symptoms: " + "; ".join(parts) + "." + flags

def evidence_score(events):
    half = CFG["days"] // 2
    return sum(1.0 for s, day in events if s in RED_FLAGS and day > half) + \
           0.25 * sum(1.0 for s, day in events if s in RED_FLAGS and day <= half)

WORDNUM = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen".split())}

def numbers_in(text):
    low = (text or "").lower()
    return {int(x) for x in re.findall(r"\b\d+\b", low)} | {WORDNUM[w] for w in re.findall(r"[a-z]+", low) if w in WORDNUM}

def check_summary(text, facts):
    """Return the reasons a free-text summary is not supported by the verified facts (empty list = faithful)."""
    allowed = {f["symptom"] for f in facts["symptoms"]}
    mentioned = claimed_symptoms(text)
    ok_nums = {CFG["days"]} | {d for f in facts["symptoms"] for d in f["days"]} | {len(f["days"]) for f in facts["symptoms"]}
    reasons = []
    if not (text or "").strip():
        reasons.append("empty")
    if mentioned - allowed:
        reasons.append("unverified symptom")
    if allowed and not (mentioned & allowed):
        reasons.append("ignores verified symptoms")
    if facts["red_flags"] and not (mentioned & set(facts["red_flags"])):
        reasons.append("omits red flag")
    if numbers_in(text) - ok_nums:
        reasons.append("unsupported number")
    return reasons

def finish_agent(stage, summarise=True):
    """Shared by v1, v2 and v3: summarise verified facts, check the summary, and score escalation evidence.
    Validation runs only need the escalation score, so they skip the model-written summary."""
    if summarise:
        summaries = LLM.generate([prompt_summary(s["facts"]) for s in stage], CFG["max_new_tokens_summary"], label="Agent summarise")
    else:
        summaries = [template_summary(s["facts"]) for s in stage]
    for s, text in zip(stage, summaries):
        reasons = check_summary(text, s["facts"])
        s["summary_llm"] = text.strip()
        s["summary_issues"] = reasons
        s["summary_unsupported"] = sorted(claimed_symptoms(text) - {f["symptom"] for f in s["facts"]["symptoms"]})
        s["summary_replaced"] = bool(reasons)
        s["summary"] = template_summary(s["facts"]) if reasons else text.strip()
        s["claimed"] = {f["symptom"] for f in s["facts"]["symptoms"]}
        s["score"] = evidence_score(s["events"])
    return stage

def build_facts(verified):
    trends = trends_from_events(verified)
    half = CFG["days"] // 2
    return {"symptoms": [{"symptom": s, "days": sorted(day for x, day in verified if x == s), "trend": trends[s]}
                         for s in sorted({s for s, _ in verified})],
            "red_flags": sorted({s for s, day in verified if s in RED_FLAGS and day > half})}

PHRASE_KEYS = ("phrase", "phrase_number", "phrase number", "fragment", "number", "index", "id")

def fragments(entry):
    return [f.strip() for f in entry["text"].split(",") if f.strip()]

def _entry_user(frags):
    numbered = "\n".join(f"{i + 1}. {f}" for i, f in enumerate(frags))
    return (f"TASK: ENTRY\nAllowed symptoms: {SYMPTOM_LIST}.\nEntry phrases:\n{numbered}\n"
            "For each phrase where the patient says they HAD one of the allowed symptoms, give the phrase number and the symptom. "
            "Skip phrases where they say they did not have it, phrases that only sound like a symptom, and symptoms not on the list. "
            "Answer with JSON only.")

FEW_SHOT = [   # worked examples; none of these phrases appear in the generated diaries
    {"role": "user", "content": _entry_user(["slight cough", "no fever", "went to the gym"])},
    {"role": "assistant", "content": '{"symptoms": []}'},
    {"role": "user", "content": _entry_user(["tummy was rumbling and sore", "sick of the rain", "so drained after work"])},
    {"role": "assistant", "content": '{"symptoms": [{"phrase": 1, "symptom": "abdominal pain"}, {"phrase": 3, "symptom": "fatigue"}]}'},
    {"role": "user", "content": _entry_user(["quiet day at home"])},
    {"role": "assistant", "content": '{"symptoms": []}'},
]

def prompt_entry(entry):
    return FEW_SHOT + [{"role": "user", "content": _entry_user(fragments(entry))}]

def prompt_retry(entry, problems):
    lines = "\n".join(f"- {p}" for p in problems)
    return FEW_SHOT + [{"role": "user", "content": _entry_user(fragments(entry)) +
                        f"\nYour previous answer for these phrases had problems:\n{lines}\n"
                        "Answer again using only phrase numbers that exist and only the allowed symptom names."}]

def parse_entry_items(raw):
    obj = extract_json(raw)
    if obj is None:
        return [], True
    items = obj.get("symptoms", [])
    return (items if isinstance(items, list) else []), False

RAW_SAMPLES = []

def trusttrace_v2(diaries, retry=True, summarise=True):
    jobs = [(i, e) for i, d in enumerate(diaries) for e in d["entries"]]
    raws = LLM_ENTRY.generate([prompt_entry(e) for _, e in jobs], CFG["max_new_tokens_entry"], label="TrustTrace entries")
    RAW_SAMPLES.extend({"method": "TrustTrace entry", "entry": e["text"], "output": r} for (_, e), r in list(zip(jobs, raws))[:40])
    stage = [{"raw_events": set(), "events": set(), "rejected": {}, "failed_entries": 0, "n_entries": len(d["entries"]),
              "retried": 0, "recovered": 0, "llm_cands": set()} for d in diaries]
    retry_jobs = []
    def absorb(i, e, items, count_raw=True):
        problems, gained = [], 0
        frags = fragments(e)
        for ev in items:
            if not isinstance(ev, dict):
                continue
            key = next((x for x in PHRASE_KEYS if x in ev), None)
            k = to_day(ev.get(key)) if key else None
            if k is None or not 1 <= k <= len(frags):
                stage[i]["rejected"]["phrase not found"] = stage[i]["rejected"].get("phrase not found", 0) + 1
                problems.append(f"phrase number {ev.get(key) if key else 'missing'} does not exist")
                continue
            ev = {"entry": e["id"], "symptom": ev.get("symptom"), "quote": frags[k - 1]}
            s = canon_symptom(ev.get("symptom"))
            if s:
                stage[i]["llm_cands"].add((e["id"], k, s))
            if s and count_raw:
                stage[i]["raw_events"].add((s, e["day"]))
            ok, why = verify_event(ev, {e["id"]: e})
            if ok:
                if (s, e["day"]) not in stage[i]["events"]:
                    gained += 1
                stage[i]["events"].add((s, e["day"]))
            else:
                stage[i]["rejected"][why] = stage[i]["rejected"].get(why, 0) + 1
                if why == "unknown symptom":
                    problems.append(f"'{ev.get('symptom')}' is not an allowed symptom")
        return problems, gained
    for (i, e), raw in zip(jobs, raws):
        items, failed = parse_entry_items(raw)
        if failed:
            stage[i]["failed_entries"] += 1
        problems, _ = absorb(i, e, items)
        if failed:
            problems.append("the answer was not valid JSON")
        if problems:
            retry_jobs.append((i, e, problems))
    if retry and retry_jobs:
        raws2 = LLM_ENTRY.generate([prompt_retry(e, p) for _, e, p in retry_jobs], CFG["max_new_tokens_entry"], label="TrustTrace retries")
        for (i, e, _), raw in zip(retry_jobs, raws2):
            items, failed = parse_entry_items(raw)
            stage[i]["retried"] += 1
            _, gained = absorb(i, e, items, count_raw=False)
            stage[i]["recovered"] += gained
    for s in stage:
        s["facts"] = build_facts(s["events"])
        s["parse_failed"] = s["failed_entries"] / max(s["n_entries"], 1)
    return finish_agent(stage, summarise)

def trusttrace_agent(diaries, summarise=True):
    raws = LLM.generate([prompt_extract(d) for d in diaries], CFG["max_new_tokens_extract"], label="Agent extract")
    RAW_SAMPLES.extend({"method": "Agent v1", "diary": d["id"], "output": r} for d, r in list(zip(diaries, raws))[:5])
    stage = []
    for d, raw in zip(diaries, raws):
        by_id = {e["id"]: e for e in d["entries"]}
        obj = extract_json(raw)
        items = obj.get("events", []) if obj and isinstance(obj.get("events"), list) else []
        raw_events, verified, rejected = set(), set(), {}
        for ev in items:
            s = canon_symptom(ev.get("symptom")) if isinstance(ev, dict) else None
            e = by_id.get(norm_entry(ev.get("entry"))) if isinstance(ev, dict) else None
            if s and e:
                raw_events.add((s, e["day"]))
            ok, why = verify_event(ev, by_id)
            if ok:
                verified.add((s, e["day"]))
            else:
                rejected[why] = rejected.get(why, 0) + 1
        stage.append({"raw_events": raw_events, "events": verified, "facts": build_facts(verified), "rejected": rejected,
                      "parse_failed": obj is None, "raw": raw})
    return finish_agent(stage, summarise)

# ---------- TrustTrace v3: keyword and LLM proposals, judged by the model ----------
DESCRIPTIONS = {
    "abdominal pain": "pain or cramps in the stomach area", "diarrhoea": "frequent watery bowel movements",
    "fatigue": "tiredness or lack of energy", "nausea": "feeling queasy without being sick",
    "vomiting": "actually being sick or throwing up", "fever": "a high temperature",
    "blood in stool": "blood when going to the toilet", "weight loss": "losing body weight",
    "joint pain": "aching or sore joints", "poor sleep": "trouble sleeping",
}

def _judge_user(phrase, symptom):
    return (f'TASK: JUDGE\nDiary phrase: "{phrase}"\n'
            f"Does this phrase say the patient HAD {symptom} ({DESCRIPTIONS[symptom]})? Answer yes or no.")

JUDGE_SHOT = []   # worked examples; none of these phrases appear in the generated diaries
for _phrase, _symptom, _answer in [("stomach was churning and sore", "abdominal pain", "yes"), ("sick of the traffic", "nausea", "no"),
                                   ("fever has gone now", "fever", "no"), ("could barely keep my eyes open", "fatigue", "yes"),
                                   ("red sauce on my pasta", "blood in stool", "no"), ("fingers stiff and sore", "joint pain", "yes"),
                                   ("slept like a log", "poor sleep", "no"), ("ran to the toilet all day", "diarrhoea", "yes")]:
    JUDGE_SHOT += [{"role": "user", "content": _judge_user(_phrase, _symptom)}, {"role": "assistant", "content": _answer}]

def prompt_judge(phrase, symptom):
    return JUDGE_SHOT + [{"role": "user", "content": _judge_user(phrase, symptom)}]

def keyword_candidates(entry):
    """(phrase number, symptom) pairs proposed by the keyword lexicon, with the same negation rule as the baseline."""
    out = set()
    for k, frag in enumerate(fragments(entry), 1):
        low = frag.lower()
        for s, words in LEXICON.items():
            for w in words:
                for m in re.finditer(re.escape(w), low):
                    if not any(x in NEG_CUES for x in words_before(low, m.start())):
                        out.add((k, s))
    return out

def candidate_is_true(phrase, symptom):
    """Ground truth for one candidate. Used only to evaluate the judge, never by the agent."""
    low = phrase.lower().strip()
    return symptom in SYMPTOMS and any(f in low for f in SYMPTOMS[symptom]) and low not in NEGATIONS[symptom]

def trusttrace_v3_candidates(diaries, v2_stage, label):
    cands = []
    for i, d in enumerate(diaries):
        for e in d["entries"]:
            frags = fragments(e)
            kw = keyword_candidates(e)
            ll = {(k, s) for (eid, k, s) in v2_stage[i]["llm_cands"] if eid == e["id"]}
            for k, s in sorted(kw | ll):
                if not 1 <= k <= len(frags) or is_negated(e["text"], frags[k - 1]):
                    continue
                cands.append({"i": i, "day": e["day"], "k": k, "symptom": s, "phrase": frags[k - 1],
                              "src": "both" if (k, s) in kw and (k, s) in ll else "kw" if (k, s) in kw else "llm"})
    probs = LLM.yes_prob([prompt_judge(c["phrase"], c["symptom"]) for c in cands], label=label)
    for c, p in zip(cands, probs):
        c["p"] = float(p)
    RAW_SAMPLES.extend({"method": "TrustTrace v3 judge", "phrase": c["phrase"], "symptom": c["symptom"], "source": c["src"],
                        "p_yes": round(c["p"], 4)} for c in cands[:40])
    return cands

TAU_GRID = [round(float(x), 2) for x in np.arange(0, 1.0001, 0.05)] + [1.01]   # 1.01 = never accept from this source
SOURCES = ["both", "kw", "llm"]

def select_thresholds(cands, diaries):
    """Choose one acceptance threshold per candidate source to maximise event F1 on validation data."""
    keys = sorted({(c["i"], c["symptom"], c["day"]) for c in cands})
    index = {key: j for j, key in enumerate(keys)}
    ev_idx = np.array([index[(c["i"], c["symptom"], c["day"])] for c in cands], dtype=int)
    ev_true = np.array([(s, day) in diaries[i]["truth"] for i, s, day in keys], dtype=bool)
    probs = np.array([c["p"] for c in cands]); src = np.array([SOURCES.index(c["src"]) for c in cands], dtype=int)
    n_truth = sum(len(d["truth"]) for d in diaries)
    best = None
    for tb in TAU_GRID:
        for tk in TAU_GRID:
            for tl in TAU_GRID:
                acc = probs >= np.array([tb, tk, tl])[src]
                acc_ev = np.bincount(ev_idx, weights=acc, minlength=len(keys)) > 0 if len(keys) else np.zeros(0, bool)
                tp = int((acc_ev & ev_true).sum()); fp = int((acc_ev & ~ev_true).sum())
                p, r, f1 = prf(tp, fp, n_truth - tp)
                key = (round(f1, 6), round(p, 6), tb + tk + tl)
                if best is None or key > best[0]:
                    best = (key, {"both": tb, "kw": tk, "llm": tl}, {"precision": p, "recall": r, "f1": f1})
    return best[1], best[2]

def trusttrace_v3(diaries, cands, taus, v2_stage, summarise=True):
    stage = []
    for i, d in enumerate(diaries):
        mine = [c for c in cands if c["i"] == i]
        events = {(c["symptom"], c["day"]) for c in mine if c["p"] >= taus[c["src"]]}
        rejected = {}
        for c in mine:
            if c["p"] < taus[c["src"]]:
                rejected[f"judge said no ({c['src']})"] = rejected.get(f"judge said no ({c['src']})", 0) + 1
        stage.append({"events": events, "raw_events": {(c["symptom"], c["day"]) for c in mine}, "facts": build_facts(events),
                      "rejected": rejected, "parse_failed": v2_stage[i]["parse_failed"]})
    return finish_agent(stage, summarise)

def judge_diagnostics(cands):
    """How well the judge's P(yes) separates true and false candidates (AUC) and how calibrated it is (ECE)."""
    y = np.array([candidate_is_true(c["phrase"], c["symptom"]) for c in cands], dtype=float)
    p = np.array([c["p"] for c in cands])
    if len(y) == 0 or y.min() == y.max():
        return {"auc": float("nan"), "ece": float("nan"), "n": int(len(y))}
    order = np.argsort(p); ranks = np.empty(len(p)); ranks[order] = np.arange(1, len(p) + 1)
    for v in np.unique(p):                                   # average ranks for ties
        m = p == v; ranks[m] = ranks[m].mean()
    n1 = y.sum(); n0 = len(y) - n1
    auc = (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    bins = np.clip((p * 10).astype(int), 0, 9)
    ece = sum((bins == b).mean() * abs(y[bins == b].mean() - p[bins == b].mean()) for b in range(10) if (bins == b).any())
    return {"auc": float(auc), "ece": float(ece), "n": int(len(y)), "positives": int(n1)}

# %%
# ## 6. Evaluation

# %%
def event_counts(results, diaries):
    tp = fp = fn = 0; kinds = {"negation": 0, "distractor": 0, "unsupported": 0}
    for r, d in zip(results, diaries):
        tp += len(r["events"] & d["truth"]); fn += len(d["truth"] - r["events"])
        for ev in r["events"] - d["truth"]:
            fp += 1
            kinds["negation" if ev in d["negated"] else "distractor" if ev in d["distract"] else "unsupported"] += 1
    return tp, fp, fn, kinds

def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)

def escalation_cost(flags, diaries):
    fn = sum(1 for f, d in zip(flags, diaries) if d["escalate"] and not f)
    fp = sum(1 for f, d in zip(flags, diaries) if f and not d["escalate"])
    return (CFG["cost_missed_escalation"] * fn + CFG["cost_unneeded_review"] * fp) / len(diaries), fn, fp

def evaluate(name, results, diaries, flags=None, rng=None):
    flags = [r["escalate"] for r in results] if flags is None else flags
    tp, fp, fn, kinds = event_counts(results, diaries)
    p, r, f1 = prf(tp, fp, fn)
    claim_tp = sum(len(res["claimed"] & set(d["present"])) for res, d in zip(results, diaries))
    claim_all = sum(len(res["claimed"]) for res in results)
    cost, efn, efp = escalation_cost(flags, diaries)
    boot = {"precision": [], "recall": [], "f1": [], "cost": []}
    rng = rng or np.random.default_rng(CFG["seed"])
    n = len(diaries)
    for _ in range(CFG["n_bootstrap"]):
        idx = rng.integers(0, n, n)
        rs, ds, fs = [results[i] for i in idx], [diaries[i] for i in idx], [flags[i] for i in idx]
        bp, br, bf = prf(*event_counts(rs, ds)[:3])
        boot["precision"].append(bp); boot["recall"].append(br); boot["f1"].append(bf)
        boot["cost"].append(escalation_cost(fs, ds)[0])
    ci = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in boot.items()}
    return {"method": name, "precision": p, "recall": r, "f1": f1, "false_events": fp, "false_event_kinds": kinds,
            "claim_precision": claim_tp / claim_all if claim_all else 0.0,
            "hallucinated_claims_per_diary": (claim_all - claim_tp) / n,
            "escalation_cost": cost, "missed_escalations": efn, "unneeded_reviews": efp,
            "parse_failure_rate": float(np.mean([res.get("parse_failed", False) for res in results])), "ci95": ci}

# %%
# ## 7. Run the experiment

# %%
T0 = time.time()
base_val, base_test = [keyword_baseline(d) for d in VAL], [keyword_baseline(d) for d in TEST]
print("Running LLM only on the test set...")
llm_test = llm_only(TEST)
print("Running agent v1 (whole-diary extraction) on validation and test sets...")
v1_val, v1_test = trusttrace_agent(VAL, summarise=False), trusttrace_agent(TEST)
print("Running TrustTrace v2 (per-entry extraction with tool feedback) on validation and test sets...")
agent_val, agent_test = trusttrace_v2(VAL, summarise=False), trusttrace_v2(TEST)
print("Running TrustTrace v3 (hybrid): the model judges every keyword and LLM candidate...")
cands_val = trusttrace_v3_candidates(VAL, agent_val, "v3 judge, validation")
cands_test = trusttrace_v3_candidates(TEST, agent_test, "v3 judge, test")
taus, v3_val_fit = select_thresholds(cands_val, VAL)          # chosen on validation data only
print(f"  Judge thresholds chosen on validation: {taus} (validation F1 {v3_val_fit['f1']:.3f})")
v3_val = trusttrace_v3(VAL, cands_val, taus, agent_val, summarise=False)
v3_test = trusttrace_v3(TEST, cands_test, taus, agent_test)

# Cost-aware escalation threshold, chosen on validation data only (separately for each agent)
grid = sorted(set([0.25, 0.5, 0.75] + [float(x) for x in range(1, 7)]))
def pick_threshold(val_stage):
    costs = [escalation_cost([a["score"] >= t for a in val_stage], VAL)[0] for t in grid]
    best = min(costs)
    return min(t for t, c in zip(grid, costs) if c == best), costs
threshold, val_costs = pick_threshold(agent_val)
threshold_v1, _ = pick_threshold(v1_val)
threshold_v3, val_costs_v3 = pick_threshold(v3_val)
test_costs = [escalation_cost([a["score"] >= t for a in agent_test], TEST)[0] for t in grid]
test_costs_v3 = [escalation_cost([a["score"] >= t for a in v3_test], TEST)[0] for t in grid]

rng_eval = np.random.default_rng(CFG["seed"])
raw_view = [{"events": a["raw_events"], "claimed": {s for s, _ in a["raw_events"]}, "escalate": evidence_score(a["raw_events"]) >= threshold,
             "parse_failed": a["parse_failed"]} for a in agent_test]
RESULTS = {
    "Keyword baseline": evaluate("Keyword baseline", base_test, TEST, rng=rng_eval),
    "LLM only": evaluate("LLM only", llm_test, TEST, rng=rng_eval),
    "Agent v1 (whole diary)": evaluate("Agent v1 (whole diary)", v1_test, TEST, flags=[a["score"] >= threshold_v1 for a in v1_test], rng=rng_eval),
    "TrustTrace v2 before verification": evaluate("TrustTrace v2 before verification", raw_view, TEST, rng=rng_eval),
    "TrustTrace v2": evaluate("TrustTrace v2", agent_test, TEST, flags=[a["score"] >= threshold for a in agent_test], rng=rng_eval),
    "TrustTrace v3 (hybrid)": evaluate("TrustTrace v3 (hybrid)", v3_test, TEST, flags=[a["score"] >= threshold_v3 for a in v3_test], rng=rng_eval),
}
rejections = {}
for a in agent_test:
    for k, v in a["rejected"].items():
        rejections[k] = rejections.get(k, 0) + v
retried = sum(a["retried"] for a in agent_test); recovered = sum(a["recovered"] for a in agent_test)

# How the v3 judge behaved on test candidates, by where the candidate came from
judge_val, judge_test = judge_diagnostics(cands_val), judge_diagnostics(cands_test)
by_source = {}
for src in SOURCES:
    mine = [c for c in cands_test if c["src"] == src]
    acc = [c for c in mine if c["p"] >= taus[src]]
    by_source[src] = {"candidates": len(mine), "true_candidates": sum(candidate_is_true(c["phrase"], c["symptom"]) for c in mine),
                      "accepted": len(acc), "accepted_true": sum(candidate_is_true(c["phrase"], c["symptom"]) for c in acc),
                      "threshold": taus[src]}

def summary_stats(stage):
    issues = {}
    for a in stage:
        for r in a["summary_issues"]:
            issues[r] = issues.get(r, 0) + 1
    return issues, sum(1 for a in stage if a["summary_unsupported"]), sum(1 for a in stage if a["summary_replaced"])
summary_issue_counts, summary_catches, summary_replaced = summary_stats(agent_test)
summary_issue_counts_v3, summary_catches_v3, summary_replaced_v3 = summary_stats(v3_test)
print(f"\nDone in {(time.time() - T0) / 60:.1f} minutes. Escalation thresholds chosen on validation: "
      f"v1 {threshold_v1}, v2 {threshold}, v3 {threshold_v3}")


# %%
# ## 8. Figures

# %%
names = ["Keyword baseline", "LLM only", "Agent v1 (whole diary)", "TrustTrace v2 before verification", "TrustTrace v2", "TrustTrace v3 (hybrid)"]
short = {"Keyword baseline": "Keyword", "LLM only": "LLM only", "Agent v1 (whole diary)": "Agent v1",
         "TrustTrace v2 before verification": "v2 unverified", "TrustTrace v2": "TrustTrace v2", "TrustTrace v3 (hybrid)": "TrustTrace v3"}
colors = ["#9e9e9e", "#ef8a62", "#c2a5cf", "#fdb863", "#67a9cf", "#1b5e8a"]
fig, axs = plt.subplots(2, 3, figsize=(21, 11.5))

ax = axs[0, 0]; w = 0.13; x = np.arange(3)
for i, n in enumerate(names):
    R = RESULTS[n]; vals = [R["precision"], R["recall"], R["f1"]]
    err = [[v - R["ci95"][k][0] for v, k in zip(vals, ["precision", "recall", "f1"])],
           [R["ci95"][k][1] - v for v, k in zip(vals, ["precision", "recall", "f1"])]]
    ax.bar(x + (i - 2.5) * w, vals, w, yerr=np.clip(err, 0, None), capsize=2, label=short[n], color=colors[i])
ax.set_xticks(x); ax.set_xticklabels(["Precision", "Recall", "F1"]); ax.set_ylim(0, 1.05)
ax.set_title("Symptom-event extraction, test (95% bootstrap CI)")
ax.legend(fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.07), frameon=False)

ax = axs[0, 1]; bottom = np.zeros(len(names)); xs = np.arange(len(names))
for kind, c in zip(["negation", "distractor", "unsupported"], ["#d7191c", "#fdae61", "#7b3294"]):
    v = np.array([RESULTS[n]["false_event_kinds"][kind] for n in names], dtype=float)
    ax.bar(xs, v, bottom=bottom, label=kind, color=c); bottom += v
ax.set_xticks(xs); ax.set_xticklabels([short[n] for n in names], rotation=30, ha="right", fontsize=9)
ax.set_title("False symptom events by cause (test)"); ax.set_ylabel("Count"); ax.legend(fontsize=9)

ax = axs[0, 2]
ax.plot(grid, test_costs, "s-", color=colors[4], label="TrustTrace v2, test")
ax.plot(grid, test_costs_v3, "o-", color=colors[5], label="TrustTrace v3, test")
ax.axvline(threshold_v3, color=colors[5], ls="--", label=f"v3 threshold from validation ({threshold_v3})")
ax.axhline(RESULTS["LLM only"]["escalation_cost"], color=colors[1], ls=":", label="LLM only")
ax.axhline(RESULTS["Keyword baseline"]["escalation_cost"], color=colors[0], ls=":", label="Keyword")
ax.set(title="Escalation cost per diary (missed red flag = 5 x unneeded review)", xlabel="Evidence-score threshold", ylabel="Cost per diary")
ax.legend(fontsize=8)

ax = axs[1, 0]
truth_test = np.array([candidate_is_true(c["phrase"], c["symptom"]) for c in cands_test], dtype=bool)
p_test = np.array([c["p"] for c in cands_test])
bins = np.linspace(0, 1, 21)
ax.hist(p_test[truth_test], bins, alpha=0.7, color="#1b5e8a", label="true candidates")
ax.hist(p_test[~truth_test], bins, alpha=0.7, color="#d7191c", label="false candidates")
ax.set(title=f"v3 judge P(yes) on test candidates (AUC {judge_test['auc']:.3f})", xlabel="P(yes)", ylabel="Candidates")
ax.legend(fontsize=9)

ax = axs[1, 1]
labels = list(rejections) or ["none"]
ax.bar(np.arange(len(labels)), [rejections.get(k, 0) for k in labels], color=colors[4])
ax.set_xticks(np.arange(len(labels))); ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
ax.set_title(f"v2 evidence checker (test): {retried} entries retried, {recovered} events recovered"); ax.set_ylabel("Count")

ax = axs[1, 2]; xs = np.arange(len(SOURCES)); src_names = {"both": "keyword + LLM", "kw": "keyword only", "llm": "LLM only"}
for j, (key, lab, c) in enumerate([("candidates", "candidates", "#cccccc"), ("true_candidates", "true", "#92c5de"),
                                     ("accepted", "accepted", "#67a9cf"), ("accepted_true", "accepted and true", "#1b5e8a")]):
    ax.bar(xs + (j - 1.5) * 0.2, [by_source[s][key] for s in SOURCES], 0.2, label=lab, color=c)
ax.set_xticks(xs); ax.set_xticklabels([f"{src_names[s]}\n(threshold {taus[s]})" for s in SOURCES])
ax.set_title("v3 candidates on test, by source"); ax.set_ylabel("Count"); ax.legend(fontsize=9)

fig.suptitle(f"TrustTrace on {len(TEST)} synthetic test diaries ({CFG['model_name']}" + (", MOCK BACKEND" if BACKEND == "mock" else "") + ")",
             fontsize=15)
fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(os.path.join(CFG["out_dir"], "summary.png"), dpi=130); plt.show(); plt.close(fig)


# %%
# ## 9. Results tables, worked examples and a hand-check sheet

# %%
def pct(v): return f"{100 * v:.1f}%"
lines = [f"# TrustTrace results", "", f"Model: {CFG['model_name']} | backend: {BACKEND} | test diaries: {len(TEST)} "
         f"({sum(len(d['truth']) for d in TEST)} symptom events, {sum(d['escalate'] for d in TEST)} needing escalation)", ""]
if BACKEND == "mock":
    lines += ["**MOCK BACKEND: software test only, not model results.**", ""]
lines += ["| Method | Precision (95% CI) | Recall (95% CI) | F1 | False events (negation / distractor / unsupported) | "
          "Summary claim precision | Escalation cost (95% CI) | Missed escalations | Unneeded reviews | Parse failures |",
          "|---|---|---|---|---|---|---|---|---|---|"]
for n in names:
    R = RESULTS[n]; k = R["false_event_kinds"]; c = R["ci95"]
    lines.append(f"| {n} | {pct(R['precision'])} ({pct(c['precision'][0])}-{pct(c['precision'][1])}) | "
                 f"{pct(R['recall'])} ({pct(c['recall'][0])}-{pct(c['recall'][1])}) | {R['f1']:.3f} | "
                 f"{R['false_events']} ({k['negation']} / {k['distractor']} / {k['unsupported']}) | {pct(R['claim_precision'])} | "
                 f"{R['escalation_cost']:.3f} ({c['cost'][0]:.3f}-{c['cost'][1]:.3f}) | {R['missed_escalations']} | "
                 f"{R['unneeded_reviews']} | {pct(R['parse_failure_rate'])} |")
lines += ["", f"Escalation thresholds chosen on {len(VAL)} validation diaries: agent v1 {threshold_v1}, TrustTrace v2 {threshold}, "
          f"TrustTrace v3 {threshold_v3}.",
          f"TrustTrace v2 evidence checker on test: rejections {json.dumps(rejections)}; entries retried with tool feedback: {retried}; "
          f"symptom events recovered by retries: {recovered}.",
          f"TrustTrace v3 judge thresholds chosen on validation: {json.dumps(taus)} (validation F1 {v3_val_fit['f1']:.3f}).",
          f"TrustTrace v3 judge on test candidates: AUC {judge_test['auc']:.3f}, ECE {judge_test['ece']:.3f}, "
          f"{judge_test['n']} candidates ({judge_test.get('positives', 0)} true). Validation AUC {judge_val['auc']:.3f}.",
          "TrustTrace v3 candidates on test by source: " + "; ".join(
              f"{s}: {v['candidates']} candidates ({v['true_candidates']} true), {v['accepted']} accepted ({v['accepted_true']} true)"
              for s, v in by_source.items()) + ".",
          f"LLM-written summaries replaced by the verified-facts template: v2 {summary_replaced} of {len(TEST)} "
          f"(reasons: {json.dumps(summary_issue_counts)}); v3 {summary_replaced_v3} of {len(TEST)} "
          f"(reasons: {json.dumps(summary_issue_counts_v3)}).", ""]
report = "\n".join(lines)
with open(os.path.join(CFG["out_dir"], "summary.md"), "w") as f:
    f.write(report)
with open(os.path.join(CFG["out_dir"], "metrics.json"), "w") as f:
    json.dump({"config": CFG, "backend": BACKEND, "threshold": threshold, "threshold_v1": threshold_v1, "threshold_v3": threshold_v3,
               "rejections": rejections, "entries_retried": retried, "events_recovered": recovered, "summary_issues": summary_issue_counts,
               "summaries_with_unsupported_claims": summary_catches, "summaries_replaced": summary_replaced,
               "v3": {"judge_thresholds": taus, "validation_fit": v3_val_fit, "judge_test": judge_test, "judge_val": judge_val,
                      "by_source_test": by_source, "summary_issues": summary_issue_counts_v3,
                      "summaries_with_unsupported_claims": summary_catches_v3, "summaries_replaced": summary_replaced_v3},
               "results": RESULTS}, f, indent=2)

ex = ["# Worked examples (test set)", ""]
for d, a, l, v1, v3 in list(zip(TEST, agent_test, llm_test, v1_test, v3_test))[:3]:
    ex += [f"## {d['id']}", "```", diary_text(d), "```",
           f"**Ground truth:** {', '.join(d['present'])} | escalate: {d['escalate']}",
           f"**LLM only:** {l['summary'] or '(no valid output)'} | escalate: {l['escalate']}",
           f"**Agent v1:** {v1['summary']} | escalate: {v1['score'] >= threshold_v1}",
           f"**TrustTrace v2:** {a['summary']} | escalate: {a['score'] >= threshold} | rejected claims: {a['rejected'] or 'none'}"
           + (f" | model summary replaced ({', '.join(a['summary_issues'])}): \"{a['summary_llm']}\"" if a['summary_replaced'] else ""),
           f"**TrustTrace v3:** {v3['summary']} | escalate: {v3['score'] >= threshold_v3} | judge rejections: {v3['rejected'] or 'none'}"
           + (f" | model summary replaced ({', '.join(v3['summary_issues'])}): \"{v3['summary_llm']}\"" if v3['summary_replaced'] else ""), ""]
with open(os.path.join(CFG["out_dir"], "examples.md"), "w") as f:
    f.write("\n".join(ex))

with open(os.path.join(CFG["out_dir"], "raw_samples.jsonl"), "w") as f:
    for r in RAW_SAMPLES:
        f.write(json.dumps(r) + "\n")

with open(os.path.join(CFG["out_dir"], "human_check.csv"), "w", newline="") as f:
    wr = csv.writer(f)
    wr.writerow(["diary_id", "diary", "trusttrace_summary", "verified_events", "escalate",
                 "summary_faithful (yes/no)", "missed_red_flag (yes/no)", "notes"])
    for d, a in list(zip(TEST, v3_test))[:CFG["n_human_check"]]:   # TrustTrace v3 is the final method
        wr.writerow([d["id"], diary_text(d), a["summary"], "; ".join(f"{s} (day {day})" for s, day in sorted(a["events"], key=lambda z: (z[1], z[0]))),
                     a["score"] >= threshold_v3, "", "", ""])
print(report)
print("Saved to", os.path.abspath(CFG["out_dir"]), ": summary.md, metrics.json, summary.png, examples.md, human_check.csv, raw_samples.jsonl")
