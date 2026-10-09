"""Can a larger instruction-tuned model write a plain-English sentence about the specific clause?

The fine-tuned FLAN-T5-small copies one of 41 category templates for every input (report Section
5.1.3): ROUGE-L 0.116 against clause-specific references, no better than the template alone (0.115).
Here FLAN-T5 small, base and large are used without fine-tuning, with four prompts, on the same 150
held-out clauses and clause-specific references (analysis/sum_sample150.json, diverse_refs.json).

  split   50 clauses (seeded) to choose the model and prompt, 100 to test once.
  scores  ROUGE-L against the clause-specific reference (as in the report);
          readability: Flesch reading ease of the output, against that of the clause;
          copying: share of the output's word trigrams found word for word in the clause;
          invented numbers: share of outputs with a number that is not in the clause.
  choose  among model/prompt pairs whose development outputs are easier to read than the clauses on
          average and have an invented number in at most 5% of them, the highest ROUGE-L.

Writes runs/plain_english.json and runs/rows_<model>.json.

    cd ~/Desktop/"final project p3" && python3 retrain/plain_v2/plain_english.py
"""
import os, sys, re, json, time
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

HERE = os.path.dirname(os.path.abspath(__file__)); P = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(HERE, "runs"); os.makedirs(OUT, exist_ok=True)
DEV = os.environ.get("PLAIN_DEVICE") or ("mps" if torch.backends.mps.is_available() else "cpu")
samp = json.load(open(os.path.join(P, "analysis", "sum_sample150.json")))
refs = json.load(open(os.path.join(P, "analysis", "diverse_refs.json")))
TEMPLATES = json.load(open(os.path.join(P, "analysis", "templates.json")))
perm = np.random.default_rng(42).permutation(len(samp)); dev_idx, test_idx = sorted(perm[:50]), sorted(perm[50:])

MODELS = ["google/flan-t5-small", "google/flan-t5-base", "google/flan-t5-large"]
PROMPTS = {
    "summarize": "summarize in plain English: {clause}",
    "explain": "Explain in one plain-English sentence what this contract clause means for the people who sign it: {clause}",
    "rewrite": "Rewrite this contract clause as one short, simple sentence that a person without legal training can understand: {clause}",
    "typed": "This is a {category} clause from a contract. In one short sentence of plain English, say what it requires or allows, keeping any amounts, time periods and parties it names.\n\nClause: {clause}",
}


def lcs(a, b):
    dp = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        prev = 0
        for j in range(1, len(b) + 1):
            tmp = dp[j]; dp[j] = prev + 1 if a[i - 1] == b[j - 1] else max(dp[j], dp[j - 1]); prev = tmp
    return dp[-1]


def rouge_l(pred, gold):                     # identical to retrain/analysis_oct2026/summ_pilot.py
    p, g = pred.lower().split(), gold.lower().split()
    if not p or not g:
        return 0.0
    l = lcs(p, g)
    return 0.0 if l == 0 else 2 * (l / len(p)) * (l / len(g)) / (l / len(p) + l / len(g))


def syllables(w):
    w = re.sub(r"[^a-z]", "", w.lower())
    if not w:
        return 0
    n = len(re.findall(r"[aeiouy]+", w)) - (1 if w.endswith("e") and not w.endswith("le") else 0)
    return max(1, n)


def flesch(text):
    words = re.findall(r"[A-Za-z]+", text)
    sents = max(1, len(re.findall(r"[.!?]+", text)) or 1)
    if not words:
        return 0.0
    return 206.835 - 1.015 * len(words) / sents - 84.6 * sum(syllables(w) for w in words) / len(words)


WORDNUM = {w: str(i) for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen "
           "fourteen fifteen sixteen seventeen eighteen nineteen twenty".split())}
WORDNUM.update({"thirty": "30", "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70", "eighty": "80",
                "ninety": "90", "hundred": "100", "thousand": "1000"})


def numbers(text):
    t = text.lower().replace(",", "")
    found = set(re.findall(r"\d+(?:\.\d+)?", t))
    found |= {WORDNUM[w] for w in re.findall(r"[a-z]+", t) if w in WORDNUM}
    return found


def invented_number(out, clause):
    return int(bool(numbers(out) - numbers(clause)))


def copy_share(out, clause):
    o, c = out.lower().split(), clause.lower().split()
    tri = [tuple(o[i:i + 3]) for i in range(len(o) - 2)]
    if not tri:
        return 0.0
    cs = {tuple(c[i:i + 3]) for i in range(len(c) - 2)}
    return sum(t in cs for t in tri) / len(tri)


def score(rows, idx):
    r = [rows[i] for i in idx]
    return dict(rouge_l=float(np.mean([x["rouge_l"] for x in r])), flesch=float(np.mean([x["flesch"] for x in r])),
                flesch_clause=float(np.mean([flesch(samp[x["i"]]["clause"]) for x in r])),
                copy_share=float(np.mean([x["copy"] for x in r])),
                invented_number=float(np.mean([x["invented"] for x in r])),
                mean_words=float(np.mean([len(x["out"].split()) for x in r])))


res, t_all = {}, {}
for name in MODELS:
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSeq2SeqLM.from_pretrained(name).to(DEV).eval()
    for pk, pt in PROMPTS.items():
        rows, t0 = [], time.time()
        for i, s in enumerate(samp):
            text = pt.format(clause=s["clause"][:1200], category=s["category"])
            enc = tok(text, return_tensors="pt", truncation=True, max_length=384).to(DEV)
            with torch.no_grad():
                g = model.generate(**enc, max_new_tokens=64, num_beams=4, no_repeat_ngram_size=3)
            out = tok.decode(g[0], skip_special_tokens=True).strip()
            rows.append(dict(i=i, category=s["category"], out=out, rouge_l=rouge_l(out, refs[str(i)]),
                             flesch=flesch(out), copy=copy_share(out, s["clause"]), invented=invented_number(out, s["clause"])))
        key = f"{name.split('/')[-1]}|{pk}"
        t_all[key] = (time.time() - t0) / len(samp)
        res[key] = dict(dev=score(rows, dev_idx), test=score(rows, test_idx), seconds_per_clause=round(t_all[key], 3))
        json.dump(rows, open(os.path.join(OUT, f"rows_{key.replace('|', '_')}.json"), "w"), indent=1)
        print(key, {k: round(v, 3) for k, v in res[key]["dev"].items()}, f"{t_all[key]:.2f}s/clause ({DEV})", flush=True)
    del model
    if DEV == "mps":
        torch.mps.empty_cache()

ok = [k for k, v in res.items() if v["dev"]["flesch"] > v["dev"]["flesch_clause"] and v["dev"]["invented_number"] <= 0.05]
choice = max(ok, key=lambda k: res[k]["dev"]["rouge_l"]) if ok else None
base = dict(template_vs_diverse=float(np.mean([rouge_l(TEMPLATES[samp[i]["category"]], refs[str(i)]) for i in test_idx])))
pilot = json.load(open(os.path.join(P, "retrain", "analysis_oct2026", "summ_pilot_rows.json")))
base["finetuned_vs_diverse"] = float(np.mean([pilot[i]["ft_vs_diverse"] for i in test_idx]))
json.dump(dict(dev_n=len(dev_idx), test_n=len(test_idx), dev_idx=[int(i) for i in dev_idx], candidates=res,
               eligible=ok, chosen=choice, test_chosen=res[choice]["test"] if choice else None,
               baselines_on_test_100=base, device=DEV,
               rule="among pairs easier to read than the clause and with an invented number in at most 5% of "
                    "development outputs, the highest development ROUGE-L"),
          open(os.path.join(OUT, "plain_english.json"), "w"), indent=1)
print("chosen:", choice, res[choice]["test"] if choice else None, "| baselines on the 100:", base)
