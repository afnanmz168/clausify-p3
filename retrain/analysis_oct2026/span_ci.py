"""Span extraction with uncertainty: identical protocol to test.ipynb Cell 7,
but keeps every per-example score so the same cluster bootstrap used for
presence classification can be applied here too."""
import os, sys, re, json, string
from collections import Counter, defaultdict
import numpy as np, pandas as pd, torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.expanduser("~/Desktop/final project p3/analysis"))
import common
from common import NB

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
QA_MAXLEN, FOCUS = 320, 1200
D = common.load_all(); _test = set(D["test_titles"])
wins_by_title = {t: common.make_windows(x) for t, x in D["ctx"].items()}

rows = []
for c in D["js"]:
    t = c["title"]
    if t not in _test: continue
    wins = wins_by_title[t]
    for qa in c["paragraphs"][0]["qas"]:
        if not qa["answers"]: continue
        cat = qa["id"].rsplit("__", 1)[-1]
        for aa in qa["answers"]:
            at = aa["text"].strip()
            if not at: continue
            probe = at[:200]
            for w in wins:
                pos = w.find(probe)
                if pos != -1:
                    rows.append({"title": t, "category": cat, "question": qa["question"],
                                 "context": w, "answer_text": at, "char_start": pos})
                    break
sp = pd.DataFrame(rows)
print("answer-bearing test windows:", len(sp), flush=True)

tok = AutoTokenizer.from_pretrained(f"{NB}/outputs/span/final")
model = AutoModelForQuestionAnswering.from_pretrained(f"{NB}/outputs/span/final").to(DEVICE).eval()

def encode(b):
    ctxs = [w[max(0, cs - 150):max(0, cs - 150) + FOCUS]
            for w, cs in zip(b["context"], b["char_start"])]
    return tok(b["question"], ctxs, truncation="only_second", max_length=QA_MAXLEN)

ds = Dataset.from_pandas(sp, preserve_index=False).map(
    encode, batched=True, remove_columns=sp.columns.tolist())

norm = lambda s: " ".join("".join(ch for ch in re.sub(r"\b(a|an|the)\b", " ", s.lower())
                                  if ch not in string.punctuation).split())
def token_f1(pred, gold):
    p, g = norm(pred).split(), norm(gold).split()
    if not p or not g: return float(p == g)
    common_n = sum((Counter(p) & Counter(g)).values())
    if common_n == 0: return 0.0
    pr, rc = common_n / len(p), common_n / len(g)
    return 2 * pr * rc / (pr + rc)

sep_id, golds = tok.sep_token_id, sp["answer_text"].tolist()
f1s, ems = [], []
B, N = 32, len(ds)
for i in range(0, N, B):
    feats = [ds[j] for j in range(i, min(i + B, N))]
    batch = tok.pad({"input_ids": [f["input_ids"] for f in feats],
                     "attention_mask": [f["attention_mask"] for f in feats]},
                    padding=True, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model(**batch)
    sl, el = out.start_logits.cpu().numpy(), out.end_logits.cpu().numpy()
    for k, f in enumerate(feats):
        ids = f["input_ids"]; L = len(ids); sep = ids.index(sep_id)
        mask = np.full(L, -1e9); mask[sep + 1:] = 0.0
        s = int(np.argmax(sl[k][:L] + mask)); e = int(np.argmax(el[k][:L] + mask))
        if e < s: e = s
        e = min(e, s + 60)
        pred = tok.decode(ids[s:e + 1], skip_special_tokens=True)
        g = golds[i + k]
        f1s.append(token_f1(pred, g)); ems.append(int(norm(pred) == norm(g)))
    if (i // B) % 20 == 0: print(f"  {i}/{N}", flush=True)

sp["token_f1"] = f1s; sp["em"] = ems; sp["overlap"] = (np.array(f1s) >= 0.5).astype(int)
sp[["title", "category", "token_f1", "em", "overlap"]].to_json(f"{HERE}/span_scores.json",
                                                               orient="records")

rng = np.random.default_rng(42)
titles = sorted(sp.title.unique())
by_title = {t: sp.index[sp.title == t].values for t in titles}
def boot(col, unit):
    v = sp[col].values
    out = []
    for _ in range(10000):
        if unit == "clause":
            idx = rng.integers(0, len(v), len(v))
        else:
            pick = rng.integers(0, len(titles), len(titles))
            idx = np.concatenate([by_title[titles[p]] for p in pick])
        out.append(v[idx].mean())
    lo, hi = np.percentile(out, [2.5, 97.5])
    return float(v.mean()), float(lo), float(hi)

res = {}
for col in ["token_f1", "overlap", "em"]:
    for unit in ["contract", "clause"]:
        m, lo, hi = boot(col, unit)
        res[f"{col}_{unit}"] = dict(mean=m, lo=lo, hi=hi)
        print(f"{col:9s} [{unit:8s}] {m:.4f}  95% CI [{lo:.4f}, {hi:.4f}]", flush=True)
res["n_examples"] = int(len(sp)); res["n_contracts"] = len(titles)
json.dump(res, open(f"{HERE}/span_ci.json", "w"), indent=2)
