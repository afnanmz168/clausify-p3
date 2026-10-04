"""Summarization pilot: score the fine-tuned FLAN-T5 against TWO reference sets
on the SAME 150 held-out clauses --

  (a) the 41 self-authored category templates  -> reproduces the thesis number
  (b) 150 clause-specific references drafted independently of the model

Difference between (a) and (b) isolates how much of ROUGE-L 0.775 is template
reproduction rather than summarization.
"""
import os, sys, json, numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.expanduser("~/Desktop/final project p3/analysis"))
NB = os.path.expanduser("~/Desktop/final project p3/notebooks")

samp = pd.DataFrame(json.load(open(os.path.expanduser("~/Desktop/final project p3/analysis/sum_sample150.json"))))
refs = json.load(open(os.path.expanduser("~/Desktop/final project p3/analysis/diverse_refs.json")))
TEMPLATES = json.load(open(os.path.expanduser("~/Desktop/final project p3/analysis/templates.json")))

def lcs_len(a, b):
    dp = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        prev = 0
        for j in range(1, len(b) + 1):
            tmp = dp[j]
            dp[j] = prev + 1 if a[i-1] == b[j-1] else max(dp[j], dp[j-1])
            prev = tmp
    return dp[-1]

def rouge_l(pred, gold):
    p, g = pred.lower().split(), gold.lower().split()
    if not p or not g: return 0.0
    l = lcs_len(p, g)
    if l == 0: return 0.0
    pr, rc = l / len(p), l / len(g)
    return 2 * pr * rc / (pr + rc)

tok = AutoTokenizer.from_pretrained(f"{NB}/outputs/summarizer/final")
ft = AutoModelForSeq2SeqLM.from_pretrained(f"{NB}/outputs/summarizer/final").to("cpu").eval()
zs = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small").to("cpu").eval()

rows = []
for i in range(len(samp)):
    inp = tok(samp.loc[i, "input_text"], return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        o_ft = ft.generate(**inp, max_new_tokens=48, num_beams=2)
        o_zs = zs.generate(**inp, max_new_tokens=48, num_beams=2)
    p_ft = tok.decode(o_ft[0], skip_special_tokens=True)
    p_zs = tok.decode(o_zs[0], skip_special_tokens=True)
    cat = samp.loc[i, "category"]
    tmpl, div = TEMPLATES[cat], refs[str(i)]
    rows.append(dict(i=i, category=cat, pred_ft=p_ft, pred_zs=p_zs,
                     template=tmpl, diverse=div,
                     ft_vs_template=rouge_l(p_ft, tmpl),
                     ft_vs_diverse=rouge_l(p_ft, div),
                     zs_vs_diverse=rouge_l(p_zs, div),
                     template_vs_diverse=rouge_l(tmpl, div),
                     exact_template=int(p_ft.strip().lower() == tmpl.strip().lower())))
    if i % 25 == 0: print(f"  {i}/150", flush=True)

df = pd.DataFrame(rows)
df.to_json(f"{HERE}/summ_pilot_rows.json", orient="records", indent=1)

rng = np.random.default_rng(42)
def ci(col):
    v = df[col].values
    b = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(10000)])
    return float(v.mean()), [float(x) for x in np.percentile(b, [2.5, 97.5])]

out = {c: dict(zip(["mean", "ci"], ci(c))) for c in
       ["ft_vs_template", "ft_vs_diverse", "zs_vs_diverse", "template_vs_diverse"]}
out["exact_template_rate"] = float(df.exact_template.mean())
out["n"] = len(df)
json.dump(out, open(f"{HERE}/summ_pilot.json", "w"), indent=2)
for k, v in out.items():
    print(k, v, flush=True)
