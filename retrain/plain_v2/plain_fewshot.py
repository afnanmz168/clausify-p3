"""Exploratory follow-up to plain_english.py (not part of its pre-set choice rule).

Zero-shot FLAN-T5 mostly copied the clause. Here FLAN-T5-large is shown three worked examples first
(development clauses with their clause-specific references; none of the 100 test clauses), then asked
for the new clause. Same 100 test clauses and scores as plain_english.py. Writes runs/fewshot.json.

    cd ~/Desktop/"final project p3" && python3 retrain/plain_v2/plain_fewshot.py
"""
import os, sys, json
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.argv = [sys.argv[0]]
src = open(os.path.join(HERE, "plain_english.py")).read().split("res, t_all = {}, {}")[0]   # helpers only
pe = {"__file__": os.path.join(HERE, "plain_english.py")}; exec(compile(src, "plain_english_helpers", "exec"), pe)
samp, refs, dev_idx, test_idx = pe["samp"], pe["refs"], pe["dev_idx"], pe["test_idx"]

demos = [i for i in dev_idx if len(samp[i]["clause"]) > 150 and samp[i]["category"] != "Parties"][:3]
shot = "".join(f"Contract clause: {samp[i]['clause'][:400]}\nPlain English: {refs[str(i)]}\n\n" for i in demos)
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained("google/flan-t5-large")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large").to(DEV).eval()
rows = []
for i in test_idx:
    s = samp[i]
    prompt = ("Rewrite each contract clause as one short plain-English sentence for a person without legal "
              "training.\n\n" + shot + f"Contract clause: {s['clause'][:1000]}\nPlain English:")
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=768).to(DEV)
    with torch.no_grad():
        g = model.generate(**enc, max_new_tokens=64, num_beams=4, no_repeat_ngram_size=3)
    out = tok.decode(g[0], skip_special_tokens=True).strip()
    rows.append(dict(i=int(i), category=s["category"], out=out, rouge_l=pe["rouge_l"](out, refs[str(i)]),
                     flesch=pe["flesch"](out), copy=pe["copy_share"](out, s["clause"]),
                     invented=pe["invented_number"](out, s["clause"])))
res = dict(demos=[int(i) for i in demos], n=len(rows),
           rouge_l=float(np.mean([r["rouge_l"] for r in rows])), flesch=float(np.mean([r["flesch"] for r in rows])),
           copy_share=float(np.mean([r["copy"] for r in rows])), near_verbatim=float(np.mean([r["copy"] >= 0.9 for r in rows])),
           invented_number=float(np.mean([r["invented"] for r in rows])))
zs = json.load(open(os.path.join(HERE, "runs", "rows_flan-t5-large_explain.json")))
res["zero_shot_large_explain_near_verbatim"] = float(np.mean([zs[i]["copy"] >= 0.9 for i in test_idx]))
json.dump(dict(result=res, rows=rows), open(os.path.join(HERE, "runs", "fewshot.json"), "w"), indent=1)
print(res)
