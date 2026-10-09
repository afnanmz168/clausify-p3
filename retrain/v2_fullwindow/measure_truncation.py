"""How much of each 2,000-character presence window the original model reads at 256 tokens,
and how many positive TRAINING windows have their clause after the cut (label noise).
Writes truncation_256.json. Takes about two minutes on a CPU; no model weights are needed
beyond the tokenizer saved with the original presence model.

    cd ~/Desktop/"final project p3" && python3 retrain/v2_fullwindow/measure_truncation.py
"""
import os, sys, json, random
import numpy as np
from transformers import AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__)); P = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(P, "analysis")); import common

D = common.load_all(); train = set(D["train_titles"])
tok = AutoTokenizer.from_pretrained(os.path.join(common.NB, "outputs", "presence_mil", "final"))
MAXLEN = 256
random.seed(0)

def read_upto(q, w):
    e = tok(q, w, truncation=True, max_length=MAXLEN, return_offsets_mapping=True)
    seq, offs = e.sequence_ids(), e["offset_mapping"]
    return max(offs[i][1] for i in range(len(seq)) if seq[i] == 1)

share, qlen, pos, beyond = [], [], 0, 0
for c in D["js"]:
    t = c["title"]; W = common.make_windows(D["ctx"][t])
    for qa in c["paragraphs"][0]["qas"]:
        q = qa["question"]; cat = qa["id"].rsplit("__", 1)[-1]
        w = random.choice(W)                      # one sampled window per (contract, category)
        if len(w) == common.WIN:
            share.append(read_upto(q, w) / len(w))
        qlen.append(len(tok(q)["input_ids"]) - 2)
        spans = D["gold_spans"].get((t, cat), [])
        if t in train and spans:
            for w in W:
                hits = [w.find(s[:80]) for s in spans if s[:80] in w]
                if hits:
                    pos += 1; beyond += int(min(hits) >= read_upto(q, w))
out = {"max_len": MAXLEN, "question_tokens_median": float(np.median(qlen)),
       "share_read_median": round(float(np.median(share)), 4),
       "share_read_p10": round(float(np.percentile(share, 10)), 4),
       "share_read_p90": round(float(np.percentile(share, 90)), 4),
       "chars_read_median": round(float(np.median(share)) * common.WIN),
       "contract_never_read": round(max(0.0, 1 - float(np.median(share)) * common.WIN / common.STRIDE), 4),
       "positive_train_windows": pos, "positive_train_windows_clause_after_cut": beyond,
       "share_label_noise": round(beyond / pos, 4)}
json.dump(out, open(os.path.join(HERE, "truncation_256.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
