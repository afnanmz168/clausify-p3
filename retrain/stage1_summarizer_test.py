# Extracted from notebooks/test.ipynb cells 1, 2, 8, 9. Run from notebooks/.
# === Setup: load the split, the data, and confirm artifacts exist ===
import os, json, time, random, pickle, re, string
from collections import defaultdict, Counter
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
from datasets import Dataset
from huggingface_hub import hf_hub_download

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print("device:", DEVICE)

ART = "artifacts"
for p in [f"{ART}/split.json", f"{ART}/baseline.pkl",
          "outputs/summarizer/final"]:
    assert os.path.exists(p), f"MISSING: {p} — run build_artifacts.py or train.ipynb first."

split = json.load(open(f"{ART}/split.json"))
train_titles, test_titles = split["train"], split["test"]
_train, _test = set(train_titles), set(test_titles)

js = json.load(open(hf_hub_download("theatticusproject/cuad", repo_type="dataset",
                                    filename="CUAD_v1/CUAD_v1.json")))["data"]
print("test contracts:", len(test_titles))


# === Rebuild contexts, windows, and gold spans (deterministic from the data) ===
def cat_from_qid(qid): return qid.rsplit("__", 1)[-1]
ctx_by_title = {c["title"]: c["paragraphs"][0]["context"] for c in js}

WIN, STRIDE = 2000, 1500
def make_windows(text):
    wins, pos = [], 0
    while pos < len(text):
        wins.append(text[pos:pos + WIN])
        if pos + WIN >= len(text): break
        pos += STRIDE
    return wins or [text]
windows_by_title = {t: make_windows(x) for t, x in ctx_by_title.items()}

gold_spans = defaultdict(list)
for c in js:
    t = c["title"]
    for qa in c["paragraphs"][0]["qas"]:
        for a in qa["answers"]:
            if a["text"].strip():
                gold_spans[(t, cat_from_qid(qa["id"]))].append(a["text"])
print("windowed", len(windows_by_title), "contracts")


from transformers import AutoTokenizer
# === Model 1 summarizer: ROUGE-L on the 20% test ===
from transformers import AutoModelForSeq2SeqLM
MAX_IN, MAX_OUT = 256, 48
t5_tok = AutoTokenizer.from_pretrained("outputs/summarizer/final")
t5_model = AutoModelForSeq2SeqLM.from_pretrained("outputs/summarizer/final").to("cpu").eval()

GOLD_SUMMARIES = {
    "Document Name": "This states the official name and type of the agreement.",
    "Parties": "This identifies who is signing the contract and the role each side plays.",
    "Agreement Date": "This is the date the contract was signed.",
    "Effective Date": "This is the date the contract terms actually start applying.",
    "Expiration Date": "This is when the contract is scheduled to end.",
    "Renewal Term": "This explains how the contract continues if neither side cancels.",
    "Notice Period To Terminate Renewal": "This is how long ahead you must say you will not renew.",
    "Governing Law": "This says which country or state law decides any dispute.",
    "Most Favored Nation": "This promises you the best deal the other side gives anyone else.",
    "Non-Compete": "This restricts you from working with the other side\'s competitors.",
    "Exclusivity": "This makes one side the only allowed source or buyer for something.",
    "No-Solicit Of Customers": "You agree not to approach the other side\'s customers.",
    "Competitive Restriction Exception": "This lists where the non-compete or exclusivity does not apply.",
    "No-Solicit Of Employees": "You agree not to recruit the other side\'s employees.",
    "Non-Disparagement": "You agree not to publicly criticise the other side.",
    "Termination For Convenience": "Either side can end the contract for any reason with notice.",
    "Rofr/Rofo/Rofn": "You get the first chance to match an offer or take an opportunity.",
    "Change Of Control": "This explains what happens if one company is bought or merges.",
    "Anti-Assignment": "You cannot transfer this contract to someone else without permission.",
    "Revenue/Profit Sharing": "You share a portion of revenue or profit with the other side.",
    "Price Restrictions": "This limits what prices you can charge or change.",
    "Minimum Commitment": "You promise to buy or deliver at least a stated minimum amount.",
    "Volume Restriction": "There is a cap on how much you can sell or distribute.",
    "Ip Ownership Assignment": "Any new intellectual property created belongs to the named owner.",
    "Joint Ip Ownership": "Both sides jointly own intellectual property created under this contract.",
    "License Grant": "This grants permission to use the other side\'s intellectual property.",
    "Non-Transferable License": "You cannot pass this licence to anyone else.",
    "Affiliate License-Licensor": "The licensor\'s affiliates also grant you rights here.",
    "Affiliate License-Licensee": "Your affiliates can use the licence too.",
    "Unlimited/All-You-Can-Eat-License": "The licence is unlimited in scope or volume.",
    "Irrevocable Or Perpetual License": "The licence cannot be taken away and may last forever.",
    "Source Code Escrow": "Source code is held by a third party in case something goes wrong.",
    "Post-Termination Services": "Some services continue for a while after the contract ends.",
    "Audit Rights": "The other side can inspect your records to check compliance.",
    "Uncapped Liability": "There is no limit on how much one side may owe in damages.",
    "Cap On Liability": "There is a maximum amount one side may owe in damages.",
    "Liquidated Damages": "A fixed amount is owed for specific breaches, agreed in advance.",
    "Warranty Duration": "This is how long the warranty lasts.",
    "Insurance": "You must keep specified insurance coverage in force.",
    "Covenant Not To Sue": "You agree not to sue the other side over the listed matters.",
    "Third Party Beneficiary": "Someone not signing the contract still gets to enforce part of it.",
}

sum_rows = []
for c in js:
    if c["title"] not in _test: continue
    for qa in c["paragraphs"][0]["qas"]:
        s = GOLD_SUMMARIES.get(cat_from_qid(qa["id"]))
        if not s: continue
        for a in qa["answers"]:
            at = a["text"].strip()
            if at: sum_rows.append({"input_text": f"summarize in plain English: {at[:1200]}", "target_text": s})
sum_test = pd.DataFrame(sum_rows)

def lcs_len(a, b):
    dp = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        prev = 0
        for j in range(1, len(b) + 1):
            tmp = dp[j]; dp[j] = prev + 1 if a[i-1] == b[j-1] else max(dp[j], dp[j-1]); prev = tmp
    return dp[-1]
def rouge_l(pred, gold):
    p, g = pred.lower().split(), gold.lower().split()
    if not p or not g: return 0.0
    l = lcs_len(p, g)
    if l == 0: return 0.0
    pr, rc = l / len(p), l / len(g); return 2 * pr * rc / (pr + rc)

samp = sum_test.sample(150, random_state=SEED).reset_index(drop=True)
scores = []
for i in range(len(samp)):
    inp = t5_tok(samp.loc[i, "input_text"], return_tensors="pt", truncation=True, max_length=MAX_IN)
    out = t5_model.generate(**inp, max_new_tokens=MAX_OUT, num_beams=2)
    scores.append(rouge_l(t5_tok.decode(out[0], skip_special_tokens=True), samp.loc[i, "target_text"]))
print(f"fine-tuned summarizer ROUGE-L on 150 test clauses : {np.mean(scores):.3f}")

res = {"n": len(scores), "rouge_l_templates": round(float(np.mean(scores)), 4)}


# === Zero-shot vs fine-tuned summarizer, same clauses (qualitative) ===
zs_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small").to("cpu").eval()
demo = sum_test.sample(4, random_state=1).reset_index(drop=True)
for i in range(len(demo)):
    clause = demo.loc[i, "input_text"].replace("summarize in plain English: ", "")
    enc = t5_tok(demo.loc[i, "input_text"], return_tensors="pt", truncation=True, max_length=256)
    ft = t5_tok.decode(t5_model.generate(**enc, max_new_tokens=40)[0], skip_special_tokens=True)
    zs = t5_tok.decode(zs_model.generate(**enc, max_new_tokens=40)[0], skip_special_tokens=True)
    print(f"--- clause {i+1} ---")
    print("clause    :", clause[:160])
    print("gold      :", demo.loc[i, "target_text"])
    print("fine-tuned:", ft)
    print("zero-shot :", zs)
    print()

json.dump(res, open("../retrain/stage1_summarizer_test.json", "w"), indent=1); print(res)
