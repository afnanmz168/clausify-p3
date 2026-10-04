# Extracted from notebooks/build_artifacts.py (data section + step 4: the deployed summarizer, no eval set). Run from notebooks/.
"""
Rebuild all artifacts the standalone test.ipynb needs, trained on the 80% split:

  artifacts/split.json                 — the 80/20 contract-level split
  artifacts/baseline.pkl               — TF-IDF vectorizer + per-category LogReg
  outputs/presence_mil/final/          — window-level DistilBERT (presence)
  outputs/span/final/                  — DistilBertForQuestionAnswering (span)
  outputs/summarizer/final/            — FLAN-T5-small (summarizer)

Run once (≈80–90 min on Apple MPS). test.ipynb then loads these from disk.
"""
import os, json, time, random, pickle, ast
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print("device:", DEVICE, flush=True)

ART = "artifacts"
os.makedirs(ART, exist_ok=True)
os.makedirs("outputs/presence_mil/final", exist_ok=True)
os.makedirs("outputs/span/final", exist_ok=True)
os.makedirs("outputs/summarizer/final", exist_ok=True)

# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
js = json.load(open(hf_hub_download("theatticusproject/cuad", repo_type="dataset",
                                    filename="CUAD_v1/CUAD_v1.json")))["data"]

def cat_from_qid(qid):
    return qid.rsplit("__", 1)[-1]

titles = [c["title"] for c in js]
rng = np.random.default_rng(SEED)
perm = rng.permutation(len(titles))
n_test = int(0.20 * len(titles))
test_set = {titles[i] for i in perm[:n_test]}
train_titles = [t for t in titles if t not in test_set]
test_titles = [t for t in titles if t in test_set]
json.dump({"train": train_titles, "test": test_titles}, open(f"{ART}/split.json", "w"))
print(f"split: {len(train_titles)} train / {len(test_titles)} test", flush=True)

ctx_by_title, presence = {}, []
for c in js:
    t = c["title"]; para = c["paragraphs"][0]; ctx_by_title[t] = para["context"]
    for qa in para["qas"]:
        presence.append({"title": t, "category": cat_from_qid(qa["id"]),
                         "question": qa["question"], "label": 1 if qa["answers"] else 0})
pres_df = pd.DataFrame(presence)
cats = sorted(pres_df["category"].unique())
cat_question = pres_df.groupby("category")["question"].first().to_dict()

gold_spans = defaultdict(list)
for c in js:
    t = c["title"]
    for qa in c["paragraphs"][0]["qas"]:
        for a in qa["answers"]:
            if a["text"].strip():
                gold_spans[(t, cat_from_qid(qa["id"]))].append(a["text"])

_train = set(train_titles)


# --------------------------------------------------------------------------- #
# 4) Summarizer: FLAN-T5-small (CPU) -> save
# --------------------------------------------------------------------------- #
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
    "Non-Compete": "This restricts you from working with the other side's competitors.",
    "Exclusivity": "This makes one side the only allowed source or buyer for something.",
    "No-Solicit Of Customers": "You agree not to approach the other side's customers.",
    "Competitive Restriction Exception": "This lists where the non-compete or exclusivity does not apply.",
    "No-Solicit Of Employees": "You agree not to recruit the other side's employees.",
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
    "License Grant": "This grants permission to use the other side's intellectual property.",
    "Non-Transferable License": "You cannot pass this licence to anyone else.",
    "Affiliate License-Licensor": "The licensor's affiliates also grant you rights here.",
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
rows = []
for c in js:
    if c["title"] not in _train: continue
    for qa in c["paragraphs"][0]["qas"]:
        s = GOLD_SUMMARIES.get(cat_from_qid(qa["id"]))
        if not s: continue
        for a in qa["answers"]:
            at = a["text"].strip()
            if at:
                rows.append({"input_text": f"summarize in plain English: {at[:1200]}", "target_text": s})
sum_df = pd.DataFrame(rows)

T5_MODEL, MAX_IN, MAX_OUT, MAX_SUM_TRAIN = "google/flan-t5-small", 256, 48, 4000
t5_tok = AutoTokenizer.from_pretrained(T5_MODEL)
t5_model = AutoModelForSeq2SeqLM.from_pretrained(T5_MODEL)
s_train = sum_df.sample(min(MAX_SUM_TRAIN, len(sum_df)), random_state=SEED)
def enc_sum(b):
    mi = t5_tok(b["input_text"], truncation=True, max_length=MAX_IN)
    mi["labels"] = t5_tok(text_target=b["target_text"], truncation=True, max_length=MAX_OUT)["input_ids"]
    return mi
t5_train = Dataset.from_pandas(s_train, preserve_index=False).map(
    enc_sum, batched=True, remove_columns=s_train.columns.tolist())
t5_args = Seq2SeqTrainingArguments(output_dir="outputs/summarizer/ckpt", num_train_epochs=2,
    per_device_train_batch_size=8, learning_rate=3e-4, logging_strategy="steps", logging_steps=100,
    save_strategy="no", report_to="none", seed=SEED, use_cpu=True, fp16=False, bf16=False,
    dataloader_pin_memory=False)
t5_tr = Seq2SeqTrainer(model=t5_model, args=t5_args, train_dataset=t5_train,
                       processing_class=t5_tok, data_collator=DataCollatorForSeq2Seq(t5_tok, model=t5_model))
t0 = time.time(); t5_tr.train()
for h in t5_tr.state.log_history: print(h)
json.dump(t5_tr.state.log_history, open("../retrain/summarizer_log_history.json","w"), indent=1)
t5_model.save_pretrained("outputs/summarizer/final"); t5_tok.save_pretrained("outputs/summarizer/final")
print(f"[4/4] summarizer model saved ({(time.time()-t0)/60:.1f} min)", flush=True)

print("ALL ARTIFACTS BUILT", flush=True)

