# Auto-extracted from notebooks/train.ipynb cells 1, 9-12, 14, 19-21 + save. Run from notebooks/.
import pandas as pd
from huggingface_hub import hf_hub_download


# === Cell 1: setup + load the raw CUAD dataset ===
import os, sys, json, random, time
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
import torch

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# Device
if torch.cuda.is_available():
    DEVICE = 'cuda'
elif torch.backends.mps.is_available():
    DEVICE = 'mps'
else:
    DEVICE = 'cpu'




# === Cell 9: load CUAD_v1.json (full context + char offsets for span training) ===
json_path = hf_hub_download(
    repo_id="theatticusproject/cuad",
    repo_type="dataset",
    filename="CUAD_v1/CUAD_v1.json",
)

with open(json_path) as f:
    cuad_json = json.load(f)

js = cuad_json["data"]
ex_qa = js[0]["paragraphs"][0]["qas"][0]
print("contracts in JSON :", len(js))
print("context present   :", "context" in js[0]["paragraphs"][0],
      "| context chars:", len(js[0]["paragraphs"][0]["context"]))
print("answer has offset :", "answer_start" in (ex_qa["answers"][0] if ex_qa["answers"] else {}))

# === Cell 10: contract-level train/test split (80/20, by whole contract) ===
titles = [c["title"] for c in js]

rng = np.random.default_rng(SEED)
perm = rng.permutation(len(titles))
n_test = int(0.20 * len(titles))
test_set  = {titles[i] for i in perm[:n_test]}

train_titles = [t for t in titles if t not in test_set]
test_titles  = [t for t in titles if t in test_set]

print("total contracts :", len(titles))
print("train contracts :", len(train_titles))
print("test  contracts :", len(test_titles))

# === Cell 11: presence dataset — labels per (contract, category), full text kept per contract ===
def cat_from_qid(qid):
    return qid.rsplit("__", 1)[-1]

ctx_by_title = {}                      # title -> full contract text (stored once, ~510 entries)
presence = []
for c in js:
    title = c["title"]
    split = "test" if title in test_set else "train"
    para = c["paragraphs"][0]
    ctx_by_title[title] = para["context"]
    for qa in para["qas"]:
        presence.append({
            "title"   : title,
            "category": cat_from_qid(qa["id"]),
            "question": qa["question"],
            "label"   : 1 if qa["answers"] else 0,
            "split"   : split,
        })

pres_df = pd.DataFrame(presence)
pres_df.groupby("split")["label"].agg(count="count", positive_rate="mean")

# === Cell 12: baseline — TF-IDF over the FULL contract + per-category LogReg ===
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score

_train = set(train_titles)
_test  = set(test_titles)
tr_titles = [t for t in titles if t in _train]
te_titles = [t for t in titles if t in _test]

vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), sublinear_tf=True, min_df=2)
X_train = vec.fit_transform([ctx_by_title[t] for t in tr_titles])
X_test  = vec.transform([ctx_by_title[t] for t in te_titles])

cats = sorted(pres_df["category"].unique())
lab = pres_df.pivot_table(index="title", columns="category", values="label", aggfunc="max")
Y_train = lab.loc[tr_titles, cats].values
Y_test  = lab.loc[te_titles, cats].values

rows = []
for j, cat in enumerate(cats):
    ytr, yte = Y_train[:, j], Y_test[:, j]
    if np.unique(ytr).size < 2:
        # Only one class in train (e.g. Document Name = always present) -> predict the constant.
        pred = np.full_like(yte, int(ytr[0]))
    else:
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        clf.fit(X_train, ytr)
        pred = clf.predict(X_test)
    rows.append({
        "category" : cat,
        "test_pos" : int(yte.sum()),
        "precision": round(precision_score(yte, pred, zero_division=0), 3),
        "recall"   : round(recall_score(yte, pred, zero_division=0), 3),
        "f1"       : round(f1_score(yte, pred, zero_division=0), 3),
    })

base = pd.DataFrame(rows)
macro = pd.DataFrame([{
    "category": "** MACRO AVG **",
    "test_pos": int(base["test_pos"].sum()),
    "precision": round(base["precision"].mean(), 3),
    "recall"   : round(base["recall"].mean(), 3),
    "f1"       : round(base["f1"].mean(), 3),
}])


# === Cell 14: sliding windows + question-guided retrieval (TF-IDF cosine) ===
from sklearn.metrics.pairwise import linear_kernel

WIN, STRIDE = 2000, 1500          # 2000-char windows, 500 overlap

windows_by_title = {}
for title, text in ctx_by_title.items():
    wins, pos = [], 0
    while pos < len(text):
        wins.append(text[pos:pos + WIN])
        if pos + WIN >= len(text):
            break
        pos += STRIDE
    windows_by_title[title] = wins or [text]

all_windows = [w for ws in windows_by_title.values() for w in ws]
retr_vec = TfidfVectorizer(max_features=30000, ngram_range=(1, 2),
                           min_df=2, sublinear_tf=True).fit(all_windows)

# query per category = the category's QUESTION text (inference-available, no answer peeking)
cat_question  = pres_df.groupby("category")["question"].first().to_dict()
cat_query_vec = {c: retr_vec.transform([cat_question[c]]) for c in cats}

def retrieve_window(title, category):
    wins = windows_by_title[title]
    scores = linear_kernel(cat_query_vec[category], retr_vec.transform(wins)).ravel()
    return wins[int(scores.argmax())]

# --- sanity check: does the retrieved window actually contain the gold clause? ---
demo_t, demo_cat = train_titles[0], "Audit Rights"
win = retrieve_window(demo_t, demo_cat)
qa = next(q for q in next(c for c in js if c["title"] == demo_t)["paragraphs"][0]["qas"]
          if cat_from_qid(q["id"]) == demo_cat)
gold = qa["answers"][0]["text"] if qa["answers"] else None
print("contract         :", demo_t[:55])
print("category         :", demo_cat, "| gold present:", gold is not None)
print("retrieved window :", win[:200].replace("\n", " "))
if gold:
    print("gold in window?  :", gold[:60] in win)

# === Cell 21: build span-extraction examples (present clauses only, windowed) ===
span_examples = []
for c in js:
    title = c["title"]
    split = "test" if title in _test else "train"
    wins  = windows_by_title[title]
    for qa in c["paragraphs"][0]["qas"]:
        if not qa["answers"]:
            continue
        cat, q = cat_from_qid(qa["id"]), qa["question"]
        for a in qa["answers"]:
            atext = a["text"].strip()
            if not atext:
                continue
            probe = atext[:200]                 # match on first 200 chars
            for w in wins:
                pos = w.find(probe)
                if pos != -1:
                    span_examples.append({
                        "title": title, "split": split, "category": cat,
                        "question": q, "context": w,
                        "answer_text": atext, "char_start": pos,
                    })
                    break

span_df = pd.DataFrame(span_examples)
print(span_df.groupby("split").size().to_frame("n_examples"))

from transformers import AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
# === Cell 22: tokenize span examples -> (start_token, end_token) labels ===
QA_MODEL, QA_MAXLEN, FOCUS, MAX_SPAN_TRAIN = "distilbert-base-uncased", 320, 1200, 8000
qa_tok = AutoTokenizer.from_pretrained(QA_MODEL)

def encode_qa(batch):
    ctxs, a_starts, a_ends = [], [], []
    for w, cs, at in zip(batch["context"], batch["char_start"], batch["answer_text"]):
        start_f = max(0, cs - 150)
        ctx = w[start_f:start_f + FOCUS]
        a_s = cs - start_f
        ctxs.append(ctx); a_starts.append(a_s); a_ends.append(min(len(ctx), a_s + len(at)))
    enc = qa_tok(batch["question"], ctxs, truncation="only_second",
                 max_length=QA_MAXLEN, return_offsets_mapping=True)
    starts, ends = [], []
    for i in range(len(ctxs)):
        off, seq = enc["offset_mapping"][i], enc.sequence_ids(i)
        idx = 0
        while seq[idx] != 1: idx += 1
        c_start = idx
        while idx < len(seq) and seq[idx] == 1: idx += 1
        c_end = idx - 1
        a_s, a_e = a_starts[i], a_ends[i]
        if off[c_start][0] > a_s or off[c_end][1] < a_e:
            ts = te = 0                                   # answer not in span -> CLS
        else:
            j = c_start
            while j <= c_end and off[j][0] <= a_s: j += 1
            ts = j - 1
            j = c_end
            while j >= c_start and off[j][1] >= a_e: j -= 1
            te = j + 1
        starts.append(ts); ends.append(te)
    enc["start_positions"], enc["end_positions"] = starts, ends
    enc.pop("offset_mapping")
    return enc

n_train = int((span_df["split"] == "train").sum())
sp_train = span_df[span_df.split == "train"].sample(min(MAX_SPAN_TRAIN, n_train), random_state=SEED)
sp_test  = span_df[span_df.split == "test"]

qa_train = Dataset.from_pandas(sp_train, preserve_index=False).map(
    encode_qa, batched=True, remove_columns=sp_train.columns.tolist())
qa_test = Dataset.from_pandas(sp_test, preserve_index=False).map(
    encode_qa, batched=True, remove_columns=sp_test.columns.tolist())

# sanity: decode the labelled span back to text for 3 examples (should match the gold answer start)
for i in range(3):
    s, e = qa_train[i]["start_positions"], qa_train[i]["end_positions"]
    print("label span ->", qa_tok.decode(qa_train[i]["input_ids"][s:e+1])[:90])

# === Cell 23: train DistilBertForQuestionAnswering (span extractor) ===
from transformers import AutoModelForQuestionAnswering

qa_model = AutoModelForQuestionAnswering.from_pretrained(QA_MODEL)

def qa_collator(features):
    batch = qa_tok.pad(
        {"input_ids":      [f["input_ids"]      for f in features],
         "attention_mask": [f["attention_mask"] for f in features]},
        padding=True, return_tensors="pt",
    )
    batch["start_positions"] = torch.tensor([f["start_positions"] for f in features])
    batch["end_positions"]   = torch.tensor([f["end_positions"]   for f in features])
    return batch

qa_args = TrainingArguments(
    output_dir="outputs/span/ckpt",
    num_train_epochs=2, per_device_train_batch_size=16,
    per_device_eval_batch_size=16, learning_rate=3e-5,
    logging_strategy="steps", logging_steps=100,
    eval_strategy="no", save_strategy="no", report_to="none", seed=SEED,
    fp16=False, bf16=False, dataloader_pin_memory=False,
)
qa_trainer = Trainer(model=qa_model, args=qa_args, train_dataset=qa_train,
                     processing_class=qa_tok, data_collator=qa_collator)

t0 = time.time()
qa_trainer.train()
for h in qa_trainer.state.log_history: print(h)
print(f"\n[done] span training took {(time.time()-t0)/60:.1f} min")


qa_model.save_pretrained("outputs/span/final"); qa_tok.save_pretrained("outputs/span/final")
json.dump(qa_trainer.state.log_history, open("../retrain/span_log_history.json","w"), indent=1)
print("saved span/final")
