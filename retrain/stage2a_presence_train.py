# Auto-extracted from notebooks/train.ipynb cells 1, 9-18 + save. Run from notebooks/.
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
print(pd.concat([macro, base.sort_values("f1", ascending=False)], ignore_index=True).to_string())

# === Cell 13: richer baseline anchors — micro, weighted, and tiered macro-F1 ===
P = np.zeros_like(Y_test)
for j in range(len(cats)):
    ytr = Y_train[:, j]
    if np.unique(ytr).size < 2:
        P[:, j] = int(ytr[0])
    else:
        clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(X_train, ytr)
        P[:, j] = clf.predict(X_test)

supp   = Y_test.sum(0)
common = supp >= 10
summary = pd.DataFrame([
    {"metric": f"micro-F1  (all {Y_test.size:,} cells equal)",
     "value": round(f1_score(Y_test.ravel(), P.ravel(), zero_division=0), 3)},
    {"metric": "weighted-F1  (by support)",
     "value": round(f1_score(Y_test, P, average="weighted", zero_division=0), 3)},
    {"metric": "macro-F1  (all 41 categories)",
     "value": round(f1_score(Y_test, P, average="macro", zero_division=0), 3)},
    {"metric": f"macro-F1  (>=10 test pos, {int(common.sum())} cats)",
     "value": round(f1_score(Y_test[:, common], P[:, common], average="macro", zero_division=0), 3)},
])
print(summary)

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

# === Cell 15: apply retrieval to all rows + measure hit-rate on positives ===
# Precompute each contract's window TF-IDF matrix ONCE (avoid re-vectorizing 41x)
win_mat = {t: retr_vec.transform(ws) for t, ws in windows_by_title.items()}

def best_window_idx(title, category):
    scores = linear_kernel(cat_query_vec[category], win_mat[title]).ravel()
    return int(scores.argmax())

# gold span per (title, category) — used ONLY to score retrieval quality, never to build the window
gold_by_key = {}
for c in js:
    t = c["title"]
    for qa in c["paragraphs"][0]["qas"]:
        gold_by_key[(t, cat_from_qid(qa["id"]))] = (
            qa["answers"][0]["text"] if qa["answers"] else None
        )

t0 = time.time()
windows_col, hit, npos = [], 0, 0
for r in presence:                       # the list built in Cell 11
    t, cat = r["title"], r["category"]
    w = windows_by_title[t][best_window_idx(t, cat)]
    windows_col.append(w)
    g = gold_by_key[(t, cat)]
    if g:
        npos += 1
        hit += int(g[:80] in w)

pres_df["window"] = windows_col
print(f"applied retrieval to {len(pres_df):,} rows in {time.time()-t0:.1f}s")
print(f"retrieval hit-rate on positives : {hit}/{npos} = {hit/npos:.1%}")

# === Cell 16: fix retrieval — compare query = question vs category-name, top1 vs top3 ===
from collections import defaultdict

# all gold spans per (title, category)  [previously only kept the first]
gold_spans = defaultdict(list)
for c in js:
    t = c["title"]
    for qa in c["paragraphs"][0]["qas"]:
        cat = cat_from_qid(qa["id"])
        for a in qa["answers"]:
            if a["text"].strip():
                gold_spans[(t, cat)].append(a["text"])

def hit_rate(query_by_cat, topk=1):
    qv = {c: retr_vec.transform([query_by_cat[c]]) for c in cats}
    hit = npos = 0
    for (t, cat), spans in gold_spans.items():
        npos += 1
        scores = linear_kernel(qv[cat], win_mat[t]).ravel()
        top = scores.argsort()[::-1][:topk]
        joined = " ".join(windows_by_title[t][i] for i in top)
        hit += int(any(s[:80] in joined for s in spans))
    return hit / npos

strategies = [
    ("question        (top1)", cat_question,                              1),
    ("category name   (top1)", {c: c for c in cats},                      1),
    ("category name   (top3)", {c: c for c in cats},                      3),
    ("name+question   (top3)", {c: c + " " + cat_question[c] for c in cats}, 3),
]
print(pd.DataFrame([{"strategy": s, "topk": k, "hit_rate": round(hit_rate(q, k), 3)}
                    for s, q, k in strategies]).to_string())   # Table 4.1 of the report

# === Cell 17: build window-level training examples (MIL, TRAIN split only) ===
import random as _random
_random.seed(SEED)

def has_span(spans, w):
    return any(s[:80] in w for s in spans)

NEG_PER_BAG = 2     # sampled negative windows per (contract, category)

win_examples = []
for c in js:
    title = c["title"]
    if title not in _train:                      # TRAIN ONLY
        continue
    wins = windows_by_title[title]
    for qa in c["paragraphs"][0]["qas"]:
        cat   = cat_from_qid(qa["id"])
        q     = qa["question"]
        spans = gold_spans.get((title, cat), [])
        if spans:                                 # present
            pos_idx = [i for i, w in enumerate(wins) if has_span(spans, w)]
            neg_idx = [i for i in range(len(wins)) if i not in pos_idx]
            for i in pos_idx:
                win_examples.append({"question": q, "window": wins[i], "label": 1})
            for i in _random.sample(neg_idx, min(NEG_PER_BAG, len(neg_idx))):
                win_examples.append({"question": q, "window": wins[i], "label": 0})
        else:                                     # absent — all windows negative
            for i in _random.sample(range(len(wins)), min(NEG_PER_BAG, len(wins))):
                win_examples.append({"question": q, "window": wins[i], "label": 0})

win_df = pd.DataFrame(win_examples)
print(pd.DataFrame({
    "n_examples":    [len(win_df)],
    "positives":     [int(win_df["label"].sum())],
    "positive_rate": [round(win_df["label"].mean(), 3)],
}))

# === Cell 18: train window-level DistilBERT (question, window) -> contains-clause ===
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          DataCollatorWithPadding, Trainer, TrainingArguments)
from datasets import Dataset

MODEL_NAME = "distilbert-base-uncased"
MAX_LEN, BATCH, EPOCHS, LR = 256, 16, 2, 2e-5
MAX_TRAIN = 24000          # subsample cap (keep ~39% positive). Raise for more data / more time.

# stratified subsample to keep training tractable on a laptop
if len(win_df) > MAX_TRAIN:
    win_df_s = (win_df.groupby("label", group_keys=False)
                .apply(lambda g: g.sample(round(MAX_TRAIN * len(g) / len(win_df)),
                                          random_state=SEED)))
else:
    win_df_s = win_df
print("training on", len(win_df_s), "examples | positive rate",
      round(win_df_s["label"].mean(), 3))

tok = AutoTokenizer.from_pretrained(MODEL_NAME)
wds = Dataset.from_pandas(win_df_s[["question", "window", "label"]], preserve_index=False).shuffle(seed=SEED)
wsplit = wds.train_test_split(test_size=0.05, seed=SEED)

def enc(b):
    e = tok(b["question"], b["window"], truncation=True, max_length=MAX_LEN)
    e["labels"] = b["label"]
    return e

wtr = wsplit["train"].map(enc, batched=True, remove_columns=wsplit["train"].column_names)
wva = wsplit["test"].map(enc,  batched=True, remove_columns=wsplit["test"].column_names)

mil_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

args = TrainingArguments(
    output_dir="outputs/presence_mil/ckpt",
    num_train_epochs=EPOCHS, per_device_train_batch_size=BATCH,
    per_device_eval_batch_size=BATCH, learning_rate=LR,
    eval_strategy="epoch", logging_strategy="steps", logging_steps=100,
    save_strategy="no", report_to="none", seed=SEED,
    fp16=False, bf16=False, dataloader_pin_memory=False,
)
mil_trainer = Trainer(model=mil_model, args=args, train_dataset=wtr, eval_dataset=wva,
                      processing_class=tok, data_collator=DataCollatorWithPadding(tok))

t0 = time.time()
mil_trainer.train()
for h in mil_trainer.state.log_history: print(h)
print(f"\n[done] window-level training took {(time.time()-t0)/60:.1f} min")

# === Save baseline + presence model (subset of the notebook's save cell) ===
import pickle
os.makedirs("artifacts", exist_ok=True)
json.dump({"train": train_titles, "test": test_titles}, open("artifacts/split.json", "w"))
classifiers = {}
for j, cat in enumerate(cats):
    ytr = Y_train[:, j]
    if np.unique(ytr).size < 2:
        classifiers[cat] = ("const", int(ytr[0]))
    else:
        classifiers[cat] = ("clf", LogisticRegression(max_iter=1000, class_weight="balanced").fit(X_train, ytr))
pickle.dump({"vec": vec, "classifiers": classifiers, "cats": cats, "cat_question": cat_question},
            open("artifacts/baseline.pkl", "wb"))
mil_model.save_pretrained("outputs/presence_mil/final"); tok.save_pretrained("outputs/presence_mil/final")
json.dump(mil_trainer.state.log_history, open("../retrain/presence_log_history.json","w"), indent=1)
print("saved baseline.pkl + presence_mil/final")
