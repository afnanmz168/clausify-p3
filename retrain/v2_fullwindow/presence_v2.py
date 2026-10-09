"""Stage 2A presence, version 2: read the WHOLE window, and hold out a validation split.

Two changes from stage2a_presence_train.py / stage2a_presence_test.py, and nothing else:

1. MAX_LEN. The original model reads (question, 2000-char window) pairs at max_length=256.
   The question alone takes ~53 tokens, so only ~986 characters (49%) of each window are read,
   ~34% of every contract is never read, and 23.2% of positive training windows have their gold
   clause past the cut (the label says "present" for text the model never sees).
   At max_length=512 the whole window fits (question p90 71 tokens + ~405 window tokens).

2. VALIDATION SPLIT. 20% of the 408 training contracts (81, seed 42) are held out, so the epoch,
   pooling rule and thresholds can be chosen on data that is neither training nor test.
   The model trains on the other 327 contracts. The 102 test contracts are unchanged.

Everything else is identical: 2000/1500 windows, Cell-17 example construction, 24,000-example
stratified cap, DistilBERT, LR 2e-5, batch 16, seed 42.
Epochs: trains 3 and keeps the epoch with the lowest validation-window loss.

Output (in retrain/v2_fullwindow/runs/<tag>/):
  model/                 selected checkpoint
  window_scores.npz      P(present) for every (contract, category, window) of the val + test contracts
  train_info.json        split sizes, truncation statistics, log history, selected epoch

Run from the notebooks/ folder:
  caffeinate -i python3 -u ../retrain/v2_fullwindow/presence_v2.py --max-len 512 --tag L512 \
      2>&1 | tee ../retrain/v2_fullwindow/L512.log
Then:  python3 ../retrain/v2_fullwindow/tune_and_test.py --tag L512
"""
import os, sys, json, time, random, argparse, shutil
import numpy as np, pandas as pd, torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
P = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(P, "analysis"))
import common

ap = argparse.ArgumentParser()
ap.add_argument("--max-len", type=int, default=512)
ap.add_argument("--tag", default=None)
ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--max-train", type=int, default=24000)
ap.add_argument("--smoke", action="store_true", help="tiny end-to-end run to check the script")
args = ap.parse_args()
TAG = args.tag or f"L{args.max_len}"
if args.smoke:
    TAG += "_smoke"
OUT = os.path.join(HERE, "runs", TAG); os.makedirs(OUT, exist_ok=True)

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print(f"device {DEVICE} | max_len {args.max_len} | tag {TAG}", flush=True)

# --- data, the unchanged 408/102 split, and the new validation split -------------------------
D = common.load_all()
split = json.load(open(os.path.join(common.NB, "artifacts", "split.json")))
assert sorted(split["train"]) == sorted(D["train_titles"]) and sorted(split["test"]) == sorted(D["test_titles"])
train_titles, test_titles = D["train_titles"], D["test_titles"]

rng = np.random.default_rng(SEED)
perm = rng.permutation(len(train_titles))
n_val = int(0.20 * len(train_titles))
val_titles = [train_titles[i] for i in sorted(perm[:n_val])]
fit_titles = [t for t in train_titles if t not in set(val_titles)]
if args.smoke:
    fit_titles, val_titles, test_titles = fit_titles[:20], val_titles[:4], test_titles[:4]
json.dump({"fit": fit_titles, "val": val_titles, "test": test_titles},
          open(os.path.join(OUT, "val_split.json"), "w"), indent=0)
print(f"fit {len(fit_titles)} | val {len(val_titles)} | test {len(test_titles)} contracts", flush=True)

cats, cat_question = D["cats"], D["cat_question"]
windows = {t: common.make_windows(D["ctx"][t]) for t in fit_titles + val_titles + test_titles}

# --- window examples (Cell 17 construction) ------------------------------------------------
def examples(titles):
    Dx = dict(D); Dx["train_titles"] = titles
    return common.build_window_examples(Dx)

def cap(df, n):
    if len(df) <= n:
        return df
    return (df.groupby("label", group_keys=False)
              .apply(lambda g: g.sample(round(n * len(g) / len(df)), random_state=SEED)))

max_train = 200 if args.smoke else args.max_train
tr_df = cap(examples(fit_titles), max_train)
va_df = cap(examples(val_titles), 60 if args.smoke else 3000)
print(f"train windows {len(tr_df)} (pos rate {tr_df.label.mean():.3f}) | val windows {len(va_df)}", flush=True)

# --- how much of each window does this max_len read? ---------------------------------------
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          DataCollatorWithPadding, Trainer, TrainingArguments)
from datasets import Dataset
MODEL_NAME = "distilbert-base-uncased"
tok = AutoTokenizer.from_pretrained(MODEL_NAME)

def chars_read(q, w):
    e = tok(q, w, truncation=True, max_length=args.max_len, return_offsets_mapping=True)
    seq, offs = e.sequence_ids(), e["offset_mapping"]
    return max(offs[i][1] for i in range(len(seq)) if seq[i] == 1)

pos = tr_df[tr_df.label == 1].sample(min(2000, int(tr_df.label.sum())), random_state=SEED)
read_share, beyond = [], 0
gold = D["gold_spans"]
q2cat = {v: k for k, v in cat_question.items()}
for q, w in zip(pos.question, pos.window):
    r = chars_read(q, w)
    if len(w) == common.WIN:
        read_share.append(r / len(w))
    # find the earliest gold start in this window (same 80-char test as the labels use)
    cat = q2cat.get(q)
    starts = [w.find(s[:80]) for t in fit_titles for s in gold.get((t, cat), []) if s[:80] in w] if cat else []
    if starts and min(starts) >= r:
        beyond += 1
trunc = {"median_share_of_window_read": round(float(np.median(read_share)), 4) if read_share else None,
         "positive_windows_checked": len(pos),
         "positive_windows_with_clause_past_cut": beyond,
         "share_label_noise": round(beyond / max(1, len(pos)), 4)}
print("truncation:", trunc, flush=True)

# --- train ---------------------------------------------------------------------------------
def enc(b):
    e = tok(b["question"], b["window"], truncation=True, max_length=args.max_len)
    e["labels"] = b["label"]
    return e

tr_ds = Dataset.from_pandas(tr_df[["question", "window", "label"]], preserve_index=False).shuffle(seed=SEED)
va_ds = Dataset.from_pandas(va_df[["question", "window", "label"]], preserve_index=False)
tr_ds = tr_ds.map(enc, batched=True, remove_columns=tr_ds.column_names)
va_ds = va_ds.map(enc, batched=True, remove_columns=va_ds.column_names)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
ckpt_dir = os.path.join(OUT, "ckpt")
targs = TrainingArguments(
    output_dir=ckpt_dir, num_train_epochs=1 if args.smoke else args.epochs,
    per_device_train_batch_size=16, per_device_eval_batch_size=32, learning_rate=args.lr,
    eval_strategy="epoch", save_strategy="epoch", save_only_model=True,
    load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
    logging_strategy="steps", logging_steps=100, report_to="none", seed=SEED,
    fp16=False, bf16=False, dataloader_pin_memory=False,
)
# MPS keeps a buffer per distinct tensor shape. Free-length padding gave hundreds of shapes, memory grew
# into swap and steps slowed from 2 s to 14 s. Padding to a multiple of 64 caps it at a few shapes.
from transformers import TrainerCallback
class FreeMPS(TrainerCallback):
    def on_step_end(self, a, state, control, **kw):
        if DEVICE == "mps" and state.global_step % 50 == 0:
            torch.mps.empty_cache()
trainer = Trainer(model=model, args=targs, train_dataset=tr_ds, eval_dataset=va_ds,
                  processing_class=tok, data_collator=DataCollatorWithPadding(tok, pad_to_multiple_of=64),
                  callbacks=[FreeMPS()])
t0 = time.time()
trainer.train()
train_min = (time.time() - t0) / 60
evals = [h for h in trainer.state.log_history if "eval_loss" in h]
best = min(evals, key=lambda h: h["eval_loss"])
print(f"[train] {train_min:.1f} min | eval loss by epoch:",
      [(round(h["epoch"], 2), round(h["eval_loss"], 4)) for h in evals],
      f"| selected epoch {best['epoch']:.0f}", flush=True)
model = trainer.model.to(DEVICE).eval()
model.save_pretrained(os.path.join(OUT, "model")); tok.save_pretrained(os.path.join(OUT, "model"))
shutil.rmtree(ckpt_dir, ignore_errors=True)

# --- score every window of every val + test contract, for all 41 categories -----------------
score_titles = val_titles + test_titles
rows = [(ti, ci, wi) for ti, t in enumerate(score_titles) for ci in range(len(cats))
        for wi in range(len(windows[t]))]
print(f"[score] {len(rows):,} (question, window) pairs", flush=True)
p = np.zeros(len(rows), dtype=np.float32); B = 32
t0 = time.time()
for i in range(0, len(rows), B):
    chunk = rows[i:i + B]
    qs = [cat_question[cats[ci]] for _, ci, _ in chunk]
    ws = [windows[score_titles[ti]][wi] for ti, _, wi in chunk]
    e = tok(qs, ws, truncation=True, max_length=args.max_len, padding=True, pad_to_multiple_of=64,
            return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        p[i:i + B] = F.softmax(model(**e).logits, dim=-1)[:, 1].float().cpu().numpy()
    if DEVICE == "mps" and (i // B) % 200 == 0:
        torch.mps.empty_cache()
    if (i // B) % 500 == 0:
        done = i + len(chunk); rate = done / max(1e-9, time.time() - t0)
        print(f"  {done:,}/{len(rows):,} | {rate:.0f} pairs/s | ~{(len(rows) - done) / rate / 60:.0f} min left", flush=True)
score_min = (time.time() - t0) / 60
r = np.array(rows, dtype=np.int32)
np.savez(os.path.join(OUT, "window_scores.npz"), title_idx=r[:, 0], cat_idx=r[:, 1], win_idx=r[:, 2], p=p,
         titles=np.array(score_titles), cats=np.array(cats),
         is_val=np.array([t in set(val_titles) for t in score_titles]))
json.dump({"tag": TAG, "max_len": args.max_len, "lr": args.lr, "epochs": args.epochs, "smoke": args.smoke,
           "n_fit": len(fit_titles), "n_val": len(val_titles), "n_test": len(test_titles),
           "n_train_windows": len(tr_df), "n_val_windows": len(va_df), "truncation": trunc,
           "selected_epoch": best["epoch"], "train_minutes": round(train_min, 1),
           "score_minutes": round(score_min, 1), "log_history": trainer.state.log_history},
          open(os.path.join(OUT, "train_info.json"), "w"), indent=1)
print(f"[done] scored in {score_min:.1f} min -> {OUT}", flush=True)
