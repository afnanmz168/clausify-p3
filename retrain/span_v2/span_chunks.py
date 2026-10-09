"""Retrain the span (clause-text) model on the chunks it actually reads when the system runs.

The first span model was trained on 1,200-character focus windows with the answer placed about 150
characters in, and never saw a chunk without an answer. In real use it reads plain chunks of the
2,000-character window the presence model chose (offsets 0 and 800), where the clause can start
anywhere, or not at all. When the clause sits in the second half of the window it succeeded only
5.4% of the time (report, Section 5.2.1).

Here every training example is such a chunk:
  * the answer's window is one of the 2,000-character windows that contain the answer's start,
    chosen at random, so the answer can sit anywhere in it;
  * the window is cut into the same chunks as at run time;
  * a chunk that contains the answer's start is labelled with the real token positions (an answer
    that runs past the chunk is cut at the chunk's end); a chunk that does not is labelled
    "no answer" (both positions on [CLS]), as in SQuAD 2.0.

The model is trained on the 327 "fit" contracts of the re-run (runs/L512/val_split.json). The epoch
and the scoring rule used at run time are chosen on the 81 validation contracts by
evaluate_span.py; the 102 test contracts are scored once.

    cd ~/Desktop/"final project p3" && python3 retrain/span_v2/span_chunks.py [--smoke]
"""
import os, sys, json, time, argparse, random
import numpy as np, pandas as pd, torch

HERE = os.path.dirname(os.path.abspath(__file__)); P = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(P, "analysis")); import common

ap = argparse.ArgumentParser()
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--epochs", type=int, default=2)
ap.add_argument("--n_answers", type=int, default=6000)
ap.add_argument("--max_len", type=int, default=384)
ap.add_argument("--lr", type=float, default=3e-5)
args = ap.parse_args()

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
OUT = os.path.join(HERE, "runs", "smoke" if args.smoke else "chunks")
os.makedirs(OUT, exist_ok=True)
CHUNK, CHUNK_STRIDE = 1200, 800


def chunk_starts(n):
    """Chunk offsets inside a window of n characters: every 800, plus one that reaches the end."""
    s = list(range(0, max(1, n - CHUNK + 1), CHUNK_STRIDE))
    if s[-1] + CHUNK < n:
        s.append(n - CHUNK)
    return s


D = common.load_all()
sp = json.load(open(os.path.join(P, "retrain", "v2_fullwindow", "runs", "L512", "val_split.json")))
fit_titles = sp["fit"]
if args.smoke:
    fit_titles = fit_titles[:15]
cat_of = lambda q: q.rsplit("__", 1)[-1]
by_title = {c["title"]: c for c in D["js"]}

# --- one (window, answer) pair per answer, the window chosen at random among those holding its start
rng = random.Random(SEED)
answers = []
for t in fit_titles:
    text = D["ctx"][t]; wins = common.make_windows(text)
    for qa in by_title[t]["paragraphs"][0]["qas"]:
        for a in qa["answers"]:
            at = a["text"]
            if not at.strip():
                continue
            s0 = a["answer_start"]; e0 = s0 + len(at)
            hold = [k for k, w in enumerate(wins) if k * common.STRIDE <= s0 < k * common.STRIDE + len(w)]
            if not hold:
                continue
            k = rng.choice(hold); off = k * common.STRIDE
            answers.append(dict(title=t, category=cat_of(qa["id"]), question=qa["question"], window=wins[k],
                                a_start=s0 - off, a_end=min(e0 - off, len(wins[k]))))
rng.shuffle(answers)
answers = answers[:args.n_answers]

rows = []
for x in answers:
    w = x["window"]
    for cs in chunk_starts(len(w)):
        rows.append(dict(question=x["question"], context=w[cs:cs + CHUNK],
                         a_start=x["a_start"] - cs, a_end=x["a_end"] - cs))
df = pd.DataFrame(rows)
print(f"fit contracts {len(fit_titles)} | answers {len(answers)} | chunk examples {len(df)}", flush=True)

from transformers import (AutoTokenizer, AutoModelForQuestionAnswering, Trainer, TrainingArguments,
                          TrainerCallback)
from datasets import Dataset
MODEL = "distilbert-base-uncased"
tok = AutoTokenizer.from_pretrained(MODEL)


def encode(b):
    enc = tok(b["question"], b["context"], truncation="only_second", max_length=args.max_len,
              return_offsets_mapping=True)
    starts, ends = [], []
    for i in range(len(b["context"])):
        off, seq = enc["offset_mapping"][i], enc.sequence_ids(i)
        ctx = [j for j, s in enumerate(seq) if s == 1]
        c0, c1 = ctx[0], ctx[-1]
        a_s, a_e = b["a_start"][i], b["a_end"][i]
        if not (off[c0][0] <= a_s < off[c1][1]):
            starts.append(0); ends.append(0); continue           # no answer starts in this chunk
        ts = next(j for j in ctx if off[j][1] > a_s)
        a_e = min(a_e, off[c1][1])                                # answer cut at the chunk's end
        te = max(j for j in ctx if off[j][0] < a_e)
        starts.append(ts); ends.append(max(te, ts))
    enc["start_positions"], enc["end_positions"] = starts, ends
    enc.pop("offset_mapping")
    return enc


ds = Dataset.from_pandas(df, preserve_index=False).map(encode, batched=True, remove_columns=list(df.columns))
pos = int(sum(1 for s in ds["start_positions"] if s > 0))
print(f"positive chunks {pos} ({pos / len(ds):.1%}), no-answer chunks {len(ds) - pos}", flush=True)
for i in range(3):                                                # labels decode back to the answer
    s, e = ds[i]["start_positions"], ds[i]["end_positions"]
    if s:
        print("  label ->", tok.decode(ds[i]["input_ids"][s:e + 1])[:90], flush=True)


def collate(features):
    batch = tok.pad({"input_ids": [f["input_ids"] for f in features],
                     "attention_mask": [f["attention_mask"] for f in features]},
                    padding=True, pad_to_multiple_of=64, return_tensors="pt")
    batch["start_positions"] = torch.tensor([f["start_positions"] for f in features])
    batch["end_positions"] = torch.tensor([f["end_positions"] for f in features])
    return batch


class FreeMPS(TrainerCallback):              # see presence_v2.py: MPS memory grows otherwise
    def on_step_end(self, a, state, control, **kw):
        if DEVICE == "mps" and state.global_step % 50 == 0:
            torch.mps.empty_cache()


model = AutoModelForQuestionAnswering.from_pretrained(MODEL)
targs = TrainingArguments(
    output_dir=os.path.join(OUT, "ckpt"), num_train_epochs=1 if args.smoke else args.epochs,
    per_device_train_batch_size=16, learning_rate=args.lr, save_strategy="epoch", save_only_model=True,
    eval_strategy="no", logging_strategy="steps", logging_steps=100, report_to="none", seed=SEED,
    fp16=False, bf16=False, dataloader_pin_memory=False,
)
trainer = Trainer(model=model, args=targs, train_dataset=ds, processing_class=tok, data_collator=collate,
                  callbacks=[FreeMPS()])
t0 = time.time()
trainer.train()
minutes = (time.time() - t0) / 60
print(f"[train] {minutes:.1f} min", flush=True)
json.dump(dict(fit_contracts=len(fit_titles), answers=len(answers), chunk_examples=len(ds),
               positive_chunks=pos, epochs=targs.num_train_epochs, max_len=args.max_len, lr=args.lr,
               train_minutes=round(minutes, 1), log_history=trainer.state.log_history),
          open(os.path.join(OUT, "train_info.json"), "w"), indent=1)
print("checkpoints:", sorted(os.listdir(os.path.join(OUT, "ckpt"))), flush=True)
