"""A dedicated classifier for the app's clause-by-clause mode.

Clause mode used to reuse the presence model: it asked all 41 yes/no questions about one clause and
picked the category whose score stood out most (z-score against other categories' clauses). That
picks the right category for 44.9% of held-out CUAD clauses (clause_calibration_v2.json).

Here a classifier is trained for the actual task, "which of the 41 types is this clause?", with a
42nd class, "none", for paragraphs that are not any annotated clause (definitions, notices,
signature blocks...), so the app can say "unrecognized" for a reason the model learnt.

  data   clause texts (>= 20 characters) of the 327 "fit" contracts of the re-run, labelled with
         their category; "none" = lines of the same contracts (40-2,000 characters) that overlap
         no annotated clause, 2,000 of them.
  models (a) TF-IDF (words 1-2) + logistic regression, C chosen on validation;
         (b) DistilBERT fine-tuned for 42 classes (256 tokens, 3 epochs, best epoch on validation).
  choose on a balanced sample of the 81 validation contracts (20 clauses per category, the same
         recipe as the test), by 41-way accuracy.
  test   once, on exactly the 721 test clauses that gave 44.9% (calibrate_clause_mode.py: seed 42,
         20 per category from the 102 test contracts), plus 200 "none" lines of the test contracts.

    cd ~/Desktop/"final project p3" && python3 retrain/clause_v2/clause_classifier.py tfidf
    cd ~/Desktop/"final project p3" && python3 retrain/clause_v2/clause_classifier.py bert      # ~25 min, MPS
    cd ~/Desktop/"final project p3" && python3 retrain/clause_v2/clause_classifier.py test      # choose, test once
"""
import os, sys, re, json, random, time, pickle
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); P = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(P, "analysis")); import common
STAGE = sys.argv[1] if len(sys.argv) > 1 else "tfidf"
OUT = os.path.join(HERE, "runs"); os.makedirs(OUT, exist_ok=True)
SEED, NONE = 42, "none"
D = common.load_all(); CATS = D["cats"]; LABELS = CATS + [NONE]
sp = json.load(open(os.path.join(P, "retrain", "v2_fullwindow", "runs", "L512", "val_split.json")))
FIT, VAL, TEST = set(sp["fit"]), set(sp["val"]), set(sp["test"])
cat_of = lambda q: q.rsplit("__", 1)[-1]


# ------------------------------------------------------------------ data
def clause_examples(titles):
    out = []
    for c in D["js"]:
        if c["title"] not in titles:
            continue
        for qa in c["paragraphs"][0]["qas"]:
            for a in qa["answers"]:
                t = a["text"].strip()
                if len(t) >= 20:
                    out.append((t, cat_of(qa["id"]), c["title"]))
    return out


def none_lines(titles, n, seed):
    """Lines of these contracts (40-2,000 characters) that overlap no annotated clause."""
    pool = []
    for c in D["js"]:
        if c["title"] not in titles:
            continue
        ctx = c["paragraphs"][0]["context"]
        spans = [(a["answer_start"], a["answer_start"] + len(a["text"]))
                 for qa in c["paragraphs"][0]["qas"] for a in qa["answers"] if a["text"].strip()]
        for m in re.finditer(r"[^\n]+", ctx):
            s, e = m.start(), m.end(); t = m.group().strip()
            if 40 <= len(t) <= 2000 and not any(s < b and a < e for a, b in spans):
                pool.append((t, NONE, c["title"]))
    random.Random(seed).shuffle(pool)
    return pool[:n]


def balanced_eval(titles, per_cat, seed):
    """per_cat clause texts per category, the recipe of calibrate_clause_mode.py."""
    by = defaultdict(list)
    for t, c, _ in clause_examples(titles):
        by[c].append(t)
    rng = random.Random(seed)
    return [(t, c) for c in CATS for t in rng.sample(by[c], min(per_cat, len(by[c])))]


def app_test_clauses():
    """Exactly the 721 test clauses behind the 44.9% (calibrate_clause_mode.py, same code path)."""
    titles = [c["title"] for c in D["js"]]
    perm = np.random.default_rng(SEED).permutation(len(titles))
    test = {titles[i] for i in perm[:int(0.2 * len(titles))]}
    data = {"train": defaultdict(list), "test": defaultdict(list)}
    for c in D["js"]:
        split = "test" if c["title"] in test else "train"
        for qa in c["paragraphs"][0]["qas"]:
            for a in qa["answers"]:
                t = a["text"].strip()
                if len(t) >= 20:
                    data[split][cat_of(qa["id"])].append(t)
    rng = random.Random(SEED)
    sample = lambda d, k: [(cat, t) for cat in CATS for t in rng.sample(d[cat], min(k, len(d[cat])))]
    sample(data["train"], 40)                      # advances the generator exactly as the original did
    te = sample(data["test"], 20)
    assert test == TEST and len(te) == 721, (len(te), test == TEST)
    return [(t, c) for c, t in te]


train = clause_examples(FIT) + none_lines(FIT, 2000, SEED)
val = balanced_eval(VAL, 20, SEED + 1) + [(t, c) for t, c, _ in none_lines(VAL, 200, SEED + 1)]
same_text_labels = defaultdict(set)                # a span can be annotated for several categories
for t, c, _ in clause_examples(FIT | VAL | TEST):
    same_text_labels[t].add(c)


def scores(probs, data):
    """41-way accuracy (arg-max over the 41 types, as the 44.9% was measured), top-3, the share of
    real clauses the model calls "none", accuracy on "none" lines, and accuracy counting any category
    the same text is annotated with."""
    y = np.array([LABELS.index(c) for _, c in data]); real = y < len(CATS)
    p41 = probs[:, :len(CATS)]; top = p41.argmax(1)
    acc = float((top[real] == y[real]).mean())
    top3 = float(np.mean([y[i] in np.argsort(-p41[i])[:3] for i in np.where(real)[0]]))
    anyl = float(np.mean([CATS[top[i]] in same_text_labels.get(data[i][0], {data[i][1]}) for i in np.where(real)[0]]))
    full = probs.argmax(1)
    return dict(n_clauses=int(real.sum()), acc41=acc, top3=top3, acc_any_label=anyl,
                clauses_called_none=float((full[real] == len(CATS)).mean()),
                n_none=int((~real).sum()), none_acc=float((full[~real] == len(CATS)).mean()) if (~real).any() else None)


# ------------------------------------------------------------------ (a) TF-IDF + logistic regression
if STAGE == "tfidf":
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=50000)
    Xtr = vec.fit_transform([t for t, _, _ in train]); ytr = [LABELS.index(c) for _, c, _ in train]
    Xva = vec.transform([t for t, _ in val])
    res = {}
    for C in (1, 3, 10, 30, 100):
        clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(Xtr, ytr)
        res[C] = (scores(clf.predict_proba(Xva), val), clf)
        print(f"tfidf C={C}: {res[C][0]}", flush=True)
    C = max(res, key=lambda k: res[k][0]["acc41"])
    pickle.dump({"vec": vec, "clf": res[C][1], "labels": LABELS, "C": C}, open(os.path.join(OUT, "tfidf.pkl"), "wb"))
    json.dump({"val_by_C": {str(k): v[0] for k, v in res.items()}, "chosen_C": C,
               "n_train": len(train), "n_train_none": sum(1 for _, c, _ in train if c == NONE)},
              open(os.path.join(OUT, "tfidf_val.json"), "w"), indent=1)

# ------------------------------------------------------------------ (b) DistilBERT, 42 classes
if STAGE == "bert":
    import torch
    from datasets import Dataset
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments,
                              DataCollatorWithPadding, TrainerCallback)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    DEV = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    enc = lambda b: tok(b["text"], truncation=True, max_length=256)
    tr = Dataset.from_dict({"text": [t for t, _, _ in train], "label": [LABELS.index(c) for _, c, _ in train]}).map(enc, batched=True)
    va = Dataset.from_dict({"text": [t for t, _ in val], "label": [LABELS.index(c) for _, c in val]}).map(enc, batched=True)
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=len(LABELS))

    class FreeMPS(TrainerCallback):
        def on_step_end(self, a, state, control, **kw):
            if DEV == "mps" and state.global_step % 50 == 0:
                torch.mps.empty_cache()

    def metrics(ev):
        lg = ev.predictions; pr = np.exp(lg - lg.max(1, keepdims=True)); pr /= pr.sum(1, keepdims=True)
        return {"acc41": scores(pr, val)["acc41"]}

    args = TrainingArguments(output_dir=os.path.join(OUT, "bert_ckpt"), num_train_epochs=3, per_device_train_batch_size=16,
                             per_device_eval_batch_size=32, learning_rate=3e-5, eval_strategy="epoch", save_strategy="epoch",
                             save_only_model=True, load_best_model_at_end=True, metric_for_best_model="acc41",
                             logging_strategy="steps", logging_steps=100, report_to="none", seed=SEED,
                             fp16=False, bf16=False, dataloader_pin_memory=False)
    trainer = Trainer(model=model, args=args, train_dataset=tr, eval_dataset=va, processing_class=tok,
                      data_collator=DataCollatorWithPadding(tok, pad_to_multiple_of=64), compute_metrics=metrics,
                      callbacks=[FreeMPS()])
    t0 = time.time(); trainer.train(); minutes = (time.time() - t0) / 60
    trainer.save_model(os.path.join(OUT, "bert_best")); tok.save_pretrained(os.path.join(OUT, "bert_best"))
    lg = trainer.predict(va).predictions; pr = np.exp(lg - lg.max(1, keepdims=True)); pr /= pr.sum(1, keepdims=True)
    json.dump({"val": scores(pr, val), "train_minutes": round(minutes, 1), "n_train": len(train),
               "epochs_eval": [h for h in trainer.state.log_history if "eval_acc41" in h]},
              open(os.path.join(OUT, "bert_val.json"), "w"), indent=1)
    print("bert val:", scores(pr, val), f"{minutes:.1f} min", flush=True)

# ------------------------------------------------------------------ choose on validation, test once
if STAGE == "test":
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    DEV = "mps" if torch.backends.mps.is_available() else "cpu"
    tv, bv = json.load(open(os.path.join(OUT, "tfidf_val.json"))), json.load(open(os.path.join(OUT, "bert_val.json")))
    tf = pickle.load(open(os.path.join(OUT, "tfidf.pkl"), "rb"))
    tok = AutoTokenizer.from_pretrained(os.path.join(OUT, "bert_best"))
    bert = AutoModelForSequenceClassification.from_pretrained(os.path.join(OUT, "bert_best")).to(DEV).eval()

    def bert_probs(texts):
        out = []
        for i in range(0, len(texts), 32):
            e = tok(texts[i:i + 32], truncation=True, max_length=256, padding=True, return_tensors="pt").to(DEV)
            with torch.no_grad():
                out.append(torch.softmax(bert(**e).logits.float(), -1).cpu().numpy())
        return np.vstack(out)

    cand = {"tfidf": lambda x: tf["clf"].predict_proba(tf["vec"].transform(x)), "bert": bert_probs}
    cand["average"] = lambda x: (cand["tfidf"](x) + cand["bert"](x)) / 2
    vx = [t for t, _ in val]
    val_scores = {k: scores(f(vx), val) for k, f in cand.items()}
    choice = max(val_scores, key=lambda k: val_scores[k]["acc41"])
    test = app_test_clauses() + [(t, c) for t, c, _ in none_lines(TEST, 200, SEED + 2)]
    res = {k: scores(f([t for t, _ in test]), test) for k, f in cand.items()}
    cal_old = json.load(open(os.path.join(os.path.dirname(P), "final project app p3", "clause_calibration_v2.json")))["eval"]
    out = dict(validation=val_scores, chosen=choice, test_chosen=res[choice], test_all=res,
               old_presence_zscore=dict(acc41=cal_old["calibrated_acc"], top3=cal_old["calibrated_top3"], n=cal_old["n_test_clauses"]),
               tfidf_C=tf["C"], bert_train_minutes=bv["train_minutes"], n_train=bv["n_train"])
    risk = json.load(open(os.path.join(P, "analysis", "risk_levels.json")))
    pr = cand[choice]([t for t, _ in test]); top = pr[:, :len(CATS)].argmax(1)
    real = [i for i, (_, c) in enumerate(test) if c != NONE]
    out["test_chosen"]["risk_level_acc"] = float(np.mean([risk[CATS[top[i]]] == risk[test[i][1]] for i in real]))
    json.dump(out, open(os.path.join(OUT, "evaluation.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "test_all"}, indent=1))
