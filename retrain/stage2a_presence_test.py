# Auto-extracted from notebooks/test.ipynb cells 1-6 (presence only). Run from notebooks/.
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
          "outputs/presence_mil/final"]:
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


# === Load the TF-IDF baseline; compute its probability per (test contract, category) ===
bl = pickle.load(open(f"{ART}/baseline.pkl", "rb"))
vec, classifiers, cats, cat_question = bl["vec"], bl["classifiers"], bl["cats"], bl["cat_question"]

te_titles = [t for t in [c["title"] for c in js] if t in _test]
X_test = vec.transform([ctx_by_title[t] for t in te_titles])
Y_test = np.array([[int(bool(gold_spans.get((t, cat), []))) for cat in cats] for t in te_titles])

proba_tfidf = np.zeros_like(Y_test, dtype=float)
for j, cat in enumerate(cats):
    kind, obj = classifiers[cat]
    proba_tfidf[:, j] = float(obj) if kind == "const" else obj.predict_proba(X_test)[:, 1]
print("TF-IDF test probabilities:", proba_tfidf.shape)


# === Model 2A presence: max-pool the window-level DistilBERT over all test windows ===
from transformers import AutoTokenizer, AutoModelForSequenceClassification
mil_tok = AutoTokenizer.from_pretrained("outputs/presence_mil/final")
mil_model = AutoModelForSequenceClassification.from_pretrained("outputs/presence_mil/final").to(DEVICE).eval()
MAX_LEN = 256

pairs = []
for t in te_titles:
    for cat in cats:
        q = cat_question[cat]
        for w in windows_by_title[t]:
            pairs.append((t, cat, q, w))
pdf = pd.DataFrame(pairs, columns=["title", "category", "question", "window"])

probs = np.zeros(len(pdf)); B = 32
t0 = time.time()
for i in range(0, len(pdf), B):
    q = pdf["question"].iloc[i:i+B].tolist(); w = pdf["window"].iloc[i:i+B].tolist()
    enc = mil_tok(q, w, truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        logits = mil_model(**enc).logits
    probs[i:i+B] = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()
pdf["p1"] = probs
bag = pdf.groupby(["title", "category"])["p1"].max().reset_index()
p_trans = {(t, c): p for t, c, p in zip(bag["title"], bag["category"], bag["p1"])}
print(f"presence inference: {len(pdf)} (question, window) pairs in {(time.time()-t0)/60:.1f} min")


# === Ensemble TF-IDF + transformer (parameter-free) — micro-F1 on the 20% test ===
from sklearn.metrics import f1_score
keys = [(te_titles[i], cats[j]) for i in range(len(te_titles)) for j in range(len(cats))]
a = np.array([proba_tfidf[i, j] for i in range(len(te_titles)) for j in range(len(cats))])
b = np.array([p_trans.get(k, 0.0) for k in keys])
y = np.array([int(bool(gold_spans.get(k, []))) for k in keys])
micro = lambda pred: round(f1_score(y, pred, zero_division=0), 3)

ens_table = pd.DataFrame([
    {"model": "Ensemble AND  (min)",    "micro_F1": micro((np.minimum(a, b) >= 0.5).astype(int))},
    {"model": "TF-IDF baseline",        "micro_F1": micro((a >= 0.5).astype(int))},
    {"model": "Ensemble AVG  (mean)",   "micro_F1": micro(((a + b) / 2 >= 0.5).astype(int))},
    {"model": "Transformer (max-pool)", "micro_F1": micro((b >= 0.5).astype(int))},
    {"model": "Ensemble OR   (max)",    "micro_F1": micro((np.maximum(a, b) >= 0.5).astype(int))},
]).sort_values("micro_F1", ascending=False).reset_index(drop=True)
print(ens_table.to_string())


# === Confusion matrix + test metrics (AND-ensemble) — the "matrix test" ===
from sklearn.metrics import confusion_matrix
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

y_pred = (np.minimum(a, b) >= 0.5).astype(int)
cm = confusion_matrix(y, y_pred); tn, fp, fn, tp = cm.ravel()

print("CONFUSION MATRIX (AND-ensemble, 80/20 test set)")
print(f"                 Predicted Absent   Predicted Present")
print(f"Actually Absent      TN = {tn:>5d}         FP = {fp:>5d}")
print(f"Actually Present     FN = {fn:>5d}         TP = {tp:>5d}\n")

acc = (tp + tn) / (tp + tn + fp + fn); prec = tp / (tp + fp); rec = tp / (tp + fn)
spec = tn / (tn + fp); f1 = 2 * prec * rec / (prec + rec)
print("TEST METRICS (from the confusion matrix):")
print(f"  Accuracy    : {acc:.4f}")
print(f"  Precision   : {prec:.4f}")
print(f"  Recall      : {rec:.4f}")
print(f"  Specificity : {spec:.4f}")
print(f"  F1 Score    : {f1:.4f}")

fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                fontsize=18, color="white" if cm[i, j] > cm.max()/2 else "black")
ax.set_xticks([0, 1]); ax.set_xticklabels(["Absent", "Present"])
ax.set_yticks([0, 1]); ax.set_yticklabels(["Absent", "Present"])
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix — AND-Ensemble (80/20 Test)")
plt.colorbar(im, ax=ax); plt.tight_layout(); plt.savefig("../retrain/stage2a_confusion_matrix.png", dpi=150)



# === Per-category F1 (AND-ensemble) + save everything ===
from sklearn.metrics import precision_score, recall_score
P_and = y_pred.reshape(len(te_titles), len(cats)); Y = y.reshape(len(te_titles), len(cats))
P_tf = (a >= 0.5).astype(int).reshape(Y.shape); P_tr = (b >= 0.5).astype(int).reshape(Y.shape)
percat = pd.DataFrame([{"category": c, "test_pos": int(Y[:, j].sum()),
    "precision": round(precision_score(Y[:, j], P_and[:, j], zero_division=0), 3),
    "recall": round(recall_score(Y[:, j], P_and[:, j], zero_division=0), 3),
    "f1": round(f1_score(Y[:, j], P_and[:, j], zero_division=0), 3)} for j, c in enumerate(cats)]
).sort_values(["f1", "test_pos"], ascending=False)
print(percat.to_string())
prf = lambda p: {"micro_f1": round(f1_score(y, p), 4), "precision": round(precision_score(y, p), 4), "recall": round(recall_score(y, p), 4)}
res = {"ensemble_table": ens_table.to_dict("records"),
       "transformer": prf((b >= 0.5).astype(int)), "tfidf": prf((a >= 0.5).astype(int)),
       "and_ensemble": prf(y_pred),
       "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
       "metrics": {"accuracy": round(acc, 4), "precision": round(prec, 4), "recall": round(rec, 4),
                   "specificity": round(spec, 4), "f1": round(f1, 4)},
       "macro_f1_and": round(f1_score(Y, P_and, average="macro", zero_division=0), 4),
       "per_category": percat.to_dict("records")}
json.dump(res, open("../retrain/stage2a_presence_test.json", "w"), indent=1)
np.savez("../retrain/stage2a_probs_seed42.npz", a=a, b=b, y=y)
print("saved stage2a_presence_test.json")
