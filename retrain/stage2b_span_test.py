# Auto-extracted from notebooks/test.ipynb cells 1, 2, 7. Run from notebooks/.
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
          "outputs/span/final"]:
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
# === Model 2B span extractor: token-F1 / Exact Match / overlap on the 20% test ===
from transformers import AutoModelForQuestionAnswering
QA_MAXLEN, FOCUS = 320, 1200
qa_tok = AutoTokenizer.from_pretrained("outputs/span/final")
qa_model = AutoModelForQuestionAnswering.from_pretrained("outputs/span/final").to(DEVICE).eval()

span_rows = []
for c in js:
    title = c["title"]
    if title not in _test: continue
    wins = windows_by_title[title]
    for qa in c["paragraphs"][0]["qas"]:
        if not qa["answers"]: continue
        q = qa["question"]
        for aa in qa["answers"]:
            at = aa["text"].strip()
            if not at: continue
            probe = at[:200]
            for w in wins:
                pos = w.find(probe)
                if pos != -1:
                    span_rows.append({"question": q, "context": w, "answer_text": at, "char_start": pos}); break
sp_test = pd.DataFrame(span_rows)

def encode_qa(batch):
    ctxs = []
    for w, cs in zip(batch["context"], batch["char_start"]):
        start_f = max(0, cs - 150); ctxs.append(w[start_f:start_f + FOCUS])
    return qa_tok(batch["question"], ctxs, truncation="only_second", max_length=QA_MAXLEN)
qa_test = Dataset.from_pandas(sp_test, preserve_index=False).map(
    encode_qa, batched=True, remove_columns=sp_test.columns.tolist())

def normalize(s):
    s = s.lower(); s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in string.punctuation); return " ".join(s.split())
def token_f1(pred, gold):
    p, g = normalize(pred).split(), normalize(gold).split()
    if not p or not g: return float(p == g)
    common = sum((Counter(p) & Counter(g)).values())
    if common == 0: return 0.0
    pr, rc = common / len(p), common / len(g); return 2 * pr * rc / (pr + rc)

sep_id = qa_tok.sep_token_id; golds = sp_test["answer_text"].tolist()
em = f1 = overlap = 0; N = len(qa_test); B = 32
for i in range(0, N, B):
    feats = [qa_test[j] for j in range(i, min(i + B, N))]
    batch = qa_tok.pad({"input_ids": [f["input_ids"] for f in feats],
                        "attention_mask": [f["attention_mask"] for f in feats]},
                       padding=True, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = qa_model(**batch)
    sl = out.start_logits.cpu().numpy(); el = out.end_logits.cpu().numpy()
    for k, f in enumerate(feats):
        ids = f["input_ids"]; L = len(ids); sep = ids.index(sep_id)
        mask = np.full(L, -1e9); mask[sep + 1:] = 0.0
        s = int(np.argmax(sl[k][:L] + mask)); e = int(np.argmax(el[k][:L] + mask))
        if e < s: e = s
        e = min(e, s + 60)
        pred_text = qa_tok.decode(ids[s:e + 1], skip_special_tokens=True)
        g = golds[i + k]
        em += int(normalize(pred_text) == normalize(g)); tf = token_f1(pred_text, g)
        f1 += tf; overlap += int(tf >= 0.5)

print(f"span extractor on {N} test examples (answer-bearing windows):")
print(f"  Exact Match     : {em/N:.1%}")
print(f"  token-F1        : {f1/N:.3f}")
print(f"  overlap (F1>=.5): {overlap/N:.1%}")

res = {"n": N, "exact_match": round(em/N, 4), "token_f1": round(f1/N, 4), "overlap_f1_ge_0.5": round(overlap/N, 4)}
json.dump(res, open("../retrain/stage2b_span_test.json", "w"), indent=1); print(res)
