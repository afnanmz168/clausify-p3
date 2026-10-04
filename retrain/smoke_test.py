"""Quick check that every saved model loads and produces a sensible output.

    cd ~/Desktop/"final project p3"
    python3 retrain/smoke_test.py

Runs on CPU in about a minute. Exit code 0 = all checks passed.
"""
import os, pickle, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          AutoModelForQuestionAnswering, AutoModelForSeq2SeqLM)

NB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "notebooks")
CLAUSE = ("This Agreement shall be governed by and construed in accordance with the laws of the "
          "State of New York, without regard to its conflict of laws principles.")
OTHER = "Supplier shall deliver the Products to the Distributor's warehouse within thirty days."
Q_GOV = ('Highlight the parts (if any) of this contract related to "Governing Law" that should be '
         'reviewed by a lawyer. Details: Which state/country\'s law governs the interpretation of '
         'the contract?')
failures = []


def check(name, ok, detail):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        failures.append(name)


# 1) TF-IDF + logistic-regression baseline -------------------------------------------
t0 = time.time()
bl = pickle.load(open(os.path.join(NB, "artifacts", "baseline.pkl"), "rb"))
kind, clf = bl["classifiers"]["Governing Law"]
p_gov, p_oth = (clf.predict_proba(bl["vec"].transform([x]))[0, 1] for x in (CLAUSE, OTHER))
check("TF-IDF baseline", p_gov > p_oth,
      f"P(Governing Law) = {p_gov:.2f} on a governing-law clause vs {p_oth:.2f} on a delivery clause "
      f"({len(bl['cats'])} categories, {time.time()-t0:.1f}s)")

# 2) Presence classifier (window-level DistilBERT) ------------------------------------
d = os.path.join(NB, "outputs", "presence_mil", "final")
tok = AutoTokenizer.from_pretrained(d); m = AutoModelForSequenceClassification.from_pretrained(d).eval()
enc = tok([Q_GOV, Q_GOV], [CLAUSE, OTHER], padding=True, return_tensors="pt")
with torch.no_grad():
    p = F.softmax(m(**enc).logits, -1)[:, 1].numpy()
check("Presence classifier", p[0] > 0.5 and p[0] > p[1],
      f"P(present) = {p[0]:.2f} on the governing-law clause vs {p[1]:.2f} on the delivery clause")

# 3) Span extractor (DistilBERT QA) ----------------------------------------------------
d = os.path.join(NB, "outputs", "span", "final")
tok = AutoTokenizer.from_pretrained(d); m = AutoModelForQuestionAnswering.from_pretrained(d).eval()
ctx = OTHER + " " + CLAUSE
enc = tok(Q_GOV, ctx, return_tensors="pt", truncation="only_second", max_length=320)
with torch.no_grad():
    out = m(**enc)
sep = int((enc["input_ids"][0] == tok.sep_token_id).nonzero()[0])
mask = torch.full_like(out.start_logits[0], -1e9); mask[sep + 1:] = 0
s, e = int((out.start_logits[0] + mask).argmax()), int((out.end_logits[0] + mask).argmax())
span = tok.decode(enc["input_ids"][0][s:max(e, s) + 1], skip_special_tokens=True)
check("Span extractor", "new york" in span.lower() or "governed" in span.lower(),
      f'extracted "{span[:80]}"')

# 4) Summarizer (FLAN-T5-small) --------------------------------------------------------
d = os.path.join(NB, "outputs", "summarizer", "final")
tok = AutoTokenizer.from_pretrained(d); m = AutoModelForSeq2SeqLM.from_pretrained(d).eval()
enc = tok("summarize in plain English: " + CLAUSE, return_tensors="pt", truncation=True, max_length=256)
with torch.no_grad():
    gen = tok.decode(m.generate(**enc, max_new_tokens=48, num_beams=2)[0], skip_special_tokens=True)
check("Summarizer", len(gen.split()) >= 4, f'output "{gen}"')

print("\nALL MODELS OK" if not failures else f"\nFAILED: {', '.join(failures)}")
sys.exit(1 if failures else 0)
