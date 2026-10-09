"""Repeat the Chapter 5 analyses on the re-run (512-token) models, and calibrate the scores.

The first-setup versions live in retrain/analysis_oct2026/. Here every analysis uses the setups
of the re-run, with the thresholds chosen on the 81 validation contracts (runs/L512/decision.json):

  RECALL      transformer alone, tuned for micro-F2 (the app's default)
  RECALL_ENS  average of transformer and TF-IDF, tuned for micro-F2
  BALANCED    both models above their thresholds (AND), tuned for micro-F1
  TFIDF       TF-IDF alone, tuned for micro-F1

Writes runs/L512/downstream.json and the app's calibration map runs/L512/calibration.json.
Window choice for the span step is the arg-max window of the saved window scores, exactly as in
the app; only the span model is run here (a few minutes on MPS).

    cd ~/Desktop/"final project p3" && python3 retrain/v2_fullwindow/downstream_v2.py
"""
import os, sys, re, json, string, pickle
from collections import Counter
import numpy as np, pandas as pd, torch
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

HERE = os.path.dirname(os.path.abspath(__file__)); P = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(P, "analysis")); import common
RUN = os.path.join(HERE, "runs", "L512")
D = common.load_all(); cats = D["cats"]; C = len(cats)
sp = json.load(open(os.path.join(RUN, "val_split.json"))); val_t, test_t = sp["val"], sp["test"]
dec = json.load(open(os.path.join(RUN, "decision.json")))["objectives"]
risk = json.load(open(os.path.join(P, "analysis", "risk_levels.json"))); band = np.array([risk[c] for c in cats])
Yv, Yt = common.labels(D, val_t, cats), common.labels(D, test_t, cats)

# ---------------------------------------------------------------- scores
z = np.load(os.path.join(RUN, "window_scores.npz")); titles = list(z["titles"])
df = pd.DataFrame({"t": z["title_idx"], "c": z["cat_idx"], "w": z["win_idx"], "p": z["p"]})
best = df.loc[df.groupby(["t", "c"]).p.idxmax()]
B = np.zeros((len(titles), C)); W = np.zeros((len(titles), C), dtype=int)
B[best.t.values, best.c.values] = best.p.values; W[best.t.values, best.c.values] = best.w.values
vi = [titles.index(t) for t in val_t]; ti = [titles.index(t) for t in test_t]
Bv, Bt, Wt = B[vi], B[ti], W[ti]
bl = pickle.load(open(os.path.join(RUN, "baseline_C10.pkl"), "rb"))
def tfidf(ts):
    X = bl["vec"].transform([D["ctx"][t] for t in ts])
    return np.column_stack([np.full(len(ts), float(o)) if k == "const" else o.predict_proba(X)[:, 1]
                            for k, o in (bl["classifiers"][c] for c in cats)])
At = tfidf(test_t)
th = lambda obj, key: np.array([dec[obj][key][c] for c in cats])
SETUPS = {
    "RECALL": (Bt >= th("f2", "t_transformer")).astype(int),
    "RECALL_ENS": ((At + Bt) / 2 >= th("f2", "t_avg")).astype(int),
    "BALANCED": ((At >= th("f1", "t_tfidf")) & (Bt >= th("f1", "t_transformer"))).astype(int),
    "TFIDF": (At >= th("f1", "t_tfidf")).astype(int),
}
SCORE = {"RECALL": Bt, "RECALL_ENS": (At + Bt) / 2, "BALANCED": np.minimum(At, Bt), "TFIDF": At}

def prf(y, p):
    tp = int(((y == 1) & (p == 1)).sum()); fp = int(((y == 0) & (p == 1)).sum()); fn = int(((y == 1) & (p == 0)).sum())
    pr = tp / (tp + fp) if tp + fp else 0.0; rc = tp / (tp + fn) if tp + fn else 0.0
    return dict(tp=tp, fp=fp, fn=fn, precision=pr, recall=rc, f1=2 * pr * rc / (pr + rc) if pr + rc else 0.0)

out = {"setups": list(SETUPS), "n_test": len(test_t), "n_val": len(val_t)}
for name, Pr in SETUPS.items():                                  # sanity: overall scores match results.json
    out.setdefault("overall", {})[name] = prf(Yt, Pr)

# ---------------------------------------------------------------- 1. risk-weighted
out["by_band"] = {}
for b in ["High", "Medium", "Low"]:
    idx = np.where(band == b)[0]
    out["by_band"][b] = {"n_categories": int(len(idx)), "n_test_positives": int(Yt[:, idx].sum())}
    for name, Pr in SETUPS.items():
        out["by_band"][b][name] = prf(Yt[:, idx], Pr[:, idx])
print("High-risk missed:", {k: out["by_band"]["High"][k]["fn"] for k in SETUPS})

# ---------------------------------------------------------------- 2. rare categories
train_freq = np.array([sum(bool(D["gold_spans"].get((t, c), [])) for t in D["train_titles"]) for c in cats])
order = np.argsort(train_freq, kind="stable"); rare, comm = order[:10], order[10:]
out["rare_tail"] = {"rare_categories": [cats[j] for j in rare]}
for grp, idx in [("rare10", rare), ("common31", comm)]:
    out["rare_tail"][grp] = {name: prf(Yt[:, idx], Pr[:, idx]) for name, Pr in SETUPS.items()}
out["rare_tail"]["per_category_rare"] = [
    {"category": cats[j], "test_pos": int(Yt[:, j].sum()),
     **{f"{n}_F1": round(prf(Yt[:, [j]], SETUPS[n][:, [j]])["f1"], 3) for n in ("RECALL", "BALANCED")},
     **{f"{n}_tp": prf(Yt[:, [j]], SETUPS[n][:, [j]])["tp"] for n in ("RECALL", "BALANCED")}} for j in rare]

# ---------------------------------------------------------------- 3. calibration (fit on validation only)
def calib(s, y, nbins=10):
    s, y = s.ravel(), y.ravel(); edges = np.linspace(0, 1, nbins + 1); ece = 0.0; rows = []
    for i in range(nbins):
        m = (s >= edges[i]) & ((s < edges[i + 1]) if i < nbins - 1 else (s <= 1.0))
        if m.sum():
            ece += m.sum() / len(s) * abs(y[m].mean() - s[m].mean())
            rows.append(dict(bin=[float(edges[i]), float(edges[i + 1])], n=int(m.sum()), conf=float(s[m].mean()), acc=float(y[m].mean())))
    return dict(ece=float(ece), brier=float(((s - y) ** 2).mean()), bins=rows)
iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(Bv.ravel(), Yv.ravel())
platt = LogisticRegression().fit(np.log(np.clip(Bv.ravel(), 1e-6, 1 - 1e-6) / np.clip(1 - Bv.ravel(), 1e-6, 1)).reshape(-1, 1), Yv.ravel())
logit = lambda s: np.log(np.clip(s, 1e-6, 1 - 1e-6) / np.clip(1 - s, 1e-6, 1)).reshape(-1, 1)
out["calibration"] = {
    "transformer_raw": calib(Bt, Yt),
    "transformer_isotonic": calib(iso.predict(Bt.ravel()).reshape(Bt.shape), Yt),
    "transformer_platt": calib(platt.predict_proba(logit(Bt.ravel()))[:, 1].reshape(Bt.shape), Yt),
    "fitted_on": "81 validation contracts (3,321 decisions)",
}
for k in ("transformer_raw", "transformer_isotonic", "transformer_platt"):
    print(f"{k:22s} test ECE {out['calibration'][k]['ece']:.3f}  Brier {out['calibration'][k]['brier']:.3f}")
json.dump({"method": "isotonic regression on the max-pooled transformer score, fitted on the 81 validation contracts",
           "x": [float(x) for x in iso.X_thresholds_], "y": [float(y) for y in iso.y_thresholds_],
           "test_ece_raw": round(out["calibration"]["transformer_raw"]["ece"], 4),
           "test_ece_calibrated": round(out["calibration"]["transformer_isotonic"]["ece"], 4)},
          open(os.path.join(RUN, "calibration.json"), "w"), indent=1)

# ---------------------------------------------------------------- 4. error examples (app default)
wins = {t: common.make_windows(D["ctx"][t]) for t in test_t}
Pr, S = SETUPS["RECALL"], SCORE["RECALL"]
fps = sorted([(i, j) for i in range(len(test_t)) for j in range(C) if Yt[i, j] == 0 and Pr[i, j] == 1], key=lambda k: -S[k])
fns = sorted([(i, j) for i in range(len(test_t)) for j in range(C) if Yt[i, j] == 1 and Pr[i, j] == 0], key=lambda k: S[k])
out["errors"] = {"n_false_positives": len(fps), "n_false_negatives": len(fns),
                 "false_positives": [dict(contract=test_t[i], category=cats[j], score=float(S[i, j]),
                                          threshold=float(th("f2", "t_transformer")[j]),
                                          window_text=" ".join(wins[test_t[i]][Wt[i, j]].split())[:600]) for i, j in fps[:6]],
                 "false_negatives": [dict(contract=test_t[i], category=cats[j], score=float(S[i, j]),
                                          threshold=float(th("f2", "t_transformer")[j]),
                                          gold_clause=" ".join(D["gold_spans"][(test_t[i], cats[j])][0].split())[:600]) for i, j in fns[:6]]}

# ---------------------------------------------------------------- 5. end to end (span model in the chosen window)
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
qtok = AutoTokenizer.from_pretrained(os.path.join(common.NB, "outputs", "span", "final"))
qa = AutoModelForQuestionAnswering.from_pretrained(os.path.join(common.NB, "outputs", "span", "final")).to(DEV).eval()
norm = lambda s: " ".join("".join(c for c in re.sub(r"\b(a|an|the)\b", " ", s.lower()) if c not in string.punctuation).split())
def token_f1(p, g):
    p, g = norm(p).split(), norm(g).split()
    if not p or not g: return float(p == g)
    n = sum((Counter(p) & Counter(g)).values())
    return 0.0 if n == 0 else 2 * (n / len(p)) * (n / len(g)) / (n / len(p) + n / len(g))
def extract(q, w):                                              # identical to analysis_oct2026/e2e.py
    chunks = [w[s:s + 1200] for s in range(0, max(1, len(w) - 1200 + 1), 800)] or [w]
    e = qtok([q] * len(chunks), chunks, truncation="only_second", max_length=320, padding=True, return_tensors="pt").to(DEV)
    with torch.no_grad(): o = qa(**e)
    sl, el, ids = o.start_logits.cpu().numpy(), o.end_logits.cpu().numpy(), e["input_ids"].cpu().numpy()
    bc, bt = -1e18, ""
    for k in range(len(chunks)):
        row = list(ids[k]); L = int(e["attention_mask"][k].sum()); sep = row.index(qtok.sep_token_id)
        m = np.full(len(row), -1e9); m[sep + 1:L] = 0.0
        s_, e_ = int(np.argmax(sl[k] + m)), int(np.argmax(el[k] + m)); e_ = max(e_, s_); e_ = min(e_, s_ + 60)
        if float(sl[k][s_] + el[k][e_]) > bc: bc, bt = float(sl[k][s_] + el[k][e_]), qtok.decode(row[s_:e_ + 1], skip_special_tokens=True)
    return bt
gold = [(i, j) for i in range(len(test_t)) for j in range(C) if Yt[i, j] == 1]
rng = np.random.default_rng(42); boot_idx = [rng.integers(0, len(test_t), len(test_t)) for _ in range(10000)]
out["end_to_end"] = {}
for name in ("RECALL", "BALANCED"):
    rows = []
    for n, (i, j) in enumerate([g for g in gold if SETUPS[name][g] == 1]):
        t, c = test_t[i], cats[j]; w = wins[t][Wt[i, j]]; gs = D["gold_spans"][(t, c)]
        f1 = max(token_f1(extract(D["cat_question"][c], w), g) for g in gs)
        rows.append(dict(i=i, title=t, category=c, window=int(Wt[i, j]), token_f1=f1, span_ok=int(f1 >= 0.5),
                         window_has_gold=int(any(s[:80] in w for s in gs))))
        if n % 300 == 0: print(f"  {name} span {n}", flush=True)
    for x in rows:                                   # where the clause starts inside the chosen window
        w = wins[x["title"]][x["window"]]; gs = D["gold_spans"][(x["title"], x["category"])]
        pos = [w.find(g[:80]) for g in gs if g[:80] in w]; x["clause_start"] = min(pos) if pos else -1
    pd.DataFrame(rows).to_json(os.path.join(RUN, f"e2e_rows_{name}.json"), orient="records")
    r = pd.DataFrame(rows); n_gold = len(gold); n_pres = len(r); n_ok = int(r.span_ok.sum())
    per = np.zeros(len(test_t)); tot = np.zeros(len(test_t))
    for (i, j) in gold: tot[i] += 1
    for x in rows: per[x["i"]] += x["span_ok"]
    bs = [per[b].sum() / tot[b].sum() for b in boot_idx]
    wc = r[r.window_has_gold == 1]
    out["end_to_end"][name] = dict(
        n_gold=n_gold, n_presence=n_pres, n_span_ok=n_ok, presence_recall=n_pres / n_gold,
        span_given_presence=n_ok / n_pres, end_to_end=n_ok / n_gold,
        end_to_end_ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
        mean_token_f1_given_presence=float(r.token_f1.mean()),
        window_correct=int(len(wc)), window_correct_rate=len(wc) / n_pres,
        span_ok_given_window_correct=float(wc.span_ok.mean()), window_wrong=int(n_pres - len(wc)),
        span_ok_given_window_wrong=int(r[r.window_has_gold == 0].span_ok.sum()))
    print(name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out["end_to_end"][name].items()}, flush=True)

json.dump(out, open(os.path.join(RUN, "downstream.json"), "w"), indent=1)
print("wrote", os.path.join(RUN, "downstream.json"))
