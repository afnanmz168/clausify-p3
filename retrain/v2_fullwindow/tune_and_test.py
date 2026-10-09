"""Tune the TF-IDF baseline and the transformer on the SAME validation split, then test once.

Reads runs/<tag>/window_scores.npz and runs/<tag>/val_split.json from presence_v2.py.
Takes seconds; no GPU.

Rules fixed before looking at any test number (the same for both models):
  * Baseline: TF-IDF (same settings as the original) + per-category logistic regression,
    fitted on the 327 fit contracts. C chosen from {0.1, 0.3, 1, 3, 10} on validation.
  * Transformer: pooling chosen from {max, top-2 mean, top-3 mean, top-5 mean} on validation.
  * Thresholds: one global threshold per model chosen on validation (grid 0.02-0.98), then a
    per-category threshold for every category with at least 8 validation positives
    (the others keep the global one, so rare categories are not fitted to noise).
  * Ensemble: rule chosen on validation from AND, OR, AVG and MIXED
    (MIXED = OR for the 8 High-risk categories, AND for the rest), using each model's own thresholds.
  * Two objectives, both reported: micro-F1 (comparable with every earlier number) and
    micro-F2 (counts a missed clause as worse than a false alarm; the setting for the app).
  * The test set is scored once per configuration. 10,000-resample cluster bootstrap over contracts.

Usage (from anywhere):  python3 tune_and_test.py --tag L512
"""
import os, sys, json, argparse
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

HERE = os.path.dirname(os.path.abspath(__file__))
P = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(P, "analysis"))
import common

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="L512")
args = ap.parse_args()
RUN = os.path.join(HERE, "runs", args.tag)

D = common.load_all()
sp = json.load(open(os.path.join(RUN, "val_split.json")))
fit_t, val_t, test_t = sp["fit"], sp["val"], sp["test"]
cats = D["cats"]; C = len(cats)
Yv = common.labels(D, val_t, cats); Yt = common.labels(D, test_t, cats)
risk = json.load(open(os.path.join(P, "analysis", "risk_levels.json")))
band = np.array([risk[c] for c in cats])
HIGH = band == "High"
SMOKE = len(test_t) < 50

# ---------------------------------------------------------------- metrics
def counts(Pr, Y):
    return ((Pr == 1) & (Y == 1)).sum(), ((Pr == 1) & (Y == 0)).sum(), ((Pr == 0) & (Y == 1)).sum()

def fbeta(Pr, Y, beta=1.0):
    tp, fp, fn = counts(Pr, Y); b2 = beta * beta
    return 0.0 if tp == 0 else float((1 + b2) * tp / ((1 + b2) * tp + b2 * fn + fp))

def report(Pr, Y):
    tp, fp, fn = counts(Pr, Y)
    common_c = Y.sum(0) >= 10
    def macro(mask):
        return float(np.mean([fbeta(Pr[:, j], Y[:, j]) for j in np.where(mask)[0]])) if mask.any() else None
    hi_pos = int(Y[:, HIGH].sum()); hi_tp = int(((Pr == 1) & (Y == 1))[:, HIGH].sum())
    return {"micro_f1": round(fbeta(Pr, Y), 4), "micro_f2": round(fbeta(Pr, Y, 2), 4),
            "precision": round(tp / max(1, tp + fp), 4), "recall": round(tp / max(1, tp + fn), 4),
            "macro_f1_all41": round(macro(np.ones(C, bool)), 4),
            "macro_f1_ge10pos": round(macro(common_c), 4) if macro(common_c) is not None else None,
            "n_cats_ge10pos": int(common_c.sum()),
            "high_risk_recall": round(hi_tp / max(1, hi_pos), 4), "high_risk_missed": hi_pos - hi_tp,
            "high_risk_total": hi_pos, "tp": int(tp), "fp": int(fp), "fn": int(fn)}

rng = np.random.default_rng(42); NB = 10000
BOOT = [rng.integers(0, len(test_t), len(test_t)) for _ in range(NB)]
def per_contract(Pr, Y):
    return (((Pr == 1) & (Y == 1)).sum(1), ((Pr == 1) & (Y == 0)).sum(1), ((Pr == 0) & (Y == 1)).sum(1))
def boot_f1(Pr, Y, beta=1.0):
    tp, fp, fn = per_contract(Pr, Y); b2 = beta * beta
    return np.array([(1 + b2) * tp[i].sum() / max(1, (1 + b2) * tp[i].sum() + b2 * fn[i].sum() + fp[i].sum())
                     for i in BOOT])
def boot_high_recall(Pr, Y):
    tp = ((Pr == 1) & (Y == 1))[:, HIGH].sum(1); pos = Y[:, HIGH].sum(1)
    return np.array([tp[i].sum() / max(1, pos[i].sum()) for i in BOOT])
def ci(x):
    return [round(float(np.percentile(x, 2.5)), 6), round(float(np.percentile(x, 97.5)), 6)]

# ---------------------------------------------------------------- threshold tuning (validation only)
GRID = np.round(np.arange(0.02, 0.99, 0.02), 2)
def tune(Sv, beta):
    g = max(GRID, key=lambda t: (fbeta((Sv >= t).astype(int), Yv, beta), -abs(t - 0.5)))
    th = np.full(C, g)
    for j in range(C):
        if Yv[:, j].sum() >= 8:
            th[j] = max(GRID, key=lambda t: (fbeta((Sv[:, j] >= t).astype(int), Yv[:, j], beta), -abs(t - g)))
    return float(g), th

# ---------------------------------------------------------------- TF-IDF baseline, refitted on the fit contracts
vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), sublinear_tf=True, min_df=2)
Xf = vec.fit_transform([D["ctx"][t] for t in fit_t])
Xv = vec.transform([D["ctx"][t] for t in val_t]); Xt = vec.transform([D["ctx"][t] for t in test_t])
Yf = common.labels(D, fit_t, cats)
def tfidf(Cr):
    Av, At = np.zeros((len(val_t), C)), np.zeros((len(test_t), C))
    for j in range(C):
        if np.unique(Yf[:, j]).size < 2:
            Av[:, j] = At[:, j] = float(Yf[0, j]); continue
        m = LogisticRegression(max_iter=2000, class_weight="balanced", C=Cr).fit(Xf, Yf[:, j])
        Av[:, j] = m.predict_proba(Xv)[:, 1]; At[:, j] = m.predict_proba(Xt)[:, 1]
    return Av, At
TF = {Cr: tfidf(Cr) for Cr in [0.1, 0.3, 1.0, 3.0, 10.0]}

# ---------------------------------------------------------------- transformer: pool window scores
z = np.load(os.path.join(RUN, "window_scores.npz"))
titles = list(z["titles"]); assert list(z["cats"]) == cats
df = pd.DataFrame({"t": z["title_idx"], "c": z["cat_idx"], "p": z["p"]})
df = df.sort_values(["t", "c", "p"], ascending=[True, True, False])
df["r"] = df.groupby(["t", "c"]).cumcount()
def pooled(k):
    s = df[df.r < k].groupby(["t", "c"]).p.mean() if k else df.groupby(["t", "c"]).p.max()
    M = np.zeros((len(titles), C)); M[s.index.get_level_values(0), s.index.get_level_values(1)] = s.values
    vi = [titles.index(t) for t in val_t]; ti = [titles.index(t) for t in test_t]
    return M[vi], M[ti]
POOL = {"max": pooled(0), "top2": pooled(2), "top3": pooled(3), "top5": pooled(5)}

# ---------------------------------------------------------------- select on validation, test once
out = {"tag": args.tag, "n_fit": len(fit_t), "n_val": len(val_t), "n_test": len(test_t),
       "rules": "see module docstring", "configs": {}}
pred = {}
def add(name, Pr, chosen):
    pred[name] = Pr
    out["configs"][name] = {"chosen_on_validation": chosen, "test": report(Pr, Yt)}

# untuned rows (threshold 0.5, max pooling, C=1): the direct before/after of reading the whole window
add("tfidf_untuned", (TF[1.0][1] >= .5).astype(int), {"C": 1.0, "threshold": 0.5})
add("transformer_untuned", (POOL["max"][1] >= .5).astype(int), {"pool": "max", "threshold": 0.5})

DEPLOY = {}
for obj, beta in [("f1", 1.0), ("f2", 2.0)]:
    # baseline: choose C, then thresholds
    bestC = max(TF, key=lambda Cr: fbeta((TF[Cr][0] >= tune(TF[Cr][0], beta)[1]).astype(int), Yv, beta))
    Av, At = TF[bestC]; gA, thA = tune(Av, beta)
    add(f"tfidf_tuned_{obj}", (At >= thA).astype(int), {"C": bestC, "global_threshold": gA,
        "per_category_thresholds": dict(zip(cats, thA.round(2).tolist()))})
    # transformer: choose pooling, then thresholds
    bestP = max(POOL, key=lambda k: fbeta((POOL[k][0] >= tune(POOL[k][0], beta)[1]).astype(int), Yv, beta))
    Bv, Bt = POOL[bestP]; gB, thB = tune(Bv, beta)
    add(f"transformer_tuned_{obj}", (Bt >= thB).astype(int), {"pool": bestP, "global_threshold": gB,
        "per_category_thresholds": dict(zip(cats, thB.round(2).tolist()))})
    # ensemble rule
    av, bv, at, bt = Av >= thA, Bv >= thB, At >= thA, Bt >= thB
    gAvg, thAvg = tune((Av + Bv) / 2, beta)
    rules = {"AND": (av & bv, at & bt), "OR": (av | bv, at | bt),
             "MIXED": (np.where(HIGH, av | bv, av & bv), np.where(HIGH, at | bt, at & bt)),
             "AVG": ((Av + Bv) / 2 >= thAvg, (At + Bt) / 2 >= thAvg)}
    val_scores = {r: round(fbeta(v.astype(int), Yv, beta), 4) for r, (v, _) in rules.items()}
    bestR = max(val_scores, key=val_scores.get)
    add(f"ensemble_tuned_{obj}", rules[bestR][1].astype(int), {"rule": bestR, "validation_scores": val_scores,
        "tfidf_C": bestC, "transformer_pool": bestP})
    # everything the app needs to reproduce this exact decision rule
    DEPLOY[obj] = {"rule": bestR, "tfidf_C": bestC, "transformer_pool": bestP,
                   "t_tfidf": dict(zip(cats, thA.round(2).tolist())),
                   "t_transformer": dict(zip(cats, thB.round(2).tolist())),
                   "t_avg": dict(zip(cats, thAvg.round(2).tolist()))}

# bootstrap: every config's micro-F1, and paired differences vs the tuned baseline
for name, Pr in pred.items():
    out["configs"][name]["test"]["micro_f1_ci95"] = ci(boot_f1(Pr, Yt))
    out["configs"][name]["test"]["high_risk_recall_ci95"] = ci(boot_high_recall(Pr, Yt))
# each objective is compared on its own metric: micro-F1 for the f1 rows, micro-F2 for the f2 rows
for obj, beta in [("f1", 1.0), ("f2", 2.0)]:
    base = boot_f1(pred[f"tfidf_tuned_{obj}"], Yt, beta)
    for m in [f"transformer_tuned_{obj}", f"ensemble_tuned_{obj}"]:
        d = boot_f1(pred[m], Yt, beta) - base
        out["configs"][m]["test"]["vs_tuned_tfidf"] = {
            "metric": f"micro-{obj.upper()}",
            "diff": round(fbeta(pred[m], Yt, beta) - fbeta(pred[f"tfidf_tuned_{obj}"], Yt, beta), 4),
            "ci95": ci(d), "p_gt0": round(float((d > 0).mean()), 3)}
d = boot_f1(pred["transformer_untuned"], Yt) - boot_f1(pred["tfidf_untuned"], Yt)
out["configs"]["transformer_untuned"]["test"]["vs_tfidf_untuned"] = {
    "diff": round(fbeta(pred["transformer_untuned"], Yt) - fbeta(pred["tfidf_untuned"], Yt), 4),
    "ci95": ci(d), "p_gt0": round(float((d > 0).mean()), 3)}

# reference rows: the original (August/October) models on the same test contracts
if not SMOKE:
    o = np.load(os.path.join(P, "retrain", "stage2a_probs_seed42.npz"))
    a0, b0 = o["a"].reshape(102, C), o["b"].reshape(102, C)
    assert (o["y"].reshape(102, C) == Yt).all()
    out["reference_original"] = {"tfidf_408_C1_t0.5": report((a0 >= .5).astype(int), Yt),
                                 "transformer256_408_max_t0.5": report((b0 >= .5).astype(int), Yt),
                                 "and_ensemble_original": report((np.minimum(a0, b0) >= .5).astype(int), Yt)}

json.dump(out, open(os.path.join(RUN, "results.json"), "w"), indent=1)

# deployable decision rules (f1 = best overall F1, f2 = recall-first) and the fitted TF-IDF model,
# in the same pickle format as notebooks/artifacts/baseline.pkl
json.dump({"tag": args.tag, "max_len": json.load(open(os.path.join(RUN, "train_info.json")))["max_len"],
           "objectives": DEPLOY}, open(os.path.join(RUN, "decision.json"), "w"), indent=1)
import pickle
for Cr in sorted({d["tfidf_C"] for d in DEPLOY.values()}):
    clf = {}
    for j, cat in enumerate(cats):
        if np.unique(Yf[:, j]).size < 2:
            clf[cat] = ("const", int(Yf[0, j]))
        else:
            clf[cat] = ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", C=Cr).fit(Xf, Yf[:, j]))
    pickle.dump({"vec": vec, "classifiers": clf, "cats": cats, "cat_question": D["cat_question"], "C": Cr},
                open(os.path.join(RUN, f"baseline_C{Cr:g}.pkl"), "wb"))

# readable table
rows = []
for name, r in list(out.get("reference_original", {}).items()):
    rows.append(("original: " + name, r))
for name, c in out["configs"].items():
    rows.append((name, c["test"]))
lines = ["| configuration | micro-F1 [95% CI] | P | R | F2 | macro-F1 (>=10 pos) | High-risk recall (missed) |",
         "|---|---|---|---|---|---|---|"]
for name, r in rows:
    cis = f" [{r['micro_f1_ci95'][0]:.3f}, {r['micro_f1_ci95'][1]:.3f}]" if "micro_f1_ci95" in r else ""
    lines.append(f"| {name} | {r['micro_f1']:.3f}{cis} | {r['precision']:.3f} | {r['recall']:.3f} | "
                 f"{r['micro_f2']:.3f} | {r['macro_f1_ge10pos']} | {r['high_risk_recall']:.3f} ({r['high_risk_missed']}) |")
lines.append("")
for name, c in out["configs"].items():
    ch = {k: v for k, v in c["chosen_on_validation"].items() if k != "per_category_thresholds"}
    extra = c["test"].get("vs_tuned_tfidf") or c["test"].get("vs_tfidf_untuned")
    lines.append(f"- **{name}**: chosen on validation {ch}" + (f"; vs baseline {extra}" if extra else ""))
open(os.path.join(RUN, "results.md"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
print("\nwrote", os.path.join(RUN, "results.json"))
