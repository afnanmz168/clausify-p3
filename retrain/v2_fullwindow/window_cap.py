"""What does the app's 30-window limit cost, and can TF-IDF choose which later windows to read?

The app scores at most 30 windows (about 45,500 characters) to stay fast, but every number in the
report scores whole contracts. Because runs/L512/window_scores.npz holds the transformer's score for
every window, any reading rule can be replayed here without running the model again.

Rules compared (recall-first transformer, the app's default, with its per-category thresholds):
  whole      every window (what the report measures)
  first30    the first 30 windows (what the app read until now)
  tfidf_k    the first 30 windows, plus, for each clause type, the k later windows that the re-run
             TF-IDF model (runs/L512/baseline_C10.pkl, applied to single windows) rates most likely
             to contain that type. Each extra window is one transformer pass per clause type, so k
             extra windows cost about as much as k more windows for every type.
  cosine_k   the same, ranking the later windows by TF-IDF cosine similarity to the CUAD question.

k is chosen on the 81 validation contracts: the smallest k whose micro-F2 is within 0.005 of reading
every window. The 102 test contracts are scored once. Writes runs/L512/window_cap.json.

    cd ~/Desktop/"final project p3" && python3 retrain/v2_fullwindow/window_cap.py
"""
import os, sys, json, pickle
import numpy as np, pandas as pd
from sklearn.metrics.pairwise import linear_kernel
HERE = os.path.dirname(os.path.abspath(__file__)); P = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(P, "analysis")); import common
R = os.path.join(HERE, "runs", "L512")
CAP, KS = 30, [0, 2, 3, 5, 8, 10, 15, 20]

D = common.load_all(); cats = D["cats"]; C = len(cats)
sp = json.load(open(os.path.join(R, "val_split.json"))); val_t, test_t = sp["val"], sp["test"]
thr = np.array([json.load(open(os.path.join(R, "decision.json")))["objectives"]["f2"]["t_transformer"][c] for c in cats])
risk = json.load(open(os.path.join(P, "analysis", "risk_levels.json"))); high = np.array([risk[c] == "High" for c in cats])
bl = pickle.load(open(os.path.join(R, "baseline_C10.pkl"), "rb"))
z = np.load(os.path.join(R, "window_scores.npz")); titles = list(z["titles"])
df = pd.DataFrame({"t": z["title_idx"], "c": z["cat_idx"], "w": z["win_idx"], "p": z["p"]})
S = {t: g.pivot(index="w", columns="c", values="p").reindex(columns=range(C)).values for t, g in df.groupby("t")}

q_vec = {}
def rankers(t):
    """TF-IDF scores of the windows after the first 30: (n_extra, C) for each ranker."""
    wins = common.make_windows(D["ctx"][t])[CAP:]
    X = bl["vec"].transform(wins)
    lr = np.column_stack([np.zeros(len(wins)) if k == "const" else o.predict_proba(X)[:, 1]
                          for k, o in (bl["classifiers"][c] for c in cats)])
    if not q_vec:
        q_vec["Q"] = bl["vec"].transform([D["cat_question"][c] for c in cats])
    cos = linear_kernel(X, q_vec["Q"])
    return {"tfidf": lr, "cosine": cos}


def contract_scores(t, rule, k):
    s = S[titles.index(t)]
    if rule == "whole" or len(s) <= CAP:
        return s.max(0)
    first = s[:CAP].max(0)
    if k == 0:
        return first
    rk = RK[t][rule]; out = first.copy()
    for j in range(C):
        pick = np.argsort(-rk[:, j], kind="stable")[:k] + CAP
        out[j] = max(first[j], s[pick, j].max())
    return out


def evaluate(ts, rule, k):
    Y = common.labels(D, ts, cats)
    Pm = np.array([(contract_scores(t, rule, k) >= thr).astype(int) for t in ts])
    tp = int((Pm & Y).sum()); fp = int((Pm & (1 - Y)).sum()); fn = int(((1 - Pm) & Y).sum())
    longs = np.array([len(S[titles.index(t)]) > CAP for t in ts])
    hy, hp = Y[:, high], Pm[:, high]
    extra = sum(min(k, max(0, len(S[titles.index(t)]) - CAP)) for t in ts) if rule not in ("whole",) else None
    return dict(micro_f1=2 * tp / (2 * tp + fp + fn), micro_f2=5 * tp / (5 * tp + 4 * fn + fp),
                high_found=float((hp & hy).sum() / hy.sum()), high_missed=int(hy.sum() - (hp & hy).sum()),
                high_found_long=float((hp[longs] & hy[longs]).sum() / hy[longs].sum()), n_long=int(longs.sum()),
                extra_windows_per_type=extra)


RK = {t: rankers(t) for t in val_t + test_t if len(S[titles.index(t)]) > CAP}
out = {"cap": CAP, "n_val": len(val_t), "n_test": len(test_t)}
val = {"whole": evaluate(val_t, "whole", 0)}
for rule in ("tfidf", "cosine"):
    for k in KS:
        val[f"{rule}_{k}"] = evaluate(val_t, rule, k)
target = val["whole"]["micro_f2"] - 0.005
ok = [(k, rule) for rule in ("tfidf", "cosine") for k in KS if val[f"{rule}_{k}"]["micro_f2"] >= target]
k_best, rule_best = min(ok) if ok else (max(KS), "tfidf")
out["validation"] = val
out["chosen"] = dict(rule=rule_best, k=k_best, criterion="smallest k with validation micro-F2 within 0.005 of reading every window")
out["test"] = {"whole": evaluate(test_t, "whole", 0), "first30": evaluate(test_t, "tfidf", 0),
               "chosen": evaluate(test_t, rule_best, k_best)}
lens = np.array([len(S[titles.index(t)]) for t in test_t])
out["test"]["windows_read"] = dict(chosen_max=int(min(lens.max(), CAP + k_best)), whole_max=int(lens.max()))
json.dump(out, open(os.path.join(R, "window_cap.json"), "w"), indent=1)
for k, v in val.items():
    print(f"val {k:10s} F1 {v['micro_f1']:.3f} F2 {v['micro_f2']:.3f} High {v['high_found']:.1%} long {v['high_found_long']:.1%}")
print("chosen:", out["chosen"])
for k, v in out["test"].items():
    if k != "windows_read":
        print(f"test {k:8s} F1 {v['micro_f1']:.3f} F2 {v['micro_f2']:.3f} High {v['high_found']:.1%} (missed {v['high_missed']}) long {v['high_found_long']:.1%}")
