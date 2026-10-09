"""Paired comparison of the 512-token and 256-token transformers trained under the same protocol
(same 327 contracts, same validation split, same tuning rules), on the same 102 test contracts.
Rebuilds each model's test decisions from window_scores.npz and decision.json, then runs the same
10,000-resample cluster bootstrap as tune_and_test.py. Writes compare_window.json."""
import os, sys, json
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); P = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(P, "analysis")); import common
D = common.load_all(); cats = D["cats"]; C = len(cats)
risk = json.load(open(os.path.join(P, "analysis", "risk_levels.json"))); HIGH = np.array([risk[c] == "High" for c in cats])

def scores(tag):
    R = os.path.join(HERE, "runs", tag); sp = json.load(open(os.path.join(R, "val_split.json")))
    z = np.load(os.path.join(R, "window_scores.npz")); titles = list(z["titles"])
    df = pd.DataFrame({"t": z["title_idx"], "c": z["cat_idx"], "p": z["p"]}).sort_values(["t", "c", "p"], ascending=[True, True, False])
    df["r"] = df.groupby(["t", "c"]).cumcount()
    def pool(k):
        s = df[df.r < k].groupby(["t", "c"]).p.mean() if k else df.groupby(["t", "c"]).p.max()
        M = np.zeros((len(titles), C)); M[s.index.get_level_values(0), s.index.get_level_values(1)] = s.values
        return M[[titles.index(t) for t in sp["test"]]]
    dec = json.load(open(os.path.join(R, "decision.json")))["objectives"]
    res = json.load(open(os.path.join(R, "results.json")))["configs"]
    out = {"untuned": (pool(0) >= 0.5).astype(int)}
    for obj in ("f1", "f2"):
        k = {"max": 0, "top2": 2, "top3": 3, "top5": 5}[res[f"transformer_tuned_{obj}"]["chosen_on_validation"]["pool"]]
        out[obj] = (pool(k) >= np.array([dec[obj]["t_transformer"][c] for c in cats])).astype(int)
    return out, sp["test"]

S512, test = scores("L512"); S256, test2 = scores("L256"); assert test == test2
Y = common.labels(D, test, cats)
rng = np.random.default_rng(42); BOOT = [rng.integers(0, len(test), len(test)) for _ in range(10000)]
def per(P):
    return ((P == 1) & (Y == 1)).sum(1), ((P == 1) & (Y == 0)).sum(1), ((P == 0) & (Y == 1)).sum(1), ((P == 1) & (Y == 1))[:, HIGH].sum(1)
hpos = Y[:, HIGH].sum(1)
def metric(P, beta, idx=None):
    tp, fp, fn, htp = per(P); idx = slice(None) if idx is None else idx; b2 = beta * beta
    return (1 + b2) * tp[idx].sum() / ((1 + b2) * tp[idx].sum() + b2 * fn[idx].sum() + fp[idx].sum())
def high(P, idx=None):
    _, _, _, htp = per(P); idx = slice(None) if idx is None else idx
    return htp[idx].sum() / hpos[idx].sum()
out = {}
for name, beta in [("untuned", 1.0), ("f1", 1.0), ("f2", 2.0)]:
    a, b = S512[name], S256[name]
    d = np.array([metric(a, beta, i) - metric(b, beta, i) for i in BOOT])
    h = np.array([high(a, i) - high(b, i) for i in BOOT])
    out[name] = {"metric": "micro-F2" if beta == 2 else "micro-F1",
                 "L512": round(metric(a, beta), 4), "L256": round(metric(b, beta), 4),
                 "diff": round(metric(a, beta) - metric(b, beta), 4),
                 "ci95": [round(float(np.percentile(d, 2.5)), 6), round(float(np.percentile(d, 97.5)), 6)],
                 "p_gt0": round(float((d > 0).mean()), 3),
                 "high_recall_diff": round(high(a) - high(b), 4),
                 "high_recall_ci95": [round(float(np.percentile(h, 2.5)), 6), round(float(np.percentile(h, 97.5)), 6)]}
json.dump(out, open(os.path.join(HERE, "compare_window.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
