"""Paired cluster bootstrap over the 102 test contracts for the presence models
(Table 5.14). Reads the per-(contract, category) probabilities saved by
stage2a_presence_test.py and writes retrain/stage2a_bootstrap.json."""
import os, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(HERE)
d = np.load(f"{R}/stage2a_probs_seed42.npz"); a, b, y = d["a"], d["b"], d["y"]
n, C = 102, 41
A = (a >= .5).astype(int).reshape(n, C)            # TF-IDF baseline
B = (b >= .5).astype(int).reshape(n, C)            # transformer (max-pool)
E = (np.minimum(a, b) >= .5).astype(int).reshape(n, C)  # AND-ensemble
Y = y.reshape(n, C)

def f1(P, Yy):
    tp = (P * Yy).sum(); fp = (P * (1 - Yy)).sum(); fn = ((1 - P) * Yy).sum()
    return 2 * tp / (2 * tp + fp + fn)

rng = np.random.default_rng(42); NB = 10000
idxs = [rng.integers(0, n, n) for _ in range(NB)]
res = {}
for k, (P1, P2) in {"AND_vs_TFIDF": (E, A), "TFIDF_vs_transformer": (A, B),
                    "AND_vs_transformer": (E, B)}.items():
    diffs = np.array([f1(P1[i], Y[i]) - f1(P2[i], Y[i]) for i in idxs])
    res[k] = {"diff": round(f1(P1, Y) - f1(P2, Y), 4),
              "ci95": [round(float(np.percentile(diffs, 2.5)), 4), round(float(np.percentile(diffs, 97.5)), 4)],
              "p_gt0": round(float((diffs > 0).mean()), 3)}
for nm, P in [("AND", E), ("TFIDF", A), ("transformer", B)]:
    s = np.array([f1(P[i], Y[i]) for i in idxs])
    res[nm] = {"f1": round(f1(P, Y), 4),
               "ci95": [round(float(np.percentile(s, 2.5)), 3), round(float(np.percentile(s, 97.5)), 3)]}
res["seed"] = 42; res["resamples"] = NB
print(json.dumps(res, indent=1)); json.dump(res, open(f"{R}/stage2a_bootstrap.json", "w"), indent=1)
