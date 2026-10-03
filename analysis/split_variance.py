"""Dataset-split variance, which is NOT the same as model-seed variance.

The thesis reports one contract-level 80/20 split (seed 42). With 510 contracts
a different draw could move the numbers. Retraining the transformer per split is
infeasible, but the TF-IDF baseline refits in seconds -- and since the baseline
is the thesis' reference point, its split-to-split spread bounds how much the
comparison itself could move.
"""
import os, sys, json, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import common

D = common.load_all()
titles = D["titles"]; cats = D["cats"]; ctx = D["ctx"]; gold = D["gold_spans"]
Yall = np.array([[int(bool(gold.get((t,c),[]))) for c in cats] for t in titles])

def run_split(seed):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(titles)); n_test = int(0.20*len(titles))
    test = set(titles[i] for i in perm[:n_test])
    tr = [i for i,t in enumerate(titles) if t not in test]
    te = [i for i,t in enumerate(titles) if t in test]
    vec = TfidfVectorizer(max_features=20000, ngram_range=(1,2), sublinear_tf=True, min_df=2)
    Xtr = vec.fit_transform([ctx[titles[i]] for i in tr])
    Xte = vec.transform([ctx[titles[i]] for i in te])
    Ytr, Yte = Yall[tr], Yall[te]
    P = np.zeros_like(Yte)
    for j in range(len(cats)):
        y = Ytr[:,j]
        if np.unique(y).size < 2: P[:,j] = int(y[0])
        else: P[:,j] = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xtr,y).predict(Xte)
    tp=int(((Yte==1)&(P==1)).sum()); fp=int(((Yte==0)&(P==1)).sum()); fn=int(((Yte==1)&(P==0)).sum())
    micro = 2*tp/(2*tp+fp+fn) if tp else 0.0
    # macro over categories with >=10 test positives
    f1s=[]
    for j in range(len(cats)):
        if Yte[:,j].sum() < 10: continue
        a=int(((Yte[:,j]==1)&(P[:,j]==1)).sum()); b=int(((Yte[:,j]==0)&(P[:,j]==1)).sum()); c_=int(((Yte[:,j]==1)&(P[:,j]==0)).sum())
        pr=a/(a+b) if a+b else 0; rc=a/(a+c_) if a+c_ else 0
        f1s.append(2*pr*rc/(pr+rc) if pr+rc else 0)
    return micro, float(np.mean(f1s)), len(f1s)

N = 20
rows = []
for s in range(N):
    m, mac, ncat = run_split(s)
    rows.append(dict(seed=s, micro_f1=m, macro_f1_common=mac, n_common_cats=ncat))
    print(f"  split seed {s:2d}: micro-F1 {m:.4f}  macro-F1(>=10 pos, {ncat} cats) {mac:.4f}", flush=True)

mi = np.array([r["micro_f1"] for r in rows])
res = dict(n_splits=N, micro_mean=float(mi.mean()), micro_sd=float(mi.std(ddof=1)),
           micro_min=float(mi.min()), micro_max=float(mi.max()), micro_range=float(mi.max()-mi.min()),
           thesis_split_seed42=0.7747, rows=rows)
print(f"\nTF-IDF baseline across {N} contract-level splits:")
print(f"  mean {mi.mean():.4f}  SD {mi.std(ddof=1):.4f}  min {mi.min():.4f}  max {mi.max():.4f}  range {mi.max()-mi.min():.4f}")
print(f"  the thesis' split (seed 42) gives 0.7747")
json.dump(res, open(f"{HERE}/split_variance.json","w"), indent=2)
