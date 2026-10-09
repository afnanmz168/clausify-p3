"""How many of the app's cards (default setting) show a calibrated chance under 50% or 20%, and how
many of those are real clauses? The recall-first thresholds are set per clause type while the
calibration map is shared, so some types are reported at a low chance. The app labels cards under
50% "Possible - check" (model_utils.POSSIBLE_BELOW). Writes runs/L512/cards.json.

    cd ~/Desktop/"final project p3" && python3 retrain/v2_fullwindow/card_chance.py
"""
import os, sys, json
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); P = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(P, "analysis")); import common
R = os.path.join(HERE, "runs", "L512")
D = common.load_all(); cats = D["cats"]
test_t = json.load(open(os.path.join(R, "val_split.json")))["test"]
thr = json.load(open(os.path.join(R, "decision.json")))["objectives"]["f2"]["t_transformer"]
cal = json.load(open(os.path.join(R, "calibration.json")))
risk = json.load(open(os.path.join(P, "analysis", "risk_levels.json")))
z = np.load(os.path.join(R, "window_scores.npz")); titles = list(z["titles"])
B = pd.DataFrame({"t": z["title_idx"], "c": z["cat_idx"], "p": z["p"]}).groupby(["t", "c"]).p.max().unstack().values
Y = common.labels(D, test_t, cats)

rows = []
for i, t in enumerate(test_t):
    for j, c in enumerate(cats):
        p = B[titles.index(t), j]
        if p >= thr[c]:                       # a card in the app's default setting
            chance = float(min(0.99, max(0.01, np.interp(p, cal["x"], cal["y"]))))
            rows.append(dict(category=c, chance=chance, real=int(Y[i, j]), risk=risk[c]))
r = pd.DataFrame(rows)
high_real = int(r[(r.risk == "High")].real.sum())
out = dict(n_cards=len(r), n_real=int(r.real.sum()), high_real_found=high_real, high_total=int(Y[:, [cats.index(c) for c in cats if risk[c] == "High"]].sum()))
for name, lim in (("below_50", 0.5), ("below_20", 0.2)):
    s = r[r.chance < lim]
    out[name] = dict(n=len(s), share=len(s) / len(r), real=int(s.real.sum()), real_share=float(s.real.mean()),
                     high_real=int(s[s.risk == "High"].real.sum()))
s = r[r.chance >= 0.5]
out["at_least_50"] = dict(n=len(s), real_share=float(s.real.mean()))
out["high_recall_without_possible"] = (high_real - out["below_50"]["high_real"]) / out["high_total"]
out["lowest_non_compete_chance"] = float(r[r.category == "Non-Compete"].chance.min())
json.dump(out, open(os.path.join(R, "cards.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
