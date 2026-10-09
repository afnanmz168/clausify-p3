"""Where does the clause sit inside the window the presence model chose, and does that explain the
span model's lower success with the re-run model? Reads the end-to-end rows of the first setup
(analysis_oct2026/e2e_rows_AND.json) and of the app's setting (runs/L512/e2e_rows_RECALL.json).
Writes runs/L512/span_position.json.

    cd ~/Desktop/"final project p3" && python3 retrain/v2_fullwindow/span_position.py
"""
import os, sys, json
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); P = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(P, "analysis")); import common
D = common.load_all()
v1 = pd.DataFrame(json.load(open(os.path.join(P, "retrain", "analysis_oct2026", "e2e_rows_AND.json"))))
v2 = pd.read_json(os.path.join(HERE, "runs", "L512", "e2e_rows_RECALL.json"))
def clause_start(r):
    w = common.make_windows(D["ctx"][r["title"]])[r["window"]]; gs = D["gold_spans"][(r["title"], r["category"])]
    p = [w.find(g[:80]) for g in gs if g[:80] in w]; return min(p) if p else -1
v1["clause_start"] = v1.apply(clause_start, axis=1)
out = {}
for name, df in [("first_setup", v1), ("app", v2)]:
    c = df[df.clause_start >= 0]; a, b = c[c.clause_start < 1000], c[c.clause_start >= 1000]
    out[name] = dict(n_right_window=int(len(c)), share_clause_in_second_half=float(len(b) / len(c)),
                     span_ok_first_half=float(a.span_ok.mean()), span_ok_second_half=float(b.span_ok.mean()),
                     n_first_half=int(len(a)), n_second_half=int(len(b)))
json.dump(out, open(os.path.join(HERE, "runs", "L512", "span_position.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
