"""Score the 102 test contracts through the app's own presence code (model_utils.predict_presence
and _decide_v2, default recall-first setting), with the windows the app really reads: the first 30,
plus 5 later windows per clause type picked by TF-IDF. This is the app's figure, as opposed to the
whole-contract figures of the report. Runs on MPS if available. Writes runs/L512/app_presence_test.json.

    cd ~/Desktop/"final project p3" && python3 retrain/v2_fullwindow/app_presence_test.py
"""
import os, sys, json, time
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); P = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(P, "analysis")); import common
sys.path.insert(0, os.path.join(os.path.dirname(P), "final project app p3"))
import model_utils as mu
mu.DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

D = common.load_all(); cats = D["cats"]
test_t = json.load(open(os.path.join(HERE, "runs", "L512", "val_split.json")))["test"]
risk = json.load(open(os.path.join(P, "analysis", "risk_levels.json")))
assert mu.v2_available() and mu.default_rule() == "RECALL"
Y = common.labels(D, test_t, cats)
Pm = np.zeros_like(Y); rows = []; t0 = time.time()
for i, t in enumerate(test_t):
    text = D["ctx"][t]
    sc, _ = mu.predict_presence(text, version="v2")
    for j, c in enumerate(cats):
        Pm[i, j] = int(mu._decide_v2("RECALL", c, sc[c], None)[0])
    rows.append(dict(title=t, windows=len(mu.all_windows(text)), scores={c: round(sc[c], 6) for c in cats}))
    if i % 10 == 0:
        print(f"  {i}/{len(test_t)}  {time.time() - t0:.0f}s", flush=True)
high = np.array([risk[c] == "High" for c in cats]); longs = np.array([r["windows"] > mu.MAX_WINDOWS for r in rows])
tp = int((Pm & Y).sum()); fp = int((Pm & (1 - Y)).sum()); fn = int(((1 - Pm) & Y).sum())
hy, hp = Y[:, high], Pm[:, high]
out = dict(max_windows=mu.MAX_WINDOWS, extra_windows=mu.EXTRA_WINDOWS, n_test=len(test_t), n_long=int(longs.sum()),
           micro_f1=2 * tp / (2 * tp + fp + fn), micro_f2=5 * tp / (5 * tp + 4 * fn + fp), tp=tp, fp=fp, fn=fn,
           high_found=float((hp & hy).sum() / hy.sum()), high_missed=int(hy.sum() - (hp & hy).sum()),
           high_found_long=float((hp[longs] & hy[longs]).sum() / hy[longs].sum()),
           minutes=round((time.time() - t0) / 60, 1), device=mu.DEVICE, rows=rows)
json.dump(out, open(os.path.join(HERE, "runs", "L512", "app_presence_test.json"), "w"), indent=1)
print({k: v for k, v in out.items() if k != "rows"})
