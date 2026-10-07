"""Reshape the per-(contract, category) probabilities written by
stage2a_presence_test.py into runs/base24k_seed42/probs.npz, the format read by
e2e.py, rare_tail.py, risk_threshold_calib.py and error_examples.py.
Run once after stage2a_presence_test.py."""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); P = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(P, "analysis")); import common
D = common.load_all()
n = np.load(os.path.join(P, "retrain", "stage2a_probs_seed42.npz"))
a = n["a"].reshape(102, 41); b = n["b"].reshape(102, 41); y = n["y"].reshape(102, 41)
out = os.path.join(HERE, "runs", "base24k_seed42"); os.makedirs(out, exist_ok=True)
np.savez(os.path.join(out, "probs.npz"), proba_trans=b, proba_tfidf=a, Y=y,
         titles=np.array(D["test_titles"]), cats=np.array(D["cats"]))
print("wrote", os.path.join(out, "probs.npz"))
