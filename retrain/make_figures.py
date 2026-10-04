"""Redraw the report figures from the October 2026 re-training results.

Writes into project_report/images/ (the August originals stay in thesis/figures/).
Run from anywhere:  python retrain/make_figures.py
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "project_report", "images")
BLUE, ORANGE, GREEN, GREY = "#4c6ef5", "#e8590c", "#2f9e44", "#868e96"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 10})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), dpi=150)
    plt.close(fig)
    print("wrote", name)


res = json.load(open(os.path.join(HERE, "stage2a_presence_test.json")))

# ---- presence model comparison -------------------------------------------------
label = {"Ensemble AND  (min)": "AND\nensemble", "TF-IDF baseline": "TF-IDF\nbaseline",
         "Ensemble AVG  (mean)": "AVG\nensemble", "Ensemble OR   (max)": "OR\nensemble",
         "Transformer (max-pool)": "Transformer\n(max-pool)"}
rows = sorted(res["ensemble_table"], key=lambda r: -r["micro_F1"])
colors = [GREEN if "AND" in r["model"] else BLUE if "TF-IDF" in r["model"] else GREY for r in rows]
fig, ax = plt.subplots(figsize=(6, 3.2))
bars = ax.bar([label[r["model"]] for r in rows], [r["micro_F1"] for r in rows], color=colors, width=0.6)
for b, r in zip(bars, rows):
    ax.text(b.get_x() + b.get_width() / 2, r["micro_F1"] + 0.003, f'{r["micro_F1"]:.3f}',
            ha="center", va="bottom", fontsize=8, fontweight="bold")
ax.set_ylim(0.6, 0.83); ax.set_ylabel("micro-F1")
ax.set_title("Presence models on the 20% test set")
save(fig, "ensemble_comparison.png")

# ---- per-category F1, top 12 -------------------------------------------------------
pc = sorted(res["per_category"], key=lambda r: (-r["f1"], -r["test_pos"]))[:12][::-1]
short = lambda c: c.replace("Post-Termination Services", "Post-Termination Serv.")
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.barh([short(r["category"]) for r in pc], [r["f1"] for r in pc], color=BLUE, height=0.6)
for i, r in enumerate(pc):
    ax.text(r["f1"] + 0.01, i, f'{r["f1"]:.3f}', va="center", fontsize=7)
ax.set_xlim(0, 1.12); ax.set_xlabel("F1 score"); ax.tick_params(axis="y", labelsize=8)
ax.set_title("Per-category F1 on the 20% test set (top 12 categories)")
save(fig, "per_category_f1.png")

# ---- confusion matrix --------------------------------------------------------------
c = res["confusion"]; cm = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
fig, ax = plt.subplots(figsize=(4.6, 3.8))
im = ax.imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=16,
                color="white" if cm[i, j] > cm.max() / 2 else "black")
ax.set_xticks([0, 1]); ax.set_xticklabels(["Absent", "Present"])
ax.set_yticks([0, 1]); ax.set_yticklabels(["Absent", "Present"])
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_title("AND-ensemble, 20% test set", fontsize=10)
fig.colorbar(im, ax=ax)
save(fig, "confusion_matrix.png")


# ---- training curves (logged every 100 steps; validation at each epoch end) --------
def loss_curve(hist_file, title, fname, steps_per_epoch):
    h = json.load(open(os.path.join(HERE, hist_file)))
    tr = [(x["step"], x["loss"]) for x in h if "loss" in x and "step" in x]
    va = [(x["step"], x["eval_loss"]) for x in h if "eval_loss" in x]
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.plot(*zip(*tr), "-o", ms=3, color=BLUE, label="Training loss (every 100 steps)")
    if va:
        ax.plot(*zip(*va), "--s", ms=5, color=ORANGE, label="Validation loss (end of epoch)")
    for e in range(1, 3):
        ax.axvline(e * steps_per_epoch, color="#ced4da", lw=0.8, zorder=0)
    ax.set_xlabel("Training step"); ax.set_ylabel("Cross-entropy loss")
    ax.set_title(title); ax.legend(fontsize=8)
    save(fig, fname)


loss_curve("presence_log_history.json", "Presence classifier (DistilBERT window-level) training",
           "presence_loss.png", 1425)
if os.path.exists(os.path.join(HERE, "summarizer_log_history.json")):
    loss_curve("summarizer_log_history.json", "Summarizer (FLAN-T5-small) fine-tuning",
               "summarizer_loss.png", 500)

# ---- threshold sweep + reliability diagram (re-run analysis output) -------------------
src = os.path.join(HERE, "analysis_oct2026", "threshold_calibration.png")
if os.path.exists(src):
    import shutil; shutil.copy(src, os.path.join(OUT, "threshold_calibration.png"))
    print("copied threshold_calibration.png")
