"""Regenerate the IEEE paper's two bar charts from the stored result files."""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures"
L512 = ROOT / "retrain/v2_fullwindow/runs/L512"

NAVY, LIGHT, GREEN, ORANGE, RED = "#1F2761", "#A9B3D6", "#4CAF7F", "#E8A03A", "#D94F5C"
INK, GREY, GRID = "#141a3a", "#6b7080", "#e6e8f0"
plt.rcParams.update({"font.size": 15, "axes.edgecolor": "#d0d4e4", "axes.labelcolor": GREY,
                     "xtick.color": GREY, "ytick.color": GREY})

cfg = json.load(open(L512 / "results.json"))["configs"]
T = lambda k: cfg[k]["test"]

# First-setup values (256 tokens, 408 contracts, threshold 0.5): report Tables 5.1 and 5.12.
FIRST = {"TF-IDF baseline": (0.7747, 62), "Transformer": (0.6924, 28),
         "AND ensemble": (0.7789, 65), "OR ensemble": (0.6916, 25)}

# ---------------- Figure 1: micro-F1, first setup vs whole window + tuned ----------------
models = ["TF-IDF baseline", "Transformer", "Ensemble"]
first = [FIRST["TF-IDF baseline"][0], FIRST["Transformer"][0], FIRST["AND ensemble"][0]]
tuned = [T("tfidf_tuned_f1")["micro_f1"], T("transformer_tuned_f1")["micro_f1"],
         T("ensemble_tuned_f1")["micro_f1"]]

fig, ax = plt.subplots(figsize=(8, 4.9), dpi=200)
h = 0.36
ys = range(len(models))
for i, (a, b) in enumerate(zip(first, tuned)):
    ax.barh(i - h / 2, a, h, color=LIGHT)
    ax.barh(i + h / 2, b, h, color=NAVY)
    ax.text(a + 0.003, i - h / 2, f"{a:.3f}", va="center", fontsize=13, color=GREY)
    ax.text(b + 0.003, i + h / 2, f"{b:.3f}", va="center", fontsize=14, color=INK, weight="bold")
ax.set_yticks(list(ys), models)
ax.invert_yaxis()
ax.set_xlim(0.65, 0.835)
ax.set_xlabel("held-out micro-F1 (102 test contracts)")
ax.grid(axis="x", color=GRID)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=LIGHT), plt.Rectangle((0, 0), 1, 1, color=NAVY)],
          labels=["First setup: 256 tokens, threshold 0.5", "Whole window, tuned on validation"],
          loc="lower center", bbox_to_anchor=(0.42, 1.0), ncol=2, fontsize=11, frameon=False,
          handlelength=1.2, columnspacing=1.2)
fig.tight_layout()
fig.savefig(OUT / "chart_leaderboard.png")
plt.close(fig)

# ---------------- Figure 2: High-risk clauses missed (of 176) ----------------------------
rows = [
    ("AND\nensemble", FIRST["AND ensemble"][1], RED),
    ("TF-IDF\n", FIRST["TF-IDF baseline"][1], RED),
    ("Transformer\n", FIRST["Transformer"][1], ORANGE),
    ("OR\nensemble", FIRST["OR ensemble"][1], ORANGE),
    ("Balanced\n(AND, F1)", T("ensemble_tuned_f1")["high_risk_missed"], RED),
    ("TF-IDF\n(tuned, F1)", T("tfidf_tuned_f1")["high_risk_missed"], RED),
    ("Ensemble\n(AVG, F2)", T("ensemble_tuned_f2")["high_risk_missed"], ORANGE),
    ("Transformer\n(F2, app)", T("transformer_tuned_f2")["high_risk_missed"], GREEN),
]
assert T("ensemble_tuned_f1")["high_risk_total"] == 176
fig, ax = plt.subplots(figsize=(10, 5), dpi=200)
xs = [0, 1, 2, 3, 4.6, 5.6, 6.6, 7.6]
for x, (lab, missed, col) in zip(xs, rows):
    pct = 100 * missed / 176
    ax.bar(x, pct, 0.72, color=col)
    ax.text(x, pct + 0.8, f"{pct:.1f}%", ha="center", fontsize=13, color=INK, weight="bold")
ax.set_xticks(xs, [r[0] for r in rows], fontsize=10.5)
ax.set_ylabel("High-risk clauses missed (%)")
ax.set_ylim(0, 54)
ax.axvline(3.8, color="#c4c8d8", lw=1, ls="--")
ax.text(1.5, 51, "First setup (threshold 0.5)", ha="center", fontsize=12, color=GREY)
ax.text(6.1, 51, "Re-run, chosen on validation", ha="center", fontsize=12, color=GREY)
ax.grid(axis="y", color=GRID)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "chart_highrisk.png")
plt.close(fig)
print("wrote", OUT / "chart_leaderboard.png", OUT / "chart_highrisk.png")
print("tuned F1:", tuned)
