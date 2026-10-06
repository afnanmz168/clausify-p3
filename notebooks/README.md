# CUAD Full Pipeline — Notebooks (EDA + 3 Models)

**AI-Powered Legal Document Analysis (CUAD) · BRAC University · Phase 3**

Two notebooks are provided:

| Notebook | Contents |
|---|---|
| **`train.ipynb`** | **Training** — data load, wide→long melt, EDA, the 80/20 split, and training of all three models (TF-IDF + DistilBERT presence, DistilBERT-QA span, FLAN-T5 summarizer). Keeps the training outputs. |
| **`test.ipynb`** | **Testing** — everything measured on the held-out 20%: max-pool inference, AND-ensemble, the **confusion matrix** + metrics, span token-F1/EM, and summarizer ROUGE-L. Keeps the result outputs. **Run `train.ipynb` first in the same kernel.** |

The section below describes the pipeline as one sequence; `train.ipynb` holds the training half of the cells and `test.ipynb` the testing half.

This pipeline builds the complete three-model CUAD system **end-to-end on the full dataset** (510 contracts), starting from raw data, with exploratory analysis up front and an honest baseline-vs-transformer comparison at every stage. It was developed and run on an Apple-Silicon laptop (MPS).

> **What makes this notebook different from a "train and quote the accuracy" run:** it keeps the *decisions and dead ends*. We measure a ceiling before training a doomed model, we build a strong baseline before any transformer, and we report the caveat behind every headline number. That reasoning is the substance.

---

## How to run

```bash
pip install torch transformers datasets scikit-learn pandas numpy huggingface_hub
jupyter notebook train.ipynb   # then test.ipynb
```

Then **Run All**, top to bottom. The notebook downloads the CUAD dataset from Hugging Face automatically on first run (~40 MB JSON + a 4 MB CSV).

**Notes before you run:**
- **No token required.** The first cell's `login()` is commented out — it only silences an anonymous-rate-limit warning. If you want it, set `HF_TOKEN` as an environment variable rather than pasting a token into the notebook.
- **Hardware.** Uses `distilbert-base-uncased` and `google/flan-t5-small` so it runs on a laptop. On a CUDA GPU, swap to `microsoft/deberta-v3-base` / `google/flan-t5-base` (each model string is marked in its cell) for the report's headline backbones.
- **Runtime.** Roughly **1.5 hours** of training on Apple MPS: presence ~52 min, span ~23 min, summarizer ~8 min on CPU. The TF-IDF baseline trains in under a minute. A full evaluation pass in `test.ipynb` adds ~60 min, dominated by max-pool inference over ~150,000 (question, window) pairs.
- **Memory.** MPS does not release memory between large trainings. **Restart the kernel** (or use `use_cpu=True`) before the summarizer if you hit an out-of-memory error — see the troubleshooting section.

---

## What the notebook produces

```
outputs/
├── presence_mil/final/   # trained DistilBERT window-level presence classifier
├── span/final/           # trained DistilBertForQuestionAnswering span extractor
└── summarizer/final/     # fine-tuned FLAN-T5-small summarizer

artifacts/
├── split.json            # the seeded 80/20 contract-level split
└── baseline.pkl          # the fitted TF-IDF vectorizer + 41 logistic regressions
```
All three checkpoints are saved, so `test.ipynb` can regenerate every reported number from disk
without re-training.

---

## Cell-by-cell map

The notebook is organized as a numbered sequence. Each cell prints **one focused output** so it stays readable.

### Part A — Data & EDA (cells 1–8)
| Cell | What it does |
|---|---|
| 1 | Setup: imports, seed, MPS/CUDA device detection, load the raw `theatticusproject/cuad` dataset. |
| 2 | List the repo's files (the default `load_dataset` exposes only PDFs — the real data is in other files). |
| 3 | Download `master_clauses.csv` (the 83-column wide label table) and show its shape: `(510, 83)`. |
| 4 | List all 83 columns — reveals the `<Category>` / `<Category>-Answer` pairing. |
| 5 | Inspect one category's cells: clause text is a stringified list (`'[]'` or `"['clause']"`), answer is `Yes`/`No`. |
| 6 | **Melt** wide → long, pairing columns *by position* (robust to a header typo). 510 × 41 = **20,910** rows. |
| 7 | Look at the melted table. |
| 8 | Per-category positive rate — confirms severe imbalance (Document Name 100% … Source Code Escrow 2.5%). |

### Part B — Stage 2A: Presence classification (cells 9–20)
| Cell | What it does |
|---|---|
| 9 | Load `CUAD_v1.json` (full contract text + character offsets). |
| 10 | **Contract-level** train/test split (408 / 102, **80/20**) — no clause leaks across the boundary. |
| 11 | Build the presence dataset (labels per contract×category; full text kept per contract). |
| 12 | **Baseline:** TF-IDF over the *whole* contract + per-category Logistic Regression. |
| 13 | Richer baseline anchors: **micro-F1 0.775**, weighted 0.777, macro 0.572, macro-on-common 0.666. |
| 14 | Sliding windows + question-guided retrieval (TF-IDF cosine), with a leakage-free sanity check. |
| 15 | Apply retrieval to all rows; measure **retrieval hit-rate** — the upper bound on transformer recall. |
| 16 | Diagnose the low hit-rate; compare query strategies → best is **category-name top-3 = 64%**. |
| 17 | Build window-level MIL training examples (gold offset labels which window is positive — supervision, not leakage). |
| 18 | **Train** the window-level DistilBERT `(question, window) → contains-clause`. |
| 19 | **Max-pool inference** over all test windows; per-category comparison to the baseline. |
| 20 | **Ensemble** TF-IDF + transformer (parameter-free rules). AND-rule leads at **micro-F1 0.776** — but see the significance note below: it ties the baseline. |
| 20B | **Confusion matrix + test metrics** for the AND-ensemble: Accuracy 85.49%, Precision 77.18%, Recall 78.04%, Specificity 89.03%, F1 0.7761. Heatmap plot. |

### Part C — Stage 2B: Span extraction (cells 21–24)
| Cell | What it does |
|---|---|
| 21 | Build span-extraction examples (present clauses, windowed). 11,156 train / 2,667 test. |
| 22 | Tokenize → token `(start, end)` labels via offset mapping; sanity-decode the labelled spans. |
| 23 | **Train** `DistilBertForQuestionAnswering`. |
| 24 | Evaluate: **token-F1 0.763**, Exact Match 35.5%, overlap (F1≥0.5) 82.7%. |

### Part D — Stage 1: Summarizer (cells 25–28)
| Cell | What it does |
|---|---|
| 25 (A) | Rebuild the clause→summary pairs from the 41-template gold seed (also the post-restart entry point). |
| 26 (B) | **Train** FLAN-T5-small (on CPU, to dodge the MPS memory limit). |
| 27 | **ROUGE-L 0.775** on the test set. |
| 28 | **Zero-shot vs fine-tuned** side by side — reveals the result is template reproduction, not abstraction. |

---

## Results — and what each number actually means

| Stage | Model | Headline | Honest reading |
|---|---|---|---|
| **2A Presence** | TF-IDF + DistilBERT, AND-ensemble | **micro-F1 0.776, Accuracy 85.49%** | Transformer *alone* loses (0.694 < TF-IDF 0.775) — a real, significant gap. But the ensemble's 0.776 **ties** the 0.775 baseline (bootstrap 95% CI [−0.007, +0.009]); it earns its place through a balanced error profile, not a higher score. Confusion matrix: TP=1052, TN=2523, FP=311, FN=296. |
| **2B Span** | DistilBERT-QA | **token-F1 0.763** | Genuine extraction skill; the transformer is *essential* (TF-IDF can't point at spans). Measures extraction given the answer-bearing window — **not** the full-document precision@80%-recall benchmark. |
| **1 Summarizer** | FLAN-T5-small | **ROUGE-L 0.775** | High score = reproducing the 41 category templates, not summarizing the specific clause. Needs a larger, clause-diverse target set + bigger model. |

**The three findings in one line each:**
1. For document-level *presence*, a simple lexical baseline beats a windowed transformer — confirmed by a cluster bootstrap (+0.081, 95% CI [+0.063, +0.099], significant in 10,000/10,000 resamples). The transformer pays off only as a precision filter, and even then the ensemble ties rather than beats the baseline.
2. For *span extraction*, the transformer is irreplaceable and performs well even distilled.
3. For *summarization*, a tiny template seed turns the task into classification — the high metric is real but means less than it looks.

The full narrative, with the retrieval-ceiling investigation and the methodology, is in the project's `REPORT.md` / `REPORT.pdf`, and in full detail in `thesis/main.pdf`. The significance testing lives in `analysis/bootstrap_ci.py`.

---

## Key design decisions (why it's built this way)

- **Strong baseline first.** TF-IDF on the whole document (no length limit) sets the bar; every transformer result is judged against it, not against a vacuum.
- **Measure the ceiling before training.** The retrieval hit-rate (64%) told us a windowed transformer's recall was capped *below* the baseline — so we pivoted to max-pool MIL instead of training a doomed model.
- **Contract-level 80/20 split** (408 train / 102 test) so held-out contracts are genuinely unseen.
- **Leakage discipline:** gold offsets may label *training* windows, but are never used to build the model's input or to pick the inference window; retrieval uses only the category/question.
- **Parameter-free ensemble** (AND/OR/AVG) so there's nothing tuned on the test set.
- **Measure small margins, don't hedge about them.** A cluster bootstrap over contracts turned the ensemble's 0.001 lead into a quantified tie.
- **Honest dual eval for the summarizer** (ROUGE-L *and* a qualitative comparison) because ROUGE alone is misleading with only 41 targets.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `MPS backend out of memory` | MPS doesn't release memory between large trainings | **Restart the kernel** (Kernel → Restart) and re-run from cell 25, or set `use_cpu=True` in the summarizer's `TrainingArguments`. The execution count resetting to `[1]` confirms the restart worked. |
| `NonMatchingSplitsSizesError` on load | Dataset metadata vs file mismatch | Already handled — `load_dataset(..., verification_mode="no_checks")`. |
| `grad_norm = nan` / loss explodes | DeBERTa-v3 is unstable on Apple MPS | Use the default `distilbert-base-uncased`; switch to DeBERTa only on CUDA. |
| HF Hub rate-limit warning | Anonymous requests | Cosmetic; ignore, or set `HF_TOKEN` env var. |
| `Trainer(tokenizer=...)` TypeError | `transformers 5.x` renamed the arg | Already uses `processing_class=...`. |

---

## Honesty statement

Every reported number is on the **held-out test set** of 102 unseen contracts. Where a metric flatters the model (the summarizer's ROUGE; the small presence ensemble margin) the notebook says so — and in the ensemble's case we went further and measured it, finding the margin is statistically a tie. The transformer is reported as *losing* on presence because it does — and the more interesting result (the complementary ensemble) is what we kept. Backbones were downsized for laptop runtime, which is documented so the numbers are read at the right scale.
