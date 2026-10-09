# AI-Powered Legal Document Analysis System (CUAD)

**Clause Detection, Category-Level Risk Prioritization and Plain-Language Explanation**
B.Sc. Final Year Project (CSE400) · Department of Computer Science and Engineering · BRAC University

Jerin Aktar (22101279) · Afnan Mazumdar (24141229) · Shoyeb Hasan Sayem (22101386)
Supervisor: Utsha Kumar Roy

---

## What this project is

Commercial contracts run to tens of thousands of characters, and the clauses that matter most are buried
inside them. CUAD (510 expert-annotated contracts, 41 clause categories) frames this as an extraction
benchmark. This project goes further: it decides **what is in a contract**, **where each clause is**,
and **what it means and how risky it is** for the signer, and ships it as the **Clausify** app.

| Stage | Model | Job | Held-out result (102 test contracts) |
|---|---|---|---|
| **2A** | TF-IDF + DistilBERT (MIL) ensemble, tuned on a validation split | Which of the 41 categories are present? | micro-F1 **0.809**, accuracy **88.43%**; recall-first setting finds **90.3%** of High-risk clauses |
| **2B** | DistilBERT-QA | Locate the exact clause text | token-F1 **0.764** (given the window) |
| **1** | FLAN-T5-small | One plain-English sentence per clause | ROUGE-L **0.775** (41 templates) |

All numbers come from a **contract-level 80/20 split** (408 training / 102 test contracts). The
Stage 2A numbers come from the re-run in `retrain/v2_fullwindow/` (report Section 5.3.3): the
transformer reads the whole 2,000-character window (512 tokens instead of 256, which read only about
half of it), and both models are tuned on 81 validation contracts taken from the training split. The
first setup's numbers (TF-IDF 0.775, transformer 0.692, AND-ensemble 0.779), confidence intervals,
seed/budget/split controls, end-to-end survival (22.0%), risk-weighted results and every caveat are
in the report.

---

## What is in this folder

| Folder / file | Contents |
|---|---|
| **`project_report/`** | **The full project report** (LaTeX, BRAC CSE400 template) — `main.pdf`, ~100 pages |
| `paper/`, `IEEE_Paper_Clausify.pdf` | Conference-style paper |
| `notebooks/` | `train.ipynb`, `test.ipynb`, `build_artifacts.py`; `artifacts/` (split, TF-IDF baseline); `outputs/` (trained models) |
| `retrain/` | Re-training and re-testing scripts with **exact terminal commands** (`retrain/README.md`), logs and results; `v2_fullwindow/` is the 512-token re-run with validation tuning (the final presence results and the app's thresholds); `analysis_oct2026/` re-runs every analysis on the October models; `smoke_test.py` checks every model; `verify_report.py` checks every reported number |
| `analysis/` | August 2026 control runs (seeds, training budget, split variance, Longformer cost, app speed) |
| `data/` | The 80/20 split as CSV |
| `USABILITY_PROTOCOL.md` | Protocol for a small user test of the app |
| `demo_clause.txt` | 18-clause demonstration contract (used by the app's tests) |
| `P25301051-p3-presentation-15-slides.pptx`, `VIVA_QA*` | Presentation (with speaker notes) and viva question bank |

The application lives in its own repository: **[clausify-p3-app](https://github.com/afnanmz168/clausify-p3-app)**
(Streamlit; all four models, PDF/Word upload, highlighted contract view, risk report download,
missing-protection checklist). Trained weights: **[af123Af/clausify-models](https://huggingface.co/af123Af/clausify-models)**.
The re-run presence model (`notebooks/outputs/presence_v2/`) and its TF-IDF model
(`notebooks/artifacts/baseline_v2.pkl`) are in that repository as `presence_v2/` and
`baseline_v2/baseline.pkl` (uploaded 9 October 2026; same SHA-256 as the local files), so the app
downloads the final models, with the calibrated chance, when it runs from GitHub.

---

## Quick start

Check that the trained models load and work (about a minute):

```bash
python3 retrain/smoke_test.py
```

Re-train or re-test any model: see `retrain/README.md`.

Rebuild the report PDF:

```bash
cd project_report && tectonic main.tex
```

---

## Requirements

Python 3.13, PyTorch 2.8, transformers 5.12.1, scikit-learn 1.6.1 (see `requirements.txt`). Developed and
trained on an Apple M4 laptop (MPS); no CUDA needed.

*Research prototype — not legal advice.*
