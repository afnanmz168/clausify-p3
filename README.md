# AI-Powered Legal Document Analysis System (CUAD)

**Clause Detection, Category-Level Risk Prioritization and Plain-Language Explanation**
B.Sc. Final Year Project (CSE400) · Department of Computer Science and Engineering · BRAC University

Jerin Aktar (22101279) · Afnan Mazumdar (24141229) · Shoyeb Hasan Sayem (22101386)
Supervisor: Utsho Kumar Roy

---

## What this project is

Commercial contracts run to tens of thousands of characters, and the clauses that matter most are buried
inside them. CUAD (510 expert-annotated contracts, 41 clause categories) frames this as an extraction
benchmark. This project goes further: it decides **what is in a contract**, **where each clause is**,
and **what it means and how risky it is** for the signer, and ships it as the **Clausify** app.

| Stage | Model | Job | Held-out result (102 test contracts) |
|---|---|---|---|
| **2A** | TF-IDF + DistilBERT (MIL) AND-ensemble | Which of the 41 categories are present? | micro-F1 **0.779**, accuracy **85.65%** |
| **2B** | DistilBERT-QA | Locate the exact clause text | token-F1 **0.764** (given the window) |
| **1** | FLAN-T5-small | One plain-English sentence per clause | ROUGE-L **0.775** (41 templates) |

All numbers come from a **contract-level 80/20 split** (408 training / 102 test contracts) and the
October 2026 re-training. The full evaluation — confidence intervals, seed/budget/split controls,
end-to-end survival (21.7%), risk-weighted results and every caveat — is in the report.

---

## What is in this folder

| Folder / file | Contents |
|---|---|
| **`project_report/`** | **The full project report** (LaTeX, BRAC CSE400 template) — `main.pdf`, ~100 pages |
| `paper/`, `IEEE_Paper_Clausify.pdf` | Conference-style paper |
| `notebooks/` | `train.ipynb`, `test.ipynb`, `build_artifacts.py`; `artifacts/` (split, TF-IDF baseline); `outputs/` (trained models) |
| `retrain/` | Re-training and re-testing scripts with **exact terminal commands** (`retrain/README.md`), logs and results; `smoke_test.py` checks every model |
| `analysis/` | Scripts behind every statistic in the report (bootstrap, seeds, end-to-end, risk-weighted, ablations) |
| `data/` | The 80/20 split as CSV |
| `review/` | Risk-taxonomy review sheet and agreement script for legal reviewers |
| `demo_clause.txt` | 18-clause demonstration contract (used by the app's tests) |
| `P25301051 p3.pptx`, `PRESENTATION_SPEECH*`, `SPEECH_slides_*`, `VIVA_QA*` | Presentation and viva material |

The application lives in its own repository: **[clausify-p3-app](https://github.com/afnanmz168/clausify-p3-app)**
(Streamlit; all four models, PDF/Word upload, highlighted contract view, risk report download,
missing-protection checklist). Trained weights: **[af123Af/clausify-models](https://huggingface.co/af123Af/clausify-models)**.

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

*Research prototype — not legal advice. The risk taxonomy has not been reviewed by a lawyer.*
