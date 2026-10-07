# Re-training and re-testing the three models

Every script here is extracted from `notebooks/train.ipynb`, `notebooks/test.ipynb` and
`notebooks/build_artifacts.py`, with the same settings as the report (seed 42, same 80/20
contract-level split). Each **train** script overwrites the model in `notebooks/outputs/*/final/`.

**Run every command from the `notebooks/` folder**, one at a time, and keep the terminal open
until it finishes. Logs are written next to the scripts in `retrain/`.

```bash
cd ~/Desktop/"final project p3"/notebooks
```

## 0. Check that all saved models work (about 1 minute)

```bash
python3 ../retrain/smoke_test.py
```

Expected: four `[PASS]` lines and `ALL MODELS OK`.

## 1. Stage 2A: Presence (TF-IDF baseline + DistilBERT)

Train (about 50 minutes on Apple MPS; the baseline is printed in the first minute):

```bash
python3 -u ../retrain/stage2a_presence_train.py 2>&1 | tee ../retrain/stage2a_presence_train.log
```

Test (about 50 minutes; scores ~150,000 question/window pairs):

```bash
python3 -u ../retrain/stage2a_presence_test.py 2>&1 | tee ../retrain/stage2a_presence_test.log
```

Results: `retrain/stage2a_presence_test.json`, confusion matrix `retrain/stage2a_confusion_matrix.png`.

| Metric | Aug 2026 | Oct 2026 |
|---|---|---|
| TF-IDF baseline micro-F1 | 0.775 | 0.775 (deterministic) |
| Transformer (max-pool) micro-F1 | 0.694 | 0.692 |
| AND-ensemble micro-F1 / accuracy | 0.776 / 85.49% | 0.779 / 85.65% |

## 2. Stage 2B: Span extractor (DistilBERT-QA)

Train (about 25 minutes on MPS):

```bash
python3 -u ../retrain/stage2b_span_train.py 2>&1 | tee ../retrain/stage2b_span_train.log
```

Test (about 2 minutes):

```bash
python3 -u ../retrain/stage2b_span_test.py 2>&1 | tee ../retrain/stage2b_span_test.log
```

Results: `retrain/stage2b_span_test.json`.

| Metric | Aug 2026 | Oct 2026 |
|---|---|---|
| token-F1 / Exact Match / overlap | 0.763 / 35.5% / 82.7% | 0.764 / 35.5% / 82.8% |

## 3. Stage 1: Summarizer (FLAN-T5-small)

Train (about 8 minutes on CPU; built exactly like the deployed model in `build_artifacts.py`,
with no evaluation set):

```bash
python3 -u ../retrain/stage1_summarizer_train.py 2>&1 | tee ../retrain/stage1_summarizer_train.log
```

Test (about 3 minutes on CPU; ROUGE-L on 150 test clauses plus 4 side-by-side examples):

```bash
python3 -u ../retrain/stage1_summarizer_test.py 2>&1 | tee ../retrain/stage1_summarizer_test.log
```

Results: `retrain/stage1_summarizer_test.json`. Aug 2026: ROUGE-L 0.775 against the 41 templates.

## 4. After all three are trained

Check the new models, then redraw the report figures:

```bash
python3 ../retrain/smoke_test.py
python3 ../retrain/make_figures.py
```

The analyses behind every Chapter 5 table (run from `retrain/analysis_oct2026/`, one at a time):

```bash
cd ../retrain/analysis_oct2026
python3 make_probs.py                                           # once: reshape the presence scores for the scripts below
python3 bootstrap_ci.py                                         # Table 5.14, cluster bootstrap (seconds)
python3 -u span_ci.py 2>&1 | tee span_ci.log                    # Table 5.5 intervals
python3 -u summ_pilot.py 2>&1 | tee summ_pilot.log              # Tables 5.6-5.7 (~5 min)
python3 -u e2e.py AND 2>&1 | tee e2e.log                        # Tables 5.8-5.9 (~40 min)
python3 -u risk_threshold_calib.py 2>&1 | tee risk_threshold_calib.log   # Table 5.10, Figure 5.4
python3 -u error_examples.py 2>&1 | tee error_examples.log      # Table 5.11
python3 -u rare_tail.py 2>&1 | tee rare_tail.log                # Table 5.12
caffeinate -i python3 -u pooling_ablation.py 2>&1 | tee pooling_ablation.log  # Table 5.13 (~55 min; keep the Mac awake)
python3 -u app_bench.py 2>&1 | tee app_bench.log                # Table 4.6, app speed with the current app (~3 min)
```

Finally, check every number in the report against these result files (from the project folder):

```bash
cd ~/Desktop/"final project p3"
python3 retrain/verify_report.py        # expected last line: "... checks, 0 mismatched"
```

## Notes

- Run one script at a time. Two trainings at once share the GPU and memory and can stall.
- MPS training is not bit-reproducible: a rerun matches the reported scores to about the third
  decimal place, not exactly.
