# Stage 2A v2: read the whole window, tune both models on one validation split

**Why.** The original presence model pairs each 2,000-character window with the category question
and truncates at 256 tokens. The question uses ~53 tokens, so the model reads only ~986 characters
(median 49%) of each window. About 34% of every contract is never read, and 23.2% of positive
training windows have their gold clause after the cut, so the label says "present" for text the
model never sees. Measured on the training split with the trained tokenizer.

**What changes.** `max_length` 256 → 512 (the whole window fits), and 81 of the 408 training
contracts are held out as a validation split. Epoch, pooling rule, thresholds, TF-IDF `C` and the
ensemble rule are all chosen on validation, for both models in the same way, before the test set is
scored. Everything else matches `stage2a_presence_train.py`. The original models and results are not
touched; output goes to `runs/<tag>/`.

## Run (from `notebooks/`)

```bash
cd ~/Desktop/"final project p3"/notebooks
caffeinate -i python3 -u ../retrain/v2_fullwindow/presence_v2.py --max-len 512 --tag L512 2>&1 | tee ../retrain/v2_fullwindow/L512.log
python3 ../retrain/v2_fullwindow/tune_and_test.py --tag L512
```

About 5 hours on the M4 (training ~2.2 h for 3 epochs, scoring ~255,000 pairs ~2.4 h).
Keep the Mac on power with the lid open.

Optional control, so the before/after uses the same 327 training contracts (about 2.5 hours):

```bash
caffeinate -i python3 -u ../retrain/v2_fullwindow/presence_v2.py --max-len 256 --tag L256 2>&1 | tee ../retrain/v2_fullwindow/L256.log
python3 ../retrain/v2_fullwindow/tune_and_test.py --tag L256
```

## Output

`runs/<tag>/results.md` is the table for the report. Rows:

| row | meaning |
|---|---|
| `original: ...` | the current report's models (256 tokens, 408 contracts, threshold 0.5) |
| `tfidf_untuned`, `transformer_untuned` | refitted on 327 contracts, threshold 0.5, max-pooling: the direct effect of reading the whole window |
| `*_tuned_f1` | everything chosen on validation to maximise micro-F1 |
| `*_tuned_f2` | everything chosen on validation to maximise micro-F2 (a missed clause counts more than a false alarm; the app setting) |

`results.json` also holds the chosen per-category thresholds, which the app can load.
