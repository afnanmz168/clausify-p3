#!/bin/zsh
# Remaining October 2026 re-runs, strictly one at a time (avoids CPU/MPS contention).
cd ~/Desktop/"final project p3"/notebooks
while pgrep -f calibrate_clause_mode.py >/dev/null; do sleep 30; done
python3 -u ../retrain/stage1_summarizer_train.py > ../retrain/stage1_summarizer_train.log 2>&1; echo "summarizer train exit=$?"
python3 -u ../retrain/stage1_summarizer_test.py  > ../retrain/stage1_summarizer_test.log 2>&1;  echo "summarizer test exit=$?"
cd ../retrain/analysis_oct2026
python3 -u summ_pilot.py        > summ_pilot.log 2>&1;       echo "summ_pilot exit=$?"
python3 -u pooling_ablation.py  > pooling_ablation.log 2>&1; echo "pooling exit=$?"
python3 -u e2e.py TRANS         > e2e_TRANS.log 2>&1;        echo "e2e TRANS exit=$?"
