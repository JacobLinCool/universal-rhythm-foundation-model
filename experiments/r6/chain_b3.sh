#!/usr/bin/env bash
# VM-B (restart after a lost runtime): planner from the Hub, the four ablations (each resumes
# from the Hub), their evaluations, then RL v2 on the same panels.
set -u
LOG=/content/logs; X=/content/urfm/experiments; R=/content/runs
mkdir -p $LOG $R
note() { echo "$(date -u +%FT%T) $*" >> $LOG/chain.log; }
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('JacobLinCool/urfm-runs', 'r6-plan/best.pt', local_dir='$R')
" > $LOG/fetch-plan.log 2>&1
note "planner fetched rc=$?"
for spec in "r6-a10k|" "r6-noplan|--sections false" "r6-noref|--ref false" \
            "r6-oct15|--p-octave 0.15 --octave-factors"; do
  NAME=${spec%%|*} STEPS=10000 EXTRA="${spec#*|}" bash $X/r6/train_decoder.sh
done
note "ablations done"
PL=$R/r6-plan/best.pt
for n in r6-a10k r6-noplan r6-noref r6-oct15; do
  NAME=$n CKPT=$R/$n/best.pt PLANNER=$PL LEVEL=ablation bash $X/r6/eval.sh
done
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('JacobLinCool/urfm-runs', 'r5-rl2/best.pt', local_dir='$R')
" > $LOG/fetch-rl2.log 2>&1
NAME=r5-rl2 CKPT=$R/r5-rl2/best.pt LEVEL=full bash $X/r6/eval.sh
note "chain_b done"
