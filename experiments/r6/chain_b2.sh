#!/usr/bin/env bash
# VM-B after the ablations: evaluate them, then RL v2 (R5) on the same panels for the new
# metrics (placement, subdivision recall, structure alignment).
set -u
LOG=/content/logs; X=/content/urfm/experiments; R=/content/runs
note() { echo "$(date -u +%FT%T) $*" >> $LOG/chain.log; }
until grep -q "ablations done" $LOG/chain.log; do sleep 60; done
PL=$R/r6-plan/best.pt
for n in r6-a10k r6-noplan r6-noref r6-oct15; do
  NAME=$n CKPT=$R/$n/best.pt PLANNER=$PL LEVEL=ablation bash $X/r6/eval.sh
done
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('JacobLinCool/urfm-runs', 'r5-rl2/best.pt', local_dir='$R')
" > $LOG/fetch-rl2.log 2>&1
NAME=r5-rl2 CKPT=$R/r5-rl2/best.pt LEVEL=full bash $X/r6/eval.sh
note "chain_b2 done"
