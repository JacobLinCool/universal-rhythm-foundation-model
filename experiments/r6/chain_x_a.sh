#!/usr/bin/env bash
# R6.1 VM-A: decoder fine-tune with the extended section plan, then evaluation with planner v3.
set -u
LOG=/content/logs; X=/content/urfm/experiments; R=/content/runs
mkdir -p $LOG $R
note() { echo "$(date -u +%FT%T) $*" >> $LOG/chain.log; }
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('JacobLinCool/urfm-runs', 'r6-base/best.pt', local_dir='$R')
" > $LOG/fetch-base.log 2>&1
NAME=r6x-base STEPS=15000 EXTRA="--init $R/r6-base/best.pt --plan-extra true --lr 1.5e-4 --warmup 500" \
  bash $X/r6/train_decoder.sh
# planner v3 trains on VM-B; it is done when its metrics on the Hub reach step 10000
until python -c "
import json, sys
from huggingface_hub import hf_hub_download
p = hf_hub_download('JacobLinCool/urfm-runs', 'r6-plan3/metrics.jsonl', force_download=True)
rows = [json.loads(l) for l in open(p)]
sys.exit(0 if rows and rows[-1]['step'] >= 10000 else 1)
" > /dev/null 2>&1; do sleep 120; done
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('JacobLinCool/urfm-runs', 'r6-plan3/best.pt', local_dir='$R', force_download=True)
" > $LOG/fetch-plan3.log 2>&1
note "planner3 ready"
NAME=r6x-base CKPT=$R/r6x-base/best.pt PLANNER=$R/r6-plan3/best.pt LEVEL=audio3 TEMP=1.0 PT=0.7 \
  bash $X/r6/eval.sh
note "chain_x_a done"
