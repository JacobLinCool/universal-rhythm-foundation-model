#!/usr/bin/env bash
# VM-B: planner v2 (tolerant finest-subdivision targets), then r6-base with it on P1 audio
# (seed 0) and P7 x1/2. Waits for chain_b3.
set -u
LOG=/content/logs; DATA=/content/data/urfm-ingest-v0; X=/content/urfm/experiments; R=/content/runs
note() { echo "$(date -u +%FT%T) $*" >> $LOG/chain.log; }
until grep -q "chain_b done" $LOG/chain.log; do sleep 60; done
python -m urfm.train.planner --data $DATA --out $R/r6-plan2 --steps 10000 --workers 11 \
  --predicted-timing /content/data/urfm-timing-v1 --hub-repo JacobLinCool/urfm-runs \
  > $LOG/r6-plan2.log 2>&1
note "trained r6-plan2 rc=$?"
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('JacobLinCool/urfm-runs', 'r6-base/best.pt', local_dir='$R')
" > $LOG/fetch-base.log 2>&1
mkdir -p $R/r6-base-plan2
cp $R/r6-base/best.pt $R/r6-base-plan2/best.pt
NAME=r6-base-plan2 CKPT=$R/r6-base-plan2/best.pt PLANNER=$R/r6-plan2/best.pt LEVEL=ablation \
  bash $X/r6/eval.sh
note "chain_b4 done"
