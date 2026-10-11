#!/usr/bin/env bash
# R6.1 VM-B: planner v3 (extended plan), then the matched baseline: r6-base + planner v2 at
# decoder temperature 1.0 on the same panels.
set -u
LOG=/content/logs; DATA=/content/data/urfm-ingest-v0; X=/content/urfm/experiments; R=/content/runs
mkdir -p $LOG $R
note() { echo "$(date -u +%FT%T) $*" >> $LOG/chain.log; }
if [ ! -d $DATA/data/osu ]; then
  HF_XET_HIGH_PERFORMANCE=1 python -c "from urfm.train.data import download_shards; download_shards()" \
    > $LOG/download.log 2>&1
fi
python -c "
from huggingface_hub import snapshot_download, hf_hub_download
snapshot_download('JacobLinCool/urfm-timing-v1', repo_type='dataset', local_dir='/content/data/urfm-timing-v1')
for f in ('r6-base/best.pt', 'r6-plan2/best.pt'):
    hf_hub_download('JacobLinCool/urfm-runs', f, local_dir='$R')
" > $LOG/fetch.log 2>&1
python -m urfm.train.planner --data $DATA --out $R/r6-plan3 --steps 10000 --workers 11 --extras true \
  --predicted-timing /content/data/urfm-timing-v1 --hub-repo JacobLinCool/urfm-runs \
  > $LOG/r6-plan3.log 2>&1
note "trained r6-plan3 rc=$?"
NAME=r6-base CKPT=$R/r6-base/best.pt PLANNER=$R/r6-plan2/best.pt LEVEL=audio3 TEMP=1.0 PT=0.7 TAG=-p2t1 \
  bash $X/r6/eval.sh
note "chain_x_b done"
