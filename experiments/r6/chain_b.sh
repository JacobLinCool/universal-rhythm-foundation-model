#!/usr/bin/env bash
# VM-B: song planner, then the four 10k-step ablations (SPEC.md).
set -u
LOG=/content/logs; DATA=/content/data/urfm-ingest-v0; X=/content/urfm/experiments; R=/content/runs
mkdir -p $LOG $R
note() { echo "$(date -u +%FT%T) $*" >> $LOG/chain.log; }
if [ ! -d $DATA/data/osu ]; then
  HF_XET_HIGH_PERFORMANCE=1 python -c "from urfm.train.data import download_shards; download_shards()" \
    > $LOG/download.log 2>&1
  note "download rc=$?"
fi
python -c "
from huggingface_hub import snapshot_download, hf_hub_download
snapshot_download('JacobLinCool/urfm-timing-v1', repo_type='dataset', local_dir='/content/data/urfm-timing-v1')
hf_hub_download('JacobLinCool/urfm-runs', 'r4a-full-15m/best.pt', local_dir='$R')
" > $LOG/fetch.log 2>&1
note "fetched rc=$?"
python -m urfm.train.planner --data $DATA --out $R/r6-plan --steps 10000 --workers 10 \
  --predicted-timing /content/data/urfm-timing-v1 --hub-repo JacobLinCool/urfm-runs \
  > $LOG/r6-plan.log 2>&1
note "trained r6-plan rc=$?"
for spec in "r6-a10k|" "r6-noplan|--sections false" "r6-noref|--ref false" \
            "r6-oct15|--p-octave 0.15 --octave-factors"; do
  NAME=${spec%%|*} STEPS=10000 EXTRA="${spec#*|}" bash $X/r6/train_decoder.sh
done
note "ablations done"
