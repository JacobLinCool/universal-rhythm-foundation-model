#!/usr/bin/env bash
# R6.2: song-level GRPO from r6x-base (+ planner v3 for evaluation); osu! presets on the side.
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
from huggingface_hub import hf_hub_download
for f in ('r6x-base/best.pt', 'r6-plan3/best.pt', 'beat-v1/best.pt'):
    hf_hub_download('JacobLinCool/urfm-runs', f, local_dir='$R')
" > $LOG/fetch.log 2>&1
note "fetched rc=$?"
(python $X/r6/osu_defaults.py --out $R/r6/osu_defaults.json > $LOG/osu_defaults.log 2>&1; note "osu defaults rc=$?") &
python -m urfm.train.rl_song --init $R/r6x-base/best.pt --steps ${STEPS:-200} \
  --out $R/r6x-rl --hub-repo JacobLinCool/urfm-runs > $LOG/r6x-rl.log 2>&1
note "trained r6x-rl rc=$?"
wait
NAME=r6x-rl CKPT=$R/r6x-rl/best.pt PLANNER=$R/r6-plan3/best.pt LEVEL=audio3 TEMP=1.0 PT=0.7 \
  bash $X/r6/eval.sh
note "chain_rl_song done"
