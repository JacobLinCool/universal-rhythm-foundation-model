#!/usr/bin/env bash
# R6 decoder: NAME=r6-base STEPS=40000 EXTRA="..." bash train_decoder.sh
# Defaults are the full R6 recipe (SPEC.md); ablations override through EXTRA (later flags win).
set -u
LOG=/content/logs; DATA=/content/data/urfm-ingest-v0; R=/content/runs
mkdir -p $LOG $R
note() { echo "$(date -u +%FT%T) $*" >> $LOG/chain.log; }
if [ ! -d $DATA/data/osu ]; then
  HF_XET_HIGH_PERFORMANCE=1 python -c "from urfm.train.data import download_shards; download_shards()" \
    > $LOG/download.log 2>&1
  note "download rc=$?"
fi
if [ ! -f $R/r4a-full-15m/best.pt ] || [ ! -d /content/data/urfm-timing-v1/data ]; then
  python -c "
from huggingface_hub import snapshot_download, hf_hub_download
snapshot_download('JacobLinCool/urfm-timing-v1', repo_type='dataset', local_dir='/content/data/urfm-timing-v1')
hf_hub_download('JacobLinCool/urfm-runs', 'r4a-full-15m/best.pt', local_dir='$R')
" > $LOG/fetch.log 2>&1
  note "fetched rc=$?"
fi
note "start $NAME"
python -m urfm.train.train --data $DATA --sources taiko1000 osu --size 15m \
  --steps ${STEPS:-40000} --warmup 1000 --batch-size 64 --max-dec-len 2048 --workers ${WORKERS:-10} \
  --log-every 100 --eval-every 2000 --ckpt-every 2000 --song-context true --cond true --texture true \
  --canonical true --taiko-boost 2.0 \
  --p-speed 0.5 --p-local 0.3 --p-octave 0.5 --octave-factors 0.5 0.6 2 0.25 0.25 0.15 \
  --predicted-timing /content/data/urfm-timing-v1 --p-predicted 0.5 --fast-boost 2 \
  --sections true --ref true --ctx-noise 0.5 --song-layers 1 --windows-per-visit 4 \
  --init $R/r4a-full-15m/best.pt --init-partial true --resume true ${EXTRA:-} \
  --out $R/$NAME --hub-repo JacobLinCool/urfm-runs > $LOG/$NAME.log 2>&1
note "trained $NAME rc=$?"
