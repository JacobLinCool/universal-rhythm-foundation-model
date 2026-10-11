#!/usr/bin/env bash
# VM-A after the 45M chain: fine-tune the best generator (r4a, 15M) on audio-predicted grids.
# (45M reached a lower validation loss but generated worse charts, see LOG.md.)
set -u
LOG=/content/logs; X=/content/urfm/experiments; R=/content/runs
note() { echo "$(date -u +%FT%T) $*" >> $LOG/chain.log; }
sleep 60; while pgrep -f "[q]ueue_a3.sh" > /dev/null; do sleep 60; done
until python -c "
from huggingface_hub import snapshot_download, hf_hub_download
snapshot_download('JacobLinCool/urfm-timing-v1', repo_type='dataset', local_dir='/content/data/urfm-timing-v1')
hf_hub_download('JacobLinCool/urfm-runs', 'r4a-full-15m/best.pt', local_dir='$R')
" > $LOG/timing-download.log 2>&1; do sleep 300; done
note "timing maps + r4a downloaded"
NAME=r4a-ft-pred SIZE=15m STEPS=3000 BATCH=64 \
  EXTRA="--warmup 200 --lr 1e-4 --init $R/r4a-full-15m/best.pt --p-speed 0.5 --p-local 0.3 --p-octave 0.15 --predicted-timing /content/data/urfm-timing-v1 --p-predicted 0.3" \
  bash $X/r4/chain_r4.sh
