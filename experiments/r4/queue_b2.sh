#!/usr/bin/env bash
# VM-B after queue_b: audio-predicted timing maps for every song (DP decoding), for
# predicted-grid fine-tuning of the 45M model.
set -u
LOG=/content/logs; DATA=/content/data/urfm-ingest-v0; X=/content/urfm/experiments; R=/content/runs
note() { echo "$(date -u +%FT%T) $*" >> $LOG/chain.log; }
sleep 60; while pgrep -f "[q]ueue_b.sh" > /dev/null; do sleep 30; done
python $X/r4/predict_timing.py --ckpt $R/beat-v1/best.pt --data $DATA \
  --out /content/data/urfm-timing-v1 --hub-repo JacobLinCool/urfm-timing-v1 > $LOG/predict.log 2>&1
note "predict timing rc=$?"
