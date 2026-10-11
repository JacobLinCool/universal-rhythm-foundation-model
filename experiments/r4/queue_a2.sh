#!/usr/bin/env bash
# VM-A: finish predicted timing maps, then the 45M R4 model.
set -u
LOG=/content/logs; X=/content/urfm/experiments; R=/content/runs
note() { echo "$(date -u +%FT%T) $*" >> $LOG/chain.log; }
while pgrep -f "[p]redict_timing.py" > /dev/null; do sleep 30; done
note "predict done (queue_a2)"
NAME=r4-45m SIZE=45m STEPS=16000 \
  EXTRA="--warmup 1000 --p-speed 0.5 --p-local 0.3 --p-octave 0.15 --predicted-timing /content/data/urfm-timing-v1 --p-predicted 0.25" \
  bash $X/r4/chain_r4.sh
