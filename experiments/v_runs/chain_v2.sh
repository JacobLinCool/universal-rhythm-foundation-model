#!/usr/bin/env bash
# V2: Taiko + osu!taiko (same roles), 15M; Taiko test-set evaluation.
set -u
LOG=/content/logs
DATA=/content/data/urfm-ingest-v0
NAME=v2-taiko-osutaiko
note() { echo "$(date -u +%FT%T) $*" >> $LOG/chain.log; }
note "start $NAME"
python -m urfm.train.train --data $DATA --sources taiko1000 osu --mechanics taiko osu_taiko \
  --size 15m --steps 6000 --warmup 400 --batch-size 64 --max-dec-len 2048 --workers 9 \
  --log-every 100 --eval-every 500 --out /content/runs/$NAME --hub-repo JacobLinCool/urfm-runs \
  > $LOG/$NAME.log 2>&1
rc=$?; note "trained $NAME rc=$rc"
[ $rc -ne 0 ] && { note "abort $NAME"; exit $rc; }
python /content/urfm/experiments/eval_gen/evaluate.py --ckpt /content/runs/$NAME/best.pt \
  --data $DATA --sources taiko1000 --mechanics taiko --batch 128 --samples 16 \
  --out /content/runs/$NAME/eval > $LOG/$NAME.eval.log 2>&1
note "evaluated $NAME rc=$?"
