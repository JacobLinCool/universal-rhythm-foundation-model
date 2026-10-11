#!/usr/bin/env bash
# V1' (Taiko only, fixed pipeline) then V2 (Taiko + osu!taiko), each followed by test-set evaluation.
set -u
LOG=/content/logs
DATA=/content/data/urfm-ingest-v0
EVAL=/content/urfm/experiments/eval_gen/evaluate.py
while pgrep -f "[i]ngest.py osu" > /dev/null; do sleep 30; done
echo "$(date -u +%FT%T) ingest finished" >> $LOG/chain.log

train() {  # name, extra args...
  local name=$1; shift
  echo "$(date -u +%FT%T) start $name" >> $LOG/chain.log
  python -m urfm.train.train --data $DATA --batch-size 64 --workers 10 --log-every 100 \
    --eval-every 500 --out /content/runs/$name --hub-repo JacobLinCool/urfm-runs "$@" \
    > $LOG/$name.log 2>&1
  echo "$(date -u +%FT%T) trained $name rc=$?" >> $LOG/chain.log
  python $EVAL --ckpt /content/runs/$name/best.pt --data $DATA --sources taiko1000 \
    --mechanics taiko --batch 128 --samples 16 --out /content/runs/$name/eval \
    > $LOG/$name.eval.log 2>&1
  echo "$(date -u +%FT%T) evaluated $name rc=$?" >> $LOG/chain.log
}

train v1b-taiko --sources taiko1000 --size 10m --steps 6000 --warmup 300
train v2-taiko-osutaiko --sources taiko1000 osu --mechanics taiko osu_taiko --size 15m \
  --steps 8000 --warmup 400
