#!/usr/bin/env bash
# V3: all games with audio (Taiko + osu! std/taiko/catch/mania) at one model size, then
# evaluation on Taiko and osu! test songs and cross-game generation.
#   bash chain_v3.sh <size> <steps> <batch> [wait-for-pattern-in-chain.log]
set -u
SIZE=$1; STEPS=$2; BATCH=$3; WAIT=${4:-}
NAME=v3-all-$SIZE
LOG=/content/logs
DATA=/content/data/urfm-ingest-v0
X=/content/urfm/experiments
note() { echo "$(date -u +%FT%T) $*" >> $LOG/chain.log; }
if [ -n "$WAIT" ]; then
  until grep -q "$WAIT" $LOG/chain.log 2>/dev/null; do sleep 30; done
fi
note "start $NAME"
python -m urfm.train.train --data $DATA --sources taiko1000 osu --size $SIZE --steps $STEPS \
  --warmup 500 --batch-size $BATCH --max-dec-len 2048 --workers 9 --log-every 100 \
  --eval-every 1000 --out /content/runs/$NAME --hub-repo JacobLinCool/urfm-runs > $LOG/$NAME.log 2>&1
rc=$?; note "trained $NAME rc=$rc"
[ $rc -ne 0 ] && { note "abort $NAME"; exit $rc; }
CK=/content/runs/$NAME/best.pt
python $X/eval_gen/evaluate.py --ckpt $CK --data $DATA --sources taiko1000 --mechanics taiko \
  --batch 128 --samples 16 --out /content/runs/$NAME/eval-taiko > $LOG/$NAME.eval-taiko.log 2>&1
note "eval-taiko $NAME rc=$?"
python $X/eval_gen/evaluate.py --ckpt $CK --data $DATA --sources osu --max-songs 120 \
  --mechanics osu_taiko osu_mania osu_catch osu_std --batch 128 --samples 24 \
  --out /content/runs/$NAME/eval-osu > $LOG/$NAME.eval-osu.log 2>&1
note "eval-osu $NAME rc=$?"
python $X/crossgame/generate.py --ckpt $CK --data $DATA --songs 24 \
  --out /content/runs/$NAME/crossgame > $LOG/$NAME.crossgame.log 2>&1
note "evaluated $NAME rc=$?"
