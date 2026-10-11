#!/usr/bin/env bash
# Round 2 fair transfer comparison: identical 15M model / steps / tiers / loader; only the
# training data differs.   bash chain_r2.sh <name> <sources...>
set -u
NAME=$1; shift
LOG=/content/logs
DATA=/content/data/urfm-ingest-v0
X=/content/urfm/experiments
mkdir -p $LOG
note() { echo "$(date -u +%FT%T) $*" >> $LOG/chain.log; }
note "start $NAME"
python -m urfm.train.train --data $DATA --sources "$@" --size 15m --steps 8000 --warmup 500 \
  --batch-size 64 --max-dec-len 2048 --workers 10 --log-every 100 --eval-every 1000 ${EXTRA:-} \
  --out /content/runs/$NAME --hub-repo JacobLinCool/urfm-runs > $LOG/$NAME.log 2>&1
rc=$?; note "trained $NAME rc=$rc"
[ $rc -ne 0 ] && { note "abort $NAME"; exit $rc; }
CK=/content/runs/$NAME/best.pt
python $X/eval_gen/evaluate.py --ckpt $CK --data $DATA --sources taiko1000 --mechanics taiko \
  --batch 128 --samples 16 --out /content/runs/$NAME/eval-taiko > $LOG/$NAME.eval-taiko.log 2>&1
note "eval-taiko $NAME rc=$?"
if [ "$*" != "taiko1000" ]; then
  python $X/eval_gen/evaluate.py --ckpt $CK --data $DATA --sources osu --max-songs 120 \
    --mechanics osu_taiko osu_mania osu_catch osu_std --batch 128 --samples 24 \
    --out /content/runs/$NAME/eval-osu > $LOG/$NAME.eval-osu.log 2>&1
  note "eval-osu $NAME rc=$?"
  python $X/crossgame/generate.py --ckpt $CK --data $DATA --songs 24 \
    --out /content/runs/$NAME/crossgame > $LOG/$NAME.crossgame.log 2>&1
fi
note "evaluated $NAME rc=$?"
