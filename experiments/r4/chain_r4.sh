#!/usr/bin/env bash
# R4 note model: train, then evaluate (P1 oracle timing, P1 audio timing, P4 speed 1.25,
# P6 osu!).   NAME=... SIZE=15m STEPS=8000 EXTRA="--p-speed 0.5 ..." bash chain_r4.sh
set -u
LOG=/content/logs; DATA=/content/data/urfm-ingest-v0; X=/content/urfm/experiments; R=/content/runs
BEAT=${BEAT:-$R/beat-v1/best.pt}
mkdir -p $LOG
note() { echo "$(date -u +%FT%T) $*" >> $LOG/chain.log; }
if [ ! -d $DATA/data/osu ]; then
  HF_XET_HIGH_PERFORMANCE=1 python -c "from urfm.train.data import download_shards; download_shards()" \
    > $LOG/download.log 2>&1
  note "download rc=$?"
fi
note "start $NAME"
python -m urfm.train.train --data $DATA --sources taiko1000 osu --size ${SIZE:-15m} \
  --steps ${STEPS:-8000} --warmup 500 --batch-size ${BATCH:-64} --max-dec-len 2048 --workers 10 \
  --log-every 100 --eval-every 1000 --song-context true --cond true --texture true \
  --canonical true --taiko-boost 2.0 ${EXTRA:-} \
  --out $R/$NAME --hub-repo JacobLinCool/urfm-runs > $LOG/$NAME.log 2>&1
rc=$?; note "trained $NAME rc=$rc"
[ $rc -ne 0 ] && exit $rc
CK=$R/$NAME/best.pt
EV="python $X/eval_gen/evaluate.py --ckpt $CK --data $DATA --sources taiko1000 --mechanics taiko --batch 128"
$EV --samples 16 --out $R/$NAME/eval-taiko > $LOG/$NAME.eval-taiko.log 2>&1
note "eval-taiko $NAME rc=$?"
if [ -f $BEAT ]; then
  $EV --samples 16 --timing audio --beat-ckpt $BEAT --out $R/$NAME/eval-taiko-audio \
    > $LOG/$NAME.eval-taiko-audio.log 2>&1
  note "eval-taiko-audio $NAME rc=$?"
fi
$EV --samples 8 --speed 1.25 --out $R/$NAME/eval-taiko-x1.25 > $LOG/$NAME.eval-x125.log 2>&1
note "eval-x1.25 $NAME rc=$?"
python $X/eval_gen/evaluate.py --ckpt $CK --data $DATA --sources osu --max-songs 120 \
  --mechanics osu_taiko osu_mania osu_catch osu_std --batch 128 --samples 8 \
  --out $R/$NAME/eval-osu > $LOG/$NAME.eval-osu.log 2>&1
note "eval-osu $NAME rc=$?"
