#!/usr/bin/env bash
# R4 stage 1: data, canonicalisation statistics, beat model training, whole-song evaluation.
set -u
LOG=/content/logs; DATA=/content/data/urfm-ingest-v0; X=/content/urfm/experiments; R=/content/runs
mkdir -p $LOG $R/r4
note() { echo "$(date -u +%FT%T) $*" >> $LOG/chain.log; }
note "start beat chain"
if [ ! -d $DATA/data/osu ]; then
  HF_XET_HIGH_PERFORMANCE=1 python -c "from urfm.train.data import download_shards; download_shards()" \
    > $LOG/download.log 2>&1
  note "download rc=$?"
fi
python -m urfm.beat.train --data $DATA --out $R/beat-v1 --steps ${STEPS:-8000} --batch-size 32 \
  --workers 10 --hub-repo JacobLinCool/urfm-runs > $LOG/beat-v1.log 2>&1
rc=$?; note "beat train rc=$rc"
[ $rc -ne 0 ] && exit $rc
python $X/r4/eval_beat.py --ckpt $R/beat-v1/best.pt --data $DATA --split test \
  --out $R/beat-v1/eval-test > $LOG/beat-eval.log 2>&1
note "beat eval rc=$?"
python $X/r4/eval_beat.py --ckpt $R/beat-v1/best.pt --data $DATA --split test --warp \
  --out $R/beat-v1/eval-warp > $LOG/beat-eval-warp.log 2>&1
note "beat eval warp rc=$?"
python $X/r4/canon_stats.py --data $DATA --out $R/r4/canon.json > $LOG/canon.log 2>&1
note "canon rc=$?"
