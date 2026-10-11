#!/usr/bin/env bash
# VM-B after queue_b: R3 on the osu! panel (P6) and every audio-timing panel again with the
# final (DP) beat decoding, for R3 and r4a/r4b.
set -u
LOG=/content/logs; DATA=/content/data/urfm-ingest-v0; X=/content/urfm/experiments; R=/content/runs
note() { echo "$(date -u +%FT%T) $*" >> $LOG/chain.log; }
sleep 60; while pgrep -f "[q]ueue_b.sh" > /dev/null; do sleep 30; done
python $X/eval_gen/evaluate.py --ckpt $R/hub/ft-r3-song/best.pt --data $DATA --sources osu --max-songs 120 \
  --mechanics osu_taiko osu_mania osu_catch osu_std --batch 128 --samples 0 \
  --out $R/r3-base/eval-osu > $LOG/r3.eval-osu.log 2>&1
note "r3 eval-osu rc=$?"
for run in r4a-full-15m r4b-noaug-15m; do
  python $X/eval_gen/evaluate.py --ckpt $R/$run/best.pt --data $DATA --sources taiko1000 --mechanics taiko \
    --batch 128 --samples 0 --timing audio --beat-ckpt $R/beat-v1/best.pt \
    --out $R/$run/eval-taiko-audio-dp > $LOG/$run.eval-audio-dp.log 2>&1
  note "eval-audio-dp $run rc=$?"
done
