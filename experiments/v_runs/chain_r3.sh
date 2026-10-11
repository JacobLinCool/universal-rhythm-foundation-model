#!/usr/bin/env bash
# (1) decoding temperature sweep on the pretrained+fine-tuned checkpoint (eval only);
# (2) R3: all games with song-level context, compared with R2-all.
set -u
LOG=/content/logs; DATA=/content/data/urfm-ingest-v0; X=/content/urfm/experiments
mkdir -p $LOG
note() { echo "$(date -u +%FT%T) $*" >> $LOG/chain.log; }
for t in 0.3 0.6; do
  python $X/eval_gen/evaluate.py --ckpt /content/runs/ft-r2-all/best.pt --data $DATA --sources taiko1000 \
    --mechanics taiko --batch 128 --samples 0 --temperature $t --out /content/runs/ft-r2-all/eval-t$t \
    > $LOG/sweep-t$t.log 2>&1
  note "sweep t=$t rc=$?"
done
EXTRA="--song-context true" bash $X/v_runs/chain_r2.sh r3-all-song taiko1000 osu
