#!/usr/bin/env bash
# R5 RL v2: rhythm and count-based fast rewards (v1 traded rhythm variety and fast passages for
# F1), from the same fast-boost fine-tune; then the evaluation panels.
set -u
LOG=/content/logs; DATA=/content/data/urfm-ingest-v0; X=/content/urfm/experiments; R=/content/runs
BEAT=$R/beat-v1/best.pt; NAME=${NAME:-r5-rl2}
note() { echo "$(date -u +%FT%T) $*" >> $LOG/chain.log; }
python -m urfm.train.rl --init $R/r5-sft/best.pt --steps ${RL_STEPS:-150} --prompts 16 --group 8 \
  --lr 1e-5 --kl ${KL:-0.05} --out $R/$NAME --hub-repo JacobLinCool/urfm-runs > $LOG/$NAME.log 2>&1
note "trained $NAME rc=$?"
EVA="python $X/eval_gen/evaluate.py --ckpt $R/$NAME/best.pt --data $DATA --sources taiko1000 --mechanics taiko --batch 128 --samples 0"
$EVA --timing audio --beat-ckpt $BEAT --out $R/$NAME/eval-taiko-audio > $LOG/$NAME.eval-audio.log 2>&1
note "eval-audio $NAME rc=$?"
$EVA --out $R/$NAME/eval-taiko > $LOG/$NAME.eval-taiko.log 2>&1
note "eval-taiko $NAME rc=$?"
$EVA --speed 1.25 --out $R/$NAME/eval-taiko-x1.25 > $LOG/$NAME.eval-x125.log 2>&1
note "eval-x1.25 $NAME rc=$?"
python $X/r4/diag_fine.py --ckpt $R/$NAME/best.pt --data-files 2 --out $R/$NAME/diag-fine > $LOG/$NAME.diag-fine.log 2>&1
note "diag-fine $NAME rc=$?"
python $X/eval_gen/evaluate.py --ckpt $R/$NAME/best.pt --data $DATA --sources osu --max-songs 120 \
  --mechanics osu_taiko osu_mania osu_catch osu_std --batch 128 --samples 0 \
  --out $R/$NAME/eval-osu > $LOG/$NAME.eval-osu.log 2>&1
note "eval-osu $NAME rc=$?"
