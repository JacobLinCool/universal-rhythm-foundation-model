#!/usr/bin/env bash
# VM-A after r6-base: full evaluation, then GRPO with and without the structure reward, each
# evaluated in full (SPEC.md). Waits for the base run to finish.
set -u
LOG=/content/logs; X=/content/urfm/experiments; R=/content/runs
note() { echo "$(date -u +%FT%T) $*" >> $LOG/chain.log; }
until grep -q "trained r6-base rc=0" $LOG/chain.log; do sleep 60; done
until python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('JacobLinCool/urfm-runs', 'r6-plan/best.pt', local_dir='$R')
" > $LOG/fetch-plan.log 2>&1; do sleep 300; done
note "planner fetched"
PL=$R/r6-plan/best.pt
NAME=r6-base CKPT=$R/r6-base/best.pt PLANNER=$PL LEVEL=full bash $X/r6/eval.sh
RLARGS="--steps ${RL_STEPS:-300} --prompts 16 --group 8 --lr 1e-5 --kl 0.05 --w-f1 0.25 \
  --w-density 0 --w-fast 0 --w-rhythm 0.5 --w-place 1 --w-plan 0.5 --w-stag 0.5 \
  --p-octave 0.5 --octave-factors 0.5 0.6 2 0.25 0.25 0.15 --fast-boost 2 \
  --hub-repo JacobLinCool/urfm-runs"
python -m urfm.train.rl --init $R/r6-base/best.pt $RLARGS --w-struct 1 --out $R/r6-rl \
  > $LOG/r6-rl.log 2>&1
note "trained r6-rl rc=$?"
NAME=r6-rl CKPT=$R/r6-rl/best.pt PLANNER=$PL LEVEL=full bash $X/r6/eval.sh
python -m urfm.train.rl --init $R/r6-base/best.pt $RLARGS --w-struct 0 --out $R/r6-rl-nostruct \
  > $LOG/r6-rl-nostruct.log 2>&1
note "trained r6-rl-nostruct rc=$?"
NAME=r6-rl-nostruct CKPT=$R/r6-rl-nostruct/best.pt PLANNER=$PL LEVEL=full bash $X/r6/eval.sh
note "chain_a done"
