#!/usr/bin/env bash
# R6 evaluation of one checkpoint (SPEC.md).
#   NAME=r6-base CKPT=/content/runs/r6-base/best.pt PLANNER=/content/runs/r6-plan/best.pt \
#   LEVEL=full|ablation bash eval.sh
# full: P1 audio seeds 0-2, P1 oracle, P7 x1/2 and x2, P6 osu!; ablation: P1 audio seed 0, P7 x1/2.
set -u
LOG=/content/logs; DATA=/content/data/urfm-ingest-v0; X=/content/urfm/experiments; R=/content/runs
BEAT=$R/beat-v1/best.pt
note() { echo "$(date -u +%FT%T) $*" >> $LOG/chain.log; up; }
# evaluation results go to the Hub as soon as they exist (runtimes can disappear)
up() { python -c "
from huggingface_hub import HfApi
HfApi().upload_folder(repo_id='JacobLinCool/urfm-runs', folder_path='$R/$NAME', path_in_repo='$NAME',
    allow_patterns=['eval-*/rows.jsonl', 'eval-*/report.md', 'eval-*/summary.json'])
" > /dev/null 2>&1 || true; }
[ -f $BEAT ] || python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('JacobLinCool/urfm-runs', 'beat-v1/best.pt', local_dir='$R')
" > $LOG/fetch-beat.log 2>&1
P=""; [ -n "${PLANNER:-}" ] && P="--planner $PLANNER"
EV="python $X/eval_gen/evaluate.py --ckpt $CKPT --data $DATA --sources taiko1000 --mechanics taiko --batch 128 --samples 8 $P --temperature ${TEMP:-0.85} --plan-temperature ${PT:-0.7}"
O=$R/$NAME
if [ "${LEVEL:-full}" = audio3 ]; then
  $EV --timing audio --beat-ckpt $BEAT --seeds 0 1 2 --out $O/eval-audio${TAG:-} > $LOG/$NAME.eval-audio${TAG:-}.log 2>&1
  note "eval-audio${TAG:-} $NAME rc=$?"
  $EV --notation 0.5 --out $O/eval-notation-x0.5${TAG:-} > $LOG/$NAME.eval-x0.5${TAG:-}.log 2>&1
  note "eval-notation-x0.5${TAG:-} $NAME rc=$?"
elif [ "${LEVEL:-full}" = full ]; then
  $EV --timing audio --beat-ckpt $BEAT --seeds 0 1 2 --out $O/eval-audio > $LOG/$NAME.eval-audio.log 2>&1
  note "eval-audio $NAME rc=$?"
  $EV --out $O/eval-oracle > $LOG/$NAME.eval-oracle.log 2>&1
  note "eval-oracle $NAME rc=$?"
  for f in 0.5 2; do
    $EV --notation $f --out $O/eval-notation-x$f > $LOG/$NAME.eval-x$f.log 2>&1
    note "eval-notation-x$f $NAME rc=$?"
  done
  python $X/eval_gen/evaluate.py --ckpt $CKPT --data $DATA --sources osu --max-songs 120 \
    --mechanics osu_taiko osu_mania osu_catch osu_std --batch 128 --samples 4 $P \
    --out $O/eval-osu > $LOG/$NAME.eval-osu.log 2>&1
  note "eval-osu $NAME rc=$?"
else
  $EV --timing audio --beat-ckpt $BEAT --out $O/eval-audio > $LOG/$NAME.eval-audio.log 2>&1
  note "eval-audio $NAME rc=$?"
  $EV --notation 0.5 --out $O/eval-notation-x0.5 > $LOG/$NAME.eval-x0.5.log 2>&1
  note "eval-notation-x0.5 $NAME rc=$?"
fi
