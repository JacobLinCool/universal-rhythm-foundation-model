#!/usr/bin/env bash
# Best-so-far recipe: all games + song context pre-training (r3) -> Taiko fine-tune -> t=0.6.
# Control: the Taiko-only continuation re-evaluated at t=0.6.
set -u
LOG=/content/logs; DATA=/content/data/urfm-ingest-v0; X=/content/urfm/experiments
mkdir -p $LOG
note() { echo "$(date -u +%FT%T) $*" >> $LOG/chain.log; }
python $X/eval_gen/evaluate.py --ckpt /content/runs/ft-r2-taiko/best.pt --data $DATA --sources taiko1000 \
  --mechanics taiko --batch 128 --samples 0 --temperature 0.6 --out /content/runs/ft-r2-taiko/eval-t0.6 \
  > $LOG/ctrl-t0.6.log 2>&1
note "control t0.6 rc=$?"
NAME=ft-r3-song
note "start $NAME"
python -m urfm.train.train --data $DATA --sources taiko1000 --size 15m --steps 4000 --warmup 200 \
  --lr 1e-4 --batch-size 64 --max-dec-len 2048 --workers 10 --log-every 100 --eval-every 500 \
  --song-context true --init /content/runs/r3-all-song/best.pt --out /content/runs/$NAME \
  --hub-repo JacobLinCool/urfm-runs > $LOG/$NAME.log 2>&1
rc=$?; note "trained $NAME rc=$rc"
[ $rc -ne 0 ] && { note "abort $NAME"; exit $rc; }
python $X/eval_gen/evaluate.py --ckpt /content/runs/$NAME/best.pt --data $DATA --sources taiko1000 \
  --mechanics taiko --batch 128 --samples 0 --temperature 0.6 --out /content/runs/$NAME/eval-t0.6 \
  > $LOG/$NAME.eval.log 2>&1
note "eval $NAME rc=$?"
cd $X/eval_gen && python export_playable.py --ckpt /content/runs/$NAME/best.pt --songs 6 \
  --tag "URFM R3+FT" --out /content/runs/$NAME/playable > $LOG/$NAME.export.log 2>&1
cd /content/runs/$NAME && tar czf /content/playable-best.tgz --exclude=song.ogg playable
note "evaluated $NAME rc=$?"
