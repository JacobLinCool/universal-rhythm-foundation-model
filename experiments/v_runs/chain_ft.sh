#!/usr/bin/env bash
# Pretraining effect on Taiko: fine-tune r2-all (all games) and r2-taiko (Taiko only) with the
# identical recipe, so total steps and recipe match and only the first 8k steps differ.
set -u
LOG=/content/logs; DATA=/content/data/urfm-ingest-v0; X=/content/urfm/experiments
mkdir -p $LOG
note() { echo "$(date -u +%FT%T) $*" >> $LOG/chain.log; }
python -c "from huggingface_hub import hf_hub_download; import shutil, os; os.makedirs('/content/runs/r2-taiko', exist_ok=True); shutil.copy(hf_hub_download('JacobLinCool/urfm-runs', 'r2-taiko/best.pt'), '/content/runs/r2-taiko/best.pt')"
for base in r2-all r2-taiko; do
  NAME=ft-$base
  note "start $NAME"
  python -m urfm.train.train --data $DATA --sources taiko1000 --size 15m --steps 4000 --warmup 200 \
    --lr 1e-4 --batch-size 64 --max-dec-len 2048 --workers 10 --log-every 100 --eval-every 500 \
    --init /content/runs/$base/best.pt --out /content/runs/$NAME --hub-repo JacobLinCool/urfm-runs \
    > $LOG/$NAME.log 2>&1
  rc=$?; note "trained $NAME rc=$rc"
  [ $rc -ne 0 ] && { note "abort $NAME"; exit $rc; }
  python $X/eval_gen/evaluate.py --ckpt /content/runs/$NAME/best.pt --data $DATA --sources taiko1000 \
    --mechanics taiko --batch 128 --samples 16 --out /content/runs/$NAME/eval-taiko > $LOG/$NAME.eval.log 2>&1
  note "evaluated $NAME rc=$?"
done
