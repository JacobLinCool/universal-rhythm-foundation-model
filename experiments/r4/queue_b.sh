#!/usr/bin/env bash
# VM-B after r4a: no-augmentation ablation, then the R3 baseline on the R4 panels.
set -u
LOG=/content/logs; DATA=/content/data/urfm-ingest-v0; X=/content/urfm/experiments; R=/content/runs
note() { echo "$(date -u +%FT%T) $*" >> $LOG/chain.log; }
while pgrep -f "[c]hain_r4.sh" > /dev/null; do sleep 30; done
NAME=r4b-noaug-15m SIZE=15m STEPS=8000 EXTRA="" bash $X/r4/chain_r4.sh
python -c "
from huggingface_hub import hf_hub_download
print(hf_hub_download('JacobLinCool/urfm-runs', 'ft-r3-song/best.pt', local_dir='$R/hub'))" \
  > $LOG/r3-download.log 2>&1
CK=$R/hub/ft-r3-song/best.pt
EV="python $X/eval_gen/evaluate.py --ckpt $CK --data $DATA --sources taiko1000 --mechanics taiko --batch 128 --samples 0"
$EV --out $R/r3-base/eval-taiko > $LOG/r3.eval-taiko.log 2>&1; note "r3 eval-taiko rc=$?"
$EV --timing audio --beat-ckpt $R/beat-v1/best.pt --out $R/r3-base/eval-taiko-audio > $LOG/r3.eval-audio.log 2>&1
note "r3 eval-audio rc=$?"
$EV --speed 1.25 --out $R/r3-base/eval-taiko-x1.25 > $LOG/r3.eval-x125.log 2>&1; note "r3 eval-x1.25 rc=$?"
