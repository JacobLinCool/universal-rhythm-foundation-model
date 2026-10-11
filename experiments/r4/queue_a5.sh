#!/usr/bin/env bash
# VM-A: final multi-seed comparison on P1 with audio timing (seeds 1 and 2; seed 0 exists).
set -u
LOG=/content/logs; DATA=/content/data/urfm-ingest-v0; X=/content/urfm/experiments; R=/content/runs
note() { echo "$(date -u +%FT%T) $*" >> $LOG/chain.log; }
sleep 60; while pgrep -f "[q]ueue_a4.sh" > /dev/null; do sleep 60; done
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('JacobLinCool/urfm-runs', 'ft-r3-song/best.pt', local_dir='$R/hub')" > $LOG/r3-download.log 2>&1
for pair in "r3-base:$R/hub/ft-r3-song/best.pt" "r4a-full-15m:$R/r4a-full-15m/best.pt" "r4a-ft-pred:$R/r4a-ft-pred/best.pt"; do
  name=${pair%%:*}; ck=${pair#*:}
  python $X/eval_gen/evaluate.py --ckpt $ck --data $DATA --sources taiko1000 --mechanics taiko --batch 128 \
    --samples 0 --timing audio --beat-ckpt $R/beat-v1/best.pt --seeds 1 2 \
    --out $R/$name/eval-taiko-audio-s12 > $LOG/$name.audio-s12.log 2>&1
  note "audio-s12 $name rc=$?"
done
