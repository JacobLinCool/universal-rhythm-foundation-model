#!/usr/bin/env bash
# Diagnostic: does sampling the section plan (temperature) cost section contrast? r6-a10k,
# P1 audio timing, seed 0, plan temperature 0 and 0.3 (the ablation eval used 0.7).
set -u
LOG=/content/logs; DATA=/content/data/urfm-ingest-v0; X=/content/urfm/experiments; R=/content/runs
N=${N:-r6-a10k}
for t in 0 0.3; do
  python $X/eval_gen/evaluate.py --ckpt $R/$N/best.pt --data $DATA --sources taiko1000 --mechanics taiko \
    --batch 128 --samples 0 --planner $R/r6-plan/best.pt --plan-temperature $t \
    --timing audio --beat-ckpt $R/beat-v1/best.pt --out $R/$N/eval-audio-pt$t > $LOG/$N.pt$t.log 2>&1
  echo "$(date -u +%FT%T) diag plan-temp $t $N rc=$?" >> $LOG/chain.log
done
python -c "
from huggingface_hub import HfApi
HfApi().upload_folder(repo_id='JacobLinCool/urfm-runs', folder_path='$R/$N', path_in_repo='$N',
    allow_patterns=['eval-*/rows.jsonl', 'eval-*/report.md', 'eval-*/summary.json'])
" > /dev/null 2>&1
