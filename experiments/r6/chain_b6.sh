#!/usr/bin/env bash
# VM-B after chain_b5: r6-rl + planner v2 with decoder temperature 1.0 (P1 audio, seed 0).
set -u
LOG=/content/logs; DATA=/content/data/urfm-ingest-v0; X=/content/urfm/experiments; R=/content/runs
until grep -q "chain_b5 done" $LOG/chain.log; do sleep 30; done
N=r6-rl
python $X/eval_gen/evaluate.py --ckpt $R/$N/best.pt --data $DATA --sources taiko1000 --mechanics taiko \
  --batch 128 --samples 0 --planner $R/r6-plan2/best.pt --plan-temperature 0.7 --temperature 1.0 \
  --timing audio --beat-ckpt $R/beat-v1/best.pt --out $R/$N-p2/eval-audio-t1.0 > $LOG/$N.t1.log 2>&1
echo "$(date -u +%FT%T) diag $N plan2 decoder-temp 1.0 rc=$?" >> $LOG/chain.log
python -c "
from huggingface_hub import HfApi
HfApi().upload_folder(repo_id='JacobLinCool/urfm-runs', folder_path='$R/$N-p2', path_in_repo='$N-p2',
    allow_patterns=['eval-*/rows.jsonl', 'eval-*/report.md', 'eval-*/summary.json'])
" > /dev/null 2>&1
echo "$(date -u +%FT%T) chain_b6 done" >> $LOG/chain.log
