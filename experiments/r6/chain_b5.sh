#!/usr/bin/env bash
# VM-B after chain_b4: r6-rl with planner v2 at plan temperature 0.7 and 1.0 (P1 audio, seed 0)
# -- does a less peaked plan reduce long monotone stretches?
set -u
LOG=/content/logs; DATA=/content/data/urfm-ingest-v0; X=/content/urfm/experiments; R=/content/runs
note() { echo "$(date -u +%FT%T) $*" >> $LOG/chain.log; }
until grep -q "chain_b4 done" $LOG/chain.log; do sleep 60; done
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('JacobLinCool/urfm-runs', 'r6-rl/best.pt', local_dir='$R')
" > $LOG/fetch-rl.log 2>&1
N=r6-rl
for t in 0.7 1.0; do
  python $X/eval_gen/evaluate.py --ckpt $R/$N/best.pt --data $DATA --sources taiko1000 --mechanics taiko \
    --batch 128 --samples 0 --planner $R/r6-plan2/best.pt --plan-temperature $t \
    --timing audio --beat-ckpt $R/beat-v1/best.pt --out $R/$N-p2/eval-audio-pt$t > $LOG/$N.p2.pt$t.log 2>&1
  note "diag $N plan2 temp $t rc=$?"
done
python -c "
from huggingface_hub import HfApi
HfApi().upload_folder(repo_id='JacobLinCool/urfm-runs', folder_path='$R/$N-p2', path_in_repo='$N-p2',
    allow_patterns=['eval-*/rows.jsonl', 'eval-*/report.md', 'eval-*/summary.json'])
" > /dev/null 2>&1
note "chain_b5 done"
