#!/usr/bin/env bash
# R5: fast-passage oversampling fine-tune, then GRPO on the 15M model; evaluations.
set -u
LOG=/content/logs; DATA=/content/data/urfm-ingest-v0; X=/content/urfm/experiments; R=/content/runs
BEAT=$R/beat-v1/best.pt
mkdir -p $LOG $R
note() { echo "$(date -u +%FT%T) $*" >> $LOG/chain.log; }
if [ ! -d $DATA/data/osu ]; then
  HF_XET_HIGH_PERFORMANCE=1 python -c "from urfm.train.data import download_shards; download_shards()" > $LOG/download.log 2>&1
fi
python -c "
from huggingface_hub import snapshot_download, hf_hub_download
snapshot_download('JacobLinCool/urfm-timing-v1', repo_type='dataset', local_dir='/content/data/urfm-timing-v1')
for f in ('r4a-ft-pred/best.pt', 'beat-v1/best.pt'):
    hf_hub_download('JacobLinCool/urfm-runs', f, local_dir='$R')
" > $LOG/fetch.log 2>&1
note "fetched rc=$?"
EVA="python $X/eval_gen/evaluate.py --data $DATA --sources taiko1000 --mechanics taiko --batch 128 --samples 0"
# 0. baseline with the new fast-interval metric
$EVA --ckpt $R/r4a-ft-pred/best.pt --timing audio --beat-ckpt $BEAT --out $R/r4a-ft-pred/eval-taiko-audio-v2 > $LOG/base.eval-audio.log 2>&1
note "eval-audio baseline rc=$?"
# 1. supervised fine-tune with fast passages oversampled
python -m urfm.train.train --data $DATA --sources taiko1000 osu --size 15m --steps 2000 --warmup 100 \
  --lr 1e-4 --batch-size 64 --max-dec-len 2048 --workers 10 --log-every 100 --eval-every 500 \
  --song-context true --cond true --texture true --canonical true --taiko-boost 2.0 \
  --p-speed 0.5 --p-local 0.3 --p-octave 0.15 --predicted-timing /content/data/urfm-timing-v1 \
  --p-predicted 0.3 --fast-boost 4 --init $R/r4a-ft-pred/best.pt \
  --out $R/r5-sft --hub-repo JacobLinCool/urfm-runs > $LOG/r5-sft.log 2>&1
note "trained r5-sft rc=$?"
$EVA --ckpt $R/r5-sft/best.pt --timing audio --beat-ckpt $BEAT --out $R/r5-sft/eval-taiko-audio > $LOG/r5-sft.eval-audio.log 2>&1
note "eval-audio r5-sft rc=$?"
# 2. GRPO from the fine-tuned model
python -m urfm.train.rl --init $R/r5-sft/best.pt --steps ${RL_STEPS:-150} --prompts 16 --group 8 \
  --lr 1e-5 --kl 0.05 --out $R/r5-rl --hub-repo JacobLinCool/urfm-runs > $LOG/r5-rl.log 2>&1
note "trained r5-rl rc=$?"
$EVA --ckpt $R/r5-rl/best.pt --timing audio --beat-ckpt $BEAT --out $R/r5-rl/eval-taiko-audio > $LOG/r5-rl.eval-audio.log 2>&1
note "eval-audio r5-rl rc=$?"
$EVA --ckpt $R/r5-rl/best.pt --out $R/r5-rl/eval-taiko > $LOG/r5-rl.eval-taiko.log 2>&1
note "eval-taiko r5-rl rc=$?"
$EVA --ckpt $R/r5-rl/best.pt --speed 1.25 --out $R/r5-rl/eval-taiko-x1.25 > $LOG/r5-rl.eval-x125.log 2>&1
note "eval-x1.25 r5-rl rc=$?"
for ck in r5-sft r5-rl; do
  python $X/r4/diag_fine.py --ckpt $R/$ck/best.pt --data-files 2 --out $R/$ck/diag-fine > $LOG/$ck.diag-fine.log 2>&1
  note "diag-fine $ck rc=$?"
done
python $X/eval_gen/evaluate.py --ckpt $R/r5-rl/best.pt --data $DATA --sources osu --max-songs 120 \
  --mechanics osu_taiko osu_mania osu_catch osu_std --batch 128 --samples 0 \
  --out $R/r5-rl/eval-osu > $LOG/r5-rl.eval-osu.log 2>&1
note "eval-osu r5-rl rc=$?"
