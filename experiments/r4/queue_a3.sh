#!/usr/bin/env bash
# VM-A: the 45M R4 model (augmentation on; predicted-grid fine-tuning comes later).
# Batch 16 x 4 accumulation: 64 or 32 at 2048 decoder tokens does not fit the L4 at 45M.
set -u
X=/content/urfm/experiments
rm -rf /content/runs/r4-45m
NAME=r4-45m SIZE=45m STEPS=16000 BATCH=16 \
  EXTRA="--grad-accum 4 --warmup 1000 --p-speed 0.5 --p-local 0.3 --p-octave 0.15" \
  bash $X/r4/chain_r4.sh
