# Model, data pipeline and training (v0)

The v0 system is one audio-conditioned encoder-decoder for all games. Game, input space,
difficulty and density are given as prefix tokens. Every event's tokens put the shared
interaction part first and the game-specific action after it
([tokenize/api.py](../../src/urfm/tokenize/api.py)).

## Inputs

| Input | Where it comes from | Shape |
| --- | --- | --- |
| log-mel | ingest (`urfm.data.features`, `mel-v1`): 22.05 kHz, hop 256 (86.13 fps), 128 bins, uint8 dB | `[1380, 128]` per 16 s window |
| timing features | `tokenizer.frame_features(timing, start_us, n)`: beat/measure phase, tempo from the chart's TimingMap (ground truth in training; beat tracker or known BPM at inference) | `[1380, F]` |
| prefix | `tokenizer.prefix`: `GAME`, `LANES` (lane games), `TIER`, `DENS` | 3–4 tokens |
| context | the previous events before the window, re-encoded (up to `CONTEXT_EVENTS`), then `SEP` | ≤ ~200 tokens |
| target | the window's events, then `EOS`; loss is computed only here | ≤ `max_dec_len` |

Windows come from `urfm.tokenize.windows()`. They start on a measure boundary and end on the
last measure boundary that is at most 16 s later. Training and generation tile songs the same
way, so every event belongs to exactly one window.

## Architecture (`urfm.model.ChartModel`)

```
mel uint8 ─dequantize (GPU)─► Conv1d(128→d,k3) ─GELU─► Conv1d(d→d,k3,s2) ─GELU─┐
timing feats [T,F] ─pairwise concat→[T/2,2F]─► Linear(2F→d) ─────────────────(+)── + learned pos
                                                                                │
                                                     N_enc × pre-LN self-attn/FFN ─► memory
tokens ─embed (optionally factorized)+ learned pos─► N_dec × [causal self-attn, cross-attn, FFN]
                                                     ─► LN ─► tied output projection
```

- All attention uses `F.scaled_dot_product_attention` with head dimension 64, so flash and
  efficient kernels apply under bf16.
- Positions are learned and absolute. Encoder positions mark time inside the window; the
  timing features carry the beat grid.
- `memory_tokens > 0` appends learned memory slots to the encoder output. This is the hook for
  song-level context, which v0 does not use.
- Incremental decoding works in two steps:
  - `prefill` handles left-padded prompts and fills a preallocated KV cache.
  - `step` decodes one token per row.
  - Cross-attention K/V are projected once per window.

| Preset | d | enc | dec | heads | Params |
| --- | --- | --- | --- | --- | --- |
| `5m` | 256 | 3 | 2 | 4 | 5.4M |
| `10m` | 320 | 4 | 3 | 5 | 11.1M |
| `15m` | 384 | 4 | 3 | 6 | 15.8M |
| `45m` | 512 | 6 | 6 | 8 | 46.3M |

Parameter counts assume a vocabulary of 801. Embeddings are under 2% of every preset; the rest
is Transformer layers.

## Data pipeline (`urfm.train.data`)

1. **Index pass** (`build_index`): reads only the `chartset` and `duration_s` columns. Each
   chart gets a weight such that mechanics *m* receives a share of windows ∝ `n_m^alpha`
   (default `alpha = 0.5`). This keeps osu!standard from dominating.
2. **`WindowStream`** (an `IterableDataset`):
   - Each DataLoader worker owns a disjoint subset of Parquet row groups and loops over them
     in shuffled order. Row groups hold 64 songs.
   - Each visited chart is prepared once (`prepare`: prefix, the tokenizer's audio-wide
     windows, events sorted by onset); each window then encodes only the events in its
     range, so a window costs O(events in window) (~1 ms).
   - For each chart in a loaded group, it draws `floor(weight + U)` random windows.
   - Mel windows are zero-copy views of the uint8 array.
   - Failures are counted per mechanics (`PIPELINE`) and shipped with each batch as
     `stats`; a full pass that produces no sample raises instead of looping forever.
3. **Batching** happens in the worker:
   - A buffer of `8 × batch_size` samples is sorted by token length, so a batch holds samples
     of similar length.
   - Each batch is padded to a fixed bucket (128 / 256 / 384 / 512 / 768 / 1024). This keeps
     shapes static for `torch.compile`.
   - Batches are shuffled before they are yielded.
4. **Transfer to the GPU**: `DataLoader(batch_size=None, pin_memory, persistent_workers,
   prefetch_factor=4)`. Batches move to the GPU with `non_blocking=True`. Mel crosses
   host→device as uint8, half the bytes of fp16, and is dequantized on the GPU.
5. **Truncation**: if a window exceeds `max_dec_len`, the context is dropped first (whole),
   then the body is cut at an event boundary (`encode_window`). The rate is logged overall
   (`truncated_frac`) and per mechanics (`pipeline["<mechanics>/truncated_frac"]`).
6. **Index cache**: `build_index` indexes shards in parallel processes and caches each
   shard's index next to it, so later runs start in seconds.

## Training (`python -m urfm.train.train`)

- **Optimizer and schedule**: AdamW (fused on CUDA, β = (0.9, 0.95), no weight decay on
  positions, norms or biases), linear warmup then cosine decay to `min_lr_ratio`, gradient
  clipping, optional gradient accumulation.
- **Precision and compilation**: bf16 autocast, TF32 matmuls, `torch.compile(model.loss)`
  (one graph per decoder bucket).
- **Metrics**: kept on the GPU and synchronised every `log_every` steps. Each line of
  `<out>/metrics.jsonl` records:
  - `loss`, `acc`, `lr`
  - `samples_per_s`, `target_tokens_per_s`
  - `data_wait_frac`: the share of wall time spent waiting for the loader; it should stay
    well under 0.05
  - `truncated_frac`
  - `gpu_util` (pynvml or nvidia-smi sampled every second), `gpu_mem_gb`
  - `val_loss`, `val_acc` every `eval_every` steps, on a fixed set of validation windows
- **Checkpoints**: `<out>/latest.pt` holds the model state, model config and run metadata
  (train config, parameter counts, vocabulary hash, chart counts). `--hub-repo` uploads it
  together with the metrics to a private model repo from a background thread.

## Song context (`--song-context true`, off by default)

A 16 s window plus 32 context events can join windows smoothly, but cannot plan a whole song:
repeating a chorus motif, shaping the density arc, knowing where it is in the song. With
`ModelConfig.song_context`, every window also sees a summary of the whole song.

**Inputs** (`urfm.train.data`):

- `song_summary(mel)` mean-pools the record's uint8 mel into blocks of `SONG_POOL` = 172
  frames (~2 s) and returns `[S, 128]` uint8 blocks plus their start times. It is computed once
  per prepared chart, or once per generation `Job`.
- Songs longer than `SONG_MAX` = 256 blocks (~8.5 min) keep uniformly spaced blocks.
- `song_marker(...)` marks the blocks overlapping the current window, so the model knows where
  it is in the song.
- Batches are padded to a fixed `[B, 256, 128]` and also carry `song_marker` and `song_len`.
  Shapes stay static for `torch.compile`, and the inputs cross host→device as uint8.

**Model** (`ChartModel.encode_song`, `ChartModel.memory_for`):

- Per block: Linear(128 → d) over dequantized dB, a learned position, and a marker embedding.
- `song_layers` = 2 pre-LN encoder blocks with a key padding mask.
- The outputs are appended to the window memory, each side with a learned source embedding.
- Decoder cross-attention, the prefill path and the KV-cache path all use the same memory key
  mask, so padded blocks never receive attention.

**Compatibility.** Disabled models have exactly the original parameters (tested), so earlier
checkpoints still load and behave the same. The batches always carry the small song tensors;
models without song context ignore them.

**Cost** (15m preset, CPU, B=8, decoder length 512):

| | Parameters | Training step |
| --- | --- | --- |
| Off | 16.15M | 0.64 s |
| On | 19.85M (+3.70M) | 0.70 s (+8%) |

The extra cross-attention covers 256 more keys.

### Texture units (`--texture true`, R4)

Mean-pooled mel says how loud and bright a stretch is, not what rhythm it has, so it could not
tell a 16th-note verse from an 8th-note chorus. With `ModelConfig.song_feat_dim`, song units
are beat-synchronous (`urfm.data.texture.song_texture`): one unit per 2 beats of the timing
map, holding the spectral flux (rise in dB) in 4 bands at 24 positions per beat (16ths, 32nds
and triplets) plus the mean mel in 32 bands, 224 float16 values per unit, at most 512 units.
Positions are in beats, so a song played faster has the same units at other times: tempo
augmentation only moves them.

## Continuous conditions (`--cond true`, R4)

`ModelConfig.cond_dim` conditions enter through `CondEmbed` (see the tokenizer document for the
scales): an MLP over the masked linear and slow Fourier features of each z-score, whose output
replaces the embedding of the `COND` prefix token (`ModelConfig.cond_token`). Training batches
carry `cond [B, 7]` and `cond_mask [B, 7]`; generation jobs carry `Job.cond`.

## Augmentation (R4, `urfm.train.data.augment_chart`)

Applied per chart visit, in this order:

1. **Canonical tempo** (`--canonical`): visual tempo tricks folded into scroll
   (`urfm.data.canonical`, [timing.md](timing.md)).
2. **Predicted grid** (`--p-predicted`, with `--predicted-timing`): the chart is re-notated on
   the audio-predicted timing map of its song (only songs whose notes fit that grid), with
   scroll values recomputed to keep on-screen speeds; this is the grid generation will get.
3. **BPM octave** (`--p-octave`): the same chart written at double or half BPM.
4. **Tempo warp** (`--p-speed`, `--p-local`): the song 0.8–1.6x faster (log-uniform), and up to
   six beat spans of half a beat to a measure at their own speed (0.8–1.25x). Beats do not move;
   the timing map changes and the mel is read through the inverse warp
   (`urfm.ir.transform.warp_timing`). Tempos stay below 420 BPM so windows fit `MAX_BEATS`.

## Song planner and structure memory (R6)

- **Planner** (`urfm.model.planner`, trained by `python -m urfm.train.planner`): a bidirectional
  Transformer (d 192, 2 layers, 1.07M parameters) over every section of a song. Input per
  section: `urfm.data.sections.section_audio` (onset envelope in 4 bands folded onto one beat at
  48 phases, band means and maxima, 32 mel bands, log tempo and section length), plus the game
  embedding and the condition embedding. Output: the five section-plan values. It is trained
  on human charts with the decoder's augmentation, so it also learns which subdivision a passage
  needs on whatever grid it is given. Section contrast is decided here, with the whole song in
  view.
- **Decoder** (`--sections true --ref true`): the window's plan comes from the planner; the
  structure-memory block gives it an earlier section to repeat. `--ctx-noise 0.5` corrupts
  context and memory in half the samples (drop 10%, don/ka swap 10%, 16th shift 5%) because at
  inference both are the model's own output.
- **Size**: `--song-layers 1` keeps decoder (18.53M) plus planner under 20M.
- **Warm start**: `--init ckpt --init-partial true` copies equal-shaped tensors and the leading
  rows of tensors that grew (token embeddings).
- **Octave policy**: `--octave-factors 0.5 .6 2 .25 0.25 .15` re-notates at half, double or
  quarter BPM (within 40–336 BPM), so 32nds, 48ths and 64ths appear in training.

## Generation (`urfm.train.generate`)

```python
from urfm.train.generate import Job, generate, to_chartset, write_native
from urfm.train.train import load_checkpoint

model, meta = load_checkpoint("latest.pt", device="cuda")
job = Job(mel=mel_uint8, timing=timing, mechanics="taiko", prefix=tok.prefix(cs, reference_chart))
(chart,) = generate(model, tok, [job], temperature=0.9, top_p=0.95, device="cuda")
write_native(to_chartset(chart, timing), "out.tja")
```

- **Batching**: several jobs (songs × target games) are generated in parallel, one window per
  job per round. Each window's context is the job's own previously generated events.
- **Grammar**: decoding is constrained by the tokenizer's grammar state machine for the job's
  mechanics and lane count, so every emitted group is valid for that game.
- **Window boundaries**: events outside `[start, next start)` are discarded, so window
  boundaries never produce duplicates.

## Running on Colab (L4)

The ingest job writes shards to `/content/data/urfm-ingest-v0` on the same VM, and the trainer
reads them there. On a fresh VM, fetch them first:

```python
from urfm.train.data import download_shards

download_shards(sources=("taiko1000", "osu"))
```

```bash
# V1: Taiko only, 10M
python -m urfm.train.train --sources taiko1000 --size 10m --batch-size 64 --steps 6000 \
    --workers 10 --out /content/runs/v1-taiko --hub-repo JacobLinCool/urfm-runs
# V2: Taiko + osu!taiko (same roles), 15M
python -m urfm.train.train --sources taiko1000 osu --mechanics taiko osu_taiko --size 15m \
    --batch-size 64 --steps 8000 --workers 10 --out /content/runs/v2-taiko-osutaiko
# V3 / scaling sweep: all games, 5m / 15m / 45m
python -m urfm.train.train --sources taiko1000 osu --size 15m --batch-size 64 --steps 12000 \
    --workers 10 --out /content/runs/v3-all-15m
```

Two small runs, for example 5m and 15m, can share one L4 as separate processes to fill it.
Keep `--workers` at or below the VM core count divided by the number of concurrent runs.
