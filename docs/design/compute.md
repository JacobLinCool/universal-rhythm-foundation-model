# Compute and data plane

## Where things live

| Thing | Location | Never |
| --- | --- | --- |
| Source code, specs, small fixtures | this repo (local) → uploaded to the VM by `scripts/colab.py sync` | — |
| Raw corpora (osu! dumps, packs, Taiko-1000, simai, …) | downloaded **on the Colab VM**, local SSD under `/content/data` | downloaded to the local machine |
| Processed IR shards, feature caches, datasets | built on the VM, published to the **Hugging Face Hub** (private/gated as licences require) | copied between local and Colab |
| Checkpoints | VM disk; uploaded to the HF Hub from the VM | pulled locally unless small and needed |
| Results (raw evidence JSONL, tables, figures) | small; brought back locally via `colab download` into `runs/` | — |

The Hub is the only shared store. A new VM pulls exactly the dataset revision it needs (pinned
by commit hash) straight from the Hub.

## Access

- **Google / Colab:** Application Default Credentials (`gcloud auth application-default login`),
  used through `colab --auth adc`. Session state is in `.colab/sessions.json` (git-ignored).
- **Hugging Face:** the local token (`~/.cache/huggingface/token`) is placed on the VM with
  `colab upload` (`scripts/colab.py token`). It is never printed, and never passed in argv or
  environment variables.
- Stderr from the Colab CLI can contain authenticated proxy URLs. `scripts/colab.py` redacts
  every URL.

```bash
uv run --extra colab python scripts/colab.py new          # L4 (24 GB, Ada, bf16)
uv run --extra colab python scripts/colab.py bootstrap    # code + deps + HF token, prints GPU
uv run --extra colab python scripts/colab.py exec experiments/<id>/run.py --timeout 7200
```

## Efficiency rules

Compute is billed and L4 time is scarce. Every pipeline should saturate the GPU and keep the CPU
off the critical path.

### Data processing (VM)

- Parse charts once into IR msgpack shards (`urfm.ir.serialize.to_msgpack`). Store them as
  Parquet / Arrow or WebDataset shards of roughly 100–500 MB, and stream them with
  `datasets`/`pyarrow`. Never keep millions of small files.
- Parallelise CPU-bound parsing across all VM cores (`multiprocessing` /
  `datasets.map(num_proc=os.cpu_count())`). Audio decoding goes through FFmpeg/torchaudio in
  worker processes.
- Content-hash everything (audio SHA-256, chart SHA-256) so reprocessing is incremental and
  cache hits are exact.

### Features

- Compute audio features (mel, beat activations, SSL embeddings) **on the GPU in large
  batches**, cache them as fp16 shards, and push the shards to the Hub. Do not recompute them
  inside the training loop.
- Keep feature extraction deterministic and versioned. The feature config hash is part of the
  shard name.

### Training and inference

- **Input pipeline:** `DataLoader` with `num_workers ≈ cpu_count`, `pin_memory=True`,
  `persistent_workers=True`, `prefetch_factor ≥ 4`.
  - Collate whole batches into contiguous tensors on the CPU, then move them with
    `.to(device, non_blocking=True)`.
  - Never transfer per sample, and never call `.item()` / `.cpu()` inside the hot loop.
- **Kernels:**
  - bf16 autocast (L4 supports bf16 natively) and TF32 matmuls.
  - `torch.compile` once shapes are stable, with fused optimizers (`fused=True`) and
    `scaled_dot_product_attention` (Flash/SDPA).
- **Batch shape:**
  - Bucket sequences by length to minimise padding. Pack variable-length event sequences
    where possible.
  - Pick the largest batch size that fits, using gradient accumulation if needed. A step
    should be GPU-bound: verify that data-loader wait is well under 5% of step time.
- **Logging and checkpoints:**
  - Log asynchronously and sparsely, accumulating metrics on the GPU and syncing every N
    steps.
  - Checkpoint to local disk, then upload to the Hub in a background thread.
- **Measure, don't guess:**
  - Every run records GPU utilisation and memory (`nvidia-smi` sampling or
    `torch.cuda.utilization`), throughput (samples/s, tokens/s) and data-loader stall time
    in its raw evidence.
  - Runs whose GPU utilisation stays under 80% are treated as a pipeline bug to fix before
    scaling.
