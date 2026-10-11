"""Does the model know don vs ka? Teacher-forced colour accuracy on one human chart.

The human chart is fed as decoder input (positions, durations, everything); at every
``ROLE`` token whose target is ``taiko:don`` or ``taiko:ka`` we read the model's choice
between the two. If this is near the majority baseline, the model has not learned colour
from audio (a model/data problem); if it is high, generation errors come from decoding
(error accumulation, temperature).

    uv run python experiments/benchmark/diag_color.py --ckpt runs/checkpoints/ft-r3-song.pt \\
        --song runs/v1b-taiko/playable/taiko1000_test-00004-Crystal_Hail
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import msgspec
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from song_bench import song_mel

from urfm.codecs import get
from urfm.data.features import MEL
from urfm.tokenize import build_tokenizer
from urfm.train.data import (
    collate_song,
    make_sample,
    prepare,
    song_marker,
    song_summary,
)
from urfm.train.tiers import load_tiers
from urfm.train.train import load_checkpoint


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--song", required=True)
    args = ap.parse_args()
    song = Path(args.song)
    tok = build_tokenizer()
    v = tok.vocab
    model, meta = load_checkpoint(args.ckpt, "cpu")
    tiers = load_tiers(meta)
    vocab_size = model.cfg.vocab_size
    human = get("tja").read_path(song / "human.tja")
    # older checkpoints predate control tokens: drop controls and meter changes
    plain_timing = msgspec.structs.replace(human.timing, meters=())
    cs = msgspec.structs.replace(
        human,
        timing=plain_timing,
        charts=tuple(msgspec.structs.replace(c, controls=(), timing=None) for c in human.charts),
    )
    mel = song_mel(next(song.glob("song.*")))
    blocks, block_us = song_summary(mel)
    don, ka = v.id("ROLE", "taiko:don"), v.id("ROLE", "taiko:ka")
    print("| course | ROLE targets | model acc | majority baseline | mean p(correct) |")
    print("|---|---|---|---|---|")
    for chart in cs.charts:
        if not chart.events:
            continue
        prep = prepare(tok, cs, chart, mel.shape[0], tiers)
        correct = total = n_don = 0
        p_correct: list[float] = []
        for window in prep.windows:
            s = make_sample(tok, cs, prep, mel, window, 2048)
            if (s.tokens >= vocab_size).any():
                continue
            toks = torch.from_numpy(s.tokens)[None]
            song_in = collate_song([(blocks, song_marker(block_us, *window))])
            with torch.no_grad():
                logits = model(
                    torch.from_numpy(s.mel)[None],
                    torch.from_numpy(s.feats)[None],
                    toks[:, :-1],
                    *(
                        (song_in["song"], song_in["song_marker"], song_in["song_len"])
                        if model.cfg.song_context
                        else ()
                    ),
                )[0]
            target = s.tokens[1:]
            for i in range(s.sep, len(target)):
                if target[i] not in (don, ka):
                    continue
                pair = logits[i, [don, ka]].float()
                guess = don if pair[0] >= pair[1] else ka
                prob = torch.softmax(pair, 0)[0 if target[i] == don else 1].item()
                correct += guess == target[i]
                n_don += target[i] == don
                total += 1
                p_correct.append(prob)
        if total:
            base = max(n_don, total - n_don) / total
            print(
                f"| {chart.difficulty.label} | {total} | {correct / total:.3f} | {base:.3f} "
                f"| {np.mean(p_correct):.3f} |"
            )
    _ = MEL
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
