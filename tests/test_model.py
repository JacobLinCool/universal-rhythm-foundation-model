import pytest
import torch

from urfm.model import ChartModel, preset

T = 1380  # even number of mel frames (16 s window)


@pytest.mark.parametrize(
    ("name", "lo", "hi"),
    [("5m", 4e6, 7e6), ("10m", 9e6, 13e6), ("15m", 13e6, 18e6), ("45m", 40e6, 50e6)],
)
def test_preset_sizes_are_layer_dominated(name, lo, hi):
    params = ChartModel(preset(name, vocab_size=600, frame_feat_dim=8)).num_params()
    assert lo <= params["total"] <= hi
    assert params["embedding"] < 0.05 * params["total"]
    assert params["encoder_layers"] + params["decoder_layers"] > 0.75 * params["total"]


def tiny(vocab=40):
    return ChartModel(preset("tiny", vocab_size=vocab, frame_feat_dim=2))


def batch(b=2, length=24, vocab=40):
    tokens = torch.randint(4, vocab, (b, length))
    mask = torch.zeros(b, length, dtype=torch.bool)
    mask[:, 10:] = True
    return {
        "mel": torch.randint(0, 256, (b, T, 128), dtype=torch.uint8),
        "feats": torch.randn(b, T, 2),
        "tokens": tokens,
        "loss_mask": mask,
    }


def test_forward_backward_shapes():
    model = tiny()
    b = batch()
    assert model(b["mel"], b["feats"], b["tokens"]).shape == (2, 24, 40)
    loss, correct, n = model.loss(b)
    loss.backward()
    assert n == 2 * 14 and 0 <= correct <= n
    assert model.embed.weight.grad is not None


def test_loss_averages_only_masked_targets():
    model = tiny()
    b = batch()
    loss, _, n = model.loss(b)
    logits = model(b["mel"], b["feats"], b["tokens"][:, :-1])
    ce = torch.nn.functional.cross_entropy(
        logits.transpose(1, 2), b["tokens"][:, 1:], reduction="none"
    )
    m = b["loss_mask"][:, 1:]
    assert torch.allclose(loss, ce[m].mean(), atol=1e-6)
    assert n == m.sum()


def test_incremental_decoding_matches_teacher_forcing():
    torch.manual_seed(0)
    model = tiny().eval()
    b = batch(b=2, length=12)
    memory = model.encode(b["mel"], b["feats"])
    full = model.decode(b["tokens"], memory)
    # left-pad row 1 by 3 tokens to exercise the padding path
    prompt = b["tokens"][:, :6].clone()
    pad = torch.ones_like(prompt, dtype=torch.bool)
    prompt[1, :3] = 0
    pad[1, :3] = False
    cache, logits = model.prefill(prompt, pad, memory, 32)
    assert torch.allclose(logits[0], full[0, 5], atol=1e-4)
    key_mask = torch.zeros(2, 32, dtype=torch.bool)
    key_mask[:, :6] = pad
    positions = pad.sum(1)
    key_mask[:, 6] = True
    step = model.step(b["tokens"][:, 6], positions, cache, 6, key_mask)
    assert torch.allclose(step[0], full[0, 6], atol=1e-4)


# -- song context ----------------------------------------------------------------------


def song_batch(b=2, length=24, vocab=40, blocks=(30, 7)):
    out = batch(b, length, vocab)
    song = torch.randint(0, 256, (b, 256, 128), dtype=torch.uint8)
    marker = torch.zeros(b, 256, dtype=torch.long)
    lengths = torch.tensor(blocks[:b])
    for i, n in enumerate(lengths.tolist()):
        song[i, n:] = 0
        marker[i, max(n - 3, 0) : n] = 1
    return out | {"song": song, "song_marker": marker, "song_len": lengths}


def song_tiny(vocab=40):
    return ChartModel(preset("tiny", vocab_size=vocab, frame_feat_dim=2, song_context=True))


def test_disabled_song_context_keeps_parameters():
    """Old checkpoints load into disabled models: no new parameters unless enabled."""
    plain = set(tiny().state_dict())
    disabled = ChartModel(preset("tiny", vocab_size=40, frame_feat_dim=2, song_context=False))
    assert set(disabled.state_dict()) == plain
    enabled = set(song_tiny().state_dict())
    assert plain < enabled and all(k.startswith(("song_", "source")) for k in enabled - plain)


@pytest.mark.parametrize("name", ["5m", "10m", "15m", "45m"])
def test_song_context_presets_forward(name):
    model = ChartModel(preset(name, vocab_size=600, frame_feat_dim=2, song_context=True)).eval()
    b = song_batch(vocab=600)
    with torch.no_grad():
        out = model(b["mel"], b["feats"], b["tokens"], b["song"], b["song_marker"], b["song_len"])
    assert out.shape == (2, 24, 600) and torch.isfinite(out).all()


def test_song_context_loss_and_padding_invariance():
    torch.manual_seed(0)
    model = song_tiny().eval()
    b = song_batch()
    loss, _, n = model.loss(b)
    assert torch.isfinite(loss) and n == 2 * 14
    # padded song blocks are masked: changing them must not change the output
    noisy = dict(b)
    noisy["song"] = b["song"].clone()
    noisy["song"][0, 30:] = 255
    with torch.no_grad():
        a = model(b["mel"], b["feats"], b["tokens"], b["song"], b["song_marker"], b["song_len"])
        c = model(b["mel"], b["feats"], b["tokens"], noisy["song"], b["song_marker"], b["song_len"])
    assert torch.allclose(a, c, atol=1e-5)
    with pytest.raises(ValueError):
        model(b["mel"], b["feats"], b["tokens"])


def test_song_context_incremental_decoding_matches_teacher_forcing():
    torch.manual_seed(0)
    model = song_tiny().eval()
    b = song_batch(length=12)
    memory, mask = model.memory_for(
        b["mel"], b["feats"], b["song"], b["song_marker"], b["song_len"]
    )
    full = model.decode(b["tokens"], memory, mask)
    prompt = b["tokens"][:, :6].clone()
    pad = torch.ones_like(prompt, dtype=torch.bool)
    prompt[1, :3] = 0
    pad[1, :3] = False
    cache, logits = model.prefill(prompt, pad, memory, 32, mask)
    assert torch.allclose(logits[0], full[0, 5], atol=1e-4)
    key_mask = torch.zeros(2, 32, dtype=torch.bool)
    key_mask[:, :6] = pad
    key_mask[:, 6] = True
    step = model.step(b["tokens"][:, 6], pad.sum(1), cache, 6, key_mask)
    assert torch.allclose(step[0], full[0, 6], atol=1e-4)
