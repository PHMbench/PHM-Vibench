"""Native forecast ports: real operations, state recovery and upstream equations."""
from copy import deepcopy
import importlib
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

MODELS = ("SOFTS", "FreTS", "TSMixer")


def arguments(name, **changes):
    fields = dict(seq_len=31, pred_len=7, input_dim=3)
    if name == "SOFTS":
        fields.update(d_model=8, d_core=4, d_ff=16, e_layers=2, dropout=0.0, use_norm=True, activation="gelu")
    elif name == "FreTS":
        fields.update(embed_size=8, hidden_size=16, channel_mixing=True, sparsity_threshold=0.01)
    else:
        fields.update(d_model=8, e_layers=2, dropout=0.0)
    fields.update(changes)
    return SimpleNamespace(**fields)


def build(name, **changes):
    return importlib.import_module(f"src.model_factory.MLP.{name}").Model(arguments(name, **changes), None)


@pytest.mark.parametrize("name", MODELS)
def test_forward_gradient_update_and_input_config_immutability(name):
    torch.manual_seed(73)
    args = arguments(name)
    before = deepcopy(vars(args))
    model = importlib.import_module(f"src.model_factory.MLP.{name}").Model(args, None)
    x = torch.randn(4, 31, 3, requires_grad=True)
    original = x.detach().clone()
    target = torch.randn(4, 7, 3)
    result = model(x, task_id="forecasting")
    assert result.shape == target.shape and torch.isfinite(result).all()
    F.mse_loss(result, target).backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    saved = [p.detach().clone() for p in model.parameters()]
    torch.optim.Adam(model.parameters(), lr=0.001).step()
    assert any(not torch.equal(a, b) for a, b in zip(saved, model.parameters()))
    torch.testing.assert_close(x.detach(), original, rtol=0, atol=0)
    assert vars(args) == before


@pytest.mark.parametrize("name", MODELS)
def test_eval_rng_checkpoint_and_batch_independence(name, tmp_path):
    model = build(name).eval()
    x = torch.randn(2, 31, 3)
    rng = torch.random.get_rng_state()
    with torch.no_grad():
        predicted = model(x)
        isolated = model(x[:1])
    assert torch.equal(rng, torch.random.get_rng_state())
    torch.testing.assert_close(predicted[:1], isolated, rtol=1e-5, atol=1e-6)
    checkpoint = tmp_path / "model.pt"
    torch.save(model.state_dict(), checkpoint)
    restored = build(name).eval()
    restored.load_state_dict(torch.load(checkpoint, weights_only=True), strict=True)
    with torch.no_grad():
        torch.testing.assert_close(predicted, restored(x), rtol=0, atol=0)


@pytest.mark.parametrize("name", MODELS)
@pytest.mark.parametrize("kind", ["rank", "length", "channel", "empty", "dtype", "nan", "inf", "task", "feature"])
def test_invalid_history_is_rejected_without_repair(name, kind):
    model = build(name)
    x = torch.randn(2, 31, 3)
    task, feature = "forecasting", False
    if kind == "rank":
        x = x[0]
    elif kind == "length":
        x = x[:, :-1]
    elif kind == "channel":
        x = x[:, :, :-1]
    elif kind == "empty":
        x = x[:0]
    elif kind == "dtype":
        x = x.double()
    elif kind in ("nan", "inf"):
        x[0, 0, 0] = float(kind)
    elif kind == "task":
        task = "classification"
    elif kind == "feature":
        feature = True
    with pytest.raises((ValueError, TypeError, FloatingPointError)):
        model(x, task_id=task, return_feature=feature)


@pytest.mark.parametrize("name", MODELS)
@pytest.mark.parametrize("changes", [{"seq_len": "31"}, {"pred_len": 0}, {"input_dim": True}])
def test_constructor_rejects_invalid_dimensions(name, changes):
    with pytest.raises((TypeError, ValueError)):
        build(name, **changes)


@pytest.mark.parametrize("name", MODELS)
def test_constant_history_is_finite(name):
    with torch.no_grad():
        assert torch.isfinite(build(name).eval()(torch.zeros(2, 31, 3))).all()


def _star_reference(block, x, training):
    # Literal upstream pooling, independent of STAR.forward.
    b, c, _ = x.shape
    combined = block.gen2(F.gelu(block.gen1(x)))
    if training:
        ratio = combined.softmax(1).permute(0, 2, 1).reshape(-1, c)
        indices = torch.multinomial(ratio, 1).view(b, -1, 1).permute(0, 2, 1)
        combined = combined.gather(1, indices).repeat(1, c, 1)
    else:
        combined = (combined * combined.softmax(1)).sum(1, keepdim=True).repeat(1, c, 1)
    return block.gen4(F.gelu(block.gen3(torch.cat((x, combined), -1))))


@pytest.mark.parametrize("training", [False, True])
def test_softs_pooling_matches_upstream_in_both_modes(training):
    block = build("SOFTS").layers[0].star.train(training)
    x = torch.randn(3, 5, 8)
    rng = torch.random.get_rng_state()
    expected = _star_reference(block, x, training)
    after = torch.random.get_rng_state()
    torch.random.set_rng_state(rng)
    actual = block(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(after, torch.random.get_rng_state())


def test_softs_complete_forward_reference():
    model = build("SOFTS").eval()
    x = torch.randn(2, 31, 3)
    mean = x.mean(1, keepdim=True).detach()
    std = ((x - mean).var(1, keepdim=True, unbiased=False) + 1e-5).sqrt()
    z = model.embedding(((x - mean) / std).transpose(1, 2))
    for layer in model.layers:
        z = layer.norm1(z + _star_reference(layer.star, z, False))
        y = F.linear(z, layer.conv1.weight[:, :, 0], layer.conv1.bias)
        y = F.linear(F.gelu(y), layer.conv2.weight[:, :, 0], layer.conv2.bias)
        z = layer.norm2(z + y)
    expected = model.projection(z).transpose(1, 2) * std + mean
    torch.testing.assert_close(model(x), expected, rtol=1e-5, atol=1e-6)
    with pytest.raises(TypeError):
        build("SOFTS", use_norm="false")


def _frets_reference(model, x):
    def fre_mlp(spectrum, suffix):
        r, i = getattr(model, "r" + suffix), getattr(model, "i" + suffix)
        rb, ib = getattr(model, "rb" + suffix), getattr(model, "ib" + suffix)
        real = F.relu(torch.einsum("bijd,dd->bijd", spectrum.real, r) - torch.einsum("bijd,dd->bijd", spectrum.imag, i) + rb)
        imag = F.relu(torch.einsum("bijd,dd->bijd", spectrum.imag, r) + torch.einsum("bijd,dd->bijd", spectrum.real, i) + ib)
        return torch.view_as_complex(F.softshrink(torch.stack((real, imag), -1), lambd=model.sparsity_threshold))
    z = x.permute(0, 2, 1).unsqueeze(3) * model.embeddings
    bias = z
    if model.channel_mixing:
        z = torch.fft.rfft(z.permute(0, 2, 1, 3), dim=2, norm="ortho")
        z = torch.fft.irfft(fre_mlp(z, "1"), n=model.input_dim, dim=2, norm="ortho").permute(0, 2, 1, 3)
    z = torch.fft.rfft(z, dim=2, norm="ortho")
    z = torch.fft.irfft(fre_mlp(z, "2"), n=model.seq_len, dim=2, norm="ortho")
    return model.fc((z + bias).reshape(x.shape[0], model.input_dim, -1)).permute(0, 2, 1)


@pytest.mark.parametrize("channel_mixing", [False, True])
def test_frets_complete_forward_and_gradient_match_upstream(channel_mixing):
    model = build("FreTS", channel_mixing=channel_mixing).eval()
    x = torch.randn(2, 31, 3, requires_grad=True)
    target = torch.randn(2, 7, 3)
    actual = model(x)
    actual_grad = torch.autograd.grad(F.mse_loss(actual, target), x)[0]
    reference = _frets_reference(model, x)
    expected_grad = torch.autograd.grad(F.mse_loss(reference, target), x)[0]
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-5, atol=1e-6)
    with pytest.raises(TypeError):
        build("FreTS", channel_mixing="1")


def test_tsmixer_tsl_full_reference():
    model = build("TSMixer").eval()
    x = torch.randn(2, 31, 3)
    z = x
    for block in model.blocks:
        t = F.linear(z.transpose(1, 2), block.temporal[0].weight, block.temporal[0].bias)
        t = F.linear(t.relu(), block.temporal[2].weight, block.temporal[2].bias)
        z = z + t.transpose(1, 2)
        c = F.linear(z, block.channel[0].weight, block.channel[0].bias)
        c = F.linear(c.relu(), block.channel[2].weight, block.channel[2].bias)
        z = z + c
    expected = F.linear(z.transpose(1, 2), model.projection.weight, model.projection.bias).transpose(1, 2)
    torch.testing.assert_close(model(x), expected, rtol=0, atol=0)
