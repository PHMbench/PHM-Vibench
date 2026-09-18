"""Executable classification-port tests, without mocked models or training losses."""

from copy import deepcopy
import importlib
import math
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F


PORTS = [("Transformer", "iTransformer"), ("CNN", "TimesNet"), ("CNN", "TSLANet")]


def model_args(name, **changes):
    values = dict(seq_len=32, input_dim=2, num_classes=3, d_model=8, e_layers=2, dropout=0.0)
    if name == "iTransformer":
        values.update(n_heads=2, d_ff=16, activation="gelu")
    elif name == "TimesNet":
        values.update(d_ff=16, top_k=2, num_kernels=2)
    elif name == "TSLANet":
        values.update(patch_size=4, adaptive_filter=True, apply_asb=True, apply_icb=True)
    values.update(changes)
    return SimpleNamespace(**values)


def construct(family, name, **changes):
    module = importlib.import_module(f"src.model_factory.{family}.{name}")
    return module.Model(model_args(name, **changes), metadata=None)


@pytest.mark.parametrize("family,name", PORTS)
def test_port_forward_features_gradients_and_optimizer(family, name):
    torch.manual_seed(17)
    args = model_args(name)
    before = deepcopy(vars(args))
    module = importlib.import_module(f"src.model_factory.{family}.{name}")
    model = module.Model(args, None)
    x = torch.randn(4, 32, 2, requires_grad=True)
    logits, features = model(x, file_id=torch.arange(4), task_id="classification", return_feature=True)
    assert logits.shape == (4, 3)
    assert features.shape[0] == 4 and features.ndim == 2
    assert torch.isfinite(logits).all() and torch.isfinite(features).all()
    target = torch.tensor([0, 1, 2, 0])
    loss = F.cross_entropy(logits, target)
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    gradients = [p.grad for p in model.parameters() if p.requires_grad]
    assert gradients and all(g is not None and torch.isfinite(g).all() for g in gradients)
    saved = [p.detach().clone() for p in model.parameters()]
    torch.optim.SGD(model.parameters(), lr=0.01).step()
    assert any(not torch.equal(a, b) for a, b in zip(saved, model.parameters()))
    assert vars(args) == before
    assert next(model.parameters()).device.type == "cpu"


@pytest.mark.parametrize("family,name", PORTS)
def test_eval_is_rng_independent_and_checkpoint_reconstructs(family, name, tmp_path):
    model = construct(family, name, dropout=0.2).eval()
    x = torch.randn(3, 32, 2)
    rng = torch.random.get_rng_state().clone()
    with torch.no_grad():
        first = model(x)
        logits, _ = model(x, return_feature=True)
    assert torch.equal(rng, torch.random.get_rng_state())
    torch.testing.assert_close(first, logits, rtol=0, atol=0)
    path = tmp_path / "model.pt"
    torch.save(model.state_dict(), path)
    restored = construct(family, name, dropout=0.2).eval()
    restored.load_state_dict(torch.load(path, weights_only=True), strict=True)
    with torch.no_grad():
        torch.testing.assert_close(first, restored(x), rtol=0, atol=0)


@pytest.mark.parametrize("family,name", PORTS)
@pytest.mark.parametrize("case", ["rank", "length", "channels", "empty", "dtype", "nan", "inf", "task"])
def test_invalid_signal_is_not_repaired(family, name, case):
    model = construct(family, name).eval()
    x = torch.randn(2, 32, 2)
    task = "classification"
    if case == "rank":
        x = x[0]
    elif case == "length":
        x = x[:, :-1]
    elif case == "channels":
        x = x[:, :, :1]
    elif case == "empty":
        x = x[:0]
    elif case == "dtype":
        x = x.double()
    elif case in ("nan", "inf"):
        x[0, 0, 0] = float(case)
    elif case == "task":
        task = "prediction"
    with pytest.raises((ValueError, TypeError, FloatingPointError)):
        model(x, task_id=task)


@pytest.mark.parametrize("family,name", PORTS)
@pytest.mark.parametrize("changes", [{"seq_len": "32"}, {"d_model": 0}, {"num_classes": 1}, {"num_classes": {0: 3, 1: 4}}, {"dropout": 1.0}, {"dropout": "0.1"}])
def test_invalid_constructor_configuration(family, name, changes):
    with pytest.raises((ValueError, TypeError)):
        construct(family, name, **changes)


@pytest.mark.parametrize("family,name", PORTS)
def test_single_ontology_mapping_and_zero_signal(family, name):
    model = construct(family, name, num_classes={7: 3}).eval()
    with torch.no_grad():
        result = model(torch.zeros(2, 32, 2))
    assert result.shape == (2, 3) and torch.isfinite(result).all()


def test_itransformer_reference_attention_and_feedforward():
    model = construct("Transformer", "iTransformer").eval()
    layer = model.layers[0]
    x = torch.randn(2, 5, 8)
    # Independent multi-head reference with matmul, preserving THUML post-norm order.
    q = layer.query_projection(x).reshape(2, 5, 2, 4).transpose(1, 2)
    k = layer.key_projection(x).reshape(2, 5, 2, 4).transpose(1, 2)
    v = layer.value_projection(x).reshape(2, 5, 2, 4).transpose(1, 2)
    scores = (q @ k.transpose(-1, -2) / math.sqrt(4)).softmax(-1)
    values = (scores @ v).transpose(1, 2).reshape(2, 5, 8)
    normalized = layer.norm1(x + layer.out_projection(values))
    ff1 = F.linear(normalized, layer.conv1.weight[:, :, 0], layer.conv1.bias)
    ff2 = F.linear(F.gelu(ff1), layer.conv2.weight[:, :, 0], layer.conv2.bias)
    expected = layer.norm2(normalized + ff2)
    torch.testing.assert_close(layer(x), expected, rtol=1e-5, atol=1e-6)


def test_timesnet_period_grid_reference_and_non_dc_constraint():
    module = importlib.import_module("src.model_factory.CNN.TimesNet")
    block = module.TimesBlock(4, 8, 2, 2).eval()
    x = torch.randn(2, 15, 4)
    amplitude = torch.fft.rfft(x, dim=1).abs()
    strength = amplitude.mean(0).mean(-1)
    strength[0] = -torch.inf
    bins = strength.topk(2).indices
    periods, actual_weights = module.fft_periods(x, 2)
    expected_periods = (15 // bins).tolist()
    assert periods == expected_periods
    expected_weights = amplitude.mean(-1)[:, bins]
    torch.testing.assert_close(actual_weights, expected_weights)
    outputs = []
    for period in expected_periods:
        length = math.ceil(15 / period) * period
        padded = torch.cat([x, x.new_zeros(2, length - 15, 4)], dim=1)
        grid = padded.reshape(2, length // period, period, 4).permute(0, 3, 1, 2)
        outputs.append(block.conv(grid).permute(0, 2, 3, 1).reshape(2, length, 4)[:, :15])
    expected = x + sum(expected_weights.softmax(1)[:, i, None, None] * outputs[i] for i in range(2))
    torch.testing.assert_close(block(x), expected)
    periods, _ = module.fft_periods(torch.zeros_like(x), 2)
    assert all(1 <= period <= 15 for period in periods)


def test_tslanet_spectral_reference_and_threshold_gradient():
    module = importlib.import_module("src.model_factory.CNN.TSLANet")
    block = module.AdaptiveSpectralBlock(4, True)
    x = torch.randn(2, 15, 4, requires_grad=True)
    spectrum = torch.fft.rfft(x, dim=1, norm="ortho")
    energy = spectrum.abs().pow(2).sum(-1)
    mask = (energy / (energy.median(1, keepdim=True).values + 1e-6) > block.threshold).float()
    expected = torch.fft.irfft(
        spectrum * torch.view_as_complex(block.complex_weight)
        + spectrum * mask.unsqueeze(-1) * torch.view_as_complex(block.complex_weight_high),
        n=15, dim=1, norm="ortho",
    )
    actual = block(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.square().sum().backward()
    assert block.threshold.grad is not None and torch.isfinite(block.threshold.grad).all()


@pytest.mark.parametrize("apply_asb,apply_icb", [(True, True), (True, False), (False, True)])
def test_tslanet_ablation_is_explicit_per_instance(apply_asb, apply_icb):
    model = construct("CNN", "TSLANet", apply_asb=apply_asb, apply_icb=apply_icb).eval()
    assert all(block.apply_asb is apply_asb and block.apply_icb is apply_icb for block in model.blocks)
    assert torch.isfinite(model(torch.randn(2, 32, 2))).all()
    with pytest.raises(ValueError, match="identity-only"):
        construct("CNN", "TSLANet", apply_asb=False, apply_icb=False)


@pytest.mark.parametrize("changes", [{"patch_size": 1}, {"patch_size": 33}, {"patch_size": 6, "seq_len": 32}, {"adaptive_filter": "false"}])
def test_tslanet_patch_and_boolean_contract(changes):
    with pytest.raises((ValueError, TypeError)):
        construct("CNN", "TSLANet", **changes)


def test_itransformer_invalid_head_width():
    with pytest.raises(ValueError, match="divisible"):
        construct("Transformer", "iTransformer", d_model=7)


def test_timesnet_invalid_frequency_count():
    with pytest.raises(ValueError, match="FFT bins"):
        construct("CNN", "TimesNet", top_k=17)
