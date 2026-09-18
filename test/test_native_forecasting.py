"""Native forecast equations, actual gradients, checkpoint and causal Task tests."""

from copy import deepcopy
import importlib
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

PORTS = [("MLP", "NLinear"), ("MLP", "SparseTSF"), ("MLP", "FITS"), ("RNN", "SegRNN")]


def args_for(name, **updates):
    args = dict(seq_len=32, pred_len=16, input_dim=2)
    if name in ("NLinear", "FITS"):
        args["individual"] = False
    if name == "FITS":
        args["cut_freq"] = 6
    if name == "SparseTSF":
        args.update(period_len=4, model_type="linear")
    if name == "SegRNN":
        args.update(seg_len=4, d_model=8, dropout=0.0, rnn_type="gru", dec_way="pmf", channel_id=True, revin=False)
    args.update(updates)
    return SimpleNamespace(**args)


def build(family, name, **updates):
    module = importlib.import_module(f"src.model_factory.{family}.{name}")
    return module.Model(args_for(name, **updates), None)


@pytest.mark.parametrize("family,name", PORTS)
def test_forecast_gradients_and_checkpoint(family, name, tmp_path):
    torch.manual_seed(3)
    config = args_for(name)
    original = deepcopy(vars(config))
    cls = importlib.import_module(f"src.model_factory.{family}.{name}").Model
    model = cls(config, None)
    x = torch.randn(3, 32, 2, requires_grad=True)
    target = torch.randn(3, 16, 2)
    y = model(x, task_id="forecasting")
    assert y.shape == target.shape and torch.isfinite(y).all()
    F.mse_loss(y, target).backward()
    assert torch.isfinite(x.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    before = [p.detach().clone() for p in model.parameters()]
    torch.optim.Adam(model.parameters(), lr=1e-3).step()
    assert any(not torch.equal(old, new) for old, new in zip(before, model.parameters()))
    assert vars(config) == original
    model.eval()
    with torch.no_grad():
        expected = model(x)
    path = tmp_path / "forecast.pt"
    torch.save(model.state_dict(), path)
    restored = cls(config, None).eval()
    restored.load_state_dict(torch.load(path, weights_only=True), strict=True)
    with torch.no_grad():
        torch.testing.assert_close(expected, restored(x), rtol=0, atol=0)


@pytest.mark.parametrize("family,name", PORTS)
@pytest.mark.parametrize("case", ["short", "channels", "empty", "dtype", "nan", "task", "features"])
def test_forecast_rejects_invalid_input(family, name, case):
    model = build(family, name)
    x = torch.randn(2, 32, 2)
    task, feature = "forecasting", False
    if case == "short":
        x = x[:, :-1]
    elif case == "channels":
        x = x[:, :, :1]
    elif case == "empty":
        x = x[:0]
    elif case == "dtype":
        x = x.double()
    elif case == "nan":
        x[0, 0, 0] = torch.nan
    elif case == "task":
        task = "classification"
    elif case == "features":
        feature = True
    with pytest.raises((ValueError, TypeError, FloatingPointError)):
        model(x, task_id=task, return_feature=feature)


@pytest.mark.parametrize("individual", [False, True])
def test_nlinear_matches_direct_projection(individual):
    model = build("MLP", "NLinear", individual=individual).eval()
    x = torch.randn(2, 32, 2)
    centered = x - x[:, -1:].detach()
    if individual:
        reference = torch.stack([F.linear(centered[:, :, i], layer.weight, layer.bias)
                                 for i, layer in enumerate(model.linear)], -1)
    else:
        reference = torch.einsum("btc,ht->bhc", centered, model.linear.weight) + model.linear.bias[None, :, None]
    reference = reference + x[:, -1:].detach()
    torch.testing.assert_close(model(x), reference)


@pytest.mark.parametrize("kind", ["linear", "mlp"])
def test_sparse_tsf_phasewise_reference(kind):
    model = build("MLP", "SparseTSF", model_type=kind, d_model=8).eval()
    x = torch.randn(2, 32, 2)
    mean = x.mean(1, keepdim=True)
    z = (x - mean).transpose(1, 2)
    z = z + F.conv1d(z.reshape(-1, 1, 32), model.conv.weight, padding=2).reshape(2, 2, 32)
    reference = torch.empty(2, 16, 2)
    for channel in range(2):
        for phase in range(4):
            reference[:, phase::4, channel] = model.projection(z[:, channel, phase::4])
    torch.testing.assert_close(model(x), reference + mean)


@pytest.mark.parametrize("individual", [False, True])
@pytest.mark.parametrize("horizon", [15, 16])
def test_fits_complex_reference_including_odd_total(individual, horizon):
    model = build("MLP", "FITS", individual=individual, pred_len=horizon)
    x = torch.randn(2, 32, 2)
    mean = x.mean(1, keepdim=True)
    std = ((x - mean).var(1, keepdim=True, unbiased=True) + 1e-5).sqrt()
    low = torch.fft.rfft((x - mean) / std, dim=1)[:, :6]
    result = torch.zeros(2, (32 + horizon) // 2 + 1, 2, dtype=torch.complex64)
    for c in range(2):
        layer = model.upsampler[c] if individual else model.upsampler
        result[:, :model.output_freq, c] = low[:, :, c] @ layer.weight.T + layer.bias
    all_times = torch.fft.irfft(result, n=32+horizon, dim=1) * model.length_ratio
    reference = (all_times * std + mean)[:, -horizon:]
    torch.testing.assert_close(model(x), reference)


@pytest.mark.parametrize("rnn_type", ["rnn", "gru", "lstm"])
@pytest.mark.parametrize("dec_way", ["pmf", "rmf"])
@pytest.mark.parametrize("revin", [False, True])
def test_segrnn_decoding_variants(rnn_type, dec_way, revin):
    model = build("RNN", "SegRNN", rnn_type=rnn_type, dec_way=dec_way, revin=revin).eval()
    x = torch.randn(2, 32, 2)
    y = model(x)
    assert y.shape == (2, 16, 2) and torch.isfinite(y).all()
    torch.testing.assert_close(model(x + 5), y + 5, rtol=1e-4, atol=1e-5)
    F.mse_loss(y, torch.zeros_like(y)).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_segrnn_parallel_matches_upstream_layout():
    model = build("RNN", "SegRNN").eval()
    x = torch.randn(3, 32, 2)
    z = (x - x[:, -1:].detach()).permute(0, 2, 1)
    _, hn = model.rnn(model.embedding(z.reshape(-1, 8, 4)))
    positions = torch.cat([model.pos_emb.unsqueeze(0).repeat(2, 1, 1),
                           model.channel_emb.unsqueeze(1).repeat(1, 4, 1)], -1)
    positions = positions.view(-1, 1, 8).repeat(3, 1, 1)
    _, hy = model.rnn(positions, hn.repeat(1, 1, 4).view(1, -1, 8))
    expected = model.predict(hy).view(3, 2, 16).permute(0, 2, 1) + x[:, -1:].detach()
    torch.testing.assert_close(model(x), expected, rtol=0, atol=0)


@pytest.mark.parametrize("family,name,updates", [
    ("MLP", "SparseTSF", {"period_len": 3}),
    ("RNN", "SegRNN", {"seg_len": 3}),
    ("RNN", "SegRNN", {"d_model": 7}),
    ("MLP", "FITS", {"seq_len": 1}),
    ("MLP", "FITS", {"cut_freq": 18}),
    ("MLP", "NLinear", {"individual": "false"}),
])
def test_invalid_model_config(family, name, updates):
    with pytest.raises((ValueError, TypeError)):
        build(family, name, **updates)


def construct_task(data_updates=None, task_updates=None):
    from src.task_factory.task.DG.point_forecasting import task
    args = args_for("NLinear")
    data = dict(window_size=48, normalization="none")
    data.update(data_updates or {})
    task_args = dict(name="point_forecasting", model_task_id="forecasting", loss="MSE", metrics=["mse", "mae"], optimizer="adam", lr=0.001)
    task_args.update(task_updates or {})
    return task(build("MLP", "NLinear"), SimpleNamespace(**data), args,
                SimpleNamespace(**task_args), SimpleNamespace(), SimpleNamespace(),
                {1: {"Name": "Tiny", "Dataset_id": 0, "Label": 0}})


def test_task_future_never_changes_forecast_and_target_is_horizon():
    task = construct_task().eval()
    x = torch.randn(3, 48, 2)
    batch = {"x": x, "y": torch.tensor([0, 0, 0]), "file_id": torch.tensor([1, 1, 1])}
    original = {key: value.clone() for key, value in batch.items()}
    first = task(batch)
    changed = dict(batch, x=x.clone())
    changed["x"][:, 32:] += 100
    torch.testing.assert_close(task(changed), first, rtol=0, atol=0)
    step = task._shared_step(batch, "test")
    torch.testing.assert_close(step["test_loss"], F.mse_loss(first, x[:, 32:]))
    for key in original:
        torch.testing.assert_close(batch[key], original[key], rtol=0, atol=0)


@pytest.mark.parametrize("data,task_args", [
    ({"normalization": "per_window_standardization"}, {}),
    ({"normalization": "per_window_minmax"}, {}),
    ({"train_noise_snr": 20}, {}),
    ({"evaluation_noise_snr": 20}, {}),
    ({"window_size": 32}, {}),
    ({}, {"loss": "CE"}),
    ({}, {"metrics": ["acc"]}),
    ({}, {"model_task_id": "classification"}),
])
def test_task_rejects_target_leakage_and_wrong_objectives(data, task_args):
    with pytest.raises((ValueError, TypeError)):
        construct_task(data, task_args)
