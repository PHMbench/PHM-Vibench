"""TimeMixer forecast paths and independently expressed PDM equations."""
from copy import deepcopy
from types import SimpleNamespace
import pytest
import torch
from torch.nn import functional as F
from src.model_factory.MLP.TimeMixer import Model


def args(**changes):
    values = dict(seq_len=32, pred_len=7, input_dim=3, down_sampling_window=2,
                  down_sampling_layers=2, d_model=8, d_ff=16, e_layers=2, moving_avg=3,
                  channel_independence=False, use_norm=True, dropout=0.0,
                  decomp_method="moving_avg", down_sampling_method="avg")
    values.update(changes)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("independent", [False, True])
@pytest.mark.parametrize("use_norm", [False, True])
def test_actual_gradient_step_and_no_configuration_mutation(independent, use_norm):
    configuration = args(channel_independence=independent, use_norm=use_norm)
    before = deepcopy(vars(configuration))
    model = Model(configuration, None)
    x = torch.randn(4, 32, 3, requires_grad=True)
    original = x.detach().clone()
    forecast = model(x, task_id="forecasting")
    assert forecast.shape == (4, 7, 3) and torch.isfinite(forecast).all()
    F.mse_loss(forecast, torch.randn_like(forecast)).backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    saved = [p.detach().clone() for p in model.parameters()]
    torch.optim.Adam(model.parameters(), lr=0.001).step()
    assert any(not torch.equal(a, b) for a, b in zip(saved, model.parameters()))
    assert vars(configuration) == before
    torch.testing.assert_close(x.detach(), original, rtol=0, atol=0)


def _mlp(module, x):
    return F.linear(F.gelu(F.linear(x, module[0].weight, module[0].bias)), module[2].weight, module[2].bias)


def _decompose(x, kernel):
    pad = (kernel - 1) // 2
    padded = torch.cat((x[:, :1].repeat(1, pad, 1), x, x[:, -1:].repeat(1, pad, 1)), dim=1)
    trend = padded.unfold(1, kernel, 1).mean(-1)
    return x - trend, trend


def _reference(model, x, pool_kind):
    batch = x.shape[0]
    inputs, trends, statistics = [], [], []
    raw = x
    for i in range(len(model.lengths)):
        if i:
            pool = F.avg_pool1d if pool_kind == "avg" else F.max_pool1d
            raw = pool(raw.transpose(1, 2), model.window).transpose(1, 2)
        z = raw
        if model.use_norm:
            mean = raw.mean(1, keepdim=True).detach()
            std = (raw.var(1, keepdim=True, unbiased=False) + 1e-5).sqrt().detach()
            z = (raw - mean) / std * model.norms[i].weight + model.norms[i].bias
            statistics.append((mean, std))
        if model.independent:
            z = z.transpose(1, 2).reshape(batch * model.input_dim, z.shape[1], 1)
        else:
            z, trend = _decompose(z, model.kernel)
            trends.append(trend)
        padded = F.pad(z.transpose(1, 2), (1, 1), mode="circular")
        inputs.append(F.conv1d(padded, model.embedding.weight).transpose(1, 2))
    for pdm in model.pdm:
        season, trend = [], []
        for z in inputs:
            s, t = _decompose(z, model.kernel)
            if not model.independent:
                s, t = _mlp(pdm.cross, s), _mlp(pdm.cross, t)
            season.append(s.transpose(1, 2))
            trend.append(t.transpose(1, 2))
        high_to_low = [season[0]]
        for i, layer in enumerate(pdm.season_down):
            high_to_low.append(season[i + 1] + _mlp(layer, high_to_low[-1]))
        low_to_high = [trend[-1]]
        for i, layer in enumerate(pdm.trend_up):
            low_to_high.append(trend[-i - 2] + _mlp(layer, low_to_high[-1]))
        low_to_high.reverse()
        mixed = [(s + t).transpose(1, 2) for s, t in zip(high_to_low, low_to_high)]
        inputs = [old + _mlp(pdm.out_cross, z) for old, z in zip(inputs, mixed)] if model.independent else mixed
    predictions = []
    for i, z in enumerate(inputs):
        y = F.linear(z.transpose(1, 2), model.predict[i].weight, model.predict[i].bias).transpose(1, 2)
        y = F.linear(y, model.projection.weight, model.projection.bias)
        if model.independent:
            y = y.reshape(batch, model.input_dim, model.pred_len).transpose(1, 2)
        else:
            residual = F.linear(trends[i].transpose(1, 2), model.out_res[i].weight, model.out_res[i].bias)
            residual = F.linear(residual, model.regression[i].weight, model.regression[i].bias).transpose(1, 2)
            y = y + residual
        predictions.append(y)
    result = torch.stack(predictions, -1).sum(-1)
    if model.use_norm:
        mean, std = statistics[0]
        result = (result - model.norms[0].bias) / (model.norms[0].weight + 1e-10) * std + mean
    return result


@pytest.mark.parametrize("independent", [False, True])
@pytest.mark.parametrize("pool", ["avg", "max"])
def test_full_reference_equations_and_input_gradient(independent, pool):
    model = Model(args(channel_independence=independent, down_sampling_method=pool), None).eval()
    x = torch.randn(2, 32, 3, requires_grad=True)
    target = torch.randn(2, 7, 3)
    actual = model(x)
    actual_grad = torch.autograd.grad(F.mse_loss(actual, target), x)[0]
    reference = _reference(model, x, pool)
    reference_grad = torch.autograd.grad(F.mse_loss(reference, target), x)[0]
    torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(actual_grad, reference_grad, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("independent", [False, True])
def test_eval_no_rng_or_stale_normalization_state(independent, tmp_path):
    model = Model(args(channel_independence=independent, dropout=0.2), None).eval()
    x = torch.randn(2, 32, 3)
    with torch.no_grad():
        expected = model(x)
        model(torch.randn_like(x) * 100 + 50)
        actual = model(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    rng = torch.random.get_rng_state()
    with torch.no_grad():
        model(x)
    assert torch.equal(rng, torch.random.get_rng_state())
    checkpoint = tmp_path / "timemixer.pt"
    torch.save(model.state_dict(), checkpoint)
    restored = Model(args(channel_independence=independent, dropout=0.2), None).eval()
    restored.load_state_dict(torch.load(checkpoint, weights_only=True), strict=True)
    with torch.no_grad():
        torch.testing.assert_close(restored(x), expected, rtol=0, atol=0)
        assert torch.isfinite(restored(torch.zeros_like(x))).all()


@pytest.mark.parametrize("fault", ["rank", "length", "channels", "empty", "dtype", "nan", "inf", "task", "feature"])
def test_signal_contract(fault):
    model = Model(args(), None)
    x, task, feature = torch.randn(2, 32, 3), "forecasting", False
    if fault == "rank": x = x[0]
    elif fault == "length": x = x[:, :-1]
    elif fault == "channels": x = x[:, :, :1]
    elif fault == "empty": x = x[:0]
    elif fault == "dtype": x = x.double()
    elif fault in ("nan", "inf"): x[0, 0, 0] = float(fault)
    elif fault == "task": task = "classification"
    elif fault == "feature": feature = True
    with pytest.raises((ValueError, TypeError, FloatingPointError)):
        model(x, task_id=task, return_feature=feature)


@pytest.mark.parametrize("changes", [
    {"seq_len": "32"}, {"seq_len": 31}, {"pred_len": 0}, {"down_sampling_layers": 0},
    {"down_sampling_window": 1}, {"moving_avg": 2}, {"channel_independence": 1},
    {"use_norm": "false"}, {"decomp_method": "dft_decomp"}, {"down_sampling_method": "conv"},
])
def test_explicit_supported_configuration(changes):
    with pytest.raises((ValueError, TypeError)):
        Model(args(**changes), None)
