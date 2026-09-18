"""Interpolation, patch-attention and sampling forecasts with independent equations."""
from copy import deepcopy
from types import SimpleNamespace
import importlib
import math
import pytest
import torch
from torch.nn import functional as F

PORTS = (("MLP", "NHITS"), ("Transformer", "PAttn"), ("MLP", "LightTS"))


def args_for(name, **changes):
    fields = dict(seq_len=32, pred_len=8, input_dim=3)
    if name == "NHITS":
        fields.update(n_blocks=[1, 1], n_pool_kernel_size=[4, 2], n_freq_downsample=[4, 1],
                      mlp_units=[[16, 16], [16, 16]], pooling_mode="MaxPool1d",
                      interpolation_mode="linear", dropout_prob_theta=0.0, activation="ReLU")
    elif name == "PAttn":
        fields.update(patch_size=8, stride=4, d_model=8, n_heads=2, d_ff=16, dropout=0.0, activation="gelu")
    else:
        fields.update(chunk_size=4, d_model=32)
    fields.update(changes)
    return SimpleNamespace(**fields)


def construct(family, name, **changes):
    return importlib.import_module(f"src.model_factory.{family}.{name}").Model(args_for(name, **changes), None)


@pytest.mark.parametrize("family,name", PORTS)
def test_forward_loss_update_no_config_or_input_mutation(family, name):
    torch.manual_seed(19)
    args = args_for(name)
    before = deepcopy(vars(args))
    model = importlib.import_module(f"src.model_factory.{family}.{name}").Model(args, None)
    x = torch.randn(4, 32, 3, requires_grad=True)
    source = x.detach().clone()
    prediction = model(x, task_id="forecasting")
    assert prediction.shape == (4, 8, 3) and torch.isfinite(prediction).all()
    F.mse_loss(prediction, torch.randn_like(prediction)).backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    saved = [p.detach().clone() for p in model.parameters()]
    torch.optim.Adam(model.parameters(), lr=0.001).step()
    assert any(not torch.equal(a, b) for a, b in zip(saved, model.parameters()))
    assert vars(args) == before
    torch.testing.assert_close(x.detach(), source, rtol=0, atol=0)


@pytest.mark.parametrize("family,name", PORTS)
def test_eval_rng_and_exact_parameter_reload(family, name, tmp_path):
    model = construct(family, name).eval()
    x = torch.randn(2, 32, 3)
    rng = torch.random.get_rng_state()
    with torch.no_grad():
        prediction = model(x)
    assert torch.equal(rng, torch.random.get_rng_state())
    checkpoint = tmp_path / "forecast.pt"
    torch.save(model.state_dict(), checkpoint)
    restored = construct(family, name).eval()
    restored.load_state_dict(torch.load(checkpoint, weights_only=True), strict=True)
    with torch.no_grad():
        torch.testing.assert_close(prediction, restored(x), rtol=0, atol=0)
        assert torch.isfinite(restored(torch.zeros_like(x))).all()


@pytest.mark.parametrize("family,name", PORTS)
@pytest.mark.parametrize("fault", ["rank", "length", "channels", "empty", "nan", "inf", "dtype", "task", "feature"])
def test_history_contract(family, name, fault):
    model = construct(family, name)
    x, task_id, feature = torch.randn(2, 32, 3), "forecasting", False
    if fault == "rank":
        x = x[0]
    elif fault == "length":
        x = x[:, :-1]
    elif fault == "channels":
        x = x[:, :, :1]
    elif fault == "empty":
        x = x[:0]
    elif fault in ("nan", "inf"):
        x[0, 0, 0] = float(fault)
    elif fault == "dtype":
        x = x.double()
    elif fault == "task":
        task_id = "classification"
    elif fault == "feature":
        feature = True
    with pytest.raises((TypeError, ValueError, FloatingPointError)):
        model(x, task_id=task_id, return_feature=feature)


@pytest.mark.parametrize("family,name", PORTS)
@pytest.mark.parametrize("changes", [{"seq_len": "32"}, {"pred_len": 0}, {"input_dim": True}])
def test_reject_bad_dimensions(family, name, changes):
    with pytest.raises((ValueError, TypeError)):
        construct(family, name, **changes)


def _linear_mlp(layers, value):
    for layer in layers:
        if isinstance(layer, torch.nn.Linear):
            value = F.linear(value, layer.weight, layer.bias)
        else:
            value = layer(value)
    return value


@pytest.mark.parametrize("pool", ["MaxPool1d", "AvgPool1d"])
@pytest.mark.parametrize("interpolation", ["linear", "nearest"])
def test_nhits_full_residual_reference(pool, interpolation):
    model = construct("MLP", "NHITS", pooling_mode=pool, interpolation_mode=interpolation).eval()
    x = torch.randn(2, 32, 3)
    series = x.transpose(1, 2).reshape(6, 32)
    residual, forecast = series.flip(-1), series[:, -1:]
    for block in model.blocks:
        kernel = block.pool.kernel_size
        pooling = F.max_pool1d if pool == "MaxPool1d" else F.avg_pool1d
        pooled = pooling(residual[:, None], kernel_size=kernel, stride=kernel, ceil_mode=True).squeeze(1)
        theta = _linear_mlp(block.layers, pooled)
        residual = residual - theta[:, :32]
        forecast = forecast + F.interpolate(theta[:, None, 32:], size=8, mode=interpolation).squeeze(1)
    expected = forecast.reshape(2, 3, 8).transpose(1, 2)
    torch.testing.assert_close(model(x), expected, rtol=0, atol=0)
    single_channel = construct("MLP", "NHITS", input_dim=1, pooling_mode=pool, interpolation_mode=interpolation).eval()
    single_channel.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(model(x)[:, :, :1], single_channel(x[:, :, :1]))


def test_pattn_independent_attention_and_patch_reference():
    model = construct("Transformer", "PAttn").eval()
    x = torch.randn(2, 32, 3)
    mean = x.mean(1, keepdim=True).detach()
    std = ((x - mean).var(1, keepdim=True, unbiased=False) + 1e-5).sqrt()
    history = ((x - mean) / std).permute(0, 2, 1)
    history = torch.cat((history, history[:, :, -1:].repeat(1, 1, 4)), dim=2)
    patches = history.unfold(-1, 8, 4)
    z = F.linear(patches, model.in_layer.weight, model.in_layer.bias).reshape(6, -1, 8)
    layer = model.encoder
    def projected(projection):
        return F.linear(z, projection.weight, projection.bias).reshape(6, z.shape[1], 2, 4).transpose(1, 2)
    q, k, v = projected(layer.query_projection), projected(layer.key_projection), projected(layer.value_projection)
    mixed = ((q @ k.transpose(-1, -2) / math.sqrt(4)).softmax(-1) @ v).transpose(1, 2).reshape_as(z)
    z = layer.norm1(z + F.linear(mixed, layer.out_projection.weight, layer.out_projection.bias))
    ff = F.linear(z, layer.conv1.weight[:, :, 0], layer.conv1.bias)
    ff = F.linear(F.gelu(ff), layer.conv2.weight[:, :, 0], layer.conv2.bias)
    z = model.norm(layer.norm2(z + ff))
    expected = F.linear(z.reshape(2, 3, -1), model.out_layer.weight, model.out_layer.bias).transpose(1, 2) * std + mean
    torch.testing.assert_close(model(x), expected, rtol=1e-5, atol=1e-6)


def _ie_reference(block, x):
    z = F.linear(x.transpose(1, 2), block.spatial_proj[0].weight, block.spatial_proj[0].bias)
    z = F.linear(F.leaky_relu(z), block.spatial_proj[2].weight, block.spatial_proj[2].bias).transpose(1, 2)
    z = z + F.linear(z, block.channel_proj.weight, block.channel_proj.bias)
    return F.linear(z.transpose(1, 2), block.output_proj.weight, block.output_proj.bias).transpose(1, 2)


def test_lightts_continuous_interval_sampling_reference():
    model = construct("MLP", "LightTS").eval()
    x = torch.randn(2, 32, 3)
    continuous = x.reshape(2, 8, 4, 3).permute(0, 3, 2, 1).reshape(6, 4, 8)
    interval = x.reshape(2, 4, 8, 3).permute(0, 3, 1, 2).reshape(6, 4, 8)
    first = F.linear(_ie_reference(model.layer1, continuous), model.chunk_proj1.weight, model.chunk_proj1.bias).squeeze(-1)
    second = F.linear(_ie_reference(model.layer2, interval), model.chunk_proj2.weight, model.chunk_proj2.bias).squeeze(-1)
    joined = torch.cat((first, second), -1).reshape(2, 3, -1).transpose(1, 2)
    expected = _ie_reference(model.layer3, joined) + F.linear(x.transpose(1, 2), model.ar.weight, model.ar.bias).transpose(1, 2)
    torch.testing.assert_close(model(x), expected, rtol=0, atol=0)


@pytest.mark.parametrize("changes", [
    {"n_blocks": [1]}, {"n_pool_kernel_size": [True, 2]}, {"n_freq_downsample": [0, 1]},
    {"mlp_units": [[8, 16], [8, 8]]}, {"interpolation_mode": "cubic"}, {"pooling_mode": "unknown"},
])
def test_nhits_explicit_supported_structure(changes):
    with pytest.raises(ValueError):
        construct("MLP", "NHITS", **changes)


def test_lightts_does_not_clamp_chunks_or_pad_history():
    for fields in ({"chunk_size": 3}, {"chunk_size": 16}, {"d_model": 8}, {"d_model": 18}):
        with pytest.raises(ValueError):
            construct("MLP", "LightTS", **fields)


def test_pattn_invalid_patch_geometry():
    with pytest.raises(ValueError):
        construct("Transformer", "PAttn", stride=9, patch_size=8)
