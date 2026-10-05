"""Shared TSPN configuration contracts; synthetic checks are not paper reproduction."""
from collections import OrderedDict
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from src.model_factory.X_model.Feature_extract import EntropyFeature, KurtosisFeature
from src.model_factory.X_model.TSPN import (
    Classifier, CustomBatchNorm, FeatureExtractorlayer, Model, SignalProcessingLayer,
)
from src.model_factory.model_factory import model_factory


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _args(**overrides):
    values = dict(
        type="X_model", name="TSPN", device="cpu", in_channels=1,
        in_dim=64, out_dim=64, out_channels=1, scale=4, num_classes=4,
        skip_connection=True,
        signal_processing_configs={f"layer{i}": ["WF"] for i in range(3)},
        feature_extractor_configs=["Entropy", "Mean", "Kurtosis"],
        f_c_mu=.2, f_c_sigma=.01, f_b_mu=.05, f_b_sigma=.002,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _candidate_args(**overrides):
    values = dict(
        classifier_hidden_dims=[4], classifier_activation="identity",
        classifier_bias=False, gate_parameterization="raw", gate_bias=False,
        skip_bias=False, feature_mixing="per_feature", feature_mixing_bias=False,
        feature_definitions={"Entropy": "absolute_mean_xlogx",
                             "Kurtosis": "population_moment"},
        feature_epsilon=1e-12,
    )
    values.update(overrides)
    return _args(**values)


def _legacy_state():
    """A literal pre-option checkpoint contract, independent of model.state_dict()."""
    generator = torch.Generator().manual_seed(17)
    state = OrderedDict()
    for index, input_width in enumerate([1, 4, 4]):
        prefix = f"signal_processing_layers.{index}."
        state[prefix + "weight_connection.weight"] = torch.randn(4, input_width, generator=generator) * .1
        state[prefix + "weight_connection.bias"] = torch.randn(4, generator=generator) * .1
        state[prefix + "signal_processing_modules.WF.f_c"] = torch.tensor([.12, .2, .3, .4]).reshape(1, 1, 4)
        state[prefix + "signal_processing_modules.WF.f_b"] = torch.full((1, 1, 4), .08)
        state[prefix + "skip_connection.weight"] = torch.randn(4, input_width, generator=generator) * .1
        state[prefix + "skip_connection.bias"] = torch.randn(4, generator=generator) * .1
    prefix = "feature_extractor_layers."
    state[prefix + "weight_connection.weight"] = torch.randn(4, 4, generator=generator) * .1
    state[prefix + "weight_connection.bias"] = torch.randn(4, generator=generator) * .1
    state[prefix + "norm.running_mean"] = torch.linspace(-.1, .1, 12).reshape(1, 12)
    state[prefix + "norm.running_var"] = torch.linspace(.8, 1.2, 12).reshape(1, 12)
    state["clf.clf.0.weight"] = torch.randn(128, 12, generator=generator) * .1
    state["clf.clf.0.bias"] = torch.randn(128, generator=generator) * .1
    state["clf.clf.2.weight"] = torch.randn(4, 128, generator=generator) * .1
    state["clf.clf.2.bias"] = torch.randn(4, generator=generator) * .1
    return state


def _legacy_forward(x, state, training):
    """Functional pre-option equations, including the original entropy statistic."""
    for index in range(3):
        prefix = f"signal_processing_layers.{index}."
        source = F.instance_norm(x.transpose(1, 2)).transpose(1, 2)
        mixed = F.linear(source, (state[prefix + "weight_connection.weight"] / .1).softmax(dim=0),
                         state[prefix + "weight_connection.bias"])
        omega = torch.linspace(0, .5, mixed.shape[1] // 2 + 1).reshape(1, -1, 1)
        response = torch.exp(-((omega - state[prefix + "signal_processing_modules.WF.f_c"]) /
                               (2 * state[prefix + "signal_processing_modules.WF.f_b"])) ** 2)
        x = torch.fft.irfft(torch.fft.rfft(mixed, dim=1, norm="ortho") * response, dim=1, norm="ortho")
        x = x + F.linear(source, state[prefix + "skip_connection.weight"], state[prefix + "skip_connection.bias"])
    prefix = "feature_extractor_layers."
    x = F.instance_norm(x.transpose(1, 2)).transpose(1, 2)
    x = F.linear(x, state[prefix + "weight_connection.weight"], state[prefix + "weight_connection.bias"]).transpose(1, 2)
    mean = x.mean(dim=-1)
    features = torch.cat([
        (x * x.softmax(dim=-1).log()).mean(dim=-1), mean,
        (x - mean.unsqueeze(-1)).pow(4).mean(dim=-1) / x.var(dim=-1).square(),
    ], dim=1)
    if training:
        center, variance = features.mean(dim=0), features.var(dim=0, unbiased=False)
    else:
        center = state[prefix + "norm.running_mean"]
        variance = state[prefix + "norm.running_var"]
    standardized = (features - center) / (variance.sqrt() + .1)
    hidden = F.linear(standardized, state["clf.clf.0.weight"], state["clf.clf.0.bias"]).relu()
    logits = F.linear(hidden, state["clf.clf.2.weight"], state["clf.clf.2.bias"])
    return logits, center, variance


@pytest.mark.parametrize("training", [False, True])
def test_default_strict_load_preserves_legacy_forward_and_running_statistics(training):
    state = _legacy_state()
    model = model_factory(_args(), metadata=None)
    assert set(model.state_dict()) == set(state)
    model.load_state_dict(state, strict=True)
    model.train(training)
    x = torch.randn(5, 64, 1, generator=torch.Generator().manual_seed(13))
    expected, mean, variance = _legacy_forward(x, state, training)
    torch.testing.assert_close(model(x), expected, rtol=2e-5, atol=2e-6)
    norm = model.feature_extractor_layers.norm
    if training:
        torch.testing.assert_close(norm.running_mean, .9 * state["feature_extractor_layers.norm.running_mean"] + .1 * mean)
        torch.testing.assert_close(norm.running_var, .9 * state["feature_extractor_layers.norm.running_var"] + .1 * variance)
    assert not norm.running_mean.requires_grad and norm.running_mean.grad_fn is None
    assert not norm.running_var.requires_grad and norm.running_var.grad_fn is None


@pytest.mark.parametrize("length", [63, 64])
def test_explicit_configuration_shape_parameter_breakdown_and_gradients(length):
    model = model_factory(_candidate_args(in_dim=length, out_dim=length), metadata=None)
    groups = {"gates": 0, "skips": 0, "filters": 0, "features": 0, "head": 0}
    for name, parameter in model.named_parameters():
        if name.startswith("clf."):
            group = "head"
        elif name.startswith("feature_extractor_layers."):
            group = "features"
        elif "skip_connection" in name:
            group = "skips"
        elif "signal_processing_modules" in name:
            group = "filters"
        else:
            group = "gates"
        groups[group] += parameter.numel()
    assert groups == dict(gates=36, skips=36, filters=24, features=48, head=64)
    # This verifies the configured computation; 208 parameters do not prove fidelity.
    assert sum(groups.values()) == 208
    x = torch.randn(6, length, 1, requires_grad=True)
    lengths = []
    hooks = [layer.register_forward_hook(lambda _m, _i, output: lengths.append(output.shape[1]))
             for layer in model.signal_processing_layers]
    logits = model(x)
    for hook in hooks:
        hook.remove()
    assert logits.shape == (6, 4) and lengths == [length] * 3
    for layer in model.signal_processing_layers:
        wave_filter = layer.signal_processing_modules["WF"]
        torch.testing.assert_close(wave_filter.omega.flatten(), torch.fft.rfftfreq(length))
    F.cross_entropy(logits, torch.arange(6) % 4).backward()
    assert torch.isfinite(logits).all() and torch.isfinite(x.grad).all()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


def test_feature_definitions_are_explicit_and_numerically_well_defined():
    values = torch.tensor([[[-2., -1., 1., 2.]]], dtype=torch.float64)
    torch.testing.assert_close(KurtosisFeature()(values), torch.tensor([[[.765]]], dtype=torch.float64))
    torch.testing.assert_close(KurtosisFeature("population_moment")(values), torch.tensor([[[1.36]]], dtype=torch.float64))
    x = torch.tensor([[[1., 2.]]], dtype=torch.float64)
    # log(exp(1)+exp(2)) normalizes each time sample; weights remain signed x.
    normalizer = torch.log(torch.exp(torch.tensor(1., dtype=torch.float64)) + torch.exp(torch.tensor(2., dtype=torch.float64)))
    expected = (1 * (1 - normalizer) + 2 * (2 - normalizer)) / 2
    torch.testing.assert_close(EntropyFeature()(x).squeeze(), expected)
    extreme = torch.tensor([[[1000., -1000.]]], requires_grad=True)
    value = EntropyFeature()(extreme)
    torch.testing.assert_close(value, torch.tensor([[[1_000_000.]]]))
    value.sum().backward()
    assert torch.isfinite(extreme.grad).all()


def test_absolute_mean_statistic_and_population_floor_include_zero_signal():
    x = torch.tensor([[[-2., 0., 2.]]], dtype=torch.float64, requires_grad=True)
    entropy = EntropyFeature("absolute_mean_xlogx", epsilon=.01)
    expected = 4 * torch.log(torch.tensor(2.01, dtype=torch.float64)) / 3
    torch.testing.assert_close(entropy(x).squeeze(), expected)
    zero = torch.zeros(2, 3, 64, requires_grad=True)
    result = EntropyFeature("absolute_mean_xlogx")(zero) + KurtosisFeature("population_moment")(zero)
    torch.testing.assert_close(result, torch.zeros(2, 3, 1))
    result.sum().backward()
    assert torch.isfinite(zero.grad).all()
    tiny = torch.tensor([[[-1e-8, 1e-8]]], dtype=torch.float64)
    torch.testing.assert_close(KurtosisFeature("population_moment")(tiny),
                               torch.tensor([[[1e-8]]], dtype=torch.float64))
    model = Model(_candidate_args(feature_epsilon=.01))
    for name in ("Entropy", "Kurtosis"):
        assert model.feature_extractor_modules[name].epsilon == .01


def test_raw_and_temperature_scaled_gates_follow_declared_formula():
    modules = nn.ModuleDict({"I": nn.Identity()})
    layer = SignalProcessingLayer(modules, 1, 2, False, False,
                                  gate_parameterization="raw", gate_bias=False)
    with torch.no_grad():
        layer.weight_connection.weight.copy_(torch.tensor([[1.], [2.]]))
    x = torch.tensor([[[3.]]])
    torch.testing.assert_close(layer(x), torch.tensor([[[3., 6.]]]))
    layer.gate_parameterization = "softmax"
    layer.temperature = .5
    expected = torch.tensor([[[3 / (1 + torch.exp(torch.tensor(2.))),
                                3 / (1 + torch.exp(torch.tensor(-2.)))]]])
    torch.testing.assert_close(layer(x), expected)


def test_per_feature_mixing_has_distinct_statistic_inputs():
    modules = OrderedDict(Mean=nn.AdaptiveAvgPool1d(1), Energy=nn.AdaptiveAvgPool1d(1))
    layer = FeatureExtractorlayer(modules, 1, 1, False, mixing="per_feature", mixing_bias=False)
    layer.norm = nn.Identity()
    with torch.no_grad():
        layer.weight_connections["Mean"].weight.fill_(2.)
        layer.weight_connections["Energy"].weight.fill_(-3.)
    x = torch.tensor([[[1.], [3.]]])
    torch.testing.assert_close(layer(x), torch.tensor([[4., -6.]]))


def test_two_linear_identity_head_is_explicit_and_has_no_activation_or_bias():
    head = Classifier(12, 4, hidden_dims=[4], activation="identity", bias=False)
    x = torch.randn(3, 12)
    assert isinstance(head.clf[1], nn.Identity)
    assert all(layer.bias is None for layer in head.clf if isinstance(layer, nn.Linear))
    torch.testing.assert_close(head(x), F.linear(x, head.clf[2].weight @ head.clf[0].weight))
    linear = Classifier(12, 4, hidden_dims=[], bias=False)
    assert len(linear.clf) == 1
    assert sum(parameter.numel() for parameter in linear.parameters()) == 48


def test_source_running_stats_are_frozen_batch_invariant_and_strictly_restored(tmp_path):
    torch.manual_seed(23)
    args = _candidate_args()
    model = Model(args).train()
    optimizer = torch.optim.SGD(model.parameters(), lr=.001)
    for _ in range(2):
        optimizer.zero_grad()
        loss = F.cross_entropy(model(torch.randn(6, 64, 1)), torch.arange(6) % 4)
        loss.backward()
        optimizer.step()
    model.eval()
    buffers = {name: value.clone() for name, value in model.named_buffers()}
    probe = torch.randn(1, 64, 1)
    reference = model(probe)
    other_samples = torch.randn(3, 64, 1) * 100 + 200
    torch.testing.assert_close(model(torch.cat([probe, other_samples]))[:1], reference, rtol=2e-5, atol=2e-6)
    for name, value in model.named_buffers():
        torch.testing.assert_close(value, buffers[name], rtol=0, atol=0)
    path = tmp_path / "candidate.pt"
    torch.save(model.state_dict(), path)
    restored = Model(args).eval()
    restored.load_state_dict(torch.load(path, weights_only=True), strict=True)
    torch.testing.assert_close(restored(probe), reference, rtol=0, atol=0)
    assert torch.isfinite(restored(torch.ones(2, 64, 1))).all()


def test_constant_feature_batch_has_finite_normalization_gradient():
    x = torch.tensor([[1., -2.], [1., 0.], [1., 3.]], requires_grad=True)
    norm = CustomBatchNorm(2).train()
    actual = norm(x)
    expected = (x - x.mean(dim=0)) / (x.var(dim=0, unbiased=False).sqrt() + .1)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.square().sum().backward()
    assert torch.isfinite(x.grad).all()
    torch.testing.assert_close(x.grad[:, 0], torch.zeros(3))


@pytest.mark.parametrize("overrides,match", [
    ({"gate_parameterization": "unknown"}, "gate_parameterization"),
    ({"gate_temperature": 0.}, "gate_temperature"),
    ({"gate_temperature": float("nan")}, "gate_temperature"),
    ({"feature_mixing": "unknown"}, "feature_mixing"),
    ({"classifier_hidden_dims": [0]}, "classifier_hidden_dims"),
    ({"classifier_hidden_dims": [True]}, "classifier_hidden_dims"),
    ({"classifier_activation": "invented"}, "classifier_activation"),
    ({"feature_definitions": {"Entropy": "TON"}}, "Entropy"),
    ({"feature_definitions": {"Kurtosis": "unknown"}}, "Kurtosis"),
    ({"feature_definitions": {"RMS": "unknown"}}, "feature_definitions"),
    ({"feature_epsilon": 0.}, "feature_epsilon"),
])
def test_invalid_or_unsupported_paper_configuration_fails_explicitly(overrides, match):
    with pytest.raises(ValueError, match=match):
        Model(_args(**overrides))
