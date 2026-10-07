"""MWA-CNN source-architecture, waveform, gradient and recovery contracts."""
from __future__ import annotations

import io
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

pytest.importorskip("pytorch_wavelets", reason="install phmfactory[mwa] for MWA tests")
import pywt
from pytorch_wavelets import DWT1DForward

from src.model_factory.X_model.MWA_CNN import Model


def args(**changes):
    return SimpleNamespace(**({"depth": 6, "in_channels": 1, "num_classes": 3} | changes))


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old_threads)


def test_six_stage_architecture_matches_published_shapes_and_filters():
    model = Model(args()).eval()
    assert sum(isinstance(m, DWT1DForward) for m in model.modules()) == 6
    # These are the published six-stage dimensions, not the old four-stage proxy.
    expected_convolutions = {
        "SConv1": (12, 2, 3), "SConv2": (24, 24, 3),
        "SConv3": (48, 48, 3), "SConv4": (96, 96, 3),
        "SConv5": (192, 192, 3), "SConv6": (384, 384, 3),
    }
    for name, shape in expected_convolutions.items():
        layer = getattr(model, name)
        assert tuple(layer.conv[0].weight.shape) == shape
        assert layer.conv[1].num_groups == 6
    assert model.fc.in_features == 384
    assert [m.p for m in model.modules() if isinstance(m, nn.Dropout)] == [0.1] * 5
    assert tuple(model.cSE5.conv1[0].weight.shape) == (192, 384, 1)
    assert tuple(model.cSE5.conv2[0].weight.shape) == (384, 192, 1)
    assert all(p.device.type == "cpu" for p in model.parameters())
    assert all(b.device.type == "cpu" for b in model.buffers())

    signal = np.random.default_rng(8).normal(size=(2, 1, 65)).astype("float32")
    expected_low, expected_high = pywt.dwt(signal, "db16", mode="zero", axis=-1)
    for stage in range(6):
        low, high = getattr(model, f"DWT{stage}")(torch.from_numpy(signal))
        np.testing.assert_allclose(low.numpy(), expected_low, atol=6e-7, rtol=3e-6)
        np.testing.assert_allclose(high[0].numpy(), expected_high, atol=6e-7, rtol=3e-6)


@pytest.mark.parametrize("depth", [4, 6])
def test_gradients_optimizer_step_and_checkpoint_recovery(depth):
    torch.manual_seed(42)
    model = Model(args(depth=depth))
    waveform = torch.randn(4, 128, 1, requires_grad=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    before = model.fc.weight.detach().clone()
    loss = nn.functional.cross_entropy(model(waveform), torch.tensor([0, 1, 2, 1]))
    loss.backward()
    assert waveform.grad is not None and torch.isfinite(waveform.grad).all()
    assert waveform.grad.abs().sum() > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    assert not torch.equal(before, model.fc.weight)

    model.eval()
    expected = model(waveform.detach()).detach()
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = Model(args(depth=depth)).eval()
    restored.load_state_dict(torch.load(buffer, map_location="cpu", weights_only=True), strict=True)
    torch.testing.assert_close(restored(waveform.detach()), expected, rtol=0, atol=0)
    assert restored(torch.randn(1, 32, 1)).shape == (1, 3)


def test_legacy_default_preserves_checkpoint_module_identity():
    model = Model(SimpleNamespace(in_channels=1, num_classes=3))
    assert model.depth == 4 and model.fc.in_features == 96
    assert "SConv6.conv.0.weight" in model.state_dict()
    assert not hasattr(model, "DWT4") and not hasattr(model, "SConv4")
    with pytest.raises(RuntimeError):
        Model(args()).load_state_dict(model.state_dict(), strict=True)


def test_multichannel_factory_layout_and_config_alias():
    model = Model(SimpleNamespace(depth=6, input_dim=2, num_classes=5)).eval()
    assert model(torch.randn(2, 96, 2)).shape == (2, 5)
    with pytest.raises(ValueError, match="must agree"):
        Model(args(input_dim=2))


def test_public_model_factory_constructs_explicit_six_stage_model():
    from src.model_factory.model_factory import model_factory

    config = args(type="X_model", name="MWA_CNN")
    model = model_factory(config, metadata=None).eval()
    assert model.depth == 6
    assert model(torch.randn(2, 128, 1)).shape == (2, 3)


@pytest.mark.parametrize("config", [{"depth": 5}, {"depth": True}, {"in_channels": 0}, {"num_classes": {"x": 3}}])
def test_invalid_model_identity_fails(config):
    with pytest.raises(ValueError):
        Model(args(**config))


@pytest.mark.parametrize("shape, message", [((2, 64), "expects"), ((2, 64, 2), "channels"), ((2, 31, 1), "32 measured"), ((0, 64, 1), "nonempty")])
def test_invalid_waveform_fails_without_repair(shape, message):
    with pytest.raises(ValueError, match=message):
        Model(args()).eval()(torch.randn(*shape))


def test_training_singleton_explains_attention_batchnorm_requirement():
    with pytest.raises(ValueError, match="at least 2"):
        Model(args())(torch.randn(1, 64, 1))
