"""Native ModernTCN classification and explicit multi-stage shape semantics."""
from copy import deepcopy
from types import SimpleNamespace
import pytest
import torch
from torch.nn import functional as F
from src.model_factory.CNN.ModernTCN import Model


def args(**changes):
    values = dict(seq_len=32, input_dim=3, num_classes=4, patch_size=8, patch_stride=4,
                  downsample_ratio=2, ffn_ratio=2, dims=[8, 16], num_blocks=[1, 1],
                  large_size=[7, 7], small_size=[3, 3], dropout=0.0, class_dropout=0.0)
    values.update(changes)
    return SimpleNamespace(**values)


def test_gradients_parameter_update_and_configuration_immutability():
    torch.manual_seed(22)
    configuration = args()
    before = deepcopy(vars(configuration))
    model = Model(configuration, None)
    x = torch.randn(4, 32, 3, requires_grad=True)
    source = x.detach().clone()
    logits, features = model(x, task_id="classification", return_feature=True)
    assert logits.shape == (4, 4) and features.shape == (4, 3 * 16 * 4)
    F.cross_entropy(logits, torch.tensor([0, 1, 2, 3])).backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    saved = [p.detach().clone() for p in model.parameters()]
    torch.optim.Adam(model.parameters(), lr=0.001).step()
    assert any(not torch.equal(a, b) for a, b in zip(saved, model.parameters()))
    assert vars(configuration) == before
    torch.testing.assert_close(source, x.detach(), rtol=0, atol=0)


def _conv(layer, value):
    return F.conv1d(value, layer.weight, layer.bias, stride=layer.stride,
                    padding=layer.padding, dilation=layer.dilation, groups=layer.groups)


def _bn(layer, value):
    return F.batch_norm(value, layer.running_mean, layer.running_var,
                        layer.weight, layer.bias, training=False, eps=layer.eps)


def _block_reference(block, x):
    batch, variables, width, tokens = x.shape
    packed = x.reshape(batch, variables * width, tokens)
    z = _bn(block.large[1], _conv(block.large[0], packed))
    z = z + _bn(block.small[1], _conv(block.small[0], packed))
    z = _bn(block.norm, z.reshape(batch * variables, width, tokens)).reshape(batch, variables * width, tokens)
    assert block.ffn1pw1.groups == variables and block.ffn2pw1.groups == width
    z = _conv(block.ffn1pw2, F.gelu(_conv(block.ffn1pw1, z)))
    z = z.reshape(batch, variables, width, tokens).permute(0, 2, 1, 3).reshape(batch, width * variables, tokens)
    z = _conv(block.ffn2pw2, F.gelu(_conv(block.ffn2pw1, z)))
    return x + z.reshape(batch, width, variables, tokens).permute(0, 2, 1, 3)


@pytest.mark.parametrize("stages", [1, 2])
def test_full_classification_matches_upstream_operation_order(stages):
    configuration = args(dims=[8, 16][:stages], num_blocks=[1, 1][:stages],
                         large_size=[7, 7][:stages], small_size=[3, 3][:stages])
    model = Model(configuration, None).eval()
    x = torch.randn(2, 32, 3)
    z = x.permute(0, 2, 1).unsqueeze(2)
    for index, (layers, stage) in enumerate(zip(model.downsample_layers, model.stages)):
        packed = z.reshape(6, z.shape[2], z.shape[3])
        if index == 0:
            packed = torch.cat((packed, packed[:, :, -1:].repeat(1, 1, 4)), -1)
            packed = _bn(layers[1], _conv(layers[0], packed))
        else:
            packed = _conv(layers[1], _bn(layers[0], packed))
        z = packed.reshape(2, 3, packed.shape[1], packed.shape[2])
        for block in stage:
            z = _block_reference(block, z)
    expected = F.linear(F.gelu(z).reshape(2, -1), model.classifier.weight, model.classifier.bias)
    torch.testing.assert_close(model(x), expected, rtol=1e-5, atol=1e-6)


def test_checkpoint_eval_and_rng_independence(tmp_path):
    model = Model(args(dropout=0.2, class_dropout=0.2), None)
    model(torch.randn(4, 32, 3))
    model.eval()
    x = torch.randn(2, 32, 3)
    rng = torch.random.get_rng_state()
    with torch.no_grad():
        expected = model(x)
    assert torch.equal(rng, torch.random.get_rng_state())
    path = tmp_path / "modern.pt"
    torch.save(model.state_dict(), path)
    restored = Model(args(dropout=0.2, class_dropout=0.2), None).eval()
    restored.load_state_dict(torch.load(path, weights_only=True), strict=True)
    with torch.no_grad():
        torch.testing.assert_close(restored(x), expected, rtol=0, atol=0)
        assert torch.isfinite(restored(torch.zeros_like(x))).all()


@pytest.mark.parametrize("fault", ["rank", "length", "channels", "empty", "dtype", "nan", "inf", "task"])
def test_invalid_signal(fault):
    model = Model(args(), None)
    x, task = torch.randn(2, 32, 3), "classification"
    if fault == "rank": x = x[0]
    elif fault == "length": x = x[:, :-1]
    elif fault == "channels": x = x[:, :, :1]
    elif fault == "empty": x = x[:0]
    elif fault == "dtype": x = x.double()
    elif fault in ("nan", "inf"): x[0, 0, 0] = float(fault)
    elif fault == "task": task = "forecasting"
    with pytest.raises((ValueError, TypeError, FloatingPointError)):
        model(x, task_id=task)


@pytest.mark.parametrize("changes", [
    {"dims": [8]}, {"num_blocks": []}, {"large_size": [6, 7]},
    {"small_size": [9, 3]}, {"patch_stride": 9}, {"seq_len": 31},
    {"dims": [8, 16, 32], "num_blocks": [1, 1, 1], "large_size": [7, 7, 7], "small_size": [3, 3, 3], "seq_len": 24},
    {"dropout": "0.1"}, {"ffn_ratio": True}, {"num_classes": {0: 3, 1: 4}},
])
def test_invalid_architecture_has_no_silent_repair(changes):
    with pytest.raises((ValueError, TypeError)):
        Model(args(**changes), None)
