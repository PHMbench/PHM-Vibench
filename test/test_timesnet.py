"""TimesNet: real module contracts and fixed-source classification parity.

The reference is the upstream implementation, not the port copied as an oracle.
Its two imports are relocated; numerical methods are unchanged.
"""
from copy import deepcopy
from types import SimpleNamespace
from pathlib import Path
import importlib.util
import sys

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from src.model_factory.CNN.TimesNet import Model, TimesBlock, fft_periods
# A private package name avoids Python's stdlib `test` package and global `layers`.
_reference_dir = Path(__file__).parent / "fixtures" / "timesnet_upstream"
_spec = importlib.util.spec_from_file_location(
    "_phm_timesnet_reference", _reference_dir / "__init__.py",
    submodule_search_locations=[str(_reference_dir)],
)
_reference = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _reference
_spec.loader.exec_module(_reference)
upstream = _reference.TimesNet


def arguments(**changes):
    fields = dict(seq_len=32, input_dim=2, num_classes=3, d_model=8,
                  d_ff=16, e_layers=2, dropout=0.0, top_k=2, num_kernels=2)
    fields.update(changes)
    return SimpleNamespace(**fields)


def reference_arguments(args):
    return SimpleNamespace(**vars(args), task_name="classification", pred_len=0,
                           label_len=0, enc_in=args.input_dim, num_class=args.num_classes,
                           embed="timeF", freq="h")


def reference_key(name):
    return {"enc_embedding.token.weight": "enc_embedding.value_embedding.tokenConv.weight",
            "enc_embedding.position": "enc_embedding.position_embedding.pe"}.get(name, name)


def align_from_reference(model, reference):
    source = reference.state_dict()
    mapped = {}
    for key in model.state_dict():
        value = source[reference_key(key)]
        if key == "enc_embedding.position":
            value = value[:, :model.seq_len]
        mapped[key] = value.clone()
    unused = set(source) - {reference_key(k) for k in mapped}
    assert unused == {"enc_embedding.temporal_embedding.embed.weight"}
    model.load_state_dict(mapped, strict=True)


@pytest.mark.parametrize("length,channels,depth", [(31, 1, 1), (32, 2, 2), (48, 3, 2)])
@pytest.mark.parametrize("training", [False, True])
def test_upstream_logits_loss_input_and_all_active_parameter_gradients(length, channels, depth, training):
    torch.manual_seed(38)
    args = arguments(seq_len=length, input_dim=channels, e_layers=depth, dropout=0.2)
    reference = upstream.Model(reference_arguments(args)).train(training)
    model = Model(args).train(training)
    align_from_reference(model, reference)
    x = torch.randn(3, length, channels, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_()
    y = torch.tensor([0, 1, 2])
    rng = torch.random.get_rng_state()
    actual = model(x)
    after = torch.random.get_rng_state()
    torch.random.set_rng_state(rng)
    expected = reference(x_ref, torch.ones(3, length), None, None)
    assert torch.equal(after, torch.random.get_rng_state())
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    loss = F.cross_entropy(actual, y)
    expected_loss = F.cross_entropy(expected, y)
    torch.testing.assert_close(loss, expected_loss, rtol=2e-5, atol=2e-6)
    loss.backward()
    expected_loss.backward()
    torch.testing.assert_close(x.grad, x_ref.grad, rtol=3e-5, atol=3e-6)
    original_parameters = dict(reference.named_parameters())
    for key, parameter in model.named_parameters():
        oracle = original_parameters[reference_key(key)]
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), key
        torch.testing.assert_close(parameter.grad, oracle.grad, rtol=3e-5, atol=3e-6, msg=key)


def test_period_weights_and_block_match_unmodified_upstream():
    torch.manual_seed(18)
    args = arguments(seq_len=31)
    reference = upstream.TimesBlock(reference_arguments(args))
    block = TimesBlock(args.d_model, args.d_ff, args.top_k, args.num_kernels)
    block.load_state_dict(reference.state_dict(), strict=True)
    x = torch.randn(3, 31, args.d_model)
    periods, weights = fft_periods(x, args.top_k)
    original_periods, original_weights = upstream.FFT_for_Period(x, args.top_k)
    assert periods == original_periods.tolist()
    torch.testing.assert_close(weights, original_weights, rtol=0, atol=0)
    torch.testing.assert_close(block(x), reference(x), rtol=2e-5, atol=2e-6)


def test_explicit_dc_tie_correction_not_claimed_as_upstream_parity():
    x = torch.zeros(1, 2, 1)
    # This CPU fixture reproduces upstream selection of DC despite zeroing it.
    with np.errstate(divide="raise", invalid="raise"):
        with pytest.raises(FloatingPointError):
            upstream.FFT_for_Period(x, k=1)
    periods, weights = fft_periods(x, 1)
    assert periods == [2]
    assert torch.equal(weights, torch.zeros(1, 1))
    model = Model(arguments(seq_len=2, top_k=1)).eval()
    assert torch.isfinite(model(torch.zeros(2, 2, 2))).all()


def test_optimizer_features_no_mutation_and_strict_serialization(tmp_path):
    torch.manual_seed(12)
    args = arguments(num_classes={0: 3})
    before = deepcopy(vars(args))
    model = Model(args)
    x = torch.randn(4, 32, 2)
    source = x.clone()
    logits, features = model(x, file_id=torch.arange(4), task_id="classification", return_feature=True)
    assert logits.shape == (4, 3) and features.shape == (4, 32 * 8)
    saved = [p.detach().clone() for p in model.parameters()]
    F.cross_entropy(logits, torch.tensor([0, 1, 2, 0])).backward()
    torch.optim.Adam(model.parameters(), lr=0.001).step()
    assert any(not torch.equal(a, b) for a, b in zip(saved, model.parameters()))
    assert vars(args) == before and torch.equal(x, source)
    model.eval()
    rng = torch.random.get_rng_state()
    with torch.no_grad():
        expected = model(x)
    assert torch.equal(rng, torch.random.get_rng_state())
    checkpoint = tmp_path / "timesnet.pt"
    torch.save(model.state_dict(), checkpoint)
    restored = Model(args).eval()
    restored.load_state_dict(torch.load(checkpoint, weights_only=True), strict=True)
    with torch.no_grad():
        torch.testing.assert_close(restored(x), expected, rtol=0, atol=0)
        # Permuting the same batch preserves the batch-selected periods.
        torch.testing.assert_close(restored(x.flip(0)).flip(0), expected, rtol=2e-5, atol=2e-6)
    bad = restored.state_dict()
    del bad["projection.bias"]
    with pytest.raises(RuntimeError, match="Missing key"):
        restored.load_state_dict(bad, strict=True)


@pytest.mark.parametrize("case", ["rank", "length", "channels", "empty", "dtype", "nan", "inf", "task"])
def test_invalid_request_does_not_change_model_path(case):
    model = Model(arguments())
    x, task = torch.randn(2, 32, 2), "classification"
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
    else:
        task = "forecasting"
    with pytest.raises((ValueError, TypeError, FloatingPointError)):
        model(x, task_id=task)


@pytest.mark.parametrize("changes", [
    {"seq_len": "32"}, {"seq_len": 1}, {"seq_len": 5001}, {"d_model": 7},
    {"top_k": 17}, {"e_layers": True}, {"num_kernels": 0},
    {"num_classes": 1}, {"num_classes": {0: 3, 1: 4}},
    {"dropout": "0.1"}, {"dropout": float("nan")}, {"dropout": 1.0},
])
def test_invalid_configuration_is_not_coerced(changes):
    with pytest.raises((TypeError, ValueError)):
        Model(arguments(**changes))
