from types import SimpleNamespace

import pytest
import torch

from src.model_factory.Transformer.TSLTransformer import Model


def _args(**overrides):
    values = {
        "input_dim": 2,
        "seq_len": 128,
        "patch_size": 16,
        "d_model": 32,
        "n_heads": 4,
        "num_layers": 2,
        "d_ff": 64,
        "lstm_hidden_dim": 24,
        "dropout": 0.0,
        "num_classes": 3,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_tsl_transformer_shape_feature_and_backward():
    model = Model(_args())
    x = torch.randn(4, 128, 2)
    logits, features = model(x, task_id="classification", return_feature=True)

    assert logits.shape == (4, 3)
    assert features.shape == (4, 32)
    loss = logits.square().mean()
    loss.backward()
    assert model.patch_projection.weight.grad is not None
    assert model.blocks[0].attention.in_proj_weight.grad is not None
    assert model.blocks[0].lstm.weight_ih_l0.grad is not None


def test_tsl_transformer_eval_is_deterministic():
    model = Model(_args(dropout=0.2)).eval()
    x = torch.randn(2, 128, 2)
    with torch.no_grad():
        first = model(x)
        second = model(x)
    torch.testing.assert_close(first, second)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"seq_len": 127}, "divisible"),
        ({"d_model": 30}, "n_heads"),
        ({"dropout": 1.0}, "dropout"),
        ({"num_classes": {0: 2}}, "integer"),
    ],
)
def test_tsl_transformer_rejects_invalid_configuration(overrides, message):
    with pytest.raises(ValueError, match=message):
        Model(_args(**overrides))


@pytest.mark.parametrize("shape", [(2, 64, 2), (2, 128, 1), (2, 128)])
def test_tsl_transformer_rejects_input_contract_changes(shape):
    model = Model(_args())
    with pytest.raises(ValueError, match="expects|mismatch"):
        model(torch.randn(*shape))
