from __future__ import annotations

import pytest
import torch

from src.Pipeline_02_Pretraining_Few_Shot import (
    _require_completed_stage_evaluation,
)


def _stage_result(value):
    return {
        "stage_1": {
            "checkpoint_path": "stage1.ckpt",
            "metrics": {"test_metric": value},
        }
    }


def test_completed_stage_accepts_finite_python_and_tensor_scalars() -> None:
    result = {
        "stage_1": {
            "checkpoint_path": "stage1.ckpt",
            "metrics": {
                "test_loss": 0.25,
                "test_accuracy": torch.tensor(0.75),
            },
        }
    }

    assert _require_completed_stage_evaluation(result, "test") is result


@pytest.mark.parametrize(
    "value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
        torch.tensor(float("nan")),
    ],
)
def test_completed_stage_rejects_non_finite_metrics(value) -> None:
    with pytest.raises(FloatingPointError, match="is not finite"):
        _require_completed_stage_evaluation(_stage_result(value), "test")


@pytest.mark.parametrize(
    "value",
    [
        True,
        "0.5",
        [0.5],
        torch.tensor([0.25, 0.75]),
    ],
)
def test_completed_stage_rejects_non_scalar_or_non_numeric_metrics(value) -> None:
    with pytest.raises(RuntimeError, match="scalar"):
        _require_completed_stage_evaluation(_stage_result(value), "test")


@pytest.mark.parametrize('names', [('first',), ('second', 'first'), ('first', 'second', 'extra')])
def test_request_stage_coverage_and_order_are_checked(names):
    result = {name: {'metrics': {'loss': 1.0}} for name in names}
    with pytest.raises(RuntimeError, match='coverage/order mismatch'):
        _require_completed_stage_evaluation(result, 'test', {'first': True, 'second': True})


def test_only_explicit_train_only_contract_allows_no_test_metrics(tmp_path):
    checkpoint = tmp_path / 'selected.ckpt'
    checkpoint.write_bytes(b'existence fixture')
    result = {'train': {'status': 'trained_not_evaluated', 'checkpoint_path': str(checkpoint)}}
    assert _require_completed_stage_evaluation(result, 'test', {'train': False}) is result
    with pytest.raises(RuntimeError, match='did not complete evaluation'):
        _require_completed_stage_evaluation(result, 'test', {'train': True})
    result['train']['metrics'] = {'dry_run': True}
    with pytest.raises(RuntimeError, match='without test metrics'):
        _require_completed_stage_evaluation(result, 'test', {'train': False})


def test_train_only_must_supply_actual_selected_state(tmp_path):
    result = {'train': {'status': 'trained_not_evaluated',
                        'checkpoint_path': str(tmp_path / 'missing.ckpt')}}
    with pytest.raises(RuntimeError, match='selected checkpoint'):
        _require_completed_stage_evaluation(result, 'test', {'train': False})
