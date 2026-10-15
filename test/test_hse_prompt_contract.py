from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from src.model_factory.ISFM_Prompt.embedding.HSE_prompt import HSE_prompt
from src.model_factory.ISFM_Prompt.M_02_ISFM_Prompt import Model


def _hse_args(**overrides) -> SimpleNamespace:
    values = {
        "patch_size_L": 4,
        "patch_size_C": 1,
        "num_patches": 5,
        "output_dim": 8,
        "use_prompt": True,
        "prompt_dim": 4,
        "max_dataset_ids": 4,
        "prompt_combination": "add",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.randn(2, 16, 1),
        torch.tensor([1000.0, 1200.0]),
        torch.tensor([1, 2]),
    )


class _Metadata:
    def __getitem__(self, file_id: int) -> dict[str, float | int]:
        return {"Dataset_id": 1, "Sample_rate": 1000.0}


def _model_args(**overrides) -> SimpleNamespace:
    values = vars(_hse_args()).copy()
    values.update(
        {
            "embedding": "HSE_prompt",
            "backbone": "B_04_Dlinear",
            "task_head": "H_01_Linear_cla",
            "training_stage": "pretrain",
            "num_classes": {1: 3},
        }
    )
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    "dataset_ids,match",
    [
        (None, "dataset_ids are required"),
        (torch.tensor([1]), "one ID per sample"),
        (torch.tensor([-1, 2]), "prompt table range"),
        (torch.tensor([1, 4]), "prompt table range"),
        (torch.tensor([1.5, 2.0]), "integer values"),
    ],
)
def test_prompt_enabled_path_rejects_invalid_dataset_ids(
    dataset_ids: torch.Tensor | None,
    match: str,
) -> None:
    model = HSE_prompt(_hse_args())
    signal, fs, _ = _inputs()

    with pytest.raises(ValueError, match=match):
        model(signal, fs, dataset_ids)


def test_prompt_path_rejects_non_finite_inputs() -> None:
    model = HSE_prompt(_hse_args())
    signal, fs, dataset_ids = _inputs()
    signal[0, 0, 0] = float("nan")

    with pytest.raises(ValueError, match="x contains NaN or Inf"):
        model(signal, fs, dataset_ids)


def test_eval_patch_selection_is_deterministic() -> None:
    model = HSE_prompt(_hse_args()).eval()
    signal, fs, dataset_ids = _inputs()

    with torch.no_grad():
        first = model(signal, fs, dataset_ids)
        second = model(signal, fs, dataset_ids)

    torch.testing.assert_close(first, second)


def test_prompt_disabled_path_allows_explicit_signal_only_mode() -> None:
    model = HSE_prompt(_hse_args(use_prompt=False)).eval()
    signal, fs, _ = _inputs()

    with torch.no_grad():
        output = model(signal, fs, dataset_ids=None)

    assert output.shape == (2, 5, 8)


def test_model_requires_metadata_and_explicit_components() -> None:
    with pytest.raises(ValueError, match="requires metadata"):
        Model(_model_args(), metadata=None)

    for field in ("embedding", "backbone", "task_head"):
        values = vars(_model_args()).copy()
        del values[field]
        with pytest.raises(ValueError, match=rf"model\.{field} must be explicitly"):
            Model(SimpleNamespace(**values), metadata=_Metadata())


def test_model_requires_file_id_for_metadata_resolution() -> None:
    model = Model(_model_args(), metadata=_Metadata())
    signal = torch.randn(2, 16, 1)

    with pytest.raises(ValueError, match="file_id is required"):
        model(signal, file_id=None, task_id="classification")


def test_model_uses_explicit_metadata_for_prompt_and_head() -> None:
    model = Model(_model_args(), metadata=_Metadata()).eval()
    signal = torch.randn(2, 16, 1)

    with torch.no_grad():
        logits = model(signal, file_id=7, task_id="classification")

    assert logits.shape == (2, 3)


def test_task_head_type_error_is_not_retried_with_changed_arguments() -> None:
    model = Model(_model_args(), metadata=_Metadata())

    class _BrokenHead(nn.Module):
        def forward(self, x, **kwargs):
            raise TypeError("head contract mismatch")

    model.task_head = _BrokenHead()
    with pytest.raises(TypeError, match="head contract mismatch"):
        model(torch.randn(2, 16, 1), file_id=7, task_id="classification")
