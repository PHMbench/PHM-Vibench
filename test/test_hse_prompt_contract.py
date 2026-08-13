from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from src.model_factory.ISFM_Prompt.embedding.HSE_prompt import HSE_prompt


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
