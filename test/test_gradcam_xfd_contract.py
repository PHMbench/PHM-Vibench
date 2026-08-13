from __future__ import annotations

import json

import pytest
import torch

from src.explain_factory.explainers.gradcam_xfd import GradCAM1DExplainer


class _TinyClassifier(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = torch.nn.Conv1d(1, 4, kernel_size=3, padding=1)
        self.pool = torch.nn.AdaptiveAvgPool1d(1)
        self.head = torch.nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(self.features(x)).squeeze(-1))


def test_gradcam_requires_an_explicit_target_layer() -> None:
    with pytest.raises(ValueError, match="provided explicitly"):
        GradCAM1DExplainer("")


def test_gradcam_requires_an_explicit_target_class(tmp_path) -> None:
    explainer = GradCAM1DExplainer("features")
    with pytest.raises(TypeError):
        explainer.explain(_TinyClassifier(), torch.randn(1, 1, 8), tmp_path)  # type: ignore[call-arg]


def test_gradcam_rejects_an_out_of_range_target_class(tmp_path) -> None:
    explainer = GradCAM1DExplainer("features")
    with pytest.raises(ValueError, match="outside"):
        explainer.explain(
            _TinyClassifier(),
            torch.randn(1, 1, 8),
            tmp_path,
            class_idx=2,
        )


def test_gradcam_records_the_actual_zero_target_class(tmp_path) -> None:
    explainer = GradCAM1DExplainer("features")
    result = explainer.explain(
        _TinyClassifier(),
        torch.randn(1, 1, 8),
        tmp_path,
        class_idx=0,
    )

    record = json.loads((tmp_path / "gradcam_1d.json").read_text())
    assert result.class_idx == 0
    assert record["class_idx"] == 0
    assert record["target_layer"] == "features"
