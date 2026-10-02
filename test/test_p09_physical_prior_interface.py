"""Synthetic source-interface checks, never PHM performance evidence."""
from __future__ import annotations

import copy
import json

import numpy as np
import pytest
import torch
from torch import nn

from src.task_factory.task.GFS.physical_prior import (
    adapt_support, export_source, qualify_source,
)


class NativeSource(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gain = nn.Parameter(torch.tensor([.9, 1.1]))
        self.register_buffer("source_offset", torch.tensor([.2, -.1]))

    def forward(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return x * self.gain + self.source_offset + p


class IgnoredPrompt(NativeSource):
    def forward(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return x * self.gain + self.source_offset


class MagnitudeOnlyPrompt(NativeSource):
    def forward(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return (x * self.gain + self.source_offset) * torch.exp(p.sum())


class DetachedPrompt(NativeSource):
    def forward(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return x * self.gain + self.source_offset + p.detach()


class StationaryZeroPrompt(NativeSource):
    def forward(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return x * self.gain + self.source_offset + p.square()


def fixture() -> tuple[NativeSource, torch.Tensor, list[torch.Tensor], dict]:
    model = NativeSource()
    x = torch.tensor([[1., .4], [.3, 1.1], [-.6, .8]])
    prompts = [torch.zeros(2), torch.tensor([.02, -.01])]
    arms = ("A2", "A3", "A4", "A6", "A7", "A8")
    spec = dict(
        base_classes=[0, 1], prompt_dim=2, logit_scale=2., a3_parameter_names=["gain"],
        excluded_target_folds=["fixture-held-out"], fitted_or_selected_group_ids=[1, 2],
        excluded_class_ids=[2, 3], source_training_seed=0,
        fitted_or_selected_folds=["fixture-source"],
        source_fit_description="Synthetic fixture, not a fitted PHM source",
        group_definition="Synthetic group IDs, not real acquisitions",
        physical_fields=[dict(column="fixture", unit="dimensionless", granularity="group",
                              availability="support observation", source_only_preprocessing="identity",
                              missingness_rule="reject missing")],
        adaptation=dict(offset_norm=.02, prompt_radius=.1,
                        steps={a: 0 if a == "A4" else 2 for a in arms},
                        learning_rate={a: .1 for a in arms},
                        regularization={a: .1 for a in arms},
                        trajectory_steps={"A2": [0, 1, 2], "A3": [0, 1, 2]}),
    )
    return model, x, prompts, spec


def constants() -> dict[str, torch.Tensor]:
    return dict(base_anchors=torch.eye(2), physical_injection=torch.eye(2))


def test_export_round_trip_preserves_native_state_and_explicit_provenance(tmp_path) -> None:
    model, x, prompts, spec = fixture()
    model.train()
    model.gain.requires_grad_(False)
    model.gain.grad = torch.tensor([.7, -.2])
    before = {k: t.clone() for k, t in model.state_dict().items()}
    scripted = torch.jit.script(copy.deepcopy(model).eval())
    output = tmp_path / "source"
    report = export_source(output, model, scripted, x, prompts, source_spec=spec, **constants())
    assert all(report[k] is True for k in (
        "forward_equivalent", "prompt_gradient_nonzero", "nonprompt_gradient_nonzero",
        "a3_gradient_nonzero", "native_state_preserved",
    ))
    assert model.training and not model.gain.requires_grad
    torch.testing.assert_close(model.gain.grad, torch.tensor([.7, -.2]), rtol=0, atol=0)
    for name, value in before.items():
        torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
    loaded = torch.jit.load(str(output / "encoder.pt"))
    for n in (1, 3):
        torch.testing.assert_close(loaded(x[:n], prompts[1]), model(x[:n], prompts[1]))
    saved = json.loads((output / "source.json").read_text())
    assert saved.pop("source_qualification") == report
    assert saved == spec and "source_qualification" not in spec
    with np.load(output / "constants.npz", allow_pickle=False) as arrays:
        np.testing.assert_array_equal(arrays["base_anchors"], np.eye(2))
    with pytest.raises(FileExistsError):
        export_source(output, model, scripted, x, prompts, source_spec=spec, **constants())


@pytest.mark.parametrize("model_type", [IgnoredPrompt, MagnitudeOnlyPrompt])
def test_rejects_prompt_with_no_representation_effect(model_type, tmp_path) -> None:
    _, x, prompts, spec = fixture()
    model = model_type()
    with pytest.raises(ValueError, match="Prompt must change normalized features"):
        export_source(tmp_path / "rejected", model, torch.jit.script(model), x, prompts,
                      source_spec=spec, **constants())
    assert not (tmp_path / "rejected").exists()


def test_rejects_forward_mismatch_beyond_zero_prompt() -> None:
    model, x, prompts, spec = fixture()
    exported = torch.jit.script(IgnoredPrompt())
    with pytest.raises(ValueError, match="gradient connectivity differs|forward equivalence failed"):
        qualify_source(model, exported, x, prompts, source_spec=spec, **constants())


def test_rejects_equal_forward_with_disconnected_exported_gradient() -> None:
    model, x, prompts, spec = fixture()
    with pytest.raises(ValueError, match="gradient connectivity differs: prompt"):
        qualify_source(model, torch.jit.script(DetachedPrompt()), x, prompts,
                       source_spec=spec, **constants())


def test_rejects_stationary_zero_prompt_that_cannot_train_a6() -> None:
    _, x, prompts, spec = fixture()
    model = StationaryZeroPrompt()
    with pytest.raises(ValueError, match="Prompt must change normalized features"):
        qualify_source(model, torch.jit.script(model), x, prompts, source_spec=spec, **constants())


def test_rejects_finite_features_with_overflowed_norm() -> None:
    model, x, prompts, spec = fixture()
    x[0] = 1e20
    assert torch.isfinite(model(x, prompts[0])).all()
    with pytest.raises(ValueError, match="zero or non-finite representations"):
        qualify_source(model, torch.jit.script(model), x, prompts, source_spec=spec, **constants())


@pytest.mark.parametrize("names", [["absent"], ["gain", "gain"]])
def test_rejects_invalid_a3_identity(names) -> None:
    model, x, prompts, spec = fixture()
    spec["a3_parameter_names"] = names
    with pytest.raises(ValueError, match="A3 must name distinct"):
        qualify_source(model, torch.jit.script(model), x, prompts, source_spec=spec, **constants())


def test_rejects_constant_shape_and_missing_provenance() -> None:
    model, x, prompts, spec = fixture()
    with pytest.raises(ValueError, match="finite shape"):
        qualify_source(model, torch.jit.script(model), x, prompts, source_spec=spec,
                       base_anchors=torch.eye(2), physical_injection=torch.ones(2, 3))
    del spec["physical_fields"][0]["source_only_preprocessing"]
    with pytest.raises(ValueError, match="source-only preprocessing provenance"):
        qualify_source(model, torch.jit.script(model), x, prompts, source_spec=spec, **constants())


@pytest.mark.parametrize("key,value,match", [
    ("excluded_class_ids", [0, 2], "overlap declared excluded"),
    ("fitted_or_selected_folds", ["fixture-held-out"], "overlap declared held-out"),
    ("source_training_seed", "0", "nonnegative integer"),
    ("fitted_or_selected_group_ids", ["10"], "distinct integer IDs"),
    ("adaptation", {}, "explicit bounds, budgets and schedules"),
])
def test_rejects_contradictory_source_exclusion_declarations(key, value, match) -> None:
    model, x, prompts, spec = fixture()
    spec[key] = value
    with pytest.raises(ValueError, match=match):
        qualify_source(model, torch.jit.script(model), x, prompts, source_spec=spec, **constants())


def test_delegation_calls_only_explicit_adaptation_core() -> None:
    model, x, prompts, spec = fixture()
    call = {}
    expected = object()

    def supplied_core(*args, **kwargs):
        call.update(args=args, kwargs=kwargs)
        return expected

    y = torch.tensor([2, 2, 3])
    groups = torch.tensor([10, 10, 20])
    views = torch.tensor([0, 1, 0])
    result = adapt_support(supplied_core, model, x, y, groups, views, torch.eye(2),
                           spec["base_classes"], [2, 3], arm="A7", prior=prompts[1],
                           scale=2., lr=.1, steps=2, lam=.2, radius=.1, checkpoints=[2])
    assert result is expected and call["args"][0] is model
    assert call["kwargs"]["prior"] is prompts[1]
    assert call["kwargs"]["steps"] == 2
    assert "query" not in call["kwargs"]
