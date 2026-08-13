from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pytest
import torch
import yaml

from src.model_factory import build_model
from src.model_factory.X_model.P01Alignment import CONDITIONS, Model


CONFIG_PATH = Path("configs/base/model/p01_alignment.yaml")


def _namespace(value: Any) -> Any:
    if isinstance(value, Mapping):
        return SimpleNamespace(**{key: _namespace(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_namespace(item) for item in value]
    return value


def _args(condition: str = "M5") -> SimpleNamespace:
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))["model"]
    payload = copy.deepcopy(payload)
    payload["condition"] = condition
    return _namespace(payload)


@pytest.mark.parametrize("condition", CONDITIONS)
def test_actual_config_reaches_every_condition_through_factory(condition: str) -> None:
    torch.manual_seed(7)
    model = build_model(_args(condition), metadata=None)
    model.eval()
    waveform = torch.randn(3, 256, 2)
    with torch.no_grad():
        logits = model(waveform)

    assert isinstance(model, Model)
    assert logits.shape == (3, 4)
    assert torch.isfinite(logits).all()
    assert model.trainable_parameter_count > 0
    state = model.get_representation_state()
    assert state["fused"].shape[0] == 3
    assert ("z_1" in state) is (condition != "M2")
    assert ("z_2" in state) is (condition != "M1")


def test_every_2d_arm_consumes_the_identical_renderer() -> None:
    torch.manual_seed(11)
    waveform = torch.randn(2, 256, 2)
    identities = []
    rendered = []
    for condition in ("M2", "M3", "M4", "M5"):
        model = build_model(_args(condition), metadata=None)
        identities.append(model.renderer_identity())
        rendered.append(model.render_2d_view(waveform))

    assert all(identity == identities[0] for identity in identities)
    assert identities[0] == {
        "n_fft": 128,
        "hop_length": 32,
        "win_length": 128,
        "window": "hann",
        "window_periodic": True,
        "center": True,
        "pad_mode": "reflect",
        "normalized": False,
        "onesided": True,
        "representation": "magnitude",
        "scaling": "log1p",
        "resize": "none",
        "normalization": "none",
    }
    assert all(torch.equal(view, rendered[0]) for view in rendered[1:])


@pytest.mark.parametrize("condition", CONDITIONS)
def test_forward_does_not_create_or_replace_trainable_modules(condition: str) -> None:
    model = build_model(_args(condition), metadata=None)
    model.eval()
    parameter_ids = {name: id(value) for name, value in model.named_parameters()}
    state_keys = set(model.state_dict())
    waveform = torch.randn(2, 256, 2)

    with torch.no_grad():
        first = model(waveform)
        second = model(waveform)

    assert parameter_ids == {name: id(value) for name, value in model.named_parameters()}
    assert state_keys == set(model.state_dict())
    assert torch.equal(first, second)


def test_m3_and_m5_have_the_same_forward_parameterization() -> None:
    m3 = build_model(_args("M3"), metadata=None)
    m5 = build_model(_args("M5"), metadata=None)

    assert m3.trainable_parameter_count == m5.trainable_parameter_count
    assert set(m3.state_dict()) == set(m5.state_dict())


def test_fail_fast_on_pairing_dimension_and_renderer_drift() -> None:
    model = build_model(_args("M5"), metadata=None)
    with pytest.raises(ValueError, match="identical shapes"):
        model.forward_paired_views(
            torch.randn(2, 256, 2), torch.randn(2, 128, 2)
        )
    with pytest.raises(ValueError, match="in_channels"):
        model(torch.randn(2, 256, 1))

    bad_renderer = _args("M5")
    bad_renderer.renderer.resize = "64x64"
    with pytest.raises(RuntimeError, match="Unsupported frozen renderer choice"):
        build_model(bad_renderer, metadata=None)


def test_missing_scientific_config_cannot_fall_back_to_defaults() -> None:
    missing_condition = _args("M5")
    del missing_condition.condition
    with pytest.raises(RuntimeError, match="model.condition is required"):
        build_model(missing_condition, metadata=None)

    missing_renderer_field = _args("M5")
    del missing_renderer_field.renderer.window
    with pytest.raises(RuntimeError, match="model.renderer.window is required"):
        build_model(missing_renderer_field, metadata=None)


@pytest.mark.parametrize(
    ("field", "value"),
    (("dropout", float("nan")), ("lambda_p", float("inf"))),
)
def test_nonfinite_scientific_config_fails_closed(
    field: str, value: float
) -> None:
    args = _args("M5")
    if field == "dropout":
        args.dropout = value
    else:
        setattr(args.alignment, field, value)
    with pytest.raises(RuntimeError, match=f"model.*{field} must be finite"):
        build_model(args, metadata=None)


def test_projection_modules_are_registered_before_forward() -> None:
    model = build_model(_args("M5"), metadata=None)
    names = set(dict(model.named_modules()))
    assert "project_1d" in names
    assert "project_2d" in names
