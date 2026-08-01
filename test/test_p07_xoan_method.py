from __future__ import annotations

import csv
from pathlib import Path

import pytest
import torch
from pydantic import ValidationError

from phmfactory.config import resolve_config
from src.config_schema.models import ExperimentConfig
from src.configs.config_utils import dict_to_namespace
from src.model_factory import build_model
from src.model_factory.X_model.UXFD.operator_attention import DictionaryIntervention


CONFIG_PATH = Path(
    "configs/experiments/p07_xoan_operator_attention/"
    "g030_executable_operator_path_smoke.yaml"
)


def _resolved() -> dict:
    return resolve_config(CONFIG_PATH).data


def _model() -> torch.nn.Module:
    config = _resolved()
    return build_model(dict_to_namespace(config["model"]), metadata=None)


def test_g030_config_resolves_validates_and_builds_exact_model() -> None:
    config = _resolved()
    validated = ExperimentConfig.model_validate(config)
    model = _model()

    assert validated.pipeline == "Pipeline_01_Fault_Diagnosis"
    assert validated.model.type == "X_model"
    assert validated.model.name == "XOANOperatorPath"
    assert model.__class__.__module__ == "src.model_factory.X_model.XOANOperatorPath"
    assert model.operator_path.num_stages == 3


@pytest.mark.parametrize("batch_size", [1, 4])
def test_default_forward_contract_is_finite_for_batch_one_and_many(batch_size: int) -> None:
    torch.manual_seed(3)
    model = _model()
    x = torch.randn(batch_size, 128, 2)
    logits = model(x, 0, "classification")

    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (batch_size, 2)
    assert torch.isfinite(logits).all()


def test_relaxed_training_reaches_selector_and_classifier_gradients() -> None:
    torch.manual_seed(5)
    model = _model()
    model.train()
    logits = model(torch.randn(4, 64, 2))
    logits.square().mean().backward()

    gate_gradients = [parameter.grad for parameter in model.operator_path.gates.parameters()]
    head_gradients = [parameter.grad for parameter in model.classifier.parameters()]
    assert gate_gradients and all(gradient is not None for gradient in gate_gradients)
    assert head_gradients and all(gradient is not None for gradient in head_gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gate_gradients + head_gradients)
    assert any(torch.count_nonzero(gradient).item() > 0 for gradient in gate_gradients)


def test_evidence_api_exports_executes_and_binds_dictionary_intervention() -> None:
    torch.manual_seed(7)
    model = _model().eval()
    x = torch.randn(3, 64, 2)
    intervention = DictionaryIntervention(replacements=((0, "I", "D1"),))
    evidence = model.forward_evidence(x, dictionary_intervention=intervention)

    assert evidence["relaxed"].shape == x.shape
    assert evidence["discrete"].shape == x.shape
    assert evidence["relaxed_logits"].shape == (3, 2)
    assert evidence["discrete_logits"].shape == (3, 2)
    assert len(evidence["serialized_paths"]) == 3
    assert evidence["score_calibration_state"] == "uncalibrated"
    assert torch.isfinite(evidence["relative_rmse"]).all()
    assert torch.isfinite(evidence["selection_entropy"]).all()
    assert torch.isfinite(evidence["predictive_entropy"]).all()
    manifest = evidence["dictionary_manifest"]
    assert manifest["dictionary_intervention"]["replacements"][0] == {
        "stage": 0,
        "registered_operator": "I",
        "executed_operator": "D1",
    }
    path, restored = model.operator_path.deserialize_executable_path(
        evidence["serialized_paths"][0]
    )
    assert restored == intervention
    replay_logits, replay_signal = model.forward_discrete(
        x[:1],
        (path,),
        dictionary_intervention=restored,
    )
    assert torch.allclose(replay_signal, evidence["discrete"][:1])
    assert torch.allclose(replay_logits, evidence["discrete_logits"][:1])


def test_selective_accept_requires_an_explicit_finite_threshold() -> None:
    model = _model()
    scores = torch.tensor([0.1, 0.5, 0.9])
    assert torch.equal(
        model.selective_accept(scores, threshold=0.5),
        torch.tensor([True, True, False]),
    )
    with pytest.raises(TypeError):
        model.selective_accept(scores)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="finite"):
        model.selective_accept(scores, threshold=float("nan"))


def test_fresh_model_strict_state_roundtrip_preserves_eval_logits() -> None:
    torch.manual_seed(11)
    first = _model().eval()
    x = torch.randn(2, 96, 2)
    expected = first(x)
    state = first.state_dict()

    second = _model().eval()
    second.load_state_dict(state, strict=True)
    actual = second(x)
    assert torch.equal(expected, actual)


def test_schema_and_runtime_reject_invalid_scientific_controls() -> None:
    config = _resolved()
    config["model"]["operator_path"]["top_k"] = 1
    with pytest.raises(ValidationError, match="top_k"):
        ExperimentConfig.model_validate(config)

    config = _resolved()
    config["model"]["operator_path"]["unregistered_control"] = True
    with pytest.raises(ValidationError, match="unregistered_control"):
        ExperimentConfig.model_validate(config)

    config = _resolved()
    config["model"]["top_k"] = 999
    with pytest.raises(ValidationError, match="unsupported model fields"):
        ExperimentConfig.model_validate(config)
    with pytest.raises(RuntimeError, match="Unsupported XOANOperatorPath model fields"):
        build_model(dict_to_namespace(config["model"]), metadata=None)

    config = _resolved()
    config["model"]["operator_pth"] = config["model"].pop("operator_path")
    with pytest.raises(ValidationError, match="operator_path"):
        ExperimentConfig.model_validate(config)


def test_model_and_config_registries_bind_exact_paths() -> None:
    with Path("src/model_factory/model_registry.csv").open(newline="", encoding="utf-8") as handle:
        model_rows = list(csv.DictReader(handle))
    matches = [
        row
        for row in model_rows
        if row["model.type"] == "X_model" and row["model.name"] == "XOANOperatorPath"
    ]
    assert len(matches) == 1
    assert matches[0]["module_path"] == "src/model_factory/X_model/XOANOperatorPath.py"

    with Path("configs/config_registry.csv").open(newline="", encoding="utf-8") as handle:
        config_rows = list(csv.DictReader(handle))
    registered_paths = {row["path"] for row in config_rows}
    assert str(CONFIG_PATH) in registered_paths
    assert "configs/base/model/xoan_operator_path.yaml" in registered_paths
