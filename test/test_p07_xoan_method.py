from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import torch
from pydantic import ValidationError

from phmfactory.config import resolve_config
from src.config_schema.models import ExperimentConfig
from src.configs.config_utils import dict_to_namespace
from src.model_factory import build_model
from src.model_factory.X_model.UXFD.operator_attention import DictionaryIntervention
from src.model_factory.X_model.XOANOperatorPath import Model as XOANModel
from src.model_factory.X_model.XOANOperatorPath import ThresholdArtifact


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
    assert validated.model.operator_path is not None
    assert validated.model.operator_path.relaxation == "sparsemax"
    assert validated.model.operator_path.dictionary_version == "2.0.0"
    assert all(len(stage) == 1 for stage in validated.model.operator_path.addable_stage_operators)


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
    assert evidence["insufficiency_score_id"] == XOANModel.INSUFFICIENCY_SCORE_ID
    assert (
        evidence["insufficiency_score_formula_sha256"]
        == XOANModel.INSUFFICIENCY_SCORE_FORMULA_SHA256
    )
    assert torch.isfinite(evidence["relative_rmse"]).all()
    assert torch.isfinite(evidence["normalized_sparsemax_selection_entropy"]).all()
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
    with pytest.raises(ValueError, match="non-empty one-dimensional"):
        model.selective_accept(torch.empty(0), threshold=0.5)
    with pytest.raises(ValueError, match="non-empty one-dimensional"):
        model.selective_accept(torch.ones(1, 1), threshold=0.5)


def _selector_kwargs(*, human_gate_snapshot: bool = True) -> dict:
    return {
        "coverage_floor": 0.5,
        "split_role": "validation",
        "score_id": XOANModel.INSUFFICIENCY_SCORE_ID,
        "score_formula_sha256": XOANModel.INSUFFICIENCY_SCORE_FORMULA_SHA256,
        "validation_split_sha256": "2" * 64,
        "dataset_sha256": "3" * 64,
        "model_checkpoint_sha256": "4" * 64,
        "resolved_config_sha256": "5" * 64,
        "protocol_sha256": "6" * 64,
        "base_dictionary_sha256": "7" * 64,
        "effective_dictionary_sha256": "8" * 64,
        "human_gate_snapshot": human_gate_snapshot,
        "created_at_utc": "2026-08-01T00:00:00Z",
        "max_selective_risk": 0.34,
    }


def test_validation_risk_coverage_calibration_is_deterministic_and_tie_aware() -> None:
    model = _model()
    scores = torch.tensor([0.1, 0.2, 0.2, 0.8, 0.9], dtype=torch.float64)
    errors = torch.tensor([0, 0, 1, 1, 1])
    curve = model.risk_coverage_curve(scores, errors)

    assert torch.equal(curve["thresholds"], torch.tensor([0.1, 0.2, 0.8, 0.9], dtype=torch.float64))
    assert torch.all(curve["coverage"][1:] >= curve["coverage"][:-1])
    assert curve["accepted_count"].tolist() == [1, 3, 4, 5]

    first = model.calibrate_abstention_threshold(scores, errors, **_selector_kwargs())
    second = model.calibrate_abstention_threshold(scores, errors, **_selector_kwargs())
    assert first == second
    assert first.selected_threshold == pytest.approx(0.2)
    assert first.validation_coverage == pytest.approx(0.6)
    assert first.validation_risk == pytest.approx(1.0 / 3.0)
    assert model.selective_accept(scores, threshold=first.selected_threshold).tolist() == [
        True,
        True,
        True,
        False,
        False,
    ]


def test_threshold_artifact_roundtrip_and_test_application_are_provenance_bound() -> None:
    model = _model()
    scores = torch.tensor([0.1, 0.2, 0.8], dtype=torch.float32)
    errors = torch.tensor([0, 1, 1])
    kwargs = _selector_kwargs()
    kwargs["max_selective_risk"] = 0.5
    artifact = model.calibrate_abstention_threshold(scores, errors, **kwargs)
    restored = ThresholdArtifact.deserialize(artifact.serialize())
    assert restored == artifact
    assert restored.artifact_sha256 == artifact.artifact_sha256
    tampered = json.loads(artifact.serialize())
    tampered["artifact"]["selected_threshold"] = 0.9
    with pytest.raises(ValueError, match="hash is invalid"):
        ThresholdArtifact.deserialize(json.dumps(tampered))
    mask = model.apply_frozen_selector(
        torch.tensor([0.1, 0.7]),
        restored,
        score_id=kwargs["score_id"],
        score_formula_sha256=kwargs["score_formula_sha256"],
        dataset_sha256=kwargs["dataset_sha256"],
        model_checkpoint_sha256=kwargs["model_checkpoint_sha256"],
        resolved_config_sha256=kwargs["resolved_config_sha256"],
        protocol_sha256=kwargs["protocol_sha256"],
        base_dictionary_sha256=kwargs["base_dictionary_sha256"],
        effective_dictionary_sha256=kwargs["effective_dictionary_sha256"],
    )
    assert mask.tolist() == [True, False]
    with pytest.raises(ValueError, match="score_id"):
        model.apply_frozen_selector(
            torch.tensor([0.1]),
            restored,
            score_id="unrelated_score",
            score_formula_sha256=kwargs["score_formula_sha256"],
            dataset_sha256=kwargs["dataset_sha256"],
            model_checkpoint_sha256=kwargs["model_checkpoint_sha256"],
            resolved_config_sha256=kwargs["resolved_config_sha256"],
            protocol_sha256=kwargs["protocol_sha256"],
            base_dictionary_sha256=kwargs["base_dictionary_sha256"],
            effective_dictionary_sha256=kwargs["effective_dictionary_sha256"],
        )
    with pytest.raises(ValueError, match="provenance mismatch"):
        model.apply_frozen_selector(
            torch.tensor([0.1]),
            restored,
            score_id=kwargs["score_id"],
            score_formula_sha256=kwargs["score_formula_sha256"],
            dataset_sha256="f" * 64,
            model_checkpoint_sha256=kwargs["model_checkpoint_sha256"],
            resolved_config_sha256=kwargs["resolved_config_sha256"],
            protocol_sha256=kwargs["protocol_sha256"],
            base_dictionary_sha256=kwargs["base_dictionary_sha256"],
            effective_dictionary_sha256=kwargs["effective_dictionary_sha256"],
        )


def test_threshold_calibration_rejects_test_split_infeasibility_and_false_gate() -> None:
    model = _model()
    scores = torch.tensor([0.1, 0.2, 0.2, 0.8, 0.9])
    errors = torch.tensor([0, 0, 1, 1, 1])
    kwargs = _selector_kwargs()
    kwargs["split_role"] = "test"
    with pytest.raises(ValueError, match="split_role='validation'"):
        model.calibrate_abstention_threshold(scores, errors, **kwargs)

    kwargs = _selector_kwargs()
    kwargs["coverage_floor"] = 0.7
    with pytest.raises(ValueError, match="No validation threshold"):
        model.calibrate_abstention_threshold(scores, errors, **kwargs)

    kwargs = _selector_kwargs()
    kwargs["score_formula_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="does not match"):
        model.calibrate_abstention_threshold(scores, errors, **kwargs)

    kwargs = _selector_kwargs()
    kwargs["coverage_floor"] = True
    with pytest.raises(TypeError, match="not boolean"):
        model.calibrate_abstention_threshold(scores, errors, **kwargs)

    kwargs = _selector_kwargs()
    kwargs["max_selective_risk"] = False
    with pytest.raises(TypeError, match="not boolean"):
        model.calibrate_abstention_threshold(scores, errors, **kwargs)

    kwargs = _selector_kwargs()
    kwargs["created_at_utc"] = "not-a-time"
    with pytest.raises(ValueError, match="ISO-8601 UTC"):
        model.calibrate_abstention_threshold(scores, errors, **kwargs)

    kwargs = _selector_kwargs(human_gate_snapshot=False)
    artifact = model.calibrate_abstention_threshold(scores, errors, **kwargs)
    with pytest.raises(ValueError, match="human gate"):
        model.apply_frozen_selector(
            scores,
            artifact,
            score_id=kwargs["score_id"],
            score_formula_sha256=kwargs["score_formula_sha256"],
            dataset_sha256=kwargs["dataset_sha256"],
            model_checkpoint_sha256=kwargs["model_checkpoint_sha256"],
            resolved_config_sha256=kwargs["resolved_config_sha256"],
            protocol_sha256=kwargs["protocol_sha256"],
            base_dictionary_sha256=kwargs["base_dictionary_sha256"],
            effective_dictionary_sha256=kwargs["effective_dictionary_sha256"],
        )


@pytest.mark.parametrize(
    ("scores", "errors", "message"),
    [
        (torch.tensor([]), torch.tensor([]), "non-empty"),
        (torch.tensor([0.1, float("nan")]), torch.tensor([0, 1]), "finite"),
        (torch.tensor([0.1, 0.2]), torch.tensor([0, 2]), "only 0 or 1"),
        (torch.tensor([0.1]), torch.tensor([0, 1]), "identical shapes"),
    ],
)
def test_risk_coverage_invalid_inputs_fail_closed(
    scores: torch.Tensor, errors: torch.Tensor, message: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        _model().risk_coverage_curve(scores, errors)


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
    config["model"]["operator_path"]["relaxation"] = "softmax"
    with pytest.raises(ValidationError, match="relaxation"):
        ExperimentConfig.model_validate(config)

    config = _resolved()
    config["model"]["operator_path"]["addable_stage_operators"] = [["MA5"]]
    with pytest.raises(ValidationError, match="same number of stages"):
        ExperimentConfig.model_validate(config)

    config = _resolved()
    config["model"]["operator_path"]["addable_stage_operators"][0] = ["I"]
    with pytest.raises(ValidationError, match="overlap"):
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
