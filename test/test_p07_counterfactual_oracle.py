from __future__ import annotations

import ast
import hashlib
import inspect
import json
from dataclasses import FrozenInstanceError, replace

import pytest
import torch

from src.model_factory.X_model.UXFD.operator_attention.executable_operator_path_1d import (
    ExecutableOperatorPath1D,
    ExecutableOperatorPathConfig,
    OperatorEdge,
)
from src.utils.p07_protocol import counterfactual_oracle as oracle_module
from src.utils.p07_protocol.counterfactual_oracle import (
    COMPLETION_STATE,
    CORRUPTION_SEED_SCOPE,
    REMOVAL_SEMANTICS,
    execute_counterfactual_oracle,
    validate_counterfactual_record,
    validate_counterfactual_result,
)
from src.utils.p07_protocol.intervention_registry import (
    WRONG_DICTIONARY_DERANGEMENT,
    build_intervention_registry,
)
from src.utils.p07_protocol.path_universe import (
    CORRUPTION_SEED_DOMAIN,
    K_STAGES,
    PathRecord,
    enumerate_equivalence_classes,
    oracle_execute_path,
)


def _square_class():
    return next(
        item
        for item in enumerate_equivalence_classes()
        if item.canonical_path == ("SQUARE",)
    )


def _selected_path() -> PathRecord:
    return next(
        member
        for member in _square_class().members
        if member.raw_path == ("SQUARE", "ABS", "ABS")
    )


@pytest.fixture(scope="module")
def registry():
    return build_intervention_registry(_square_class(), _selected_path(), 7)


@pytest.fixture()
def signal() -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260801)
    return torch.randn(3, 33, 1, generator=generator, dtype=torch.float64) / 3.0


def _dictionary_record(registry, condition: str, *, magnitude: float | None = None):
    return next(
        record
        for record in registry.dictionary_records
        if record.condition == condition
        and (magnitude is None or record.corruption_magnitude == magnitude)
    )


def _raw_sample_hashes(x: torch.Tensor) -> tuple[str, ...]:
    return tuple(
        hashlib.sha256(
            bytes(sample.contiguous().view(torch.uint8).flatten().cpu().tolist())
        ).hexdigest()
        for sample in x.detach()
    )


def _model_execute(
    x: torch.Tensor, raw_path: tuple[str, str, str], intervention=None
) -> torch.Tensor:
    module = ExecutableOperatorPath1D(
        in_channels=x.shape[-1], cfg=ExecutableOperatorPathConfig()
    ).eval()
    path = tuple(
        OperatorEdge(stage=stage, source=stage, operator=operator)
        for stage, operator in enumerate(raw_path)
    )
    return module.execute_paths(x, (path,) * x.shape[0], intervention)


def test_every_registered_condition_executes_all_samples_without_filtering(
    registry, signal: torch.Tensor
) -> None:
    records = (*registry.dictionary_records, *registry.path_records)

    for source_record in records:
        result = execute_counterfactual_oracle(signal, registry, source_record)

        assert result.output.shape == signal.shape
        assert result.output.dtype == signal.dtype
        assert result.output.device == signal.device
        assert torch.isfinite(result.output).all()
        assert result.record.completion_state == COMPLETION_STATE
        assert len(result.record.sample_keys) == signal.shape[0]
        assert result.record.sample_keys == _raw_sample_hashes(signal)
        assert result.record.input_sample_sha256 == result.record.sample_keys
        assert result.record.software_consistency_only is True
        assert result.record.evidence_eligible is False
        assert result.record.causal_claim_eligible is False
        assert result.record.physical_meaning_claimed is False
        validate_counterfactual_result(
            result, registry=registry, intervention_record=source_record
        )


def test_truth_and_exported_path_are_independently_bound(
    registry, signal: torch.Tensor
) -> None:
    source_record = registry.path_records[0]
    result = execute_counterfactual_oracle(signal, registry, source_record)

    assert result.record.truth_class_id == registry.truth_class_id
    assert result.record.truth_class_sha256 == registry.truth_class_sha256
    assert result.record.selected_path == registry.selected_path
    assert result.record.selected_path_id == registry.selected_path_id
    assert result.record.selected_path_sha256 == registry.selected_path_sha256
    assert result.record.selected_path_class_id == registry.selected_path_class_id
    assert (
        result.record.selected_path_class_sha256
        == registry.selected_path_class_sha256
    )
    assert result.record.selected_path_semantic_match is True
    assert result.record.original_path == registry.selected_path

    foreign_class = next(
        item
        for item in enumerate_equivalence_classes()
        if item.class_id != registry.truth_class_id
    )
    foreign_path = foreign_class.members[0]
    forged_registry = replace(
        registry,
        selected_path=foreign_path.raw_path,
        selected_path_id=foreign_path.raw_path_id,
        selected_path_sha256=foreign_path.raw_path_sha256,
    )
    with pytest.raises(ValueError, match="class binding|construction"):
        execute_counterfactual_oracle(signal, forged_registry, source_record)

    # Arbitrary or unrecovered exported paths cannot enter a favorable subset:
    # only an exact registry record is accepted, and one bad sample rejects all.
    with pytest.raises(TypeError, match="intervention_record"):
        execute_counterfactual_oracle(  # type: ignore[arg-type]
            signal, registry, _selected_path()
        )
    nonfinite = signal.clone()
    nonfinite[1, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        execute_counterfactual_oracle(nonfinite, registry, source_record)


def test_wrong_recovery_executes_every_registered_block_without_filtering(
    signal: torch.Tensor,
) -> None:
    truth_class = _square_class()
    selected_class = next(
        item
        for item in enumerate_equivalence_classes()
        if item.class_id != truth_class.class_id
        and item.canonical_path == ("D1",)
    )
    selected_path = next(
        item
        for item in selected_class.members
        if "D1" in item.raw_path and "SQUARE" not in item.raw_path
    )
    wrong_registry = build_intervention_registry(truth_class, selected_path, 7)

    assert wrong_registry.selected_path_semantic_match is False
    assert wrong_registry.selected_path_class_id == selected_class.class_id
    records = (*wrong_registry.dictionary_records, *wrong_registry.path_records)
    assert records
    for source_record in records:
        result = execute_counterfactual_oracle(
            signal, wrong_registry, source_record
        )
        assert result.output.shape == signal.shape
        assert result.record.truth_class_id == truth_class.class_id
        assert result.record.selected_path_class_id == selected_class.class_id
        assert result.record.selected_path_semantic_match is False
        assert result.record.completion_state == COMPLETION_STATE
        validate_counterfactual_result(
            result,
            registry=wrong_registry,
            intervention_record=source_record,
        )

    target_removal = _dictionary_record(
        wrong_registry, "essential_operator_removal"
    )
    base = _dictionary_record(wrong_registry, "supported_base")
    assert torch.equal(
        execute_counterfactual_oracle(
            signal, wrong_registry, target_removal
        ).output,
        execute_counterfactual_oracle(signal, wrong_registry, base).output,
    )


def test_path_deletion_replacement_equivalent_sham_and_control_use_public_oracle(
    registry, signal: torch.Tensor
) -> None:
    base = oracle_execute_path(signal, registry.selected_path)
    observed_conditions = set()

    for source_record in registry.path_records:
        observed_conditions.add(source_record.condition)
        result = execute_counterfactual_oracle(signal, registry, source_record)
        expected = oracle_execute_path(signal, source_record.intervened_path)
        assert torch.equal(result.output, expected)
        assert result.record.effective_path == source_record.intervened_path
        assert torch.allclose(
            result.output,
            _model_execute(signal, source_record.intervened_path),
            atol=1e-12,
            rtol=0,
        )

        if source_record.condition == "selected_edge_deletion_to_identity":
            target = source_record.target_stage
            assert result.record.stages[target].action == "path_deletion_to_identity"
            assert result.record.stages[target].executed_operator == "I"
        if source_record.condition == "registered_equivalent_raw_path_control":
            assert torch.allclose(result.output, base, atol=1e-12, rtol=0)
        if source_record.condition == "unchanged_replay_sham":
            assert torch.equal(result.output, base)

    assert observed_conditions == {
        "selected_edge_deletion_to_identity",
        "selected_edge_registered_replacement",
        "unchanged_replay_sham",
        "non_selected_stage_control",
        "registered_equivalent_raw_path_control",
    }


def test_dictionary_all_stage_removal_derangement_controls_and_secondary_addition(
    registry, signal: torch.Tensor
) -> None:
    base_record = _dictionary_record(registry, "supported_base")
    base = execute_counterfactual_oracle(signal, registry, base_record)
    assert torch.equal(base.output, oracle_execute_path(signal, registry.selected_path))

    removal = _dictionary_record(registry, "essential_operator_removal")
    assert removal.intervention is not None
    target = removal.target_operators[0]
    assert set(removal.intervention.removed) == {
        (stage, target) for stage in range(K_STAGES)
    }
    removed = execute_counterfactual_oracle(signal, registry, removal)
    expected_removed_path = tuple(
        "I" if operator == target else operator
        for operator in registry.selected_path
    )
    assert removed.record.removal_semantics == REMOVAL_SEMANTICS
    assert removed.record.effective_path == expected_removed_path
    assert torch.equal(removed.output, oracle_execute_path(signal, expected_removed_path))
    assert any(
        stage.action == "dictionary_removal_to_identity"
        for stage in removed.record.stages
    )

    absent = _dictionary_record(
        registry, "matched_absent_operator_removal_control"
    )
    absent_result = execute_counterfactual_oracle(signal, registry, absent)
    assert absent.intervention is not None
    control = absent.control_operators[0]
    assert set(absent.intervention.removed) == {
        (stage, control) for stage in range(K_STAGES)
    }
    assert torch.equal(absent_result.output, base.output)

    wrong = _dictionary_record(registry, "wrong_dictionary_derangement")
    wrong_result = execute_counterfactual_oracle(signal, registry, wrong)
    mapping = dict(WRONG_DICTIONARY_DERANGEMENT)
    expected_wrong_path = tuple(mapping[operator] for operator in registry.selected_path)
    assert wrong_result.record.effective_path == expected_wrong_path
    assert torch.equal(
        wrong_result.output, oracle_execute_path(signal, expected_wrong_path)
    )
    assert torch.allclose(
        wrong_result.output,
        _model_execute(signal, registry.selected_path, wrong.intervention),
        atol=1e-12,
        rtol=0,
    )

    sham = _dictionary_record(registry, "serialization_sham")
    assert torch.equal(
        execute_counterfactual_oracle(signal, registry, sham).output, base.output
    )
    for addition in (
        record
        for record in registry.dictionary_records
        if record.condition == "ma5_expansion_secondary"
    ):
        result = execute_counterfactual_oracle(signal, registry, addition)
        assert torch.equal(result.output, base.output)
        assert torch.allclose(
            result.output,
            _model_execute(signal, registry.selected_path, addition.intervention),
            atol=1e-12,
            rtol=0,
        )


def test_frozen_corruption_is_exact_stateless_batch_invariant_and_hash_bound(
    registry, signal: torch.Tensor
) -> None:
    source_record = _dictionary_record(
        registry, "operator_output_corruption", magnitude=0.10
    )
    keys = _raw_sample_hashes(signal)
    rng_before = torch.random.get_rng_state().clone()

    first = execute_counterfactual_oracle(
        signal, registry, source_record, sample_keys=keys
    )
    second = execute_counterfactual_oracle(
        signal, registry, source_record, sample_keys=keys
    )
    rng_after = torch.random.get_rng_state()

    assert torch.equal(first.output, second.output)
    assert first.record == second.record
    assert first.record.execution_sha256 == second.record.execution_sha256
    assert torch.equal(rng_before, rng_after)
    assert first.record.corruption_seed_domain == CORRUPTION_SEED_DOMAIN
    assert first.record.corruption_seed_scope == CORRUPTION_SEED_SCOPE
    assert all(
        len(stage.corruption_derived_seeds) == signal.shape[0]
        for stage in first.record.stages
    )
    assert torch.equal(
        first.output,
        _model_execute(signal, registry.selected_path, source_record.intervention),
    )

    individually = torch.cat(
        [
            execute_counterfactual_oracle(
                signal[index : index + 1], registry, source_record
            ).output
            for index in range(signal.shape[0])
        ]
    )
    assert torch.equal(first.output, individually)
    permutation = torch.tensor([2, 0, 1])
    permuted = execute_counterfactual_oracle(
        signal.index_select(0, permutation), registry, source_record
    )
    assert torch.equal(permuted.output, first.output.index_select(0, permutation))

    wrong_keys = list(keys)
    wrong_keys[1] = "0" * 64
    with pytest.raises(ValueError, match="root input sample content"):
        execute_counterfactual_oracle(
            signal, registry, source_record, sample_keys=wrong_keys
        )


def test_canonical_record_is_frozen_and_output_tampering_fails_closed(
    registry, signal: torch.Tensor
) -> None:
    source_record = _dictionary_record(
        registry, "operator_output_corruption", magnitude=0.20
    )
    result = execute_counterfactual_oracle(signal, registry, source_record)
    payload = json.loads(result.record.canonical_json())

    assert payload["execution_sha256"] == result.record.execution_sha256
    assert payload["truth_binding"]["truth_class_id"] == registry.truth_class_id
    assert payload["export_binding"]["selected_path_id"] == registry.selected_path_id
    assert payload["export_binding"]["selected_path_semantic_match"] is True
    assert payload["claim_boundary"] == {
        "software_consistency_only": True,
        "evidence_eligible": False,
        "causal_claim_eligible": False,
        "physical_meaning_claimed": False,
        "scope": result.record.interpretation_scope,
    }
    assert result.record.canonical_json().encode("utf-8") == (
        oracle_module.path_universe.canonical_json_bytes(result.record.to_dict())
    )
    with pytest.raises(FrozenInstanceError):
        result.record.condition = "favorable_only"  # type: ignore[misc]

    favorable = replace(result.record, evidence_eligible=True)
    with pytest.raises(ValueError, match="claim boundary"):
        validate_counterfactual_record(favorable)
    drifted_truth = replace(result.record, selected_path=("I", "I", "I"))
    with pytest.raises(ValueError, match="registry-selected path"):
        validate_counterfactual_record(drifted_truth)
    malformed_shape = replace(result.record, tensor_shape=())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="BLC shape"):
        validate_counterfactual_record(malformed_shape)

    changed_output = result.output.clone()
    changed_output[0, 0, 0] += 1.0
    with pytest.raises(ValueError, match="content no longer matches"):
        validate_counterfactual_result(replace(result, output=changed_output))


def test_shape_dtype_finite_registry_record_and_hash_checks_fail_closed(
    registry, signal: torch.Tensor
) -> None:
    source_record = registry.path_records[0]
    with pytest.raises(ValueError, match="batch,length,channels"):
        execute_counterfactual_oracle(signal.squeeze(-1), registry, source_record)
    with pytest.raises(TypeError, match="float32 or torch.float64"):
        execute_counterfactual_oracle(
            torch.ones(1, 8, 1, dtype=torch.int64), registry, source_record
        )
    with pytest.raises(ValueError, match="sample_keys must match"):
        execute_counterfactual_oracle(
            signal, registry, source_record, sample_keys=("0" * 64,)
        )
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        execute_counterfactual_oracle(
            signal,
            registry,
            source_record,
            sample_keys=("G" * 64,) * signal.shape[0],
        )

    forged_record = replace(source_record, condition="selected_edge_favorable_only")
    with pytest.raises(ValueError, match="exact hash-bound member"):
        execute_counterfactual_oracle(signal, registry, forged_record)
    other_registry = build_intervention_registry(
        _square_class(), _selected_path(), 20
    )
    with pytest.raises(ValueError, match="exact hash-bound member"):
        execute_counterfactual_oracle(
            signal, registry, other_registry.path_records[0]
        )
    valid_result = execute_counterfactual_oracle(signal, registry, source_record)
    with pytest.raises(ValueError, match="different registry"):
        validate_counterfactual_result(valid_result, registry=other_registry)


def test_oracle_source_has_no_model_executor_or_private_operator_dependency() -> None:
    source = inspect.getsource(oracle_module)
    tree = ast.parse(source)
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    imported_modules.update(
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )
    assert not any("model_factory" in module for module in imported_modules)

    forbidden_calls = {"_apply_operator", "execute_paths", "relaxed_forward"}
    observed_calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    observed_calls.update(
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    )
    assert forbidden_calls.isdisjoint(observed_calls)
    assert "oracle_apply_operator" in observed_calls
