from __future__ import annotations

import json

import pytest
import torch

from src.model_factory.X_model.UXFD.operator_attention.executable_operator_path_1d import (
    DictionaryIntervention,
    ExecutableOperatorPath1D,
    ExecutableOperatorPathConfig,
    OperatorEdge,
)


def _module(*, top_k: int = 2, execution_mode: str = "relaxed") -> ExecutableOperatorPath1D:
    return ExecutableOperatorPath1D(
        in_channels=2,
        cfg=ExecutableOperatorPathConfig(
            stage_operators=(("I", "D1", "MA3"), ("I", "D1", "MA3")),
            hidden_dim=16,
            temperature=0.7,
            top_k=top_k,
            execution_mode=execution_mode,
        ),
    )


def test_relaxed_path_is_shape_preserving_sparse_and_differentiable() -> None:
    torch.manual_seed(7)
    module = _module(top_k=2)
    x = torch.randn(3, 32, 2, requires_grad=True)

    output, trace = module(x)

    assert output.shape == x.shape
    assert len(trace.stage_weights) == 2
    for weights in trace.stage_weights:
        assert torch.allclose(weights.sum(dim=1), torch.ones(3), atol=1e-6)
        assert torch.all((weights > 0).sum(dim=1) <= 2)
    output.square().mean().backward()
    assert x.grad is not None
    assert all(parameter.grad is not None for parameter in module.gates.parameters())


def test_stage_rejects_mixed_output_types() -> None:
    with pytest.raises(ValueError, match="one input kind and one output kind"):
        ExecutableOperatorPath1D(
            in_channels=1,
            cfg=ExecutableOperatorPathConfig(stage_operators=(("I", "FFT_MAG"),)),
        )


def test_export_is_deterministic_and_executes_without_gate_weights() -> None:
    torch.manual_seed(11)
    module = _module()
    module.eval()
    x = torch.randn(4, 24, 2)

    _, trace = module.relaxed_forward(x)
    first = module.export_paths(trace)
    second = module.export_paths(trace)
    executed = module.execute_paths(x, first)

    assert first == second
    assert executed.shape == x.shape
    assert torch.isfinite(executed).all()
    assert [module.serialize_path(path) for path in first] == [
        module.serialize_path(path) for path in second
    ]


def test_top1_relaxation_has_zero_export_gap_by_construction() -> None:
    torch.manual_seed(13)
    module = _module(top_k=1)
    module.eval()
    report = module.fidelity_report(torch.randn(5, 32, 2))

    assert torch.allclose(report["relaxed"], report["discrete"], atol=1e-6, rtol=1e-6)
    assert torch.all(report["relative_rmse"] <= 1e-6)
    assert torch.isfinite(report["dictionary_insufficiency_score"]).all()
    assert torch.all(report["dictionary_insufficiency_score"] >= 0)


def test_identity_elision_defines_auditable_semantic_equivalence() -> None:
    module = _module()
    path_a = (
        OperatorEdge(stage=0, source=0, operator="I"),
        OperatorEdge(stage=1, source=1, operator="D1"),
    )
    path_b = (
        OperatorEdge(stage=0, source=0, operator="D1"),
        OperatorEdge(stage=1, source=1, operator="I"),
    )

    assert module.canonical_expression(path_a) == "D1(x)"
    assert module.canonical_expression(path_a) == module.canonical_expression(path_b)


def test_registered_path_intervention_changes_executable_output() -> None:
    module = _module()
    module.eval()
    x = torch.arange(16, dtype=torch.float32).view(1, 8, 2)
    identity_path = (
        OperatorEdge(stage=0, source=0, operator="I"),
        OperatorEdge(stage=1, source=1, operator="I"),
    )

    intervened = module.intervene_paths(
        (identity_path,), stage=0, replacement_operator="FIRST_DIFFERENCE"
    )
    original_output = module.execute_paths(x, (identity_path,))
    intervened_output = module.execute_paths(x, intervened)

    assert not torch.allclose(original_output, intervened_output)
    assert module.canonical_expression(intervened[0]) == "D1(x)"


def test_discrete_forward_is_rejected_during_training() -> None:
    module = _module(execution_mode="discrete")
    module.train()

    with pytest.raises(RuntimeError, match="evaluation-only"):
        module(torch.randn(2, 16, 2))


def test_tied_logits_export_registry_first_and_trace_cache_does_not_alias() -> None:
    module = _module(top_k=2)
    for parameter in module.gates.parameters():
        torch.nn.init.zeros_(parameter)
    module.eval()
    _, trace = module.relaxed_forward(torch.randn(2, 16, 2))

    paths = module.export_paths(trace)
    assert all(edge.operator == "I" for path in paths for edge in path)
    cached = tuple(weight.clone() for weight in module.last_trace.stage_weights)
    trace.stage_weights[0].zero_()
    assert all(
        torch.equal(before, after)
        for before, after in zip(cached, module.last_trace.stage_weights)
    )


def test_path_serialization_roundtrip_binds_effective_dictionary() -> None:
    module = _module()
    path = (
        OperatorEdge(stage=0, source=0, operator="I"),
        OperatorEdge(stage=1, source=1, operator="D1"),
    )
    intervention = DictionaryIntervention(replacements=((0, "I", "D1"),))
    serialized = module.serialize_path(path, dictionary_intervention=intervention)

    restored_path, restored_intervention = module.deserialize_executable_path(serialized)
    assert restored_path == path
    assert restored_intervention == intervention
    assert torch.equal(
        module.execute_paths(
            torch.arange(16, dtype=torch.float32).view(1, 8, 2),
            (path,),
            dictionary_intervention=intervention,
        ),
        module.execute_paths(
            torch.arange(16, dtype=torch.float32).view(1, 8, 2),
            (restored_path,),
            dictionary_intervention=restored_intervention,
        ),
    )
    with pytest.raises(ValueError, match="carries a dictionary intervention"):
        module.deserialize_path(serialized)


def test_removed_operator_is_neither_evaluated_nor_executable() -> None:
    module = ExecutableOperatorPath1D(
        in_channels=1,
        cfg=ExecutableOperatorPathConfig(
            stage_operators=(("I", "SQUARE"),),
            hidden_dim=4,
            top_k=2,
        ),
    )
    module.eval()
    x = torch.full((1, 8, 1), 1.0e20)
    intervention = DictionaryIntervention(removed=((0, "SQUARE"),))
    output, trace = module.relaxed_forward(x, dictionary_intervention=intervention)

    assert torch.isfinite(output).all()
    assert trace.stage_weights[0][0, 1].item() == 0.0
    removed_path = ((OperatorEdge(stage=0, source=0, operator="SQUARE"),),)
    with pytest.raises(ValueError, match="removed dictionary slot"):
        module.execute_paths(x, removed_path, dictionary_intervention=intervention)


def test_conflicting_dictionary_intervention_is_rejected() -> None:
    module = _module()
    intervention = DictionaryIntervention(
        removed=((0, "I"),),
        replacements=((0, "I", "D1"),),
    )
    module.eval()

    with pytest.raises(ValueError, match="remove and replace"):
        module.relaxed_forward(
            torch.randn(1, 16, 2),
            dictionary_intervention=intervention,
        )


def test_stage_chain_has_no_serialized_dead_edges() -> None:
    module = _module()
    assert all(
        edge.source == stage
        for stage, edges in enumerate(module.candidate_edges)
        for edge in edges
    )


def test_top1_training_and_invalid_numeric_contracts_fail_closed() -> None:
    module = _module(top_k=1)
    with pytest.raises(RuntimeError, match="non-trainable"):
        module.relaxed_forward(torch.randn(2, 16, 2))
    with pytest.raises(ValueError, match="eps"):
        ExecutableOperatorPath1D(
            in_channels=1,
            cfg=ExecutableOperatorPathConfig(eps=0.0),
        )
    with pytest.raises(ValueError, match="eps"):
        ExecutableOperatorPath1D(
            in_channels=1,
            cfg=ExecutableOperatorPathConfig(eps=2.0),
        )
    with pytest.raises(ValueError, match="top_k"):
        ExecutableOperatorPath1D(
            in_channels=1,
            cfg=ExecutableOperatorPathConfig(
                stage_operators=(("I", "D1"),),
                top_k=3,
            ),
        )
    with pytest.raises(TypeError, match="float32 or float64"):
        _module().relaxed_forward(torch.ones(1, 16, 2, dtype=torch.int64))
    with pytest.raises(TypeError, match="float32 or float64"):
        _module().relaxed_forward(torch.ones(1, 16, 2, dtype=torch.float16))


def test_dictionary_hash_binds_semantic_version_and_batch_one_is_supported() -> None:
    first = _module()
    second = ExecutableOperatorPath1D(
        in_channels=2,
        cfg=ExecutableOperatorPathConfig(
            dictionary_version="2.0.0",
            stage_operators=(("I", "D1", "MA3"), ("I", "D1", "MA3")),
            hidden_dim=16,
            temperature=0.7,
            top_k=2,
        ),
    )
    assert first.dictionary_sha256 != second.dictionary_sha256
    logits, trace = first.relaxed_forward(torch.randn(1, 16, 2))
    assert logits.shape == (1, 16, 2)
    assert len(first.export_paths(trace)) == 1


def test_dense_entropy_is_normalized_before_and_after_removal() -> None:
    module = ExecutableOperatorPath1D(
        in_channels=1,
        cfg=ExecutableOperatorPathConfig(
            stage_operators=(("I", "D1", "ABS"),),
            hidden_dim=4,
            top_k=1,
            entropy_weight=1.0,
            export_gap_weight=0.0,
        ),
    ).eval()
    for parameter in module.gates.parameters():
        torch.nn.init.zeros_(parameter)
    x = torch.randn(2, 16, 1)

    complete = module.fidelity_report(x)
    reduced = module.fidelity_report(
        x,
        dictionary_intervention=DictionaryIntervention(removed=((0, "ABS"),)),
    )
    assert torch.allclose(complete["selection_entropy"], torch.ones(2), atol=1e-6)
    assert torch.allclose(reduced["selection_entropy"], torch.ones(2), atol=1e-6)
    assert complete["active_candidate_counts"] == (3,)
    assert reduced["active_candidate_counts"] == (2,)


def test_deserialization_rejects_tampered_semantics_and_fractional_indices() -> None:
    module = _module()
    path = (
        OperatorEdge(stage=0, source=0, operator="I"),
        OperatorEdge(stage=1, source=1, operator="D1"),
    )
    payload = json.loads(module.serialize_path(path))
    payload["canonical_expression"] = "TAMPERED"
    with pytest.raises(ValueError, match="canonical expression"):
        module.deserialize_executable_path(json.dumps(payload))

    payload = json.loads(module.serialize_path(path))
    payload["edges"][0]["stage"] = 0.5
    with pytest.raises(ValueError, match="must be an integer"):
        module.deserialize_executable_path(json.dumps(payload))
