from __future__ import annotations

import json
from dataclasses import replace

import pytest
import torch

from src.model_factory.X_model.UXFD.operator_attention.executable_operator_path_1d import (
    DictionaryIntervention,
    ExecutableOperatorPath1D,
    ExecutableOperatorPathConfig,
    OperatorCorruption,
    OperatorEdge,
    _masked_sparsemax,
)


def _module(*, execution_mode: str = "relaxed") -> ExecutableOperatorPath1D:
    return ExecutableOperatorPath1D(
        in_channels=2,
        cfg=ExecutableOperatorPathConfig(
            stage_operators=(("I", "D1", "MA3"), ("I", "D1", "MA3")),
            addable_stage_operators=(("MA5",), ("MA5",)),
            hidden_dim=16,
            temperature=0.7,
            execution_mode=execution_mode,
        ),
    )


def _force_one_hot_identity(module: ExecutableOperatorPath1D) -> None:
    for gate in module.gates:
        for parameter in gate.parameters():
            torch.nn.init.zeros_(parameter)
        gate[-1].bias.data[0] = 2.0


def test_sparsemax_oracle_simplex_sparsity_ties_and_translation() -> None:
    logits = torch.tensor([[0.8, 0.2, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float64)
    allowed = torch.tensor([True, True, True])
    actual = _masked_sparsemax(logits, allowed)

    expected = torch.tensor(
        [[0.8, 0.2, 0.0], [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]],
        dtype=torch.float64,
    )
    assert torch.allclose(actual, expected, atol=1e-12, rtol=0.0)
    assert torch.equal(actual[0, 2:], torch.zeros(1, dtype=torch.float64))
    assert torch.all(actual >= 0)
    assert torch.allclose(actual.sum(dim=1), torch.ones(2, dtype=torch.float64))
    assert torch.allclose(_masked_sparsemax(logits + 17.0, allowed), actual, atol=1e-12)


def test_sparsemax_is_continuous_at_a_support_boundary() -> None:
    allowed = torch.tensor([True, True, True])
    distances = []
    for epsilon in (1e-2, 1e-4, 1e-6):
        left = _masked_sparsemax(
            torch.tensor([[0.6, 0.4, -epsilon]], dtype=torch.float64), allowed
        )
        right = _masked_sparsemax(
            torch.tensor([[0.6, 0.4, epsilon]], dtype=torch.float64), allowed
        )
        distances.append(float(torch.linalg.vector_norm(right - left)))
    assert distances[1] < distances[0] / 50.0
    assert distances[2] < distances[1] / 50.0


def test_sparsemax_gradcheck_away_from_support_boundaries() -> None:
    logits = torch.tensor([[0.45, 0.35, 0.20]], dtype=torch.float64, requires_grad=True)
    allowed = torch.tensor([True, True, True])
    assert torch.autograd.gradcheck(
        lambda values: _masked_sparsemax(values, allowed),
        (logits,),
        eps=1e-6,
        atol=1e-5,
        rtol=1e-3,
    )


def test_sparsemax_masks_before_projection_and_single_candidate_is_exact() -> None:
    logits = torch.tensor([[0.0, 100.0, 0.0]], dtype=torch.float64)
    masked = _masked_sparsemax(logits, torch.tensor([True, False, True]))
    single = _masked_sparsemax(logits, torch.tensor([False, True, False]))
    assert torch.equal(masked, torch.tensor([[0.5, 0.0, 0.5]], dtype=torch.float64))
    assert torch.equal(single, torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64))
    with pytest.raises(ValueError, match="at least one allowed"):
        _masked_sparsemax(logits, torch.tensor([False, False, False]))


def test_relaxed_path_is_shape_preserving_and_differentiable() -> None:
    torch.manual_seed(7)
    module = _module()
    x = torch.randn(3, 32, 2, requires_grad=True)

    output, trace = module(x)

    assert output.shape == x.shape
    assert len(trace.stage_weights) == 2
    for weights in trace.stage_weights:
        assert torch.allclose(weights.sum(dim=1), torch.ones(3), atol=1e-6)
        assert torch.all(weights >= 0)
        assert torch.equal(weights[:, -1], torch.zeros(3))
    output.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    gradients = [parameter.grad for parameter in module.gates.parameters()]
    assert gradients and all(gradient is not None for gradient in gradients)
    assert any(torch.count_nonzero(gradient).item() > 0 for gradient in gradients)


def test_stage_rejects_mixed_types_and_dictionary_shape_errors() -> None:
    with pytest.raises(ValueError, match="one input kind and one output kind"):
        ExecutableOperatorPath1D(
            in_channels=1,
            cfg=ExecutableOperatorPathConfig(
                stage_operators=(("I", "FFT_MAG"),),
                addable_stage_operators=((),),
            ),
        )
    with pytest.raises(ValueError, match="same number of stages"):
        ExecutableOperatorPath1D(
            in_channels=1,
            cfg=ExecutableOperatorPathConfig(
                stage_operators=(("I", "D1"),),
                addable_stage_operators=((), ()),
            ),
        )
    with pytest.raises(ValueError, match="overlap"):
        ExecutableOperatorPath1D(
            in_channels=1,
            cfg=ExecutableOperatorPathConfig(
                stage_operators=(("I", "D1"),),
                addable_stage_operators=(("I",),),
            ),
        )


def test_export_is_deterministic_and_executes_without_gate_weights() -> None:
    torch.manual_seed(11)
    module = _module().eval()
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


def test_singleton_sparsemax_support_has_zero_export_gap() -> None:
    torch.manual_seed(13)
    module = _module().eval()
    _force_one_hot_identity(module)
    report = module.fidelity_report(torch.randn(5, 32, 2))

    assert torch.allclose(report["relaxed"], report["discrete"], atol=1e-6, rtol=1e-6)
    assert torch.all(report["relative_rmse"] <= 1e-6)
    assert all(torch.equal(size, torch.ones_like(size)) for size in report["support_sizes"])
    assert torch.isfinite(report["dictionary_insufficiency_score"]).all()


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
    module = _module().eval()
    x = torch.arange(16, dtype=torch.float32).view(1, 8, 2)
    identity_path = (
        OperatorEdge(stage=0, source=0, operator="I"),
        OperatorEdge(stage=1, source=1, operator="I"),
    )
    intervened = module.intervene_paths(
        (identity_path,), stage=0, replacement_operator="FIRST_DIFFERENCE"
    )
    assert not torch.allclose(
        module.execute_paths(x, (identity_path,)), module.execute_paths(x, intervened)
    )
    assert module.canonical_expression(intervened[0]) == "D1(x)"
    with pytest.raises(ValueError, match="not active in the base"):
        module.intervene_paths((identity_path,), stage=0, replacement_operator="MA5")


def test_discrete_forward_is_rejected_during_training() -> None:
    module = _module(execution_mode="discrete").train()
    with pytest.raises(RuntimeError, match="evaluation-only"):
        module(torch.randn(2, 16, 2))


def test_tied_sparsemax_is_symmetric_but_export_uses_registry_order() -> None:
    module = _module().eval()
    for parameter in module.gates.parameters():
        torch.nn.init.zeros_(parameter)
    _, trace = module.relaxed_forward(torch.randn(2, 16, 2))

    assert all(
        torch.allclose(weights[:, :3], torch.full_like(weights[:, :3], 1.0 / 3.0))
        for weights in trace.stage_weights
    )
    paths = module.export_paths(trace)
    assert all(edge.operator == "I" for path in paths for edge in path)
    cached = tuple(weight.clone() for weight in module.last_trace.stage_weights)
    trace.stage_weights[0].zero_()
    assert all(
        torch.equal(before, after)
        for before, after in zip(cached, module.last_trace.stage_weights)
    )


def test_dormant_addition_is_unmasking_and_binds_execution() -> None:
    module = ExecutableOperatorPath1D(
        in_channels=1,
        cfg=ExecutableOperatorPathConfig(
            stage_operators=(("I", "D1"),),
            addable_stage_operators=(("MA5",),),
            hidden_dim=4,
        ),
    ).eval()
    path = (OperatorEdge(stage=0, source=0, operator="MA5"),)
    x = torch.randn(2, 16, 1)
    with pytest.raises(ValueError, match="inactive"):
        module.execute_paths(x, (path, path))
    with pytest.raises(ValueError, match="inactive"):
        module.serialize_path(path)

    intervention = DictionaryIntervention(added=((0, "MA5"),))
    for parameter in module.gates.parameters():
        torch.nn.init.zeros_(parameter)
    module.gates[0][-1].bias.data[2] = 2.0
    _, trace = module.relaxed_forward(x, dictionary_intervention=intervention)
    exported = module.export_paths(trace)
    assert all(item.edges == path for item in exported)
    assert module.execute_paths(x, exported).shape == x.shape
    assert all(item.dictionary_intervention == intervention for item in exported)
    assert all(
        item.effective_dictionary_sha256 == module.effective_dictionary_sha256(intervention)
        for item in exported
    )
    tampered = replace(exported[0], effective_dictionary_sha256="0" * 64)
    with pytest.raises(ValueError, match="effective dictionary hash"):
        module.execute_paths(x[:1], (tampered,))
    assert module.deserialize_executable_path(module.serialize_path(path, intervention)) == (
        path,
        intervention,
    )


def test_removed_operator_is_neither_evaluated_nor_executable() -> None:
    module = ExecutableOperatorPath1D(
        in_channels=1,
        cfg=ExecutableOperatorPathConfig(
            stage_operators=(("I", "SQUARE"),),
            addable_stage_operators=((),),
            hidden_dim=4,
        ),
    ).eval()
    x = torch.full((1, 8, 1), 1.0e20)
    intervention = DictionaryIntervention(removed=((0, "SQUARE"),))
    output, trace = module.relaxed_forward(x, dictionary_intervention=intervention)
    assert torch.isfinite(output).all()
    assert trace.stage_weights[0][0, 1].item() == 0.0
    removed_path = ((OperatorEdge(stage=0, source=0, operator="SQUARE"),),)
    with pytest.raises(ValueError, match="inactive or removed"):
        module.execute_paths(x, removed_path, dictionary_intervention=intervention)
    with pytest.raises(ValueError, match="without active candidates"):
        module.relaxed_forward(
            torch.ones(1, 8, 1),
            dictionary_intervention=DictionaryIntervention(
                removed=((0, "I"), (0, "SQUARE"))
            ),
        )
    with pytest.raises(ValueError, match="produced non-finite"):
        module.execute_paths(
            x,
            ((OperatorEdge(stage=0, source=0, operator="SQUARE"),),),
        )


def test_seeded_corruption_is_stateless_batch_invariant_and_hash_bound() -> None:
    module = ExecutableOperatorPath1D(
        in_channels=1,
        cfg=ExecutableOperatorPathConfig(
            stage_operators=(("I", "D1"),),
            addable_stage_operators=((),),
            hidden_dim=4,
        ),
    ).eval()
    _force_one_hot_identity(module)
    path = (OperatorEdge(stage=0, source=0, operator="I"),)
    paths = (path, path, path)
    x = torch.arange(24, dtype=torch.float32).reshape(3, 8, 1)
    first = DictionaryIntervention(
        corruptions=(OperatorCorruption(0, "I", magnitude=0.25, seed=17),)
    )
    second = DictionaryIntervention(
        corruptions=(OperatorCorruption(0, "I", magnitude=0.25, seed=18),)
    )

    rng_before = torch.random.get_rng_state().clone()
    output_a = module.execute_paths(x, paths, first)
    output_b = module.execute_paths(x, paths, first)
    rng_after = torch.random.get_rng_state()
    assert torch.equal(output_a, output_b)
    assert torch.equal(rng_before, rng_after)
    assert not torch.equal(output_a, module.execute_paths(x, paths, second))
    assert module.effective_dictionary_sha256(first) != module.effective_dictionary_sha256(second)

    individually = torch.cat(
        [module.execute_paths(x[index : index + 1], (path,), first) for index in range(3)]
    )
    assert torch.equal(output_a, individually)
    permutation = torch.tensor([2, 0, 1])
    permuted = module.execute_paths(x.index_select(0, permutation), paths, first)
    assert torch.equal(permuted, output_a.index_select(0, permutation))

    report = module.fidelity_report(x, first)
    assert torch.allclose(report["relaxed"], report["discrete"], atol=1e-6, rtol=1e-6)
    assert "CORRUPT[additive_gaussian_absolute,magnitude=0.25,seed=17]" in (
        module.canonical_expression(path, dictionary_intervention=first)
    )


@pytest.mark.parametrize(
    ("intervention", "message"),
    [
        (DictionaryIntervention(added=((0, "I"),)), "not a preregistered dormant"),
        (
            DictionaryIntervention(
                removed=((0, "I"),), replacements=((0, "I", "D1"),)
            ),
            "inactive slot",
        ),
        (DictionaryIntervention(replacements=((0, "I", "I"),)), "No-op"),
        (
            DictionaryIntervention(
                corruptions=(OperatorCorruption(0, "I", magnitude=0.0, seed=1),)
            ),
            "positive finite",
        ),
        (
            DictionaryIntervention(
                corruptions=(OperatorCorruption(0, "I", magnitude=0.1, seed=-1),)
            ),
            "seed",
        ),
    ],
)
def test_invalid_dictionary_interventions_fail_closed(
    intervention: DictionaryIntervention, message: str
) -> None:
    module = _module().eval()
    with pytest.raises(ValueError, match=message):
        module.relaxed_forward(torch.randn(1, 16, 2), intervention)


def test_cross_type_replacement_is_rejected() -> None:
    module = ExecutableOperatorPath1D(
        1,
        ExecutableOperatorPathConfig(
            stage_operators=(("I",),), addable_stage_operators=((),)
        ),
    )
    with pytest.raises(ValueError, match="type signature"):
        module.effective_dictionary_sha256(
            DictionaryIntervention(replacements=((0, "I", "FFT_MAG"),))
        )


def test_schema_v2_roundtrip_binds_all_intervention_fields_and_rejects_tampering() -> None:
    module = _module()
    path = (
        OperatorEdge(stage=0, source=0, operator="I"),
        OperatorEdge(stage=1, source=1, operator="I"),
    )
    intervention = DictionaryIntervention(
        added=((0, "MA5"),),
        removed=((1, "MA3"),),
        replacements=((0, "D1", "ABS"),),
        corruptions=(OperatorCorruption(0, "I", magnitude=0.125, seed=31),),
    )
    serialized = module.serialize_path(path, intervention)
    assert module.deserialize_executable_path(serialized) == (path, intervention)
    payload = json.loads(serialized)
    assert payload["schema_version"] == 2
    assert payload["dictionary_intervention"]["added"] == [
        {"stage": 0, "operator": "MA5"}
    ]
    assert payload["edges"][0]["corruption"]["seed"] == 31

    payload["dictionary_intervention"]["corruptions"][0]["seed"] = 32
    with pytest.raises(ValueError, match="effective dictionary hash"):
        module.deserialize_executable_path(json.dumps(payload))
    duplicate_key = serialized[:-1] + ',"schema_version":2}'
    with pytest.raises(ValueError, match="duplicate key"):
        module.deserialize_executable_path(duplicate_key)
    with pytest.raises(ValueError, match="carries a dictionary intervention"):
        module.deserialize_path(serialized)


def test_dictionary_hash_normalizes_order_and_aliases() -> None:
    module = _module()
    first = DictionaryIntervention(
        added=((1, "MOVING_AVERAGE_5"), (0, "MA5")),
        removed=((1, "D1"), (0, "FIRST_DIFFERENCE")),
    )
    second = DictionaryIntervention(
        added=((0, "MA5"), (1, "MA5")),
        removed=((0, "D1"), (1, "D1")),
    )
    assert module.effective_dictionary_sha256(first) == module.effective_dictionary_sha256(second)


def test_stage_chain_has_no_dead_source_edges() -> None:
    module = _module()
    assert all(
        edge.source == stage
        for stage, edges in enumerate(module.candidate_edges)
        for edge in edges
    )


def test_invalid_numeric_and_input_contracts_fail_closed() -> None:
    with pytest.raises(ValueError, match="eps"):
        ExecutableOperatorPath1D(1, ExecutableOperatorPathConfig(eps=0.0))
    with pytest.raises(ValueError, match="support_tolerance"):
        ExecutableOperatorPath1D(
            1, ExecutableOperatorPathConfig(support_tolerance=float("nan"))
        )
    with pytest.raises(ValueError, match="support_tolerance"):
        ExecutableOperatorPath1D(1, ExecutableOperatorPathConfig(support_tolerance=0.9))
    with pytest.raises(ValueError, match="relaxation"):
        ExecutableOperatorPath1D(1, ExecutableOperatorPathConfig(relaxation="softmax"))
    with pytest.raises(TypeError, match="float32 or float64"):
        _module().relaxed_forward(torch.ones(1, 16, 2, dtype=torch.int64))
    with pytest.raises(TypeError, match="float32 or float64"):
        _module().relaxed_forward(torch.ones(1, 16, 2, dtype=torch.float16))


def test_dictionary_hash_binds_semantics_and_batch_one_is_supported() -> None:
    first = _module()
    second = ExecutableOperatorPath1D(
        in_channels=2,
        cfg=ExecutableOperatorPathConfig(
            dictionary_version="3.0.0",
            stage_operators=(("I", "D1", "MA3"), ("I", "D1", "MA3")),
            addable_stage_operators=(("MA5",), ("MA5",)),
            hidden_dim=16,
            temperature=0.7,
        ),
    )
    assert first.dictionary_sha256 != second.dictionary_sha256
    logits, trace = first.relaxed_forward(torch.randn(1, 16, 2))
    assert logits.shape == (1, 16, 2)
    assert len(first.export_paths(trace)) == 1


def test_strict_checkpoint_reload_rejects_shape_compatible_dictionary_reordering() -> None:
    first = ExecutableOperatorPath1D(
        1,
        ExecutableOperatorPathConfig(
            stage_operators=(("I", "D1"),), addable_stage_operators=((),)
        ),
    )
    reordered = ExecutableOperatorPath1D(
        1,
        ExecutableOperatorPathConfig(
            stage_operators=(("D1", "I"),), addable_stage_operators=((),)
        ),
    )
    assert first.dictionary_sha256 != reordered.dictionary_sha256
    with pytest.raises(RuntimeError, match="dictionary semantic hash"):
        reordered.load_state_dict(first.state_dict(), strict=True)


def test_sparsemax_entropy_is_normalized_before_and_after_remove_add() -> None:
    module = ExecutableOperatorPath1D(
        in_channels=1,
        cfg=ExecutableOperatorPathConfig(
            stage_operators=(("I", "D1", "ABS"),),
            addable_stage_operators=(("MA5",),),
            hidden_dim=4,
            entropy_weight=1.0,
            export_gap_weight=0.0,
        ),
    ).eval()
    for parameter in module.gates.parameters():
        torch.nn.init.zeros_(parameter)
    x = torch.randn(2, 16, 1)
    complete = module.fidelity_report(x)
    reduced = module.fidelity_report(x, DictionaryIntervention(removed=((0, "ABS"),)))
    expanded = module.fidelity_report(x, DictionaryIntervention(added=((0, "MA5"),)))
    key = "normalized_sparsemax_selection_entropy"
    assert torch.allclose(complete[key], torch.ones(2), atol=1e-6)
    assert torch.allclose(reduced[key], torch.ones(2), atol=1e-6)
    assert torch.allclose(expanded[key], torch.ones(2), atol=1e-6)
    assert complete["active_candidate_counts"] == (3,)
    assert reduced["active_candidate_counts"] == (2,)
    assert expanded["active_candidate_counts"] == (4,)


def test_deserialization_rejects_tampered_expression_and_fractional_indices() -> None:
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
