from __future__ import annotations

import inspect
import json
from dataclasses import FrozenInstanceError, replace

import pytest
import torch

from src.model_factory.X_model.UXFD.operator_attention.executable_operator_path_1d import (
    DictionaryIntervention,
    ExecutableOperatorPath1D,
    OperatorCorruption,
)
from src.utils.p07_protocol import intervention_registry as registry_module
from src.utils.p07_protocol.intervention_registry import (
    CORRUPTION_RMS_LEVELS,
    INTERPRETATION_SCOPE,
    WRONG_DICTIONARY_DERANGEMENT,
    build_intervention_registry,
    provably_absent_control_operators,
    semantically_essential_operators,
    validate_intervention_registry,
)
from src.utils.p07_protocol.path_universe import (
    CORRUPTION_SEED_BY_OPTIMIZATION_SEED,
    NON_IDENTITY_OPERATORS,
    OPERATORS,
    EquivalenceClass,
    PathRecord,
    enumerate_equivalence_classes,
)


def _square_class() -> EquivalenceClass:
    return next(
        item
        for item in enumerate_equivalence_classes()
        if item.canonical_path == ("SQUARE",)
    )


def _ambiguous_selected_path() -> PathRecord:
    truth_class = _square_class()
    return next(
        member
        for member in truth_class.members
        if member.raw_path == ("SQUARE", "ABS", "ABS")
    )


def _singleton_class() -> EquivalenceClass:
    return next(
        item
        for item in enumerate_equivalence_classes()
        if item.multiplicity == 1
        and sum(operator != "I" for operator in item.members[0].raw_path) >= 2
    )


def _replace_dictionary_record(registry, index: int, **changes):
    records = list(registry.dictionary_records)
    records[index] = replace(records[index], **changes)
    return replace(registry, dictionary_records=tuple(records))


def test_ambiguous_class_uses_member_intersection_and_union_not_one_path() -> None:
    truth_class = _square_class()
    selected_path = _ambiguous_selected_path()
    member_sets = [set(member.raw_path).difference({"I"}) for member in truth_class.members]

    assert truth_class.ambiguous
    assert set(selected_path.raw_path).difference({"I"}) == {"SQUARE", "ABS"}
    assert set.intersection(*member_sets) == {"SQUARE"}
    assert set.union(*member_sets) == {"SQUARE", "ABS"}
    assert semantically_essential_operators(truth_class) == ("SQUARE",)
    assert provably_absent_control_operators(truth_class) == ("D1", "MA3", "HT")

    registry = build_intervention_registry(truth_class, selected_path, 7)
    targeted = [
        record
        for record in registry.dictionary_records
        if record.condition == "essential_operator_removal"
    ]
    controls = [
        record
        for record in registry.dictionary_records
        if record.condition == "matched_absent_operator_removal_control"
    ]
    assert [record.target_operators for record in targeted] == [("SQUARE",)]
    assert len(controls) == 1
    assert controls[0].control_operators[0] in {"D1", "MA3", "HT"}
    assert controls[0].control_operators[0] != "ABS"


def test_dictionary_registry_order_all_stage_removals_derangement_and_ma5_role() -> None:
    truth_class = _square_class()
    selected_path = _ambiguous_selected_path()
    registry = build_intervention_registry(truth_class, selected_path, 7)
    records = registry.dictionary_records

    assert [record.condition for record in records[:4]] == [
        "supported_base",
        "serialization_sham",
        "essential_operator_removal",
        "matched_absent_operator_removal_control",
    ]
    assert records[0].effective_dictionary_sha256 == records[1].effective_dictionary_sha256

    removal = records[2]
    control = records[3]
    assert removal.intervention == DictionaryIntervention(
        removed=((0, "SQUARE"), (1, "SQUARE"), (2, "SQUARE"))
    )
    control_operator = control.control_operators[0]
    assert control.intervention == DictionaryIntervention(
        removed=((0, control_operator), (1, control_operator), (2, control_operator))
    )
    assert len(removal.intervention.removed) == len(control.intervention.removed) == 3
    assert control.record_id in removal.paired_record_ids

    wrong = next(
        record for record in records if record.condition == "wrong_dictionary_derangement"
    )
    assert wrong.intervention is not None
    expected_mapping = dict(WRONG_DICTIONARY_DERANGEMENT)
    assert expected_mapping == {
        "D1": "MA3",
        "ABS": "SQUARE",
        "SQUARE": "HT",
        "MA3": "D1",
        "HT": "ABS",
    }
    for stage in range(3):
        observed = {
            registered: executed
            for item_stage, registered, executed in wrong.intervention.replacements
            if item_stage == stage
        }
        assert observed == expected_mapping

    ma5_records = [
        record for record in records if record.condition == "ma5_expansion_secondary"
    ]
    assert len(ma5_records) == 3
    assert {record.intervention.added for record in ma5_records if record.intervention} == {
        ((0, "MA5"),),
        ((1, "MA5"),),
        ((2, "MA5"),),
    }
    assert all(not record.primary_eligible for record in ma5_records)
    assert all(record.secondary_eligible for record in ma5_records)


def test_corruption_records_use_only_frozen_seed_domain_and_exact_doses() -> None:
    registry = build_intervention_registry(_square_class(), _ambiguous_selected_path(), 7)
    expected_corruption_seed = CORRUPTION_SEED_BY_OPTIMIZATION_SEED[7]
    corruption_records = [
        record
        for record in registry.dictionary_records
        if record.condition == "operator_output_corruption"
    ]

    assert tuple(record.corruption_magnitude for record in corruption_records) == (
        CORRUPTION_RMS_LEVELS
    )
    assert registry.corruption_seed == expected_corruption_seed
    for record in (*registry.dictionary_records, *registry.path_records):
        assert record.optimization_seed == 7
        assert record.corruption_seed == expected_corruption_seed
    for record in corruption_records:
        assert record.intervention is not None
        assert len(record.intervention.corruptions) == 3 * len(NON_IDENTITY_OPERATORS)
        assert all(
            isinstance(corruption, OperatorCorruption)
            and corruption.seed == expected_corruption_seed
            and corruption.magnitude == record.corruption_magnitude
            for corruption in record.intervention.corruptions
        )
        assert {
            (corruption.stage, corruption.registered_operator)
            for corruption in record.intervention.corruptions
        } == {
            (stage, operator)
            for stage in range(3)
            for operator in NON_IDENTITY_OPERATORS
        }


def test_path_registry_materializes_registered_targets_and_eligible_controls() -> None:
    truth_class = _square_class()
    selected_path = _ambiguous_selected_path()
    registry = build_intervention_registry(truth_class, selected_path, 7)
    blocks = {}
    for record in registry.path_records:
        blocks.setdefault(record.block_id, []).append(record)

    assert len(blocks) == 3
    class_member_ids = {member.raw_path_id for member in truth_class.members}
    all_record_ids = {
        record.record_id
        for record in (*registry.dictionary_records, *registry.path_records)
    }
    for block in blocks.values():
        by_condition = {record.condition: record for record in block}
        assert set(by_condition) == {
            "selected_edge_deletion_to_identity",
            "selected_edge_registered_replacement",
            "unchanged_replay_sham",
            "non_selected_stage_control",
            "registered_equivalent_raw_path_control",
        }
        deletion = by_condition["selected_edge_deletion_to_identity"]
        replacement = by_condition["selected_edge_registered_replacement"]
        sham = by_condition["unchanged_replay_sham"]
        nonselected = by_condition["non_selected_stage_control"]
        equivalent = by_condition["registered_equivalent_raw_path_control"]

        assert deletion.intervened_path[deletion.target_stage] == "I"
        assert deletion.replacement_operator == "I"
        assert replacement.replacement_operator in NON_IDENTITY_OPERATORS
        assert replacement.replacement_operator != selected_path.raw_path[
            replacement.target_stage
        ]
        assert replacement.intervened_path[replacement.target_stage] == (
            replacement.replacement_operator
        )
        assert sham.intervened_path == selected_path.raw_path
        assert nonselected.intervention_stage != nonselected.target_stage
        assert nonselected.intervention_stage is not None
        assert nonselected.intervened_path[nonselected.intervention_stage] == "I"
        assert equivalent.intervened_path_id in class_member_ids
        assert equivalent.intervened_path_id != selected_path.raw_path_id
        assert equivalent.intervened_path_id in {
            member.raw_path_id for member in truth_class.members
        }
        assert set(deletion.paired_record_ids).issubset(all_record_ids)
        assert set(replacement.paired_record_ids).issubset(all_record_ids)
        assert deletion.original_path_sha256 == selected_path.raw_path_sha256
        assert deletion.intervened_path_sha256 != selected_path.raw_path_sha256
        assert deletion.effective_dictionary_sha256 == replacement.effective_dictionary_sha256
        assert json.loads(deletion.effective_dictionary_payload_json)[
            "dictionary_intervention"
        ] is None
        assert deletion.to_dict()["claim_boundary"] == {
            "causal_claim_eligible": False,
            "physical_meaning_claimed": False,
            "scope": INTERPRETATION_SCOPE,
        }


def test_singleton_class_never_fabricates_an_equivalent_raw_path_control() -> None:
    truth_class = _singleton_class()
    selected_path = truth_class.members[0]
    registry = build_intervention_registry(truth_class, selected_path, 20)

    assert not truth_class.ambiguous
    assert any(record.decision_role == "targeted" for record in registry.path_records)
    assert all(
        record.condition != "registered_equivalent_raw_path_control"
        for record in registry.path_records
    )
    validate_intervention_registry(registry)


def test_wrong_recovery_is_retained_and_controls_follow_exported_semantics() -> None:
    truth_class = _square_class()
    selected_class = next(
        item
        for item in enumerate_equivalence_classes()
        if item.class_id != truth_class.class_id
        and item.ambiguous
        and any(operator != "I" for operator in item.members[-1].raw_path)
    )
    selected_path = selected_class.members[-1]

    registry = build_intervention_registry(truth_class, selected_path, 20)

    assert registry.truth_class_id == truth_class.class_id
    assert registry.selected_path_id == selected_path.raw_path_id
    assert registry.selected_path_class_id == selected_class.class_id
    assert registry.selected_path_class_sha256 == selected_class.class_sha256
    assert registry.selected_path_semantic_match is False
    equivalent_controls = tuple(
        record
        for record in registry.path_records
        if record.condition == "registered_equivalent_raw_path_control"
    )
    assert equivalent_controls
    assert all(
        next(
            member
            for member in selected_class.members
            if member.raw_path_id == record.intervened_path_id
        )
        for record in equivalent_controls
    )
    assert all(
        record.intervened_path_id
        not in {member.raw_path_id for member in truth_class.members}
        for record in equivalent_controls
    )
    validate_intervention_registry(registry)


def test_registry_is_deterministic_canonical_hashable_and_hash_drift_visible() -> None:
    truth_class = _square_class()
    selected_path = _ambiguous_selected_path()
    first = build_intervention_registry(truth_class, selected_path, 31)
    second = build_intervention_registry(truth_class, selected_path, 31)
    other_seed = build_intervention_registry(truth_class, selected_path, 42)

    assert first == second
    assert first.canonical_json() == second.canonical_json()
    assert first.manifest_sha256 == second.manifest_sha256
    assert first.manifest_sha256 != other_seed.manifest_sha256
    assert hash(first) == hash(second)
    parsed = json.loads(first.canonical_json())
    assert parsed["manifest_sha256"] == first.manifest_sha256
    assert all(
        item["record_sha256"]
        for item in (*parsed["dictionary_records"], *parsed["path_records"])
    )
    assert first.dictionary_records[0].canonical_json() == (
        registry_module.path_universe.canonical_json_bytes(
            first.dictionary_records[0].to_dict()
        ).decode("utf-8")
    )
    with pytest.raises(FrozenInstanceError):
        first.optimization_seed = 7  # type: ignore[misc]

    drifted = replace(first, optimization_seed=42)
    assert drifted.manifest_sha256 != first.manifest_sha256
    with pytest.raises(ValueError, match="seed|construction"):
        validate_intervention_registry(drifted)


def test_builder_has_no_model_score_dependency_and_preserves_torch_rng(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parameters = tuple(inspect.signature(build_intervention_registry).parameters)
    assert parameters == ("truth_class", "selected_path", "optimization_seed")

    def forbidden_score_path(*_args, **_kwargs):
        raise AssertionError("registry construction must not execute or score a model")

    monkeypatch.setattr(ExecutableOperatorPath1D, "relaxed_forward", forbidden_score_path)
    monkeypatch.setattr(ExecutableOperatorPath1D, "export_paths", forbidden_score_path)
    registry_module._operator_core.cache_clear()
    torch.manual_seed(20260801)
    before = torch.random.get_rng_state().clone()

    registry = build_intervention_registry(_square_class(), _ambiguous_selected_path(), 100)

    assert torch.equal(torch.random.get_rng_state(), before)
    assert registry.dictionary_records
    assert registry.path_records


def test_invalid_truth_class_path_seed_and_empty_control_pool_fail_closed() -> None:
    truth_class = _square_class()
    selected_path = _ambiguous_selected_path()
    forged_class = replace(truth_class, class_sha256="0" * 64)
    foreign_path = next(
        item.members[0]
        for item in enumerate_equivalence_classes()
        if item.class_id != truth_class.class_id
    )

    with pytest.raises(ValueError, match="registered equivalence class"):
        build_intervention_registry(forged_class, selected_path, 7)
    retained_failure = build_intervention_registry(truth_class, foreign_path, 7)
    assert retained_failure.selected_path_semantic_match is False
    with pytest.raises(ValueError, match="exact registered raw path"):
        build_intervention_registry(
            truth_class,
            replace(foreign_path, raw_path_sha256="0" * 64),
            7,
        )
    with pytest.raises(ValueError, match="25-seed namespace"):
        build_intervention_registry(truth_class, selected_path, 8)
    with pytest.raises(TypeError, match="integer"):
        build_intervention_registry(truth_class, selected_path, True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="No eligible registered candidate"):
        registry_module._matched_absent_operator(
            truth_class, "SQUARE", absent_operators=()
        )


def test_duplicate_ids_nonessential_removal_unregistered_replacement_and_missing_controls_fail(
) -> None:
    registry = build_intervention_registry(_square_class(), _ambiguous_selected_path(), 7)

    duplicate = _replace_dictionary_record(
        registry,
        1,
        record_id=registry.dictionary_records[0].record_id,
    )
    with pytest.raises(ValueError, match="duplicate record IDs"):
        validate_intervention_registry(duplicate)

    target_index = next(
        index
        for index, record in enumerate(registry.dictionary_records)
        if record.condition == "essential_operator_removal"
    )
    nonessential = _replace_dictionary_record(
        registry,
        target_index,
        target_operators=("ABS",),
        intervention=DictionaryIntervention(
            removed=((0, "ABS"), (1, "ABS"), (2, "ABS"))
        ),
    )
    with pytest.raises(ValueError, match="nonessential targeted removal"):
        validate_intervention_registry(nonessential)

    wrong_index = next(
        index
        for index, record in enumerate(registry.dictionary_records)
        if record.condition == "wrong_dictionary_derangement"
    )
    unregistered = _replace_dictionary_record(
        registry,
        wrong_index,
        intervention=DictionaryIntervention(replacements=((0, "D1", "NOT_REGISTERED"),)),
    )
    with pytest.raises(ValueError, match="Invalid registered dictionary intervention"):
        validate_intervention_registry(unregistered)

    control_index = next(
        index
        for index, record in enumerate(registry.dictionary_records)
        if record.condition == "matched_absent_operator_removal_control"
    )
    missing_control = replace(
        registry,
        dictionary_records=tuple(
            record
            for index, record in enumerate(registry.dictionary_records)
            if index != control_index
        ),
    )
    with pytest.raises(ValueError, match="missing"):
        validate_intervention_registry(missing_control)

    replacement_index = next(
        index
        for index, record in enumerate(registry.path_records)
        if record.condition == "selected_edge_registered_replacement"
    )
    path_records = list(registry.path_records)
    path_records[replacement_index] = replace(
        path_records[replacement_index], replacement_operator="MA5"
    )
    bad_path_replacement = replace(registry, path_records=tuple(path_records))
    with pytest.raises(ValueError, match="unregistered or a no-op"):
        validate_intervention_registry(bad_path_replacement)


def test_every_record_binds_registered_hashes_effective_payload_and_controls() -> None:
    registry = build_intervention_registry(_square_class(), _ambiguous_selected_path(), 113)
    all_ids = {
        record.record_id
        for record in (*registry.dictionary_records, *registry.path_records)
    }

    for record in (*registry.dictionary_records, *registry.path_records):
        payload = record.to_dict()
        assert payload["record_sha256"] == record.record_sha256
        assert len(record.original_path_sha256) == 64
        assert len(record.intervened_path_sha256) == 64
        assert len(record.base_dictionary_sha256) == 64
        assert len(record.effective_dictionary_sha256) == 64
        assert json.loads(record.effective_dictionary_payload_json)[
            "effective_dictionary_sha256"
        ] == record.effective_dictionary_sha256
        assert set(record.paired_record_ids).issubset(all_ids)
        assert payload["claim_boundary"]["causal_claim_eligible"] is False
        assert payload["claim_boundary"]["physical_meaning_claimed"] is False
        assert record.primary_eligible in {True, False}
        assert record.secondary_eligible in {True, False}
        assert all(
            operator in (*OPERATORS, "MA5")
            for operator in payload.get("target_operators", [])
        )
