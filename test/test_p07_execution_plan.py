from __future__ import annotations

from dataclasses import replace

import pytest

from src.utils.p07_protocol.execution_plan import (
    DIRG_FOLDS,
    NEURAL_DIRG_ARMS,
    PLAN_DOMAIN,
    PLAN_SCHEMA_VERSION,
    ExecutionPlan,
    build_execution_plan,
    ready_work_units,
    select_work_shard,
    validate_execution_plan,
)
from src.utils.p07_protocol.path_universe import OPTIMIZATION_SEEDS


PROTOCOL_SHA = "a" * 64


def _plan(*, approved: bool = False) -> ExecutionPlan:
    return build_execution_plan(
        protocol_sha256=PROTOCOL_SHA,
        human_gate_snapshot=approved,
        thresholds_approved_snapshot=approved,
    )


def test_plan_has_exact_frozen_counts_and_is_deterministic() -> None:
    first = _plan()
    second = _plan()
    counts = first.to_payload()["counts"]
    payload = first.to_payload()
    assert first == second
    assert first.plan_sha256 == second.plan_sha256
    assert payload["schema_version"] == PLAN_SCHEMA_VERSION == 2
    assert payload["domain"] == PLAN_DOMAIN == "P07-E7-E11-EXECUTION-PLAN-v2"
    assert counts == {
        "total": 3799,
        "by_stage": {
            "cwru_confirmatory_test": 378,
            "cwru_fit_select": 378,
            "dirg_confirmatory_test": 378,
            "dirg_fit_select": 378,
            "synthetic_confirmatory_test": 918,
            "synthetic_fit_select": 918,
            "synthetic_intervention_test": 450,
            "synthetic_threshold_calibration": 1,
        },
        "confirmatory_test": 2124,
        "fit_select": 1675,
    }
    assert len({unit.work_unit_id for unit in first.units}) == 3799


def test_every_synthetic_fit_persists_fit_only_normalization() -> None:
    plan = _plan()
    fit_units = tuple(
        unit for unit in plan.units if unit.stage == "synthetic_fit_select"
    )
    assert len(fit_units) == 918
    assert all(
        "normalization_artifact.json" in unit.required_outputs
        for unit in fit_units
    )


def test_deterministic_searches_are_not_pretended_to_be_25_fits() -> None:
    plan = _plan()
    synthetic = [
        unit
        for unit in plan.units
        if unit.stage == "synthetic_fit_select"
        and unit.arm_id == "full_216_discrete_search"
    ]
    cwru = [
        unit
        for unit in plan.units
        if unit.stage == "cwru_fit_select"
        and unit.arm_id == "full_216_discrete_search"
    ]
    dirg = [
        unit
        for unit in plan.units
        if unit.stage == "dirg_fit_select"
        and unit.arm_id == "full_216_discrete_search"
    ]
    assert len(synthetic) == 18
    assert len(cwru) == 3
    assert len(dirg) == 3
    deterministic = synthetic + cwru + dirg
    assert all(unit.optimization_seed is None for unit in deterministic)
    assert all(not unit.stochastic_fit for unit in deterministic)
    assert all(
        "joined_to_all_method_seeds" in unit.replication_policy
        for unit in deterministic
    )


def test_dirg_has_exact_fold_arm_seed_coverage_and_fit_dependencies() -> None:
    plan = _plan()
    dirg = tuple(unit for unit in plan.units if unit.stage.startswith("dirg_"))
    fits = tuple(unit for unit in dirg if unit.stage == "dirg_fit_select")
    tests = tuple(unit for unit in dirg if unit.stage == "dirg_confirmatory_test")
    neural_fits = tuple(
        unit for unit in fits if unit.arm_id in NEURAL_DIRG_ARMS
    )
    neural_tests = tuple(
        unit for unit in tests if unit.arm_id in NEURAL_DIRG_ARMS
    )
    expected_neural_keys = {
        (fold_id, arm_id, seed)
        for fold_id in DIRG_FOLDS
        for arm_id in NEURAL_DIRG_ARMS
        for seed in OPTIMIZATION_SEEDS
    }
    fit_by_key = {
        (unit.fold_id, unit.arm_id, unit.optimization_seed): unit
        for unit in neural_fits
    }
    test_by_key = {
        (unit.fold_id, unit.arm_id, unit.optimization_seed): unit
        for unit in neural_tests
    }

    assert len(dirg) == 756
    assert len(fits) == len(tests) == 378
    assert len(neural_fits) == len(neural_tests) == 375
    assert set(fit_by_key) == set(test_by_key) == expected_neural_keys
    assert all(unit.experiment_ids == ("E9", "E10", "E11") for unit in dirg)
    assert all(unit.composition_class_id is None for unit in dirg)
    assert all(unit.depends_on == () for unit in fits)
    assert all(not unit.uses_confirmatory_test for unit in fits)
    assert all(unit.stochastic_fit for unit in neural_fits)
    assert all(
        unit.replication_policy
        == "fresh_held_severity_fold_model_for_each_registered_seed"
        for unit in neural_fits
    )
    assert all(unit.uses_confirmatory_test for unit in tests)
    assert all(not unit.stochastic_fit for unit in tests)
    for key, test_unit in test_by_key.items():
        assert test_unit.depends_on == (fit_by_key[key].work_unit_id,)

    ridge_fits = {
        unit.fold_id: unit
        for unit in fits
        if unit.arm_id == "full_216_discrete_search"
    }
    ridge_tests = {
        unit.fold_id: unit
        for unit in tests
        if unit.arm_id == "full_216_discrete_search"
    }
    assert len(ridge_fits) == len(ridge_tests) == len(DIRG_FOLDS)
    assert set(ridge_fits) == set(ridge_tests) == set(DIRG_FOLDS)
    assert all(unit.optimization_seed is None for unit in ridge_fits.values())
    assert all(not unit.stochastic_fit for unit in ridge_fits.values())
    assert all(
        ridge_tests[fold_id].depends_on == (ridge_fits[fold_id].work_unit_id,)
        for fold_id in DIRG_FOLDS
    )


@pytest.mark.parametrize(
    ("human_gate", "thresholds_approved"),
    ((False, False), (True, False), (False, True)),
)
def test_dirg_test_nodes_require_both_approval_gates(
    human_gate: bool,
    thresholds_approved: bool,
) -> None:
    plan = build_execution_plan(
        protocol_sha256=PROTOCOL_SHA,
        human_gate_snapshot=human_gate,
        thresholds_approved_snapshot=thresholds_approved,
    )
    fit = next(unit for unit in plan.units if unit.stage == "dirg_fit_select")
    ready = ready_work_units(
        plan,
        completed_work_unit_ids=(fit.work_unit_id,),
        allow_confirmatory_test=True,
    )
    assert not any(unit.uses_confirmatory_test for unit in ready)


def test_approved_dirg_test_node_requires_its_completed_fit() -> None:
    plan = _plan(approved=True)
    fit = next(
        unit
        for unit in plan.units
        if unit.stage == "dirg_fit_select"
        and unit.arm_id == "proposed"
        and unit.fold_id == DIRG_FOLDS[0]
    )
    before_fit = ready_work_units(
        plan,
        completed_work_unit_ids=(),
        allow_confirmatory_test=True,
    )
    assert not any(fit.work_unit_id in unit.depends_on for unit in before_fit)

    after_fit = ready_work_units(
        plan,
        completed_work_unit_ids=(fit.work_unit_id,),
        allow_confirmatory_test=True,
    )
    dependent = [unit for unit in after_fit if fit.work_unit_id in unit.depends_on]
    assert len(dependent) == 1
    assert dependent[0].stage == "dirg_confirmatory_test"

    hidden = ready_work_units(
        plan,
        completed_work_unit_ids=(fit.work_unit_id,),
        allow_confirmatory_test=False,
    )
    assert not any(fit.work_unit_id in unit.depends_on for unit in hidden)


def test_false_gate_never_releases_confirmatory_test_nodes() -> None:
    plan = _plan(approved=False)
    ready = ready_work_units(
        plan,
        completed_work_unit_ids=(),
        allow_confirmatory_test=True,
    )
    assert ready
    assert all(not unit.uses_confirmatory_test for unit in ready)
    assert plan.evidence_execution_allowed is False


def test_approved_plan_releases_only_tests_with_completed_dependencies() -> None:
    plan = _plan(approved=True)
    fit = next(
        unit
        for unit in plan.units
        if unit.stage == "synthetic_fit_select" and unit.arm_id == "proposed"
    )
    ready = ready_work_units(
        plan,
        completed_work_unit_ids=(fit.work_unit_id,),
        allow_confirmatory_test=True,
    )
    dependent = [unit for unit in ready if fit.work_unit_id in unit.depends_on]
    assert {unit.stage for unit in dependent} == {"synthetic_confirmatory_test"}
    hidden = ready_work_units(
        plan,
        completed_work_unit_ids=(fit.work_unit_id,),
        allow_confirmatory_test=False,
    )
    assert all(not unit.uses_confirmatory_test for unit in hidden)


def test_one_pooled_validation_calibration_is_a_global_barrier() -> None:
    plan = _plan(approved=True)
    proposed_fits = tuple(
        unit
        for unit in plan.units
        if unit.stage == "synthetic_fit_select" and unit.arm_id == "proposed"
    )
    assert len(proposed_fits) == 450
    completed_fits = {unit.work_unit_id for unit in proposed_fits}
    ready = ready_work_units(
        plan,
        completed_work_unit_ids=completed_fits,
        allow_confirmatory_test=True,
    )
    calibrations = [
        unit for unit in ready if unit.stage == "synthetic_threshold_calibration"
    ]
    assert len(calibrations) == 1
    assert set(calibrations[0].depends_on) == completed_fits
    assert not any(unit.stage == "synthetic_intervention_test" for unit in ready)

    completed_with_calibration = completed_fits | {calibrations[0].work_unit_id}
    released = ready_work_units(
        plan,
        completed_work_unit_ids=completed_with_calibration,
        allow_confirmatory_test=True,
    )
    assert sum(unit.stage == "synthetic_intervention_test" for unit in released) == 450


def test_completed_set_must_be_dependency_closed() -> None:
    plan = _plan(approved=True)
    calibration = next(
        unit for unit in plan.units if unit.stage == "synthetic_threshold_calibration"
    )
    with pytest.raises(ValueError, match="dependency closed"):
        ready_work_units(
            plan,
            completed_work_unit_ids=(calibration.work_unit_id,),
            allow_confirmatory_test=True,
        )


def test_tampered_unit_or_plan_hash_is_rejected() -> None:
    plan = _plan()
    tampered_unit = replace(plan.units[0], arm_id="tampered")
    with pytest.raises(ValueError, match="deterministic ID"):
        validate_execution_plan(replace(plan, units=(tampered_unit, *plan.units[1:])))
    with pytest.raises(ValueError, match="plan hash"):
        validate_execution_plan(replace(plan, plan_sha256="b" * 64))


def test_mixed_approval_snapshot_is_not_evidence() -> None:
    plan = build_execution_plan(
        protocol_sha256=PROTOCOL_SHA,
        human_gate_snapshot=True,
        thresholds_approved_snapshot=False,
    )
    assert plan.evidence_execution_allowed is False


def test_shards_are_disjoint_complete_and_stable() -> None:
    units = _plan().units[:137]
    shards = [
        select_work_shard(units, shard_index=index, shard_count=7)
        for index in range(7)
    ]
    flattened = [unit for shard in shards for unit in shard]
    assert len(flattened) == len(units)
    assert {unit.work_unit_id for unit in flattened} == {
        unit.work_unit_id for unit in units
    }
    assert len({unit.work_unit_id for unit in flattened}) == len(units)
    assert shards == [
        select_work_shard(units, shard_index=index, shard_count=7)
        for index in range(7)
    ]


@pytest.mark.parametrize("bad", ("short", "A" * 64, "g" * 64))
def test_protocol_digest_must_be_canonical_sha256(bad: str) -> None:
    with pytest.raises(ValueError, match="protocol_sha256"):
        build_execution_plan(
            protocol_sha256=bad,
            human_gate_snapshot=False,
            thresholds_approved_snapshot=False,
        )


def test_unknown_completed_work_id_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown"):
        ready_work_units(
            _plan(),
            completed_work_unit_ids=("P07-WORK-deadbeef",),
            allow_confirmatory_test=False,
        )
