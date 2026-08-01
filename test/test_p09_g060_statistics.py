from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.p09.g060_statistics import (
    benjamini_hochberg,
    bonferroni_percentile_interval,
    bootstrap_contrasts,
    decision_gates,
    exact_one_sided_sign_flip_pvalue,
    generate_bootstrap_plan,
    prepare_core_metric_cube,
)


STATE_K = {"clean": (1, 5), "imbalance": (5,)}


def metric_table() -> pd.DataFrame:
    rows = []
    for target in (1, 5):
        for seed in (42, 123):
            for arm, offset in (("P", 1.0), ("B0", 0.0), ("B1", -1.0)):
                for episode in (0, 1, 2):
                    for state, k_values in STATE_K.items():
                        for k_shot in k_values:
                            state_value = 10.0 if state == "clean" else 30.0
                            rows.append(
                                {
                                    "target_system": target,
                                    "seed": seed,
                                    "arm": arm,
                                    "episode": episode,
                                    "core_id": f"t{target}-s{seed}-e{episode}",
                                    "support_state": state,
                                    "k_shot": k_shot,
                                    "harmonic_mean": state_value + offset,
                                }
                            )
    return pd.DataFrame(rows)


def test_state_and_k_are_equal_weighted_before_seed_and_target() -> None:
    cube = prepare_core_metric_cube(
        metric_table(),
        metric="harmonic_mean",
        arms=("P", "B0", "B1"),
        targets=(1, 5),
        seeds=(42, 123),
        episodes=(0, 1, 2),
        state_k=STATE_K,
    )
    assert cube.values.shape == (3, 2, 2, 3)
    assert np.all(cube.values[0] == 21.0)
    assert np.all(cube.values[1] == 20.0)
    assert np.all(cube.values[2] == 19.0)


def test_missing_or_duplicate_pair_blocks_analysis() -> None:
    table = metric_table()
    kwargs = {
        "metric": "harmonic_mean",
        "arms": ("P", "B0", "B1"),
        "targets": (1, 5),
        "seeds": (42, 123),
        "episodes": (0, 1, 2),
        "state_k": STATE_K,
    }
    with pytest.raises(ValueError, match="not exactly complete"):
        prepare_core_metric_cube(table.iloc[1:], **kwargs)
    with pytest.raises(ValueError, match="duplicate"):
        prepare_core_metric_cube(pd.concat((table, table.iloc[[0]])), **kwargs)


def test_shared_hierarchical_plan_is_deterministic_and_paired() -> None:
    cube = prepare_core_metric_cube(
        metric_table(),
        metric="harmonic_mean",
        arms=("P", "B0", "B1"),
        targets=(1, 5),
        seeds=(42, 123),
        episodes=(0, 1, 2),
        state_k=STATE_K,
    )
    first = generate_bootstrap_plan(
        draws=25, targets=2, seeds=2, episodes=3, analysis_seed=20260801
    )
    second = generate_bootstrap_plan(
        draws=25, targets=2, seeds=2, episodes=3, analysis_seed=20260801
    )
    assert first.sha256 == second.sha256
    contrasts = bootstrap_contrasts(
        cube,
        first,
        numerator_arm="P",
        fixed_comparator_arms=("B0", "B1"),
        composite_comparator_by_target={1: "B0", 5: "B1"},
    )
    assert np.all(contrasts["P_minus_B0"] == 1.0)
    assert np.all(contrasts["P_minus_B1"] == 2.0)
    assert set(np.unique(contrasts["P_minus_selected_composite"])) <= {1.0, 1.5, 2.0}


def test_bonferroni_interval_uses_two_sided_family_alpha() -> None:
    values = np.arange(1001, dtype=np.float64)
    lower, upper = bonferroni_percentile_interval(
        values, family_size=11, familywise_alpha=0.05
    )
    assert lower == pytest.approx(np.quantile(values, 0.05 / 22.0))
    assert upper == pytest.approx(np.quantile(values, 1.0 - 0.05 / 22.0))


def test_exact_six_target_one_sided_sign_flip_has_64_patterns() -> None:
    assert exact_one_sided_sign_flip_pvalue(np.ones(6)) == pytest.approx(1.0 / 64.0)


def test_bh_adjustment_restores_original_order_and_is_monotone() -> None:
    values = np.asarray([0.04, 0.001, 0.03, 0.20])
    adjusted = benjamini_hochberg(values)
    order = np.argsort(values)
    assert np.all(np.diff(adjusted[order]) >= 0.0)
    assert adjusted[1] == pytest.approx(0.004)


def test_decision_boundaries_follow_strict_and_nonstrict_protocol_rules() -> None:
    gates = decision_gates(
        primary_simultaneous_lower=0.0,
        base_one_sided_lower=-0.02,
        aurc_one_sided_upper=0.01,
        minimum_state_delta=-0.02,
    )
    assert gates == {
        "primary_superiority": False,
        "base_noninferiority": False,
        "aurc_noninferiority": True,
        "state_consistency": True,
        "all_confirmatory_gates": False,
    }
