from __future__ import annotations

import ast
import inspect
import json
import math
from dataclasses import FrozenInstanceError
from itertools import product

import numpy as np
import pytest

import src.utils.p07_protocol.statistics_engine as statistics_engine_module
from src.utils.p07_protocol.statistics_engine import (
    DEFAULT_BOOTSTRAP_DRAWS,
    DEFAULT_MONTE_CARLO_DRAWS,
    DEFAULT_RANDOM_SEED,
    EXACT_SIGN_FLIP_MAX_CLUSTERS,
    MAX_FAMILY_CONTRASTS,
    ContrastSpec,
    benjamini_hochberg,
    binary_auroc,
    crossed_cluster_seed_bootstrap,
    paired_hedges_gz,
    primary_cluster_sign_flip_sensitivity,
)


def _mean_endpoints(cells):
    return {name: float(np.mean(values)) for name, values in cells.items()}


def _base_cells() -> dict[str, np.ndarray]:
    base = np.arange(12, dtype=np.float64).reshape(3, 4)
    return {
        "method": 0.4 * base + np.asarray([0.0, 0.5, 1.0])[:, None],
        "reference": 0.2 * base - np.asarray([0.3, 0.0, 0.4])[:, None],
    }


def test_frozen_defaults_and_json_ready_output() -> None:
    assert DEFAULT_BOOTSTRAP_DRAWS == 10_000
    assert DEFAULT_RANDOM_SEED == 2_026_080_107
    assert DEFAULT_MONTE_CARLO_DRAWS == 100_000
    assert MAX_FAMILY_CONTRASTS == 7
    assert EXACT_SIGN_FLIP_MAX_CLUSTERS == 20

    specification = ContrastSpec("gain", "method", "reference", "higher")
    with pytest.raises(FrozenInstanceError):
        specification.contrast_id = "changed"  # type: ignore[misc]

    result = crossed_cluster_seed_bootstrap(
        _base_cells(),
        _mean_endpoints,
        (specification,),
        bootstrap_draws=16,
        random_seed=7,
    )
    encoded = json.dumps(result.to_dict(), allow_nan=False, sort_keys=True)
    assert "two_way_crossed_cluster_by_paired_seed_bootstrap" in encoded
    assert result.to_dict()["standard_error_ddof"] == 1


def test_crossed_bootstrap_is_deterministic_and_global_rng_isolated() -> None:
    specification = ContrastSpec("gain", "method", "reference", "higher")
    np.random.seed(91)
    expected_global_draw = np.random.random(5)
    np.random.seed(91)
    first = crossed_cluster_seed_bootstrap(
        _base_cells(),
        _mean_endpoints,
        (specification,),
        bootstrap_draws=256,
        random_seed=1234,
    )
    observed_global_draw = np.random.random(5)
    second = crossed_cluster_seed_bootstrap(
        _base_cells(),
        _mean_endpoints,
        (specification,),
        bootstrap_draws=256,
        random_seed=1234,
    )

    assert first == second
    np.testing.assert_array_equal(observed_global_draw, expected_global_draw)


def test_crossed_resampling_shares_cartesian_cluster_and_seed_indices() -> None:
    base = np.arange(12, dtype=np.float64).reshape(3, 4)
    cells = {"left": base, "right": 1000.0 + 3.0 * base}
    seen: list[dict[str, np.ndarray]] = []
    read_only: list[bool] = []

    def recording_callback(resampled):
        seen.append({name: values.copy() for name, values in resampled.items()})
        read_only.append(all(not values.flags.writeable for values in resampled.values()))
        return _mean_endpoints(resampled)

    crossed_cluster_seed_bootstrap(
        cells,
        recording_callback,
        (ContrastSpec("difference", "left", "right", "higher"),),
        bootstrap_draws=2,
        random_seed=17,
    )
    generator = np.random.default_rng(17)
    cluster_indices = generator.integers(0, 3, size=3)
    seed_indices = generator.integers(0, 4, size=4)
    expected_left = np.take(
        np.take(base, cluster_indices, axis=0), seed_indices, axis=1
    )
    expected_right = np.take(
        np.take(cells["right"], cluster_indices, axis=0), seed_indices, axis=1
    )

    assert len(seen) == 3  # observed evaluation plus two bootstrap evaluations
    assert all(read_only)
    np.testing.assert_array_equal(seen[1]["left"], expected_left)
    np.testing.assert_array_equal(seen[1]["right"], expected_right)
    np.testing.assert_array_equal(seen[1]["right"], 1000.0 + 3.0 * seen[1]["left"])


def test_fixed_small_example_matches_single_step_bound_math() -> None:
    cells = _base_cells()
    specification = ContrastSpec("gain", "method", "reference", "higher")
    draws = 11
    alpha = 0.20
    seed = 29
    result = crossed_cluster_seed_bootstrap(
        cells,
        _mean_endpoints,
        (specification,),
        bootstrap_draws=draws,
        random_seed=seed,
        family_alpha=alpha,
    )

    observed = float(np.mean(cells["method"]) - np.mean(cells["reference"]))
    generator = np.random.default_rng(seed)
    bootstrap_effects = []
    for _ in range(draws):
        cluster_indices = generator.integers(0, 3, size=3)
        seed_indices = generator.integers(0, 4, size=4)
        method = np.take(
            np.take(cells["method"], cluster_indices, axis=0), seed_indices, axis=1
        )
        reference = np.take(
            np.take(cells["reference"], cluster_indices, axis=0), seed_indices, axis=1
        )
        bootstrap_effects.append(float(np.mean(method) - np.mean(reference)))
    bootstrap_effects_array = np.asarray(bootstrap_effects)
    standard_error = float(np.std(bootstrap_effects_array, ddof=1))
    centered_t = (observed - bootstrap_effects_array) / standard_error
    rank = min(draws, math.ceil((1.0 - alpha) * (draws + 1)))
    critical = max(0.0, float(np.partition(centered_t, rank - 1)[rank - 1]))
    expected_lower = observed - critical * standard_error

    inference = result.contrasts[0]
    assert result.max_t_critical_value == pytest.approx(critical)
    assert inference.bootstrap_standard_error == pytest.approx(standard_error)
    assert inference.simultaneous_lower_bound == pytest.approx(expected_lower)
    assert inference.simultaneous_upper_bound is None


def test_mixed_directions_translate_one_favorable_max_t_bound() -> None:
    base = np.arange(20, dtype=np.float64).reshape(4, 5)
    cells = {
        "score_method": base + np.asarray([0.0, 0.3, 0.8, 1.2])[:, None],
        "score_reference": 0.7 * base,
        "cost_method": 0.4 * base,
        "cost_reference": 0.6 * base + np.asarray([0.1, 0.2, 0.3, 0.4])[:, None],
    }
    result = crossed_cluster_seed_bootstrap(
        cells,
        _mean_endpoints,
        (
            ContrastSpec("score_gain", "score_method", "score_reference", "higher"),
            ContrastSpec("cost_reduction", "cost_method", "cost_reference", "lower"),
        ),
        bootstrap_draws=128,
        random_seed=71,
    )
    score, cost = result.contrasts

    assert score.favorable_effect == score.raw_effect
    assert score.simultaneous_lower_bound == pytest.approx(
        score.raw_effect
        - result.max_t_critical_value * score.bootstrap_standard_error
    )
    assert score.simultaneous_upper_bound is None
    assert cost.favorable_effect == -cost.raw_effect
    assert cost.simultaneous_lower_bound is None
    assert cost.simultaneous_upper_bound == pytest.approx(
        cost.raw_effect + result.max_t_critical_value * cost.bootstrap_standard_error
    )
    assert cost.favorable_scale_lower_bound == pytest.approx(
        -cost.simultaneous_upper_bound
    )


def test_bootstrap_zero_standard_error_has_exact_point_bound() -> None:
    cells = {
        "method": np.full((3, 4), 2.0),
        "reference": np.full((3, 4), 1.0),
    }
    result = crossed_cluster_seed_bootstrap(
        cells,
        _mean_endpoints,
        (ContrastSpec("constant", "method", "reference", "higher"),),
        bootstrap_draws=32,
        random_seed=3,
    )
    inference = result.contrasts[0]

    assert result.max_t_critical_value == 0.0
    assert inference.zero_standard_error
    assert inference.bootstrap_standard_error == 0.0
    assert inference.raw_effect == 1.0
    assert inference.simultaneous_lower_bound == 1.0


def test_bootstrap_family_guard_rejects_more_than_seven() -> None:
    cells: dict[str, np.ndarray] = {}
    specifications = []
    for index in range(8):
        cells[f"left_{index}"] = np.full((2, 2), float(index + 1))
        cells[f"right_{index}"] = np.zeros((2, 2))
        specifications.append(
            ContrastSpec(
                f"contrast_{index}", f"left_{index}", f"right_{index}", "higher"
            )
        )
    with pytest.raises(ValueError, match="at most 7"):
        crossed_cluster_seed_bootstrap(
            cells, _mean_endpoints, specifications, bootstrap_draws=2
        )


@pytest.mark.parametrize(
    ("callback", "exception", "match"),
    [
        (lambda cells: {"method": 1.0}, ValueError, "omitted required endpoints"),
        (
            lambda cells: {"method": float("nan"), "reference": 0.0},
            ValueError,
            "non-finite",
        ),
        (lambda cells: [1.0, 0.0], TypeError, "nonempty mapping"),
        (
            lambda cells: {"method": True, "reference": 0.0},
            TypeError,
            "real scalar",
        ),
    ],
)
def test_bootstrap_callback_fails_closed(callback, exception, match: str) -> None:
    with pytest.raises(exception, match=match):
        crossed_cluster_seed_bootstrap(
            _base_cells(),
            callback,
            (ContrastSpec("gain", "method", "reference", "higher"),),
            bootstrap_draws=2,
        )


def test_bootstrap_rejects_nonfinite_callback_value_on_a_resample() -> None:
    calls = 0

    def fails_after_observed(cells):
        nonlocal calls
        calls += 1
        endpoints = _mean_endpoints(cells)
        if calls == 2:
            endpoints["method"] = float("inf")
        return endpoints

    with pytest.raises(ValueError, match="bootstrap draw 0.*non-finite"):
        crossed_cluster_seed_bootstrap(
            _base_cells(),
            fails_after_observed,
            (ContrastSpec("gain", "method", "reference", "higher"),),
            bootstrap_draws=2,
        )


@pytest.mark.parametrize(
    ("cells", "exception", "match"),
    [
        (
            {"a": np.asarray([[1.0, np.nan], [2.0, 3.0]]), "b": np.ones((2, 2))},
            ValueError,
            "missing or non-finite",
        ),
        (
            {"a": np.ones((2, 2)), "b": np.ones((3, 2))},
            ValueError,
            "share cluster",
        ),
        (
            {"a": np.ma.array([[1.0, 2.0], [3.0, 4.0]], mask=True), "b": np.ones((2, 2))},
            ValueError,
            "masked cells",
        ),
        (
            {"a": np.ones((2, 1)), "b": np.ones((2, 1))},
            ValueError,
            "at least two clusters and two seeds",
        ),
        (
            {"a": np.asarray([["x", "y"], ["z", "w"]]), "b": np.ones((2, 2))},
            TypeError,
            "real numeric",
        ),
    ],
)
def test_bootstrap_rejects_malformed_or_missing_paired_cells(
    cells, exception, match: str
) -> None:
    with pytest.raises(exception, match=match):
        crossed_cluster_seed_bootstrap(
            cells,
            _mean_endpoints,
            (ContrastSpec("difference", "a", "b", "higher"),),
            bootstrap_draws=2,
        )


def test_paired_hedges_gz_uses_ddof_one_and_small_sample_correction() -> None:
    reference = np.asarray([0.0, 1.0, 3.0, 6.0])
    treatment = reference + np.asarray([1.0, 2.0, 3.0, 4.0])
    result = paired_hedges_gz(treatment, reference)
    expected_sd = float(np.std([1.0, 2.0, 3.0, 4.0], ddof=1))
    expected_correction = 1.0 - 3.0 / (4.0 * (4 - 1) - 1.0)

    assert result.n_pairs == 4
    assert result.raw_effect == 2.5
    assert result.difference_standard_deviation == pytest.approx(expected_sd)
    assert result.cohen_dz == pytest.approx(2.5 / expected_sd)
    assert result.hedges_correction == pytest.approx(expected_correction)
    assert result.hedges_gz == pytest.approx(expected_correction * 2.5 / expected_sd)
    json.dumps(result.to_dict(), allow_nan=False)


def test_paired_hedges_zero_and_invalid_cases_fail_closed() -> None:
    equal = paired_hedges_gz([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert equal.raw_effect == 0.0
    assert equal.cohen_dz == 0.0
    assert equal.hedges_gz == 0.0
    with pytest.raises(ValueError, match="nonzero constant difference"):
        paired_hedges_gz([2.0, 2.0, 2.0], [1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="same paired"):
        paired_hedges_gz([1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0])
    with pytest.raises(ValueError, match="at least 3"):
        paired_hedges_gz([1.0, 2.0], [0.0, 0.0])
    with pytest.raises(ValueError, match="non-finite"):
        paired_hedges_gz([1.0, 2.0, np.inf], [0.0, 0.0, 0.0])


def test_binary_auroc_uses_average_tied_ranks() -> None:
    assert binary_auroc([0, 1, 0, 1], [0.1, 0.4, 0.4, 0.8]) == pytest.approx(0.875)
    assert binary_auroc([0, 0, 1, 1], [0.5, 0.5, 0.5, 0.5]) == 0.5
    assert binary_auroc([0, 1], [1.0, 0.0]) == 0.0


@pytest.mark.parametrize(
    ("labels", "scores", "match"),
    [
        ([1, 1, 1], [0.1, 0.2, 0.3], "both classes"),
        ([0, 1, 2], [0.1, 0.2, 0.3], "binary values"),
        ([0, 1], [0.1], "same length"),
        ([0, 1], [0.1, np.nan], "non-finite"),
        ([[0, 1]], [[0.1, 0.2]], "one-dimensional"),
    ],
)
def test_binary_auroc_rejects_invalid_inputs(labels, scores, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        binary_auroc(labels, scores)


def test_benjamini_hochberg_returns_monotone_q_values_in_original_order() -> None:
    result = benjamini_hochberg([0.01, 0.04, 0.03, 0.20], alpha=0.05)
    np.testing.assert_allclose(result.q_values, [0.04, 0.053333333333, 0.053333333333, 0.20])
    assert result.rejected == (True, False, False, False)
    assert result.to_dict()["method"] == "Benjamini-Hochberg"
    with pytest.raises(ValueError, match="closed interval"):
        benjamini_hochberg([0.1, 1.1])
    with pytest.raises(ValueError, match="non-finite"):
        benjamini_hochberg([0.1, np.nan])


def test_benjamini_hochberg_allows_more_than_seven_secondary_tests() -> None:
    p_values = np.asarray(
        [0.001, 0.20, 0.04, 0.03, 0.80, 0.002, 0.07, 0.50, 0.015, 0.90]
    )
    result = benjamini_hochberg(p_values, alpha=0.05)
    q_values = np.asarray(result.q_values)
    order = np.argsort(p_values, kind="mergesort")

    assert len(result.q_values) == 10
    assert np.all(np.diff(q_values[order]) >= 0.0)
    assert result.rejected == tuple(bool(value <= 0.05) for value in q_values)
    assert any(result.rejected)
    assert not all(result.rejected)


def _manual_sign_flip_adjusted_p(
    favorable_cluster_effects: np.ndarray,
) -> np.ndarray:
    n_clusters, n_contrasts = favorable_cluster_effects.shape
    observed_mean = np.mean(favorable_cluster_effects, axis=0)
    observed_se = np.std(favorable_cluster_effects, axis=0, ddof=1) / math.sqrt(
        n_clusters
    )
    observed_t = observed_mean / observed_se
    maxima = []
    for signs_tuple in product((-1.0, 1.0), repeat=n_clusters):
        signed = favorable_cluster_effects * np.asarray(signs_tuple)[:, None]
        mean = np.mean(signed, axis=0)
        standard_error = np.std(signed, axis=0, ddof=1) / math.sqrt(n_clusters)
        maxima.append(float(np.max(mean / standard_error)))
    maxima_array = np.asarray(maxima)
    return np.asarray(
        [np.mean(maxima_array >= np.nextafter(value, -np.inf)) for value in observed_t]
    ).reshape(n_contrasts)


def test_exact_shared_sign_flip_max_t_handles_mixed_directions() -> None:
    gain = np.repeat(np.asarray([1.0, 2.0, 3.0])[:, None], 2, axis=1)
    loss = -np.repeat(np.asarray([0.5, 1.5, 2.5])[:, None], 2, axis=1)
    result = primary_cluster_sign_flip_sensitivity(
        {"gain": gain, "loss": loss},
        {"gain": "higher", "loss": "lower"},
        family_alpha=0.20,
    )
    favorable_cluster_effects = np.column_stack((gain.mean(axis=1), -loss.mean(axis=1)))
    expected = _manual_sign_flip_adjusted_p(favorable_cluster_effects)

    assert result.method == "exact_primary_cluster_sign_flip"
    assert result.draw_count == 2**3
    assert result.p_value_reference_count == 2**3
    assert result.random_seed is None
    np.testing.assert_allclose(
        [item.family_adjusted_p_value for item in result.contrasts], expected
    )
    assert result.contrasts[0].favorable_effect > 0.0
    assert result.contrasts[1].raw_effect < 0.0
    assert result.contrasts[1].favorable_effect > 0.0
    json.dumps(result.to_dict(), allow_nan=False)


def test_exact_sign_flip_zero_se_is_explicit_and_finite_json() -> None:
    constant = np.ones((3, 2), dtype=np.float64)
    result = primary_cluster_sign_flip_sensitivity(
        {"constant": constant}, {"constant": "higher"}
    )
    inference = result.contrasts[0]

    assert inference.cluster_standard_error == 0.0
    assert inference.observed_t is None
    assert inference.zero_standard_error_case == "positive_favorable_effect"
    assert inference.family_adjusted_p_value == pytest.approx(1.0 / 8.0)
    json.dumps(result.to_dict(), allow_nan=False)


def test_monte_carlo_sign_flip_is_reproducible_and_explicit() -> None:
    cluster_effect = np.linspace(-1.0, 1.25, 21)
    seed_offsets = np.asarray([-0.2, 0.0, 0.2])
    gain = cluster_effect[:, None] + seed_offsets[None, :]
    loss = -0.7 * cluster_effect[:, None] + seed_offsets[None, :]
    arguments = (
        {"gain": gain, "loss": loss},
        {"gain": "higher", "loss": "lower"},
    )
    first = primary_cluster_sign_flip_sensitivity(
        *arguments, monte_carlo_draws=513, random_seed=991
    )
    second = primary_cluster_sign_flip_sensitivity(
        *arguments, monte_carlo_draws=513, random_seed=991
    )

    assert first == second
    assert first.method == "deterministic_monte_carlo_primary_cluster_sign_flip"
    assert first.draw_count == 513
    assert first.p_value_reference_count == 514
    assert first.random_seed == 991
    for inference in first.contrasts:
        scaled = inference.family_adjusted_p_value * 514
        assert scaled == pytest.approx(round(scaled))


def test_sign_flip_rejects_invalid_families_and_missing_cells() -> None:
    valid = np.ones((3, 2))
    with pytest.raises(ValueError, match="exactly"):
        primary_cluster_sign_flip_sensitivity(
            {"gain": valid}, {"wrong": "higher"}
        )
    with pytest.raises(ValueError, match="missing or non-finite"):
        bad = valid.copy()
        bad[0, 0] = np.nan
        primary_cluster_sign_flip_sensitivity({"gain": bad}, {"gain": "higher"})
    with pytest.raises(ValueError, match="share cluster and seed"):
        primary_cluster_sign_flip_sensitivity(
            {"a": valid, "b": np.ones((4, 2))},
            {"a": "higher", "b": "lower"},
        )
    with pytest.raises(ValueError, match="at most 7"):
        matrices = {f"c{index}": valid for index in range(8)}
        directions = {name: "higher" for name in matrices}
        primary_cluster_sign_flip_sensitivity(matrices, directions)
    with pytest.raises(ValueError, match="higher.*lower"):
        primary_cluster_sign_flip_sensitivity(
            {"gain": valid}, {"gain": "sideways"}  # type: ignore[dict-item]
        )


def test_statistics_engine_has_no_filesystem_or_model_runtime_dependency() -> None:
    source = inspect.getsource(statistics_engine_module)
    tree = ast.parse(source)
    imported_modules: list[str] = []
    called_names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported_modules.append(node.module or "")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            called_names.append(node.func.id)

    assert not any(
        token in module
        for module in imported_modules
        for token in ("pathlib", "torch", "model", "trainer", "experiment_runner")
    )
    assert "open" not in called_names
    assert "evidence_guard" not in source
    assert "paper/experiments" not in source
