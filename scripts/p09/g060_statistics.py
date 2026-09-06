"""Preregistered paired aggregation and inference utilities for P09-G060."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


DEFAULT_STATE_K: Mapping[str, tuple[int, ...]] = {
    "clean": (1, 5, 10, 20),
    "label_noise": (1, 5, 10, 20),
    "outlier": (1, 5, 10, 20),
    "imbalance": (5, 10, 20),
}


@dataclass(frozen=True)
class CoreMetricCube:
    values: np.ndarray
    arms: tuple[str, ...]
    targets: tuple[int, ...]
    seeds: tuple[int, ...]
    episodes: tuple[int, ...]


@dataclass(frozen=True)
class BootstrapPlan:
    target_indices: np.ndarray
    seed_indices: np.ndarray
    episode_indices: np.ndarray
    seed: int

    @property
    def draws(self) -> int:
        return int(self.target_indices.shape[0])

    @property
    def sha256(self) -> str:
        digest = hashlib.sha256()
        digest.update(str(self.seed).encode("ascii"))
        for value in (
            self.target_indices,
            self.seed_indices,
            self.episode_indices,
        ):
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(str(value.shape).encode("ascii"))
            digest.update(value.tobytes())
        return digest.hexdigest()


def prepare_core_metric_cube(
    table: pd.DataFrame,
    *,
    metric: str,
    arms: Sequence[str],
    targets: Sequence[int],
    seeds: Sequence[int],
    episodes: Sequence[int],
    state_k: Mapping[str, Sequence[int]] = DEFAULT_STATE_K,
) -> CoreMetricCube:
    """Validate the complete paired grid and collapse K/state within each core."""

    required = {
        "target_system",
        "seed",
        "arm",
        "episode",
        "core_id",
        "support_state",
        "k_shot",
        metric,
    }
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"metric table lacks required columns: {missing}")
    selected = table[list(required)].copy()
    if selected.empty:
        raise ValueError("metric table is empty")
    if not np.isfinite(selected[metric].to_numpy(dtype=np.float64)).all():
        raise ValueError(f"metric {metric} contains non-finite values")

    key_columns = [
        "target_system",
        "seed",
        "arm",
        "episode",
        "support_state",
        "k_shot",
    ]
    if selected.duplicated(key_columns).any():
        raise ValueError("metric table contains duplicate paired cells")
    state_k_pairs = tuple(
        (str(state), int(k_shot))
        for state, k_values in state_k.items()
        for k_shot in k_values
    )
    expected = pd.MultiIndex.from_tuples(
        [
            (
                int(target),
                int(seed),
                str(arm),
                int(episode),
                state,
                k_shot,
            )
            for target in targets
            for seed in seeds
            for arm in arms
            for episode in episodes
            for state, k_shot in state_k_pairs
        ],
        names=key_columns,
    )
    observed = pd.MultiIndex.from_frame(selected[key_columns])
    missing_cells = expected.difference(observed)
    extra_cells = observed.difference(expected)
    if len(missing_cells) or len(extra_cells):
        raise ValueError(
            "metric grid is not exactly complete: "
            f"missing={len(missing_cells)}, extra={len(extra_cells)}"
        )

    core_consistency = selected.groupby(
        ["target_system", "seed", "episode"], sort=False
    )["core_id"].nunique()
    if not bool((core_consistency == 1).all()):
        raise ValueError("core_id is not shared across arms and K/state cells")

    state_mean = selected.groupby(
        ["target_system", "seed", "arm", "episode", "support_state"],
        sort=False,
        observed=True,
    )[metric].mean()
    core_mean = state_mean.groupby(
        ["target_system", "seed", "arm", "episode"],
        sort=False,
        observed=True,
    ).mean()
    index = pd.MultiIndex.from_product(
        [targets, seeds, arms, episodes],
        names=["target_system", "seed", "arm", "episode"],
    )
    core_mean = core_mean.reindex(index)
    if core_mean.isna().any():
        raise RuntimeError("complete grid did not produce a complete core aggregate")
    values = (
        core_mean.reorder_levels(["arm", "target_system", "seed", "episode"])
        .sort_index()
        .to_numpy(dtype=np.float64)
        .reshape(len(arms), len(targets), len(seeds), len(episodes))
    )
    # sort_index follows lexical labels, so explicitly reindex into caller order.
    ordered = np.empty_like(values)
    sorted_arms = sorted(str(value) for value in arms)
    sorted_targets = sorted(int(value) for value in targets)
    sorted_seeds = sorted(int(value) for value in seeds)
    sorted_episodes = sorted(int(value) for value in episodes)
    for arm_position, arm in enumerate(arms):
        for target_position, target in enumerate(targets):
            for seed_position, seed in enumerate(seeds):
                for episode_position, episode in enumerate(episodes):
                    ordered[arm_position, target_position, seed_position, episode_position] = values[
                        sorted_arms.index(str(arm)),
                        sorted_targets.index(int(target)),
                        sorted_seeds.index(int(seed)),
                        sorted_episodes.index(int(episode)),
                    ]
    return CoreMetricCube(
        values=ordered,
        arms=tuple(str(value) for value in arms),
        targets=tuple(int(value) for value in targets),
        seeds=tuple(int(value) for value in seeds),
        episodes=tuple(int(value) for value in episodes),
    )


def generate_bootstrap_plan(
    *,
    draws: int,
    targets: int,
    seeds: int,
    episodes: int,
    analysis_seed: int,
) -> BootstrapPlan:
    if min(draws, targets, seeds, episodes) <= 0:
        raise ValueError("bootstrap dimensions must be positive")
    rng = np.random.default_rng(int(analysis_seed))
    dtype = np.min_scalar_type(max(targets, seeds, episodes) - 1)
    target_indices = rng.integers(
        targets, size=(draws, targets), dtype=dtype
    )
    seed_indices = rng.integers(
        seeds, size=(draws, targets, seeds), dtype=dtype
    )
    episode_indices = rng.integers(
        episodes, size=(draws, targets, seeds, episodes), dtype=dtype
    )
    return BootstrapPlan(
        target_indices=target_indices,
        seed_indices=seed_indices,
        episode_indices=episode_indices,
        seed=int(analysis_seed),
    )


def target_arm_means(cube: CoreMetricCube) -> np.ndarray:
    return cube.values.mean(axis=(2, 3))


def bootstrap_arm_means(cube: CoreMetricCube, plan: BootstrapPlan) -> np.ndarray:
    """Return [draw, arm, sampled-target-position] paired arm means."""

    arms, targets, seeds, episodes = cube.values.shape
    if plan.target_indices.shape[1] != targets:
        raise ValueError("bootstrap target dimension differs from metric cube")
    if plan.seed_indices.shape[2] != seeds:
        raise ValueError("bootstrap seed dimension differs from metric cube")
    if plan.episode_indices.shape[3] != episodes:
        raise ValueError("bootstrap episode dimension differs from metric cube")
    output = np.empty((plan.draws, arms, targets), dtype=np.float64)
    for draw in range(plan.draws):
        for target_position, target_index in enumerate(plan.target_indices[draw]):
            seed_means = np.empty((arms, seeds), dtype=np.float64)
            for seed_position, seed_index in enumerate(
                plan.seed_indices[draw, target_position]
            ):
                episode_index = plan.episode_indices[
                    draw, target_position, seed_position
                ]
                seed_means[:, seed_position] = cube.values[
                    :, int(target_index), int(seed_index), episode_index
                ].mean(axis=1)
            output[draw, :, target_position] = seed_means.mean(axis=1)
    return output


def bootstrap_contrasts(
    cube: CoreMetricCube,
    plan: BootstrapPlan,
    *,
    numerator_arm: str,
    fixed_comparator_arms: Sequence[str],
    composite_comparator_by_target: Mapping[int, str],
) -> dict[str, np.ndarray]:
    arm_index = {arm: index for index, arm in enumerate(cube.arms)}
    unknown = set(fixed_comparator_arms) - set(arm_index)
    unknown.update(set(composite_comparator_by_target.values()) - set(arm_index))
    if numerator_arm not in arm_index or unknown:
        raise ValueError(f"contrast references unknown arms: {sorted(unknown)}")
    if set(composite_comparator_by_target) != set(cube.targets):
        raise ValueError("composite comparator must be defined for every target")
    arm_means = bootstrap_arm_means(cube, plan)
    numerator = arm_means[:, arm_index[numerator_arm], :]
    result = {
        f"{numerator_arm}_minus_{arm}": (
            numerator - arm_means[:, arm_index[arm], :]
        ).mean(axis=1)
        for arm in fixed_comparator_arms
    }
    composite = np.empty_like(numerator)
    for draw in range(plan.draws):
        for target_position, target_index in enumerate(plan.target_indices[draw]):
            target_id = cube.targets[int(target_index)]
            comparator = composite_comparator_by_target[target_id]
            composite[draw, target_position] = arm_means[
                draw, arm_index[comparator], target_position
            ]
    result[f"{numerator_arm}_minus_selected_composite"] = (
        numerator - composite
    ).mean(axis=1)
    return result


def percentile_interval(
    samples: Sequence[float] | np.ndarray,
    *,
    alpha: float,
    sides: str = "two",
) -> tuple[float | None, float | None]:
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("bootstrap samples must be finite and non-empty")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    if sides == "two":
        lower, upper = np.quantile(values, [alpha / 2.0, 1.0 - alpha / 2.0])
        return float(lower), float(upper)
    if sides == "lower":
        return float(np.quantile(values, alpha)), None
    if sides == "upper":
        return None, float(np.quantile(values, 1.0 - alpha))
    raise ValueError("sides must be 'two', 'lower', or 'upper'")


def bonferroni_percentile_interval(
    samples: Sequence[float] | np.ndarray,
    *,
    family_size: int = 11,
    familywise_alpha: float = 0.05,
) -> tuple[float, float]:
    if family_size <= 0:
        raise ValueError("family_size must be positive")
    lower, upper = percentile_interval(
        samples, alpha=familywise_alpha / family_size, sides="two"
    )
    assert lower is not None and upper is not None
    return lower, upper


def exact_one_sided_sign_flip_pvalue(
    target_differences: Sequence[float] | np.ndarray,
) -> float:
    differences = np.asarray(target_differences, dtype=np.float64)
    if differences.ndim != 1 or differences.size == 0:
        raise ValueError("target differences must be a non-empty rank-1 array")
    if not np.isfinite(differences).all():
        raise ValueError("target differences contain non-finite values")
    observed = float(differences.mean())
    exceedances = 0
    for mask in range(1 << differences.size):
        signs = np.asarray(
            [1.0 if mask & (1 << index) else -1.0 for index in range(differences.size)]
        )
        exceedances += int(float((signs * differences).mean()) >= observed - 1.0e-15)
    return exceedances / float(1 << differences.size)


def benjamini_hochberg(
    p_values: Sequence[float] | np.ndarray,
) -> np.ndarray:
    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("p_values must be a non-empty rank-1 array")
    if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("p_values must be finite and lie in [0, 1]")
    order = np.argsort(values, kind="stable")
    ranked = values[order]
    adjusted_ranked = np.minimum.accumulate(
        (ranked * values.size / np.arange(1, values.size + 1))[::-1]
    )[::-1].clip(max=1.0)
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = adjusted_ranked
    return adjusted


def decision_gates(
    *,
    primary_simultaneous_lower: float,
    base_one_sided_lower: float,
    aurc_one_sided_upper: float,
    minimum_state_delta: float,
    base_margin: float = -0.02,
    aurc_margin: float = 0.01,
    state_margin: float = -0.02,
) -> dict[str, bool]:
    gates = {
        "primary_superiority": primary_simultaneous_lower > 0.0,
        "base_noninferiority": base_one_sided_lower > base_margin,
        "aurc_noninferiority": aurc_one_sided_upper <= aurc_margin,
        "state_consistency": minimum_state_delta >= state_margin,
    }
    gates["all_confirmatory_gates"] = all(gates.values())
    return gates


__all__ = [
    "BootstrapPlan",
    "CoreMetricCube",
    "DEFAULT_STATE_K",
    "benjamini_hochberg",
    "bonferroni_percentile_interval",
    "bootstrap_arm_means",
    "bootstrap_contrasts",
    "decision_gates",
    "exact_one_sided_sign_flip_pvalue",
    "generate_bootstrap_plan",
    "percentile_interval",
    "prepare_core_metric_cube",
    "target_arm_means",
]
