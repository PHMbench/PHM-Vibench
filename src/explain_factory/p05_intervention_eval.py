"""Exact offline P05 rule-deletion and consequent-shuffle evaluator."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.special import logsumexp
from scipy.stats import kendalltau, rankdata


def _finite(values: np.ndarray, *, name: str, ndim: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != ndim or any(size <= 0 for size in array.shape):
        raise ValueError(f"{name} must be a non-empty {ndim}-dimensional array")
    if not np.isfinite(array).all():
        raise FloatingPointError(f"{name} contains non-finite values")
    return array


def _softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    probabilities = np.exp(values - logsumexp(values))
    if not np.isfinite(probabilities).all() or not np.isclose(
        probabilities.sum(), 1.0, rtol=0.0, atol=1e-12
    ):
        raise FloatingPointError("softmax probability construction failed")
    return probabilities


def natural_log_jsd(left: np.ndarray, right: np.ndarray) -> float:
    """Natural-log Jensen-Shannon divergence between two probability vectors."""

    p = np.asarray(left, dtype=np.float64)
    q = np.asarray(right, dtype=np.float64)
    if p.ndim != 1 or q.shape != p.shape or len(p) < 2:
        raise ValueError("JSD inputs must be equal one-dimensional class vectors")
    if (
        not np.isfinite(p).all()
        or not np.isfinite(q).all()
        or np.any(p < 0.0)
        or np.any(q < 0.0)
        or not np.isclose(p.sum(), 1.0, rtol=0.0, atol=1e-12)
        or not np.isclose(q.sum(), 1.0, rtol=0.0, atol=1e-12)
    ):
        raise ValueError("JSD inputs must be finite probability vectors")
    mixture = 0.5 * (p + q)
    tiny = np.finfo(np.float64).tiny
    log_mixture = np.log(np.clip(mixture, tiny, 1.0))
    p_term = np.where(
        p > 0.0,
        p * (np.log(np.clip(p, tiny, 1.0)) - log_mixture),
        0.0,
    )
    q_term = np.where(
        q > 0.0,
        q * (np.log(np.clip(q, tiny, 1.0)) - log_mixture),
        0.0,
    )
    value = float(0.5 * (p_term.sum(dtype=np.float64) + q_term.sum(dtype=np.float64)))
    if value < -1e-14 or not math.isfinite(value):
        raise FloatingPointError("JSD result is invalid")
    return max(0.0, value)


def _normalized_firing(log_rule_firing: np.ndarray, keep_mask: np.ndarray | None = None) -> np.ndarray:
    log_values = np.asarray(log_rule_firing, dtype=np.float64)
    if log_values.ndim != 1 or len(log_values) < 2 or not np.isfinite(log_values).all():
        raise ValueError("log_rule_firing must be a finite rule vector")
    if keep_mask is None:
        keep = np.ones(len(log_values), dtype=bool)
    else:
        keep = np.asarray(keep_mask, dtype=bool)
        if keep.shape != log_values.shape or not keep.any():
            raise ValueError("rule deletion mask must retain at least one rule")
    masked = np.where(keep, log_values, -np.inf)
    firing = np.exp(masked - logsumexp(masked))
    if not np.isfinite(firing).all() or not np.isclose(firing.sum(), 1.0, atol=1e-12):
        raise FloatingPointError("normalized rule firing construction failed")
    return firing


def _scaled_average_ranks(values: np.ndarray) -> np.ndarray:
    ranks = rankdata(np.asarray(values, dtype=np.float64), method="average")
    if len(ranks) < 2:
        raise ValueError("rank matching requires at least two rules")
    return (ranks - 1.0) / (len(ranks) - 1.0)


def _tau_b(left: np.ndarray, right: np.ndarray) -> float:
    value = float(kendalltau(left, right, variant="b", nan_policy="propagate").statistic)
    return 0.0 if not math.isfinite(value) else value


def _matched_endpoint(
    attribution: np.ndarray,
    firing: np.ndarray,
    consequent: np.ndarray,
    deletion_jsd: np.ndarray,
) -> dict[str, object]:
    rules = len(attribution)
    if rules < 5:
        raise ValueError("three-rule matched control requires at least five rules")
    top_rule = int(np.argmax(attribution))
    firing_rank = _scaled_average_ranks(firing)
    consequent_rank = _scaled_average_ranks(consequent)
    distance = np.abs(firing_rank - firing_rank[top_rule]) + np.abs(
        consequent_rank - consequent_rank[top_rule]
    )
    candidates = [rule for rule in range(rules) if rule != top_rule]
    matched = sorted(candidates, key=lambda rule: (float(distance[rule]), rule))[:3]
    matched_distances = [float(distance[rule]) for rule in matched]
    matched_mean = float(np.mean(deletion_jsd[matched]))
    unmatched_mean = float(np.mean(deletion_jsd[candidates]))
    tau_a = _tau_b(attribution, deletion_jsd)
    tau_f = _tau_b(firing, deletion_jsd)
    tau_q = _tau_b(consequent, deletion_jsd)
    return {
        "top_rule": top_rule,
        "matched_rules": matched,
        "matched_distances": matched_distances,
        "matched_distance_median": float(np.median(matched_distances)),
        "matched_distance_max": float(np.max(matched_distances)),
        "top_deletion_jsd": float(deletion_jsd[top_rule]),
        "matched_pool_mean_jsd": matched_mean,
        "unmatched_non_top_mean_jsd": unmatched_mean,
        "d_top": float(deletion_jsd[top_rule] - matched_mean),
        "tau_attribution_jsd": tau_a,
        "tau_firing_jsd": tau_f,
        "tau_consequent_jsd": tau_q,
        "d_rank": float(tau_a - max(0.0, tau_f, tau_q)),
    }


def shuffle_seed(
    *, dataset: str, split: str, model_seed: int, sample_id: str
) -> int:
    value = f"P05-E2-shuffle|{dataset}|{split}|{int(model_seed)}|{sample_id}"
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big")


def generate_unique_nonidentity_permutations(
    num_rules: int,
    *,
    seed: int,
    count: int = 32,
) -> tuple[tuple[int, ...], ...]:
    if isinstance(num_rules, bool) or not isinstance(num_rules, int) or num_rules < 2:
        raise ValueError("num_rules must be an integer >= 2")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("permutation count must be a positive integer")
    available = math.factorial(num_rules) - 1
    if count > available:
        raise ValueError(
            f"cannot draw {count} unique nonidentity permutations from {available}"
        )
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    identity = tuple(range(num_rules))
    selected = set()
    while len(selected) < count:
        candidate = tuple(int(value) for value in rng.permutation(num_rules))
        if candidate != identity:
            selected.add(candidate)
    return tuple(sorted(selected))


def evaluate_rule_interventions(
    *,
    dataset: str,
    split: str,
    model_seed: int,
    sample_id: str,
    logits: np.ndarray,
    non_fuzzy_logits: np.ndarray,
    fuzzy_scale: float,
    log_rule_firing: np.ndarray,
    rule_consequents: np.ndarray,
    rule_contributions: np.ndarray,
) -> dict[str, object]:
    """Evaluate every deletion and the 32 registered shuffles for one window."""

    original_logits = _finite(logits, name="logits", ndim=1)
    residual_logits = _finite(non_fuzzy_logits, name="non_fuzzy_logits", ndim=1)
    consequents = _finite(rule_consequents, name="rule_consequents", ndim=2)
    contributions = _finite(rule_contributions, name="rule_contributions", ndim=2)
    log_firing = _finite(log_rule_firing, name="log_rule_firing", ndim=1)
    if consequents.shape != contributions.shape:
        raise ValueError("rule consequents and contributions must have identical shapes")
    rules, classes = consequents.shape
    if original_logits.shape != (classes,) or residual_logits.shape != (classes,):
        raise ValueError("logit class dimension does not match rule consequents")
    if log_firing.shape != (rules,):
        raise ValueError("log_rule_firing rule dimension mismatch")
    scale = float(fuzzy_scale)
    if not math.isfinite(scale):
        raise ValueError("fuzzy_scale must be finite")
    firing = _normalized_firing(log_firing)
    expected_contributions = firing[:, None] * consequents
    if not np.allclose(contributions, expected_contributions, rtol=1e-6, atol=1e-6):
        raise ValueError("exported rule contributions do not match firing*consequents")
    reconstructed = residual_logits + scale * contributions.sum(axis=0)
    if not np.allclose(original_logits, reconstructed, rtol=1e-6, atol=1e-6):
        raise ValueError("original logits fail the registered reconstruction gate")

    original_probability = _softmax(original_logits)
    predicted_class = int(np.argmax(original_logits))
    deletion_logits = np.empty((rules, classes), dtype=np.float64)
    deletion_jsd = np.empty(rules, dtype=np.float64)
    for rule in range(rules):
        keep = np.ones(rules, dtype=bool)
        keep[rule] = False
        deletion_firing = _normalized_firing(log_firing, keep)
        fuzzy_logits = deletion_firing @ consequents
        deletion_logits[rule] = residual_logits + scale * fuzzy_logits
        deletion_jsd[rule] = natural_log_jsd(
            original_probability,
            _softmax(deletion_logits[rule]),
        )

    attribution = np.abs(scale * contributions[:, predicted_class])
    consequent = np.abs(scale * consequents[:, predicted_class])
    primary = _matched_endpoint(attribution, firing, consequent, deletion_jsd)
    attribution_vector = np.linalg.norm(scale * contributions, axis=1)
    consequent_vector = np.linalg.norm(scale * consequents, axis=1)
    full_vector = _matched_endpoint(
        attribution_vector,
        firing,
        consequent_vector,
        deletion_jsd,
    )

    permutations = generate_unique_nonidentity_permutations(
        rules,
        seed=shuffle_seed(
            dataset=dataset,
            split=split,
            model_seed=model_seed,
            sample_id=sample_id,
        ),
        count=32,
    )
    original_fuzzy_vector = scale * contributions.sum(axis=0)
    shuffle_jsd = []
    shuffle_l1 = []
    for permutation in permutations:
        shuffled_fuzzy = firing @ consequents[np.asarray(permutation, dtype=np.int64)]
        shuffled_logits = residual_logits + scale * shuffled_fuzzy
        shuffle_jsd.append(
            natural_log_jsd(original_probability, _softmax(shuffled_logits))
        )
        shuffle_l1.append(
            float(np.abs(scale * shuffled_fuzzy - original_fuzzy_vector).sum())
        )

    return {
        "sample_id": str(sample_id),
        "predicted_class": predicted_class,
        "rule_count": rules,
        "deletion_logits": deletion_logits,
        "deletion_jsd": deletion_jsd,
        "primary_reference_class": primary,
        "full_vector_sensitivity": full_vector,
        "shuffle": {
            "seed": shuffle_seed(
                dataset=dataset,
                split=split,
                model_seed=model_seed,
                sample_id=sample_id,
            ),
            "permutations": permutations,
            "predictive_jsd": np.asarray(shuffle_jsd, dtype=np.float64),
            "predictive_jsd_mean": float(np.mean(shuffle_jsd)),
            "fuzzy_class_vector_l1_change": np.asarray(shuffle_l1, dtype=np.float64),
            "membership_invariant": True,
            "firing_invariant": True,
        },
    }
