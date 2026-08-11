import numpy as np
import pytest

from src.explain_factory.p05_intervention_eval import (
    evaluate_rule_interventions,
    generate_unique_nonidentity_permutations,
    natural_log_jsd,
    shuffle_seed,
)


def _fixture():
    log_firing = np.log(np.asarray([0.35, 0.25, 0.18, 0.12, 0.10]))
    firing = np.exp(log_firing)
    firing = firing / firing.sum()
    consequents = np.asarray(
        [
            [2.0, -1.0],
            [0.5, 0.7],
            [-0.2, 1.4],
            [0.1, -0.4],
            [0.8, 0.2],
        ]
    )
    contributions = firing[:, None] * consequents
    non_fuzzy = np.asarray([0.4, -0.1])
    scale = 0.5
    logits = non_fuzzy + scale * contributions.sum(axis=0)
    return logits, non_fuzzy, scale, log_firing, consequents, contributions


def test_natural_log_jsd_is_symmetric_zero_on_identity_and_nonnegative():
    p = np.asarray([0.8, 0.2])
    q = np.asarray([0.3, 0.7])
    assert natural_log_jsd(p, p) == pytest.approx(0.0)
    assert natural_log_jsd(p, q) == pytest.approx(natural_log_jsd(q, p))
    assert natural_log_jsd(p, q) > 0.0


def test_permutations_are_exactly_32_unique_deterministic_and_nonidentity():
    seed = shuffle_seed(dataset="XJTU", split="test", model_seed=42, sample_id="s")
    first = generate_unique_nonidentity_permutations(5, seed=seed)
    second = generate_unique_nonidentity_permutations(5, seed=seed)
    assert first == second
    assert len(first) == len(set(first)) == 32
    assert tuple(range(5)) not in first
    assert all(sorted(permutation) == list(range(5)) for permutation in first)


def test_exhaustive_deletion_matching_rank_and_shuffle_bundle():
    logits, non_fuzzy, scale, log_firing, consequents, contributions = _fixture()
    result = evaluate_rule_interventions(
        dataset="XJTU",
        split="test",
        model_seed=42,
        sample_id="bearing:0:4096",
        logits=logits,
        non_fuzzy_logits=non_fuzzy,
        fuzzy_scale=scale,
        log_rule_firing=log_firing,
        rule_consequents=consequents,
        rule_contributions=contributions,
    )

    assert result["deletion_logits"].shape == (5, 2)
    assert result["deletion_jsd"].shape == (5,)
    assert np.all(result["deletion_jsd"] >= 0.0)
    assert len(result["primary_reference_class"]["matched_rules"]) == 3
    assert len(result["primary_reference_class"]["matched_distances"]) == 3
    assert np.isfinite(result["primary_reference_class"]["d_rank"])
    assert len(result["full_vector_sensitivity"]["matched_rules"]) == 3
    assert len(result["shuffle"]["permutations"]) == 32
    assert result["shuffle"]["predictive_jsd"].shape == (32,)
    assert result["shuffle"]["membership_invariant"] is True
    assert result["shuffle"]["firing_invariant"] is True


def test_intervention_rejects_incomplete_contribution_trace():
    logits, non_fuzzy, scale, log_firing, consequents, contributions = _fixture()
    contributions = contributions.copy()
    contributions[0, 0] += 0.01
    with pytest.raises(ValueError, match=r"firing\*consequents"):
        evaluate_rule_interventions(
            dataset="XJTU",
            split="test",
            model_seed=42,
            sample_id="s",
            logits=logits,
            non_fuzzy_logits=non_fuzzy,
            fuzzy_scale=scale,
            log_rule_firing=log_firing,
            rule_consequents=consequents,
            rule_contributions=contributions,
        )


def test_permutation_generation_fails_when_32_nonidentity_draws_do_not_exist():
    with pytest.raises(ValueError, match="cannot draw 32"):
        generate_unique_nonidentity_permutations(4, seed=1)
