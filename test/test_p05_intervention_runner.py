from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import torch

from src.explain_factory.p05_intervention_eval import evaluate_rule_interventions
from src.explain_factory.p05_intervention_runner import (
    P05InterventionProvenance,
    run_p05_pilot_interventions_from_loader,
    run_p05_same_checkpoint_interventions,
)
from src.explain_factory.p05_trace_export import export_p05_trace_package
from src.explain_factory.p05_trace_runner import model_state_sha256
from src.model_factory.X_model.UXFD.fuzzy.fuzzy_reasoner import (
    FuzzyConfig,
    FuzzyReasoner,
    FuzzyTrace,
)


CONFIG_HASH = "a" * 64
CHECKPOINT_HASH = "b" * 64


@dataclass(frozen=True)
class _Output:
    logits: torch.Tensor
    non_fuzzy_logits: torch.Tensor
    fuzzy_scale: float
    fuzzy_trace: FuzzyTrace


class _ActualTraceNetwork(torch.nn.Module):
    def __init__(
        self,
        *,
        drift_on_shuffle: bool = False,
        mutate_state: bool = False,
    ) -> None:
        super().__init__()
        rule = torch.arange(10, dtype=torch.float32)
        self.consequents = torch.nn.Parameter(
            torch.stack(
                (
                    0.7 * torch.cos(rule * 0.37) + 0.03 * rule,
                    0.6 * torch.sin(rule * 0.41) - 0.02 * rule,
                ),
                dim=1,
            )
        )
        self.register_buffer(
            "centers",
            torch.tensor([[-1.0, 1.0], [-0.5, 0.5]], dtype=torch.float32),
        )
        self.register_buffer(
            "widths",
            torch.tensor([[0.7, 0.8], [0.9, 1.0]], dtype=torch.float32),
        )
        self.register_buffer(
            "antecedent_probabilities",
            torch.softmax(
                torch.arange(40, dtype=torch.float32).reshape(10, 2, 2) * 0.03,
                dim=-1,
            ),
        )
        self.register_buffer("unused_state", torch.zeros(1, dtype=torch.float32))
        self.drift_on_shuffle = drift_on_shuffle
        self.mutate_state = mutate_state
        self.calls: list[dict[str, object]] = []

    def forward_with_fuzzy_trace(
        self,
        x: torch.Tensor,
        *,
        rule_mask: torch.Tensor | None = None,
        consequent_permutation: torch.Tensor | None = None,
    ) -> _Output:
        self.calls.append(
            {
                "batch_size": int(x.shape[0]),
                "rule_mask": (
                    None
                    if rule_mask is None
                    else tuple(bool(value) for value in rule_mask.detach().cpu().tolist())
                ),
                "consequent_permutation": (
                    None
                    if consequent_permutation is None
                    else tuple(
                        tuple(int(item) for item in value)
                        if isinstance(value, list)
                        else int(value)
                        for value in consequent_permutation.detach().cpu().tolist()
                    )
                ),
            }
        )
        if self.mutate_state:
            self.unused_state.add_(1.0)

        reduced = x.mean(dim=1)
        membership = torch.stack(
            (torch.sigmoid(reduced), torch.sigmoid(-reduced)), dim=-1
        )
        antecedent_memberships = torch.einsum(
            "bfm,rfm->brf",
            membership,
            self.antecedent_probabilities,
        )
        log_rule_firing = antecedent_memberships.clamp_min(1.0e-12).log().mean(dim=-1)
        rule_firing = log_rule_firing.exp()
        if rule_mask is None:
            normalized_mask = torch.ones(
                (x.shape[0], 10), dtype=torch.bool, device=x.device
            )
        else:
            normalized_mask = rule_mask.to(device=x.device).unsqueeze(0).expand(
                x.shape[0], -1
            )
        normalized_firing = torch.softmax(
            log_rule_firing.masked_fill(~normalized_mask, -torch.inf), dim=1
        )
        if consequent_permutation is None:
            permutation = torch.arange(10, device=x.device)
        else:
            permutation = consequent_permutation.to(device=x.device)
        if permutation.ndim == 1:
            consequents = self.consequents.index_select(0, permutation)
            contribution_consequents = consequents.unsqueeze(0)
        else:
            consequents = self.consequents[permutation]
            contribution_consequents = consequents
        contributions = normalized_firing.unsqueeze(-1) * contribution_consequents
        fuzzy_logits = contributions.sum(dim=1)
        non_fuzzy_logits = torch.stack(
            (reduced[:, 0] + 0.2 * reduced[:, 1], reduced[:, 1] - 0.1 * reduced[:, 0]),
            dim=1,
        )
        returned_membership = membership
        if self.drift_on_shuffle and consequent_permutation is not None:
            returned_membership = membership.clone()
            returned_membership[0, 0, 0] += 2.0e-4
        trace = FuzzyTrace(
            reduced_features=reduced,
            membership_values=returned_membership,
            centers=self.centers,
            widths=self.widths,
            antecedent_probabilities=self.antecedent_probabilities,
            antecedent_memberships=antecedent_memberships,
            log_rule_firing=log_rule_firing,
            rule_firing=rule_firing,
            normalized_rule_firing=normalized_firing,
            rule_consequents=consequents,
            rule_contributions=contributions,
            fuzzy_logits=fuzzy_logits,
            rule_mask=normalized_mask,
            consequent_permutation=permutation,
        )
        return _Output(
            logits=non_fuzzy_logits + 0.5 * fuzzy_logits,
            non_fuzzy_logits=non_fuzzy_logits,
            fuzzy_scale=0.5,
            fuzzy_trace=trace,
        )


def _batch() -> dict[str, object]:
    return {
        "x": torch.tensor(
            [
                [[0.4, 0.1], [0.2, 0.3], [0.8, -0.1], [0.6, 0.5]],
                [[-0.2, 0.7], [0.1, 0.6], [0.3, 0.2], [0.5, -0.4]],
                [[0.9, -0.3], [0.7, 0.4], [0.5, 0.2], [0.1, 0.8]],
                [[-0.4, 0.2], [0.6, 0.9], [0.2, -0.5], [0.7, 0.3]],
            ],
            dtype=torch.float32,
        ),
        "y": torch.tensor([0, 1, 0, 1]),
        "sample_id": ["sample-3", "sample-1", "sample-4", "sample-2"],
        "record_id": ["record-3", "record-1", "record-4", "record-2"],
        "group_id": ["bearing-b", "bearing-a", "bearing-b", "bearing-a"],
        "window_start": torch.tensor([30, 10, 40, 20]),
        "window_end": torch.tensor([34, 14, 44, 24]),
    }


def _provenance(network: torch.nn.Module) -> P05InterventionProvenance:
    return P05InterventionProvenance(
        dataset="XJTU",
        split="validation",
        model_seed=20260801,
        config_sha256=CONFIG_HASH,
        checkpoint_sha256=CHECKPOINT_HASH,
        model_sha256=model_state_sha256(network),
    )


def test_fuzzy_reasoner_applies_one_consequent_permutation_per_sample() -> None:
    reasoner = FuzzyReasoner(
        dim_in=2,
        num_classes=2,
        cfg=FuzzyConfig(
            num_fuzzy_features=2,
            num_membership_functions=2,
            num_rules=10,
        ),
    )
    features = torch.tensor([[0.2, -0.1], [0.4, 0.7]], dtype=torch.float32)
    permutations = torch.stack(
        (torch.arange(9, -1, -1), torch.roll(torch.arange(10), shifts=3))
    )

    trace = reasoner.forward_with_trace(
        features,
        consequent_permutation=permutations,
    )

    assert trace.consequent_permutation.shape == (2, 10)
    assert trace.rule_consequents.shape == (2, 10, 2)
    torch.testing.assert_close(
        trace.rule_consequents,
        reasoner.rule_consequents[permutations],
    )
    torch.testing.assert_close(
        trace.rule_contributions,
        trace.normalized_rule_firing.unsqueeze(-1) * trace.rule_consequents,
    )

    invalid = permutations.clone()
    invalid[1, 0] = invalid[1, 1]
    with pytest.raises(ValueError, match="each consequent_permutation row"):
        reasoner.forward_with_trace(features, consequent_permutation=invalid)


def test_actual_runner_executes_registered_forwards_and_exports_original_trace(
    tmp_path,
) -> None:
    network = _ActualTraceNetwork()
    network.train()
    state_before = model_state_sha256(network)
    provenance = _provenance(network)

    result = run_p05_same_checkpoint_interventions(
        network=network,
        batch=_batch(),
        provenance=provenance,
        expected_window_size=4,
        require_cuda=False,
        benchmark_first_n=3,
    )

    assert network.training is True
    assert model_state_sha256(network) == state_before
    assert result.arrays["sample_id"].tolist() == [
        "sample-1",
        "sample-2",
        "sample-3",
    ]
    assert len(network.calls) == 1 + 10 + 32
    assert network.calls[0] == {
        "batch_size": 3,
        "rule_mask": None,
        "consequent_permutation": None,
    }
    for rule, call in enumerate(network.calls[1:11]):
        assert call["batch_size"] == 3
        assert call["rule_mask"].count(False) == 1
        assert call["rule_mask"][rule] is False
        assert call["consequent_permutation"] is None
    assert all(call["batch_size"] == 3 for call in network.calls[11:])
    assert all(call["rule_mask"] is None for call in network.calls[11:])
    assert all(call["consequent_permutation"] is not None for call in network.calls[11:])
    assert all(
        len(call["consequent_permutation"]) == 3
        for call in network.calls[11:]
    )

    arrays = result.arrays
    assert arrays["actual_deletion_logits"].shape == (3, 10, 2)
    assert arrays["actual_deletion_normalized_rule_firing"].shape == (3, 10, 10)
    assert arrays["actual_deletion_rule_contributions"].shape == (3, 10, 10, 2)
    assert arrays["actual_shuffle_permutations"].shape == (3, 32, 10)
    assert arrays["actual_shuffle_logits"].shape == (3, 32, 2)
    assert arrays["actual_shuffle_rule_contributions"].shape == (3, 32, 10, 2)
    assert np.max(arrays["actual_deletion_invariant_max_abs"]) <= 1.0e-6
    assert np.max(arrays["actual_shuffle_invariant_max_abs"]) <= 1.0e-6
    assert result.metadata["model_state"]["unchanged"] is True
    assert result.metadata["protocol"]["actual_forward_calls"] == len(network.calls)
    assert result.metadata["selection"] == {
        "benchmark_first_n": 3,
        "input_count": 4,
        "kind": "first_n_after_stable_sample_id_sort",
        "selected_count": 3,
    }
    assert result.metadata["conclusion_control"]["claim_decision"] == "not_performed"
    assert result.timing["performance_claim_allowed"] is False
    assert result.timing["scope"] == "diagnostic_wall_clock_boundary_only"
    assert result.timing["total_seconds"] >= 0.0
    assert len(result.semantic_sha256) == 64
    with pytest.raises(ValueError):
        arrays["logits"][0, 0] = 999.0

    for sample_index in range(3):
        offline = evaluate_rule_interventions(
            **result.c2_evaluator_kwargs(sample_index)
        )
        np.testing.assert_allclose(
            offline["deletion_logits"],
            arrays["actual_deletion_logits"][sample_index],
            atol=1.0e-6,
            rtol=1.0e-6,
        )
        np.testing.assert_array_equal(
            offline["shuffle"]["permutations"],
            arrays["actual_shuffle_permutations"][sample_index],
        )
        assert offline["shuffle"]["seed"] == int(
            arrays["actual_shuffle_seed"][sample_index]
        )
        firing = arrays["trace_normalized_rule_firing"][sample_index]
        consequents = arrays["trace_rule_consequents"]
        expected_shuffle_logits = np.asarray(
            [
                arrays["non_fuzzy_logits"][sample_index]
                + float(arrays["fuzzy_scale"])
                * (firing @ consequents[permutation])
                for permutation in arrays["actual_shuffle_permutations"][sample_index]
            ]
        )
        np.testing.assert_allclose(
            arrays["actual_shuffle_logits"][sample_index],
            expected_shuffle_logits,
            atol=1.0e-6,
            rtol=1.0e-6,
        )

    exported = export_p05_trace_package(
        tmp_path / "trace-package",
        [result.as_trace_batch()],
        config_sha256=CONFIG_HASH,
        checkpoint_sha256=CHECKPOINT_HASH,
        model_sha256=state_before,
    )
    assert exported.status == "created"
    with np.load(exported.npz_path, allow_pickle=False) as trace_arrays:
        assert trace_arrays["sample_id"].tolist() == arrays["sample_id"].tolist()
        np.testing.assert_allclose(trace_arrays["logits"], arrays["logits"])


def test_actual_runner_fails_on_per_sample_intervention_invariant_drift() -> None:
    network = _ActualTraceNetwork(drift_on_shuffle=True)

    with pytest.raises(
        ValueError,
        match=r"shuffle_batch\[0\] membership_values invariant failed.*sample-1",
    ):
        run_p05_same_checkpoint_interventions(
            network=network,
            batch=_batch(),
            provenance=_provenance(network),
            expected_window_size=4,
            require_cuda=False,
            benchmark_first_n=1,
        )

    assert network.training is True


def test_actual_runner_fails_when_forward_mutates_checkpoint_state() -> None:
    network = _ActualTraceNetwork(mutate_state=True)

    with pytest.raises(RuntimeError, match="mutated the checkpoint state"):
        run_p05_same_checkpoint_interventions(
            network=network,
            batch=_batch(),
            provenance=_provenance(network),
            expected_window_size=4,
            require_cuda=False,
            benchmark_first_n=1,
        )

    assert network.training is True


def test_actual_runner_rejects_unbound_state_and_unstable_identifiers() -> None:
    network = _ActualTraceNetwork()
    wrong_provenance = P05InterventionProvenance(
        dataset="XJTU",
        split="test",
        model_seed=42,
        config_sha256=CONFIG_HASH,
        checkpoint_sha256=CHECKPOINT_HASH,
        model_sha256="f" * 64,
    )
    with pytest.raises(ValueError, match="does not match provenance"):
        run_p05_same_checkpoint_interventions(
            network=network,
            batch=_batch(),
            provenance=wrong_provenance,
            expected_window_size=4,
            require_cuda=False,
            benchmark_first_n=1,
        )
    assert network.calls == []

    duplicate = dict(_batch())
    duplicate["sample_id"] = ["same", "same", "third", "fourth"]
    with pytest.raises(ValueError, match="sample_id values must be unique"):
        run_p05_same_checkpoint_interventions(
            network=network,
            batch=duplicate,
            provenance=_provenance(network),
            expected_window_size=4,
            require_cuda=False,
        )
    assert network.calls == []


def _pilot_partition(count: int) -> dict[str, object]:
    record_ids = [f"record-{index:04d}" for index in range(count)]
    starts = torch.arange(count, dtype=torch.int64) * 4
    ends = starts + 4
    base = torch.arange(count * 8, dtype=torch.float32).reshape(count, 4, 2)
    return {
        "x": 0.1 + base.remainder(23.0) / 17.0,
        "y": torch.arange(count, dtype=torch.int64).remainder(2),
        "sample_id": [
            f"{record_ids[index]}:{int(starts[index])}:{int(ends[index])}"
            for index in range(count)
        ],
        "record_id": record_ids,
        "group_id": [f"bearing-{index % 5}" for index in range(count)],
        "window_start": starts,
        "window_end": ends,
    }


def _batch_slice(batch: dict[str, object], start: int, stop: int) -> dict[str, object]:
    return {name: value[start:stop] for name, value in batch.items()}


def test_pilot_loader_runner_proves_full_coverage_before_one_256_batch() -> None:
    partition = _pilot_partition(257)
    later = _batch_slice(partition, 128, 257)
    earlier = _batch_slice(partition, 0, 128)
    later["sample_weight"] = torch.ones(129, dtype=torch.float64)
    earlier["window_index"] = torch.arange(128, dtype=torch.int64)
    network = _ActualTraceNetwork()

    result = run_p05_pilot_interventions_from_loader(
        network=network,
        batches=[later, earlier],
        provenance=_provenance(network),
        expected_sample_ids=partition["sample_id"],
        expected_window_size=4,
        require_cuda=False,
    )

    assert len(network.calls) == 43
    assert {call["batch_size"] for call in network.calls} == {256}
    assert result.arrays["sample_id"].tolist() == sorted(
        partition["sample_id"]
    )[:256]
    assert result.metadata["selection"] == {
        "benchmark_first_n": 256,
        "input_count": 257,
        "kind": "first_n_after_stable_sample_id_sort",
        "selected_count": 256,
    }
    assert result.metadata["protocol"]["actual_forward_calls"] == 43


def test_pilot_loader_runner_rejects_partial_or_wrong_registered_partition_pre_forward() -> None:
    partition = _pilot_partition(257)
    network = _ActualTraceNetwork()
    with pytest.raises(ValueError, match="coverage differs"):
        run_p05_pilot_interventions_from_loader(
            network=network,
            batches=[_batch_slice(partition, 0, 128)],
            provenance=_provenance(network),
            expected_sample_ids=partition["sample_id"],
            expected_window_size=4,
            require_cuda=False,
        )
    assert network.calls == []

    with pytest.raises(ValueError, match="frozen validation partition"):
        run_p05_pilot_interventions_from_loader(
            network=network,
            batches=[],
            provenance=_provenance(network),
            expected_sample_ids=partition["sample_id"],
            expected_window_size=4,
            require_cuda=True,
        )
    assert network.calls == []
