from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from src.model_factory.MoE.M_04_RoleConstrainedMoE import Model


def _args(arm: str = "P0", **overrides):  # type: ignore[no-untyped-def]
    values = {
        "input_dim": 1,
        "num_classes": 4,
        "feature_dim": 12,
        "expert_hidden_channels": 8,
        "router_hidden_dim": 10,
        "dropout": 0.0,
        "routing_temperature": 1.0,
        "scientific_arm": arm,
        "router_mode": "learned_only" if arm == "P1" else "learned_prior",
        "expert_representation_mode": (
            "homogeneous_raw" if arm == "P1" else "role_constrained"
        ),
        "semantic_alignment": [1, 2, 3, 0] if arm == "P2" else [0, 1, 2, 3],
        "compatibility_alpha": 1.0,
        "low_order_cutoff": 4.0,
        "envelope_order_band": [8.0, 120.0],
        "filter_transition_order": 0.25,
        "harmonic_order_max": 12,
        "harmonic_order_bandwidth": 0.18,
        "load_reference_hp": 3.0,
        "speed_reference_rpm": 1750.0,
        "speed_scale_rpm": 100.0,
        "load_balance_weight": 0.01,
        "physical_loss_weight": 0.0,
        "entropy_floor_weight": 0.0,
        "entropy_floor": 0.25,
        "role_prior_strength": 0.5,
        "role_prior_max": 1.0,
        "role_prior_permutation": [0, 1, 2, 3],
        "role_prior_assignment": "unspecified",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _physical(batch_size: int = 3) -> dict[str, torch.Tensor]:
    return {
        "sample_rate_hz": torch.full((batch_size,), 12_000.0),
        "rotation_speed_rpm": torch.full((batch_size,), 1_797.0),
        "load_hp": torch.zeros(batch_size),
    }


def _ready_model(arm: str = "P0") -> Model:
    model = Model(_args(arm))
    if arm in {"P0", "P2"}:
        model.set_compatibility_statistics(torch.zeros(4), torch.ones(4))
    return model


def test_order_axis_consumes_per_sample_sampling_rate_and_speed() -> None:
    model = _ready_model()
    reference = torch.zeros(2, 1, 256)
    metadata = {
        "sample_rate_hz": torch.tensor([12_000.0, 24_000.0]),
        "rotation_speed_rpm": torch.tensor([1_800.0, 3_600.0]),
        "load_hp": torch.tensor([0.0, 1.0]),
    }

    frequency_hz, order = model._frequency_axes(256, reference, metadata)

    torch.testing.assert_close(frequency_hz[1], 2.0 * frequency_hz[0])
    torch.testing.assert_close(order[0], order[1])
    assert not torch.equal(frequency_hz[0], frequency_hz[1])


def test_decisive_forward_requires_complete_noncontradictory_metadata() -> None:
    x = torch.randn(2, 256, 1)
    model = _ready_model()
    with pytest.raises(ValueError, match="missing required physical metadata"):
        model(x)
    with pytest.raises(ValueError, match="rotation_speed_rpm"):
        model(
            x,
            physical_metadata={
                "sample_rate_hz": [12_000.0, 12_000.0],
                "load_hp": [0.0, 0.0],
            },
        )

    table = {
        7: {
            "Sample_rate": 12_000,
            "Sample_Rate": 48_000,
            "rotation_speed_rpm": 1_797,
            "load_hp": 0,
        }
    }
    contradictory = Model(_args("P0"), table)
    contradictory.set_compatibility_statistics(torch.zeros(4), torch.ones(4))
    with pytest.raises(ValueError, match="contradictory sample_rate_hz"):
        contradictory(x, file_id=[7, 7])


def test_explicit_metadata_is_complete_authority_when_table_is_absent() -> None:
    model = _ready_model()
    x = torch.randn(2, 256, 1)
    logits, diagnostics = model(
        x,
        file_id=torch.tensor([97, 105]),
        physical_metadata=_physical(2),
        return_diagnostics=True,
    )
    assert logits.shape == (2, 4)
    torch.testing.assert_close(
        diagnostics["sample_rate_hz"], torch.full((2,), 12_000.0)
    )


def test_decisive_model_rejects_nonfinite_signal_and_metadata_without_repair() -> None:
    model = _ready_model()
    x = torch.randn(2, 256, 1)
    x[0, 3, 0] = float("nan")
    with pytest.raises(ValueError, match="must fail rather than be repaired"):
        model(x, physical_metadata=_physical(2))

    table = {
        8: {
            "sample_rate_hz": 12_000,
            "Sample_rate": float("nan"),
            "rotation_speed_rpm": 1_797,
            "load_hp": 0,
        }
    }
    metadata_model = Model(_args("P0"), table)
    metadata_model.set_compatibility_statistics(torch.zeros(4), torch.ones(4))
    with pytest.raises(ValueError, match="contains NaN"):
        metadata_model(
            torch.randn(1, 256, 1),
            file_id=[8],
        )


def test_compatibility_statistics_are_explicit_frozen_state() -> None:
    model = Model(_args("P0"))
    x = torch.randn(2, 256, 1)
    with pytest.raises(RuntimeError, match="fitted on train data"):
        model(x, physical_metadata=_physical(2))
    model.set_compatibility_statistics([0.1, 0.2, 0.3, 0.4], [1.0, 2.0, 3.0, 4.0])
    state = model.state_dict()
    assert bool(state["compatibility_stats_fitted"].item())
    torch.testing.assert_close(
        state["compatibility_mean"], torch.tensor([0.1, 0.2, 0.3, 0.4])
    )


def test_consistent_slot_permutation_preserves_logits_loss_and_checkpoint() -> None:
    torch.manual_seed(900)
    model = _ready_model().eval()
    x = torch.randn(3, 256, 1)
    y = torch.tensor([0, 1, 3])
    physical = _physical(3)
    with torch.no_grad():
        baseline, baseline_diagnostics = model(
            x, physical_metadata=physical, return_diagnostics=True
        )
        baseline_loss = F.cross_entropy(baseline, y)

    model.permute_slots_([2, 0, 3, 1])
    with torch.no_grad():
        permuted, diagnostics = model(
            x, physical_metadata=physical, return_diagnostics=True
        )
        permuted_loss = F.cross_entropy(permuted, y)

    torch.testing.assert_close(permuted, baseline, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(permuted_loss, baseline_loss, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        diagnostics["routing_weights"],
        baseline_diagnostics["routing_weights"][:, [2, 0, 3, 1]],
        rtol=1e-5,
        atol=1e-6,
    )

    restored = _ready_model().eval()
    restored.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
    with torch.no_grad():
        round_trip = restored(x, physical_metadata=physical)
    torch.testing.assert_close(round_trip, permuted, rtol=0.0, atol=0.0)
    assert restored.slot_to_structure.tolist() == [2, 0, 3, 1]


def test_inconsistent_expert_only_permutation_fails_before_forward() -> None:
    model = _ready_model()
    model.experts = torch.nn.ModuleList(
        [model.experts[index] for index in [1, 0, 2, 3]]
    )
    with pytest.raises(RuntimeError, match="inconsistent router/expert slot permutation"):
        model(torch.randn(2, 256, 1), physical_metadata=_physical(2))


def test_p2_changes_only_persistent_semantic_alignment() -> None:
    torch.manual_seed(901)
    p0 = _ready_model("P0")
    torch.manual_seed(901)
    p2 = _ready_model("P2")

    p0_parameters = dict(p0.named_parameters())
    p2_parameters = dict(p2.named_parameters())
    assert p0_parameters.keys() == p2_parameters.keys()
    for name in p0_parameters:
        torch.testing.assert_close(p0_parameters[name], p2_parameters[name])
    p0_state = p0.state_dict()
    p2_state = p2.state_dict()
    differing = {
        name
        for name in p0_state
        if not torch.equal(p0_state[name], p2_state[name])
    }
    assert differing == {"semantic_alignment"}


def test_p1_is_capacity_matched_and_does_not_consume_compatibility() -> None:
    torch.manual_seed(902)
    p0 = _ready_model("P0")
    torch.manual_seed(902)
    p1 = _ready_model("P1")
    assert [tuple(value.shape) for value in p0.parameters()] == [
        tuple(value.shape) for value in p1.parameters()
    ]
    x = torch.randn(2, 256, 1)
    _, diagnostics = p1(
        x, physical_metadata=_physical(2), return_diagnostics=True
    )
    torch.testing.assert_close(
        diagnostics["role_prior_logits"], torch.zeros(2, 4)
    )


def test_all_arms_execute_both_router_analysis_schedules(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    x = torch.randn(2, 256, 1)
    physical = _physical(2)
    p0 = _ready_model("P0")
    p1 = _ready_model("P1")
    calls = {"p0_generic": 0, "p1_physical_mask": 0}
    original_generic = p0._generic_router_inputs
    original_mask = p1._harmonic_order_mask

    def generic_spy(*args, **kwargs):  # type: ignore[no-untyped-def]
        calls["p0_generic"] += 1
        return original_generic(*args, **kwargs)

    def mask_spy(*args, **kwargs):  # type: ignore[no-untyped-def]
        calls["p1_physical_mask"] += 1
        return original_mask(*args, **kwargs)

    monkeypatch.setattr(p0, "_generic_router_inputs", generic_spy)
    monkeypatch.setattr(p1, "_harmonic_order_mask", mask_spy)
    p0(x, physical_metadata=physical)
    p1(x, physical_metadata=physical)
    assert calls["p0_generic"] > 0
    assert calls["p1_physical_mask"] > 0


def test_decisive_objective_has_one_balance_term_and_no_rescue_or_softmax_l1() -> None:
    model = _ready_model()
    x = torch.randn(3, 256, 1, requires_grad=True)
    logits = model(x, physical_metadata=_physical(3))
    auxiliary = model.consume_auxiliary_losses()
    assert set(auxiliary) == {"moe_load_balance"}
    assert "softmax_l1" not in auxiliary
    assert "moe_entropy_floor" not in auxiliary
    total = F.cross_entropy(logits, torch.tensor([0, 1, 2])) + sum(auxiliary.values())
    total.backward()
    assert model.consume_auxiliary_losses() == {}
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert float(x.grad.abs().sum()) > 0.0
    assert not any("physical" in name for name, _ in model.named_parameters())


def test_recovered_role_intervention_targets_mapping_not_structure_name() -> None:
    model = _ready_model().eval().permute_slots_([2, 0, 3, 1])
    x = torch.randn(2, 256, 1)
    mapping = {0: 2, 1: 0, 2: 3, 3: 1}
    with torch.no_grad():
        _, diagnostics = model.delete_recovered_role(
            x,
            0,
            mapping,
            physical_metadata=_physical(2),
        )
    assert torch.count_nonzero(
        diagnostics["effective_routing_weights"][:, 2]
    ) == 0
    with pytest.raises(ValueError, match="recovered role-to-slot"):
        model.delete_expert(x, "harmonic", physical_metadata=_physical(2))
    with pytest.raises(ValueError, match="complete bijection"):
        model.delete_recovered_role(
            x, 0, {0: 2}, physical_metadata=_physical(2)
        )
    with pytest.raises(ValueError, match="unknown recovered role"):
        model.delete_recovered_role(
            x, 9, mapping, physical_metadata=_physical(2)
        )


def test_probe_signature_is_exact_three_component_paired_response() -> None:
    model = _ready_model().eval()
    x = torch.randn(2, 256, 1)
    transformed = x.roll(shifts=5, dims=1) + 0.05 * torch.randn_like(x)
    physical = _physical(2)
    with torch.no_grad():
        _, baseline = model(
            x, physical_metadata=physical, return_diagnostics=True
        )
        _, changed = model(
            transformed, physical_metadata=physical, return_diagnostics=True
        )
        signature = model.probe_response_signature(
            x, transformed, physical_metadata=physical
        )

    expected = torch.stack(
        [
            changed["routing_weights"] - baseline["routing_weights"],
            changed["expert_features"].norm(dim=-1)
            - baseline["expert_features"].norm(dim=-1),
            1.0
            - F.cosine_similarity(
                baseline["expert_features"],
                changed["expert_features"],
                dim=-1,
                eps=1e-8,
            ),
        ],
        dim=-1,
    )
    assert signature.shape == (2, 4, 3)
    torch.testing.assert_close(signature, expected)


def test_checkpoint_missing_slot_semantics_is_rejected() -> None:
    model = _ready_model()
    incomplete = copy.deepcopy(model.state_dict())
    incomplete.pop("slot_to_structure")
    with pytest.raises(RuntimeError, match="Missing key"):
        _ready_model().load_state_dict(incomplete, strict=True)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"compatibility_alpha": 0.5}, "frozen at 1.0"),
        ({"entropy_floor_weight": 0.01}, "forbids entropy"),
        ({"physical_loss_weight": 0.1}, "fixed physical operators"),
        ({"semantic_alignment": [0, 2, 3, 1]}, "fixed-point-free"),
    ],
)
def test_decisive_configuration_rejects_semantic_drift(overrides, message) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError, match=message):
        Model(_args("P2", **overrides))
