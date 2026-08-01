from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.model_factory.MoE.M_04_RoleConstrainedMoE import Model, ROLE_NAMES
from src.model_factory.MoE.role_identification import (
    assignment_accuracy,
    build_mechanism_signature,
    canonical_role_templates,
    deletion_interaction_contrast,
    solve_role_assignment,
)
from src.model_factory.model_factory import model_factory, resolve_model_module
from src.task_factory.Default_task import Default_task


def _args(**overrides):  # type: ignore[no-untyped-def]
    values = {
        "type": "MoE",
        "name": "M_04_RoleConstrainedMoE",
        "input_dim": 2,
        "num_classes": 3,
        "feature_dim": 16,
        "expert_hidden_channels": 8,
        "router_hidden_dim": 12,
        "dropout": 0.0,
        "routing_temperature": 1.0,
        "low_cutoff": 0.12,
        "envelope_band": [0.20, 0.80],
        "filter_transition": 0.03,
        "role_prior_strength": 0.5,
        "role_prior_max": 0.75,
        "role_prior_permutation": [0, 1, 2, 3],
        "router_mode": "learned_prior",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_shape_diagnostics_and_backward() -> None:
    torch.manual_seed(0)
    model = Model(_args())
    x = torch.randn(4, 128, 2)

    logits, diagnostics = model(x, return_diagnostics=True)

    assert logits.shape == (4, 3)
    assert diagnostics["expert_features"].shape == (4, 4, 16)
    assert diagnostics["expert_logits"].shape == (4, 4, 3)
    assert diagnostics["role_cues"].shape == (4, 4)
    assert diagnostics["routing_weights"].shape == (4, 4)
    assert torch.isfinite(logits).all()
    torch.testing.assert_close(
        diagnostics["routing_weights"].sum(dim=-1), torch.ones(4)
    )

    logits.square().mean().backward()
    assert model.router[0].weight.grad is not None
    assert torch.isfinite(model.router[0].weight.grad).all()
    for expert in model.experts:
        assert expert.classifier.weight.grad is not None
        assert torch.isfinite(expert.classifier.weight.grad).all()


def test_prior_contribution_is_bounded_and_permutable() -> None:
    torch.manual_seed(1)
    model = Model(_args(role_prior_strength=0.74, role_prior_max=0.75)).eval()
    x = torch.randn(3, 128, 2)
    with torch.no_grad():
        _, diagnostics = model(x, return_diagnostics=True)

    strength = model.role_prior_strength
    assert 0.0 <= float(strength) <= 0.75
    contribution = strength * diagnostics["role_prior_logits"]
    assert float(contribution.abs().max()) <= 0.75 + 1e-6
    torch.testing.assert_close(
        diagnostics["combined_router_logits"],
        diagnostics["learned_router_logits"] + contribution,
    )

    permuted = Model(_args(role_prior_permutation=[1, 0, 2, 3])).eval()
    permuted.load_state_dict(model.state_dict(), strict=False)
    with torch.no_grad():
        _, permuted_diagnostics = permuted(x, return_diagnostics=True)
    torch.testing.assert_close(
        diagnostics["role_prior_logits"][:, 0],
        permuted_diagnostics["role_prior_logits"][:, 1],
    )


def test_fixed_role_operators_have_expected_signal_contracts() -> None:
    model = Model(_args()).eval()
    sample_index = torch.arange(128, dtype=torch.float32)
    low = torch.sin(2.0 * torch.pi * sample_index / 64.0)
    high = torch.sin(2.0 * torch.pi * sample_index / 4.0)
    x = torch.stack([low, high], dim=0).unsqueeze(1)

    low_passed = model._smooth_lowpass(x)

    assert low_passed[0].square().mean() > 10.0 * low_passed[1].square().mean()
    representations = model._role_representations(x)
    assert len(representations) == len(ROLE_NAMES)
    assert all(torch.isfinite(value).all() for value in representations)
    assert representations[1].shape[-1] == 65
    assert representations[2].shape[-1] == 128


def test_expert_deletion_zeroes_and_renormalizes_without_rerouting() -> None:
    torch.manual_seed(2)
    model = Model(_args()).eval()
    x = torch.randn(3, 128, 2)
    with torch.no_grad():
        baseline, base_diagnostics = model(x, return_diagnostics=True)
        deleted, diagnostics = model.delete_expert(x, "harmonic")

    assert baseline.shape == deleted.shape == (3, 3)
    assert torch.count_nonzero(diagnostics["effective_routing_weights"][:, 1]) == 0
    torch.testing.assert_close(
        diagnostics["effective_routing_weights"].sum(dim=-1), torch.ones(3)
    )
    torch.testing.assert_close(
        diagnostics["routing_weights"], base_diagnostics["routing_weights"]
    )
    expected = torch.sum(
        diagnostics["expert_logits"]
        * diagnostics["effective_routing_weights"].unsqueeze(-1),
        dim=1,
    )
    torch.testing.assert_close(deleted, expected)


def test_expert_deletion_without_renormalization_reuses_intact_weights() -> None:
    torch.manual_seed(22)
    model = Model(_args()).eval()
    x = torch.randn(3, 128, 2)
    with torch.no_grad():
        effects = model.deletion_effects(x, renormalize=False)
        _, diagnostics = model(x, return_diagnostics=True)

    expected = []
    for expert_index in range(4):
        weights = diagnostics["routing_weights"].clone()
        weights[:, expert_index] = 0.0
        expected.append(
            torch.sum(diagnostics["expert_logits"] * weights.unsqueeze(-1), dim=1)
        )
    torch.testing.assert_close(effects["deleted_logits"], torch.stack(expected, dim=1))


def test_deletion_effects_and_behavioral_signature_contract() -> None:
    torch.manual_seed(3)
    model = Model(_args()).eval()
    x = torch.randn(5, 128, 2)
    with torch.no_grad():
        effects = model.deletion_effects(x)
        signature = model.behavioral_signature(x)

    assert effects["baseline_logits"].shape == (5, 3)
    assert effects["deleted_logits"].shape == (5, 4, 3)
    assert effects["deletion_kl"].shape == (5, 4)
    assert torch.all(effects["deletion_kl"] >= -1e-6)
    assert signature.shape == (4, 4)
    assert torch.isfinite(signature).all()


def test_auxiliary_router_losses_are_scalar_finite_and_consumed_once() -> None:
    model = Model(_args(load_balance_weight=0.02, entropy_floor_weight=0.03))
    _ = model(torch.randn(4, 128, 2))

    losses = model.consume_auxiliary_losses()

    assert set(losses) == {"moe_load_balance", "moe_entropy_floor"}
    assert all(value.ndim == 0 and torch.isfinite(value) for value in losses.values())
    assert model.consume_auxiliary_losses() == {}


def test_default_task_respects_explicit_cpu_when_cuda_is_visible(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    class CudaSpy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.cuda_calls = 0

        def cuda(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            del args, kwargs
            self.cuda_calls += 1
            return self

        def forward(self, x, file_id=None, task_id=None):  # type: ignore[no-untyped-def]
            del file_id, task_id
            return x

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    network = CudaSpy()
    task = Default_task(
        network=network,
        args_data=SimpleNamespace(normalization="standardization"),
        args_model=SimpleNamespace(type="test", name="cuda_spy"),
        args_task=SimpleNamespace(
            loss="CE", metrics=["acc"], optimizer="adam", lr=1e-3
        ),
        args_trainer=SimpleNamespace(device="cpu", gpus=1),
        args_environment=SimpleNamespace(project="p04_cpu_device_test"),
        metadata={0: {"Name": "Dummy_Data", "Label": 1}},
    )

    assert task.network is network
    assert network.cuda_calls == 0


def test_eval_is_deterministic_and_does_not_mutate_state_dict() -> None:
    torch.manual_seed(4)
    model = Model(_args(dropout=0.2)).eval()
    x = torch.randn(2, 128, 2)
    state_before = copy.deepcopy(model.state_dict())
    with torch.no_grad():
        first = model(x)
        second = model(x)
    state_after = model.state_dict()

    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
    assert state_before.keys() == state_after.keys()
    for key in state_before:
        torch.testing.assert_close(state_before[key], state_after[key])


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"num_classes": {"cwru": 4}}, "num_classes"),
        ({"feature_dim": 0}, "positive"),
        ({"dropout": 1.0}, "dropout"),
        ({"routing_temperature": 0.0}, "temperature"),
        ({"low_cutoff": 1.0}, "low_cutoff"),
        ({"envelope_band": [0.8, 0.2]}, "envelope_band"),
        ({"role_prior_strength": 1.1}, "role_prior_strength"),
        ({"load_balance_weight": -0.1}, "non-negative"),
        ({"entropy_floor": 1.1}, "entropy_floor"),
        ({"role_prior_permutation": [0, 0, 2, 3]}, "permutation"),
        ({"router_mode": "named_roles_are_evidence"}, "router_mode"),
    ],
)
def test_rejects_invalid_configuration(overrides, message) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError, match=message):
        Model(_args(**overrides))


@pytest.mark.parametrize("shape", [(2, 128), (2, 128, 1), (2, 16, 2)])
def test_rejects_invalid_input_contract(shape) -> None:  # type: ignore[no-untyped-def]
    model = Model(_args())
    with pytest.raises(ValueError, match="expects|mismatch|at least"):
        model(torch.randn(*shape))


def test_rejects_invalid_or_all_expert_deletion() -> None:
    model = Model(_args())
    x = torch.randn(2, 128, 2)
    with pytest.raises(ValueError, match="unknown|range"):
        model.delete_expert(x, "not_a_role")
    with pytest.raises(ValueError, match="every expert"):
        model(x, expert_mask=[0.0, 0.0, 0.0, 0.0])


def test_role_assignment_and_deletion_interaction_utilities() -> None:
    mechanisms = np.repeat(np.arange(4), 3)
    response = np.eye(4)[mechanisms] * 4.0 + 0.1
    routing = np.eye(4)[mechanisms] * 0.7 + 0.075
    signatures = build_mechanism_signature(response, routing, mechanisms)
    assignment = solve_role_assignment(signatures, canonical_role_templates())

    assert assignment.role_to_expert == (0, 1, 2, 3)
    assert assignment_accuracy(assignment) == 1.0

    baseline_loss = np.zeros(mechanisms.shape[0])
    deleted_losses = np.full((mechanisms.shape[0], 4), 0.5)
    deleted_losses[np.arange(mechanisms.shape[0]), mechanisms] = 2.0
    contrast = deletion_interaction_contrast(
        baseline_loss, deleted_losses, mechanisms, assignment.role_to_expert
    )
    assert contrast.overall == pytest.approx(1.5)
    assert contrast.by_role == pytest.approx((1.5, 1.5, 1.5, 1.5))
    assert contrast.observations_by_role == (3, 3, 3, 3)


def test_public_factory_builds_registered_model() -> None:
    args = _args()
    assert resolve_model_module(args) == "src.model_factory.MoE.M_04_RoleConstrainedMoE"
    model = model_factory(args, metadata={})
    assert isinstance(model, Model)
    assert model(torch.randn(2, 128, 2)).shape == (2, 3)
