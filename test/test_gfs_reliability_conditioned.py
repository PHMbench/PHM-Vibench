from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from src.task_factory import TASK_REGISTRY
from src.task_factory.task.GFS.reliability_conditioned import (
    RELIABILITY_FEATURE_NAMES,
    SupportReliabilityConditioner,
    task as ReliabilityConditionedTask,
)


def _conditioner() -> SupportReliabilityConditioner:
    torch.manual_seed(7)
    return SupportReliabilityConditioner(
        feature_dim=2,
        adapter_rank=1,
        adapter_max_gate=0.25,
        adapter_relative_bound=0.10,
    )


def _base_weights() -> torch.Tensor:
    return torch.tensor([[1.0, 0.0], [0.0, 1.0]])


def _clean_support() -> tuple[torch.Tensor, torch.Tensor]:
    features = torch.tensor(
        [
            [0.80, 0.60],
            [0.82, 0.58],
            [0.78, 0.62],
            [-0.60, 0.80],
            [-0.58, 0.82],
            [-0.62, 0.78],
        ]
    )
    labels = torch.tensor([10, 10, 10, 11, 11, 11])
    return features, labels


def test_support_condition_has_declared_reliability_features() -> None:
    module = _conditioner()
    support, labels = _clean_support()

    state = module.condition(support, labels, _base_weights(), [10, 11])

    assert state.feature_names == RELIABILITY_FEATURE_NAMES
    assert state.reliability_features.shape == (2, 4)
    assert torch.all((state.reliability > 0.0) & (state.reliability <= 1.0))
    assert torch.all(state.temperature >= module.temperature_min)
    assert torch.all(state.temperature <= module.temperature_max)


def test_support_outlier_reduces_reliability_and_increases_temperature() -> None:
    module = _conditioner()
    clean, labels = _clean_support()
    corrupted = clean.clone()
    corrupted[2] = torch.tensor([-8.0, -8.0])

    clean_state = module.condition(clean, labels, _base_weights(), [10, 11])
    corrupt_state = module.condition(corrupted, labels, _base_weights(), [10, 11])

    assert corrupt_state.reliability[0] < clean_state.reliability[0]
    assert corrupt_state.temperature[0] > clean_state.temperature[0]
    assert corrupt_state.adapter_gate < clean_state.adapter_gate


def test_conditioning_is_query_blind_and_prediction_does_not_mutate_state() -> None:
    module = _conditioner()
    support, labels = _clean_support()
    state = module.condition(support, labels, _base_weights(), [10, 11])
    snapshot = {name: getattr(state, name).clone() for name in state.__dataclass_fields__}

    first_query = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
    second_query = torch.tensor([[-9.0, 4.0], [6.0, -3.0], [0.5, 0.5]])
    module.predict(first_query, _base_weights(), [0, 1], state)
    module.predict(second_query, _base_weights(), [0, 1], state)

    for name, before in snapshot.items():
        assert torch.equal(getattr(state, name), before)


def test_adapter_residual_obeys_declared_relative_norm_bound() -> None:
    module = _conditioner()
    with torch.no_grad():
        module.adapter_down.weight.fill_(1.0)
        module.adapter_up.weight.fill_(2.0)
    features = torch.tensor([[3.0, 4.0], [-2.0, 1.0]])
    gate = torch.tensor(module.adapter_max_gate)

    adapted = module.apply_adapter(features, gate)
    change_norm = torch.linalg.vector_norm(adapted - features, dim=1)
    allowed = (
        module.adapter_max_gate
        * module.adapter_relative_bound
        * torch.linalg.vector_norm(features, dim=1)
    )

    assert torch.all(change_norm <= allowed + 1.0e-6)


def test_base_logits_are_identical_to_frozen_no_adapter_path() -> None:
    module = _conditioner()
    support, labels = _clean_support()
    state = module.condition(support, labels, _base_weights(), [10, 11])
    query = torch.tensor([[0.4, 0.6], [0.9, -0.1]])
    base_bias = torch.tensor([0.2, -0.3])

    result = module.predict(query, _base_weights(), [0, 1], state, base_bias=base_bias)
    expected = query @ _base_weights().T + base_bias

    assert torch.allclose(result["base_logits"], expected, atol=0.0, rtol=0.0)


def test_joint_prediction_exposes_class_mapping_and_abstention() -> None:
    module = _conditioner()
    support, labels = _clean_support()
    state = module.condition(support, labels, _base_weights(), [10, 11])

    result = module.predict(
        torch.tensor([[0.8, 0.6], [-0.6, 0.8], [1.0, 0.0]]),
        _base_weights(),
        [0, 1],
        state,
    )

    assert result["joint_logits"].shape == (3, 4)
    assert result["joint_class_ids"].tolist() == [0, 1, 10, 11]
    assert result["accepted"].dtype == torch.bool
    assert result["prediction_label"].shape == (3,)


def test_missing_or_unexpected_support_class_is_a_hard_failure() -> None:
    module = _conditioner()
    support, labels = _clean_support()

    with pytest.raises(ValueError, match="exactly match"):
        module.condition(support[:3], labels[:3], _base_weights(), [10, 11])
    with pytest.raises(ValueError, match="exactly match"):
        module.condition(support, labels, _base_weights(), [10, 12])


def test_base_and_novel_ids_must_be_disjoint() -> None:
    module = _conditioner()
    support, labels = _clean_support()
    state = module.condition(support, labels, _base_weights(), [10, 11])

    with pytest.raises(ValueError, match="must be disjoint"):
        module.predict(torch.ones(1, 2), _base_weights(), [0, 10], state)


def test_trainable_parameter_count_is_only_the_low_rank_adapter() -> None:
    module = _conditioner()

    assert module.trainable_parameter_count == 4
    assert module.contract["query_blind_conditioning"] is True
    assert module.contract["base_logits_use_unadapted_features"] is True
    assert module.contract["base_logits_equal_frozen_linear_head"] is True


def test_config_constructor_caps_default_rank_for_small_feature_spaces() -> None:
    module = SupportReliabilityConditioner.from_config(feature_dim=3, config={})

    assert module.adapter_down.out_features == 3


class _DummyNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.task_head = nn.Module()
        self.task_head.mutiple_fc = nn.ModuleDict({"1": nn.Linear(2, 3, bias=True)})
        with torch.no_grad():
            self.task_head.mutiple_fc["1"].weight.copy_(
                torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
            )
            self.task_head.mutiple_fc["1"].bias.zero_()

    def forward(
        self,
        x: torch.Tensor,
        file_id: int,
        task_id: str,
        return_feature: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        del file_id, task_id
        logits = self.task_head.mutiple_fc["1"](x)
        return (logits, x) if return_feature else logits


def _ns(**kwargs: object) -> SimpleNamespace:
    return SimpleNamespace(**kwargs)


def test_lightning_binding_runs_distinct_joint_episode_path() -> None:
    network = _DummyNetwork()
    task_module = ReliabilityConditionedTask(
        network=network,
        args_data=_ns(),
        args_model=_ns(output_dim=2),
        args_task=_ns(
            type="GFS",
            name="reliability_conditioned",
            base_class_ids=[0, 1],
            novel_class_ids=[2],
            num_support=1,
            freeze_encoder_base=True,
            reliability_conditioner={"adapter_rank": 1},
            loss="CE",
            metrics=["acc"],
            optimizer="adam",
            lr=1.0e-3,
            weight_decay=0.0,
            regularization={},
        ),
        args_trainer=_ns(gpus=0),
        args_environment=_ns(seed=0),
        metadata={101: {"Dataset_id": 1, "Sample_rate": 12000, "Name": "dummy", "Label": 2}},
    )
    batch = {
        "x": torch.tensor(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
                [0.1, 0.9],
                [-1.0, 0.0],
                [-0.9, 0.1],
            ]
        ),
        "y": torch.tensor([0, 0, 1, 1, 2, 2]),
        "file_id": torch.tensor([101, 101, 101, 101, 101, 101]),
    }

    metrics = task_module._shared_step(batch, "train")

    assert torch.isfinite(metrics["train_total_loss"])
    assert set(metrics) >= {
        "train_base_acc",
        "train_novel_acc",
        "train_harmonic_mean",
        "train_support_reliability",
        "train_adapter_gate",
        "train_novel_temperature",
    }
    assert list(network.parameters())
    assert all(not parameter.requires_grad for parameter in network.parameters())
    assert any(parameter.requires_grad for parameter in task_module.conditioner.parameters())
    assert TASK_REGISTRY.get("GFS.reliability_conditioned") is ReliabilityConditionedTask
    metrics["train_total_loss"].backward()
    assert task_module.conditioner.adapter_up.weight.grad is not None
    assert torch.isfinite(task_module.conditioner.adapter_up.weight.grad).all()
