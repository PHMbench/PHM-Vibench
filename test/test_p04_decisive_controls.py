from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from src.configs.config_utils import load_config
from src.model_factory.MoE.M_04_RoleConstrainedMoE import Model
from src.trainer_factory.Default_trainer import create_early_stopping_callback


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPO_ROOT / "configs" / "experiments" / "p04"
CONFIG_PATHS = {
    "FULL": CONFIG_ROOT / "decisive_full.yaml",
    "HOMO": CONFIG_ROOT / "decisive_homogeneous.yaml",
    "RAND": CONFIG_ROOT / "decisive_random_role.yaml",
}


def _config(arm: str) -> Any:
    return load_config(CONFIG_PATHS[arm])


def _parameter_signature(model: Model) -> tuple[tuple[str, tuple[int, ...]], ...]:
    return tuple((name, tuple(parameter.shape)) for name, parameter in model.named_parameters())


def _capture_expert_inputs(model: Model, x: torch.Tensor) -> list[torch.Tensor]:
    captured: list[torch.Tensor | None] = [None] * len(model.experts)
    handles = []
    for expert_index, expert in enumerate(model.experts):
        def capture(
            _module: torch.nn.Module,
            inputs: tuple[torch.Tensor, ...],
            index: int = expert_index,
        ) -> None:
            captured[index] = inputs[0].detach().clone()

        handles.append(expert.register_forward_pre_hook(capture))
    try:
        with torch.no_grad():
            model(x)
    finally:
        for handle in handles:
            handle.remove()
    assert all(value is not None for value in captured)
    return [value for value in captured if value is not None]


def test_full_homo_rand_have_exactly_identical_trainable_parameter_shapes() -> None:
    models = {arm: Model(_config(arm).model) for arm in CONFIG_PATHS}
    signatures = {arm: _parameter_signature(model) for arm, model in models.items()}

    assert signatures["FULL"] == signatures["HOMO"] == signatures["RAND"]
    assert sum(parameter.numel() for parameter in models["FULL"].parameters()) == sum(
        parameter.numel() for parameter in models["HOMO"].parameters()
    ) == sum(parameter.numel() for parameter in models["RAND"].parameters())


def test_homogeneous_heads_are_independent_and_receive_one_standardized_raw_view() -> None:
    torch.manual_seed(501)
    model = Model(_config("HOMO").model).eval()
    x = 3.0 * torch.randn(3, 512, 2) + 4.0

    head_signatures = [
        tuple((name, tuple(parameter.shape)) for name, parameter in expert.named_parameters())
        for expert in model.experts
    ]
    assert all(signature == head_signatures[0] for signature in head_signatures[1:])
    for parameter_name, _ in head_signatures[0]:
        storages = {
            dict(expert.named_parameters())[parameter_name].data_ptr()
            for expert in model.experts
        }
        assert len(storages) == 4

    captured = _capture_expert_inputs(model, x)
    raw = model._as_bcl(x)
    expected, _ = model._standardize_window(raw)
    for representation in captured:
        torch.testing.assert_close(representation, expected)
    torch.testing.assert_close(expected.mean(dim=-1), torch.zeros(3, 2), atol=1e-6, rtol=0)
    torch.testing.assert_close(
        expected.square().mean(dim=-1), torch.ones(3, 2), atol=1e-5, rtol=0
    )


def test_homogeneous_changes_expert_inputs_but_not_router_cues() -> None:
    torch.manual_seed(502)
    full = Model(_config("FULL").model).eval()
    homogeneous = Model(_config("HOMO").model).eval()
    homogeneous.load_state_dict(full.state_dict())
    x = torch.randn(4, 512, 2)

    with torch.no_grad():
        _, full_diagnostics = full(x, return_diagnostics=True)
        _, homogeneous_diagnostics = homogeneous(x, return_diagnostics=True)

    torch.testing.assert_close(
        full_diagnostics["role_cues"], homogeneous_diagnostics["role_cues"]
    )
    torch.testing.assert_close(
        full_diagnostics["router_features"],
        homogeneous_diagnostics["router_features"],
    )
    torch.testing.assert_close(
        full_diagnostics["routing_weights"],
        homogeneous_diagnostics["routing_weights"],
    )


def test_random_role_uses_full_representations_with_only_deranged_prior() -> None:
    torch.manual_seed(503)
    full = Model(_config("FULL").model).eval()
    random_role = Model(_config("RAND").model).eval()
    random_role.load_state_dict(full.state_dict())
    x = torch.randn(4, 512, 2)

    with torch.no_grad():
        _, full_diagnostics = full(x, return_diagnostics=True)
        _, random_diagnostics = random_role(x, return_diagnostics=True)

    torch.testing.assert_close(
        full_diagnostics["expert_features"], random_diagnostics["expert_features"]
    )
    torch.testing.assert_close(
        full_diagnostics["expert_logits"], random_diagnostics["expert_logits"]
    )
    torch.testing.assert_close(
        full_diagnostics["role_cues"], random_diagnostics["role_cues"]
    )
    expected_prior = (
        2.0
        * random_diagnostics["role_cues"].index_select(
            -1, random_role.role_prior_permutation
        )
        - 1.0
    ).clamp(-1.0, 1.0)
    torch.testing.assert_close(random_diagnostics["role_prior_logits"], expected_prior)


def test_response_only_diagnostic_matches_pre_routing_expert_feature_rms() -> None:
    torch.manual_seed(504)
    model = Model(_config("FULL").model).eval()
    x = torch.randn(5, 512, 2)

    with torch.no_grad():
        default_output = model(x)
        _, diagnostics = model(x, return_diagnostics=True)
        response = model.response_only_signature(x)

    expected = diagnostics["expert_features"].square().mean(dim=-1).add(1e-8).sqrt()
    assert isinstance(default_output, torch.Tensor)
    assert default_output.shape == (5, 4)
    assert response.shape == (5, 4)
    torch.testing.assert_close(diagnostics["response_only_signature"], expected)
    torch.testing.assert_close(response, expected)


def test_decisive_configs_freeze_common_training_contract() -> None:
    configs = {arm: _config(arm) for arm in CONFIG_PATHS}
    for arm, config in configs.items():
        assert config.protocol.arm == arm
        assert config.protocol.experiment_id == "E-MINDEC"
        assert config.environment.iterations == 1
        assert config.data.window_size == 512
        assert config.data.batch_size == 64
        assert config.data.normalization == "none"
        assert config.data.split.strategy == "grouped_metadata"
        assert config.data.split.manifest_mode == "read_only"
        assert config.data.split.manifest_path is None
        assert config.data.split.manifest_sha256 is None
        assert config.data.split.partition_map.__dict__ == {
            "train": "train",
            "val": "optimization_validation",
            "test": "intervention",
        }
        assert config.model.input_dim == 2
        assert config.model.num_classes == 4
        assert config.model.feature_dim == 64
        assert config.model.expert_hidden_channels == 32
        assert config.model.router_hidden_dim == 32
        assert config.model.dropout == pytest.approx(0.10)
        assert config.model.routing_temperature == pytest.approx(1.0)
        assert config.model.role_prior_strength == pytest.approx(0.50)
        assert config.model.role_prior_max == pytest.approx(1.0)
        assert config.model.load_balance_weight == pytest.approx(0.01)
        assert config.model.entropy_floor_weight == pytest.approx(0.01)
        assert config.model.entropy_floor == pytest.approx(0.25)
        assert config.task.optimizer == "adamw"
        assert config.task.lr == pytest.approx(0.001)
        assert config.task.weight_decay == pytest.approx(0.0001)
        assert config.trainer.num_epochs == 50
        assert config.trainer.early_stopping is True
        assert config.trainer.patience == 7
        assert config.trainer.min_delta == pytest.approx(0.0001)
        assert config.trainer.monitor == "val_loss"
        assert config.trainer.deterministic is True
        assert config.trainer.gpus == 1

    assert configs["FULL"].model.expert_representation_mode == "role_constrained"
    assert configs["HOMO"].model.expert_representation_mode == "homogeneous_raw"
    assert configs["RAND"].model.expert_representation_mode == "role_constrained"
    assert list(configs["FULL"].model.role_prior_permutation) == [0, 1, 2, 3]
    assert list(configs["HOMO"].model.role_prior_permutation) == [0, 1, 2, 3]
    assert list(configs["RAND"].model.role_prior_permutation) == [1, 2, 3, 0]


def test_random_role_seed_table_is_frozen_and_every_assignment_is_deranged() -> None:
    config = _config("RAND")
    entries = config.protocol.random_role_prior_permutations
    expected_seeds = [42, 123, 456, 789, 1024, 2027, 4096, 8192, 16384, 32768]

    assert [entry.seed for entry in entries] == expected_seeds
    for entry in entries:
        permutation = list(entry.permutation)
        assert sorted(permutation) == [0, 1, 2, 3]
        assert all(index != assigned for index, assigned in enumerate(permutation))
    assert list(entries[0].permutation) == list(config.model.role_prior_permutation)
    assert entries[0].seed == config.environment.seed


def test_external_random_role_mode_rejects_non_derangements() -> None:
    config = _config("RAND")
    values = dict(config.model.__dict__)
    values["role_prior_permutation"] = [0, 2, 3, 1]
    with pytest.raises(ValueError, match="fixed-point-free"):
        Model(SimpleNamespace(**values))


def test_frozen_early_stopping_min_delta_is_consumed() -> None:
    callback = create_early_stopping_callback(
        SimpleNamespace(monitor="val_loss", patience=7, min_delta=0.0001)
    )
    assert callback.monitor == "val_loss"
    assert callback.patience == 7
    # Lightning stores a signed threshold internally for mode="min".
    assert abs(callback.min_delta) == pytest.approx(0.0001)
