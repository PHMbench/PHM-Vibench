from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from src.model_factory.X_model.P01SharedPrivate import Model
from src.task_factory.Default_task import Default_task


def _ns(**kwargs):  # type: ignore[no-untyped-def]
    return SimpleNamespace(**kwargs)


def _model_args(
    *, pairing_mode: str = "paired", private_branch_enabled: bool = True,
    shared_only_head_hidden: int = 64,
) -> SimpleNamespace:
    return _ns(
        num_classes=3,
        in_channels=1,
        encoder_dim=32,
        latent_dim=8,
        dropout=0.0,
        time_frequency=_ns(n_fft=32, hop_length=8, center=True, normalized=False),
        pairing=_ns(mode=pairing_mode),
        objective=_ns(variance_floor=0.1),
        ablation=_ns(
            private_branch_enabled=private_branch_enabled,
            shared_only_head_hidden=shared_only_head_hidden,
        ),
    )


def _task(model: Model) -> Default_task:
    metadata = {
        0: {"Name": "Dummy", "Label": 0},
        1: {"Name": "Dummy", "Label": 2},
    }
    return Default_task(
        network=model,
        args_data=_ns(),
        args_model=_model_args(),
        args_task=_ns(
            loss="CE",
            metrics=[],
            optimizer="adam",
            lr=1.0e-3,
            weight_decay=0.0,
            auxiliary_loss_weights=_ns(
                alignment=0.1,
                private_independence=0.01,
                reconstruction=0.1,
                shared_variance=0.1,
            ),
        ),
        args_trainer=_ns(gpus=0, devices=0),
        args_environment=_ns(seed=0, project="test", output_dir="/tmp/p01-test"),
        metadata=metadata,
    )


def test_forward_exposes_finite_factorized_state_and_losses() -> None:
    torch.manual_seed(0)
    model = Model(_model_args())
    logits = model(torch.randn(4, 128, 1))

    assert logits.shape == (4, 3)
    assert torch.isfinite(logits).all()
    losses = model.get_auxiliary_losses()
    assert set(losses) == set(model.auxiliary_loss_names)
    assert all(value.ndim == 0 and torch.isfinite(value) for value in losses.values())
    state = model.get_representation_state()
    assert state["shared_1d"].shape == (4, 8)
    assert state["private_2d"].shape == (4, 8)


def test_explicit_2d_source_changes_only_the_derived_view_path() -> None:
    torch.manual_seed(3)
    model = Model(_model_args())
    x = torch.randn(4, 128, 1)
    _ = model(x)
    paired_state = model.get_representation_state()
    _ = model.forward_paired_views(x, torch.flip(x, dims=[0]))
    shuffled_state = model.get_representation_state()

    assert torch.allclose(paired_state["encoded_1d"], shuffled_state["encoded_1d"])
    assert not torch.allclose(paired_state["encoded_2d"], shuffled_state["encoded_2d"])


def test_default_task_adds_weighted_auxiliary_objective_and_backpropagates() -> None:
    torch.manual_seed(1)
    model = Model(_model_args())
    task = _task(model)
    batch = {
        "x": torch.randn(4, 128, 1),
        "y": torch.tensor([0, 1, 2, 1]),
        "file_id": torch.tensor([0, 0, 0, 0]),
    }

    metrics = task._shared_step(batch, "train")
    assert metrics["train_total_loss"] >= metrics["train_loss"]
    assert "train_aux_alignment_loss" in metrics
    assert "train_aux_total_loss" in metrics

    metrics["train_total_loss"].backward()
    assert model.encoder_1d.network[0].weight.grad is not None
    assert model.encoder_2d.network[0].weight.grad is not None
    assert model.shared_1d[0].weight.grad is not None
    assert model.private_2d[0].weight.grad is not None
    assert model.reconstructor_2d[0].weight.grad is not None


def test_single_sample_auxiliary_terms_are_well_defined() -> None:
    model = Model(_model_args())
    _ = model(torch.randn(1, 64, 1))
    losses = model.get_auxiliary_losses()
    assert losses["private_independence"].item() == pytest.approx(0.0)
    assert losses["shared_variance"].item() == pytest.approx(0.0)


def test_model_rejects_unregistered_runtime_pair_shuffling() -> None:
    with pytest.raises(ValueError, match="frozen dataset-level permutation manifest"):
        Model(_model_args(pairing_mode="batch_shuffle"))


def test_shared_only_ablation_removes_private_parameters_and_losses() -> None:
    model = Model(
        _model_args(private_branch_enabled=False, shared_only_head_hidden=127)
    )
    logits = model(torch.randn(4, 128, 1))
    assert logits.shape == (4, 3)
    assert model.private_1d is None
    assert model.private_2d is None
    assert model.reconstructor_1d is None
    assert model.reconstructor_2d is None
    assert set(model.get_auxiliary_losses()) == {"alignment", "shared_variance"}
    state = model.get_representation_state()
    assert "private_1d" not in state
    assert "reconstructed_2d" not in state
