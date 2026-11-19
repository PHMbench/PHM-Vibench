from __future__ import annotations

from types import SimpleNamespace

import pytest
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from src.model_factory import build_model
from src.task_factory import build_task
from src.task_factory.task.generative.conditional_vae import gaussian_vae_loss


def _task(method: str, **task_options):
    model_args = SimpleNamespace(
        type="generative_model", name=f"phm_{method}_mlp1d", num_classes=2,
        in_channels=2, window_size=8, latent_dim=3, hidden_dim=8,
        condition_dim=4, num_fault_classes=2, num_domains=2,
    )
    options = dict(
        type="generative", name=f"conditional_{method}", optimizer="adam",
        lr=1e-3, weight_decay=0.0, kl_weight=1.0, observation_std=0.5,
    )
    options.update(task_options)
    return build_task(
        args_task=SimpleNamespace(**options),
        network=build_model(model_args, metadata=None),
        args_data=SimpleNamespace(window_size=8), args_model=model_args,
        args_trainer=SimpleNamespace(), args_environment=SimpleNamespace(),
        metadata=None,
    )


def _condition(count=2):
    return {
        "fault_label": torch.arange(count) % 2,
        "domain_id": torch.arange(count) % 2,
    }


@pytest.mark.parametrize("method", ["gan", "vae"])
def test_factory_task_sampling_survives_exact_checkpoint_round_trip(method, tmp_path):
    task = _task(method)
    checkpoint_path = tmp_path / f"{method}.pt"
    torch.save(task.state_dict(), checkpoint_path)
    restored = _task(method)
    restored.load_state_dict(torch.load(checkpoint_path, weights_only=True), strict=True)
    inputs = dict(num_samples=2, length=8, channels=2, num_steps=1, device="cpu")
    torch.manual_seed(19)
    expected = task.sample(_condition(), **inputs)
    torch.manual_seed(19)
    actual = restored.sample(_condition(), **inputs)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    assert actual.shape == (2, 2, 8)
    assert actual.dtype == torch.float32
    assert torch.isfinite(actual).all()
    assert not actual.requires_grad
    assert task.sampler_metadata()["nfe"] == 1


@pytest.mark.parametrize("method", ["gan", "vae"])
@pytest.mark.parametrize("steps", [2, 1.5, True])
def test_one_pass_tasks_reject_a_different_sampling_budget(method, steps):
    with pytest.raises((TypeError, ValueError), match="num_steps"):
        _task(method).sample(
            _condition(), num_samples=2, length=8, channels=2,
            num_steps=steps, device="cpu",
        )


def test_vae_loss_matches_gaussian_density_and_kl_in_window_units():
    torch.manual_seed(7)
    windows = torch.randn(2, 2, 8)
    reconstruction = torch.randn_like(windows)
    mean = torch.randn(2, 3)
    log_variance = torch.randn(2, 3)
    std, weight = 0.7, 0.3
    values = gaussian_vae_loss(
        windows, reconstruction, mean, log_variance,
        observation_std=std, kl_weight=weight,
    )
    likelihood = torch.distributions.Independent(
        torch.distributions.Normal(reconstruction, std), 2
    )
    posterior = torch.distributions.Independent(
        torch.distributions.Normal(mean, (0.5 * log_variance).exp()), 1
    )
    prior = torch.distributions.Independent(
        torch.distributions.Normal(torch.zeros_like(mean), torch.ones_like(mean)), 1
    )
    nll = -likelihood.log_prob(windows).mean()
    kl = torch.distributions.kl_divergence(posterior, prior).mean()
    torch.testing.assert_close(values["reconstruction_nll"], nll)
    torch.testing.assert_close(values["kl"], kl)
    torch.testing.assert_close(values["loss"], nll + weight * kl)


def test_vae_reparameterization_trains_encoder_and_decoder():
    torch.manual_seed(11)
    task = _task("vae")
    windows = torch.randn(2, 2, 8)
    reconstruction, mean, log_variance = task(windows, _condition())
    loss = gaussian_vae_loss(
        windows, reconstruction, mean, log_variance,
        observation_std=task.observation_std, kl_weight=task.kl_weight,
    )["loss"]
    loss.backward()
    for module in (task.network.encoder, task.network.decoder, task.network.condition_encoder):
        gradients = [parameter.grad for parameter in module.parameters()]
        assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
        assert sum(gradient.abs().sum().item() for gradient in gradients) > 0


def test_vae_sampling_includes_the_declared_observation_distribution():
    task = _task("vae", observation_std=0.4)
    with torch.no_grad():
        for parameter in task.network.parameters():
            parameter.zero_()
    torch.manual_seed(23)
    torch.randn(2, task.network.latent_dim)
    expected = 0.4 * torch.randn(2, 2, 8)
    torch.manual_seed(23)
    actual = task.sample(
        _condition(), num_samples=2, length=8, channels=2, num_steps=1,
    )
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    assert task.sampler_metadata()["sample_decoder_mean_only"] is False


def test_gan_manual_training_updates_each_disjoint_optimizer_once(tmp_path):
    torch.manual_seed(29)
    task = _task("gan")
    generator_optimizer, discriminator_optimizer = task.configure_optimizers()
    generator_ids = {id(p) for group in generator_optimizer.param_groups for p in group["params"]}
    discriminator_ids = {id(p) for group in discriminator_optimizer.param_groups for p in group["params"]}
    assert generator_ids.isdisjoint(discriminator_ids)
    assert generator_ids | discriminator_ids == {id(p) for p in task.network.parameters()}
    previous = {name: parameter.detach().clone() for name, parameter in task.network.named_parameters()}
    examples = [
        {"x": torch.randn(8, 2), "file_id": i, "y": i % 2, "domain_id": i % 2}
        for i in range(4)
    ]
    loader = DataLoader(examples, batch_size=2)
    trainer = pl.Trainer(
        accelerator="cpu", devices=1, max_epochs=1,
        limit_train_batches=1, limit_val_batches=1, num_sanity_val_steps=0,
        logger=False, enable_checkpointing=False, enable_progress_bar=False,
        enable_model_summary=False, default_root_dir=tmp_path,
    )
    trainer.fit(task, loader, loader)
    for optimizer in trainer.optimizers:
        assert optimizer.state
        assert {int(value["step"].item()) for value in optimizer.state.values()} == {1}
    for prefix in ("generator.", "discriminator."):
        assert any(
            not torch.equal(previous[name], parameter)
            for name, parameter in task.network.named_parameters()
            if name.startswith(prefix)
        )
    assert torch.isfinite(trainer.callback_metrics["val_loss"])
