from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from src.model_factory.X_model.P01Baselines import Model, VARIANTS


def _ns(**kwargs):  # type: ignore[no-untyped-def]
    return SimpleNamespace(**kwargs)


def _args(variant: str, **overrides) -> SimpleNamespace:  # type: ignore[no-untyped-def]
    values = {
        "num_classes": 4,
        "variant": variant,
        "in_channels": 2,
        "encoder_dim": 32,
        "head_hidden": 64,
        "projection_dim": 8,
        "contrastive_temperature": 0.1,
        "attention_heads": 4,
        "dropout": 0.0,
        "time_frequency": _ns(n_fft=32, hop_length=8, center=True, normalized=False),
    }
    values.update(overrides)
    return _ns(**values)


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_all_baseline_variants_return_finite_logits(variant: str) -> None:
    torch.manual_seed(0)
    model = Model(_args(variant))
    logits = model(torch.randn(4, 128, 2))
    assert logits.shape == (4, 4)
    assert torch.isfinite(logits).all()


def test_only_contrastive_variant_exposes_an_auxiliary_loss() -> None:
    contrastive = Model(_args("contrastive"))
    _ = contrastive(torch.randn(4, 128, 2))
    losses = contrastive.get_auxiliary_losses()
    assert set(losses) == {"contrastive_alignment"}
    assert torch.isfinite(losses["contrastive_alignment"])

    generic = Model(_args("generic_attention"))
    _ = generic(torch.randn(4, 128, 2))
    assert generic.get_auxiliary_losses() == {}


def test_contrastive_variant_uses_in_batch_negatives() -> None:
    torch.manual_seed(4)
    model = Model(_args("contrastive", dropout=0.0))
    x = torch.randn(4, 128, 2)
    _ = model.forward_paired_views(x, x)
    matched = model.get_auxiliary_losses()["contrastive_alignment"]
    _ = model.forward_paired_views(x, torch.flip(x, dims=[0]))
    mismatched = model.get_auxiliary_losses()["contrastive_alignment"]
    assert matched != mismatched


def test_explicit_pair_source_is_supported_by_paired_baselines() -> None:
    model = Model(_args("concat"))
    x = torch.randn(4, 128, 2)
    logits = model.forward_paired_views(x, torch.flip(x, dims=[0]))
    assert logits.shape == (4, 4)


def test_registered_widths_match_full_method_within_five_percent() -> None:
    from src.model_factory.X_model.P01SharedPrivate import Model as FullModel

    time_frequency = _ns(n_fft=128, hop_length=32, center=True, normalized=False)
    settings = {
        "one_d": (112, 384),
        "two_d": (128, 32),
        "concat": (80, 192),
        "generic_attention": (72, 256),
        "contrastive": (56, 384),
    }
    for num_classes in (4, 2):
        full = FullModel(
            _ns(
                num_classes=num_classes,
                in_channels=2,
                encoder_dim=64,
                latent_dim=32,
                dropout=0.1,
                time_frequency=time_frequency,
                pairing=_ns(mode="paired"),
                objective=_ns(variance_floor=0.1),
            )
        )
        target = full.trainable_parameter_count
        for variant, (encoder_dim, head_hidden) in settings.items():
            baseline = Model(
                _args(
                    variant,
                    num_classes=num_classes,
                    encoder_dim=encoder_dim,
                    head_hidden=head_hidden,
                    projection_dim=32,
                    time_frequency=time_frequency,
                )
            )
            relative_gap = abs(baseline.trainable_parameter_count - target) / target
            assert relative_gap <= 0.05, (variant, num_classes, relative_gap)
