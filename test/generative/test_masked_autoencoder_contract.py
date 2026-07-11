from argparse import Namespace

import torch

from src.model_factory.ISFM.component.MaskedAutoencoder import Model


def _build_small_mae() -> Model:
    args = Namespace(
        input_dim=2,
        patch_size=4,
        embed_dim=8,
        decoder_embed_dim=8,
        num_layers=1,
        decoder_num_layers=1,
        num_heads=2,
        decoder_num_heads=2,
        mask_ratio=0.5,
        dropout=0.0,
        max_seq_len=16,
        num_classes=3,
    )
    model = Model(args)
    model.eval()
    return model


def test_pretrain_reconstructs_one_output_per_input_patch() -> None:
    model = _build_small_mae()
    x = torch.randn(2, 16, 2)

    with torch.no_grad():
        output = model(x, mode="pretrain")

    assert output["pred"].shape == (2, 4, 8)
    assert output["mask"].shape == (2, 4)
    assert output["latent"].shape == (2, 3, 8)
    assert output["mask"].sum(dim=1).tolist() == [2.0, 2.0]
    assert torch.isfinite(output["pred"]).all()


def test_pretrain_masking_is_deterministic_for_fixed_seed() -> None:
    model = _build_small_mae()
    x = torch.randn(2, 16, 2)

    with torch.no_grad():
        torch.manual_seed(17)
        first = model(x, mode="pretrain")
        torch.manual_seed(17)
        second = model(x, mode="pretrain")

    assert torch.equal(first["mask"], second["mask"])
    assert torch.allclose(first["pred"], second["pred"])
