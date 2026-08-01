from __future__ import annotations

from argparse import Namespace

import pytest
import torch

from src.model_factory.ISFM.backbone.B_04_Dlinear import B_04_Dlinear
from src.model_factory.ISFM.embedding.E_01_HSE import E_01_HSE
from scripts.p09.g060_representation import (
    HSEDLinearGlobalHead,
    model_state_sha256,
    sha256_file,
    strict_load_model,
)


def _args() -> Namespace:
    return Namespace(
        patch_size_L=128,
        patch_size_C=1,
        num_patches=32,
        output_dim=128,
    )


def test_manifest_keyed_hse_features_ignore_global_rng_state() -> None:
    torch.manual_seed(3)
    embedding = E_01_HSE(_args()).eval()
    backbone = B_04_Dlinear(_args()).eval()
    signal = torch.randn(2, 1024, 3)
    sample_rate = torch.tensor([12000.0, 48000.0])
    start_l = torch.arange(32).repeat(2, 1) * 20
    start_c = torch.tensor([[0, 1, 2, 0] * 8, [2, 1, 0, 2] * 8])

    first = backbone(
        embedding(
            signal,
            sample_rate,
            start_indices_L=start_l,
            start_indices_C=start_c,
        )
    ).mean(dim=1)
    _ = torch.randn(2000)
    second = backbone(
        embedding(
            signal,
            sample_rate,
            start_indices_L=start_l,
            start_indices_C=start_c,
        )
    ).mean(dim=1)
    assert first.shape == (2, 128)
    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)


def test_hse_rejects_partial_or_invalid_explicit_starts() -> None:
    embedding = E_01_HSE(_args()).eval()
    signal = torch.randn(1, 1024, 2)
    starts = torch.zeros(1, 32, dtype=torch.long)
    with pytest.raises(ValueError, match="must be supplied together"):
        embedding(signal, 12000.0, start_indices_L=starts)
    invalid = starts.clone()
    invalid[0, 0] = 1000
    with pytest.raises(ValueError, match="out-of-range"):
        embedding(
            signal,
            12000.0,
            start_indices_L=invalid,
            start_indices_C=starts,
        )


def test_strict_checkpoint_load_binds_file_state_and_contract(tmp_path) -> None:
    config = {
        "patch_size_L": 8,
        "patch_size_C": 1,
        "num_patches": 4,
        "output_dim": 16,
    }
    torch.manual_seed(3)
    model = HSEDLinearGlobalHead(config)
    path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "schema_version": 1,
            "status": "completed",
            "experiment_id": "P09-G060-FULL-V1",
            "seed": 42,
            "model_state_sha256": model_state_sha256(model),
            "model_state_dict": model.state_dict(),
        },
        path,
    )
    loaded, payload = strict_load_model(
        path,
        config,
        device=torch.device("cpu"),
        expected_sha256=sha256_file(path),
        expected_contract={"experiment_id": "P09-G060-FULL-V1", "seed": 42},
    )
    assert model_state_sha256(loaded) == model_state_sha256(model)
    assert payload["checkpoint_sha256"] == sha256_file(path)

    with pytest.raises(RuntimeError, match="contract mismatch"):
        strict_load_model(
            path,
            config,
            device=torch.device("cpu"),
            expected_contract={"seed": 123},
        )
    with pytest.raises(RuntimeError, match="file SHA-256"):
        strict_load_model(
            path,
            config,
            device=torch.device("cpu"),
            expected_sha256="0" * 64,
        )

    payload_value = torch.load(path, map_location="cpu", weights_only=False)
    payload_value["model_state_sha256"] = "0" * 64
    bad_path = tmp_path / "bad.pt"
    torch.save(payload_value, bad_path)
    with pytest.raises(RuntimeError, match="model-state SHA-256"):
        strict_load_model(bad_path, config, device=torch.device("cpu"))
