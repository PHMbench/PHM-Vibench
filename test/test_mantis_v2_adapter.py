import hashlib
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.nn as nn

from src.model_factory.FoundationModel import MantisV2 as adapter


class _FakeMantisV2(nn.Module):
    load_calls = []

    def __init__(
        self,
        num_patches=32,
        return_transf_layer=2,
        output_token="combined",
        device="cpu",
        pre_training=False,
    ):
        super().__init__()
        self.hidden_dim = 8 if output_token == "combined" else 4
        self.scale = nn.Parameter(torch.ones(self.hidden_dim))
        self.init_args = {
            "num_patches": num_patches,
            "return_transf_layer": return_transf_layer,
            "output_token": output_token,
            "device": device,
            "pre_training": pre_training,
        }

    def from_pretrained(self, path, **kwargs):
        self.load_calls.append((path, kwargs))
        return self

    def forward(self, x):
        pooled = x.mean(dim=(1, 2), keepdim=False).unsqueeze(1)
        return pooled * self.scale.unsqueeze(0)


@pytest.fixture
def fake_mantis(monkeypatch):
    package = ModuleType("mantis")
    architecture = ModuleType("mantis.architecture")
    architecture.MantisV2 = _FakeMantisV2
    package.architecture = architecture
    monkeypatch.setitem(sys.modules, "mantis", package)
    monkeypatch.setitem(sys.modules, "mantis.architecture", architecture)
    _FakeMantisV2.load_calls.clear()
    return _FakeMantisV2


def _checkpoint(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "weights.bin").write_bytes(b"v2-weights")
    return checkpoint


def _args(checkpoint, **overrides):
    values = {
        "checkpoint_path": str(checkpoint),
        "seq_len": 64,
        "num_patches": 32,
        "input_channels": 2,
        "num_classes": 3,
        "freeze_backbone": True,
        "return_transf_layer": 2,
        "output_token": "combined",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_adapter_uses_v2_contract_and_returns_features(tmp_path, fake_mantis):
    checkpoint = _checkpoint(tmp_path)
    model = adapter.Model(_args(checkpoint))
    logits, features = model(
        torch.randn(5, 64, 2),
        task_id="classification",
        return_feature=True,
    )

    assert logits.shape == (5, 3)
    assert features.shape == (5, 16)
    assert fake_mantis.load_calls == [
        (
            str(checkpoint.resolve()),
            {
                "local_files_only": True,
                "num_patches": 32,
                "return_transf_layer": 2,
                "output_token": "combined",
                "device": "cpu",
                "pre_training": False,
            },
        )
    ]


def test_adapter_checks_expected_checkpoint_digest(tmp_path, fake_mantis):
    checkpoint = _checkpoint(tmp_path)
    model = adapter.Model(_args(checkpoint))
    digest = model.checkpoint_sha256
    assert len(digest) == hashlib.sha256().digest_size * 2

    adapter.Model(_args(checkpoint, checkpoint_sha256=digest.upper()))
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        adapter.Model(_args(checkpoint, checkpoint_sha256="0" * 64))


def test_adapter_freezes_backbone_and_rejects_shape_changes(tmp_path, fake_mantis):
    checkpoint = _checkpoint(tmp_path)
    model = adapter.Model(_args(checkpoint))
    model.train()

    assert not model.backbone.training
    assert all(not parameter.requires_grad for parameter in model.backbone.parameters())
    loss = model(torch.randn(4, 64, 2)).sum()
    loss.backward()
    assert all(parameter.grad is None for parameter in model.backbone.parameters())
    assert all(parameter.grad is not None for parameter in model.classifier.parameters())

    with pytest.raises(ValueError):
        model(torch.randn(4, 32, 2))
    with pytest.raises(ValueError):
        model(torch.randn(4, 64, 1))


def test_adapter_rejects_invalid_v2_settings_before_import(tmp_path, monkeypatch):
    checkpoint = _checkpoint(tmp_path)

    def fail_import(name):
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(adapter.importlib, "import_module", fail_import)
    with pytest.raises(ValueError, match="divisible"):
        adapter.Model(_args(checkpoint, seq_len=65))
    with pytest.raises(ValueError, match="output_token"):
        adapter.Model(_args(checkpoint, output_token="invalid"))
