import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.nn as nn

from src.model_factory.FoundationModel import MantisV1 as adapter


class _FakeMantisV1(nn.Module):
    load_calls = []

    def __init__(self, seq_len=512, device="cpu", pre_training=False):
        super().__init__()
        self.hidden_dim = 4
        self.scale = nn.Parameter(torch.ones(self.hidden_dim))
        self.init_args = (seq_len, device, pre_training)

    def from_pretrained(self, path, **kwargs):
        self.load_calls.append((path, kwargs))
        return self

    def forward(self, x):
        pooled = x.mean(dim=2)
        return pooled * self.scale.unsqueeze(0)


@pytest.fixture
def fake_mantis(monkeypatch):
    package = ModuleType("mantis")
    architecture = ModuleType("mantis.architecture")
    architecture.MantisV1 = _FakeMantisV1
    package.architecture = architecture
    monkeypatch.setitem(sys.modules, "mantis", package)
    monkeypatch.setitem(sys.modules, "mantis.architecture", architecture)
    _FakeMantisV1.load_calls.clear()
    return _FakeMantisV1


def _args(checkpoint, **overrides):
    values = {
        "checkpoint_path": str(checkpoint),
        "seq_len": 64,
        "input_channels": 2,
        "num_classes": 3,
        "freeze_backbone": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_adapter_loads_local_checkpoint_and_preserves_contract(tmp_path, fake_mantis):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text('{"model": "fake"}\n', encoding="utf-8")
    (checkpoint / "model.safetensors").write_bytes(b"weights")

    model = adapter.Model(_args(checkpoint))
    x = torch.randn(5, 64, 2)
    logits, features = model(x, task_id="classification", return_feature=True)

    assert logits.shape == (5, 3)
    assert features.shape == (5, 8)
    assert len(model.checkpoint_sha256) == 64
    assert model.provenance["checkpoint_sha256"] == model.checkpoint_sha256
    assert fake_mantis.load_calls == [
        (
            str(checkpoint.resolve()),
            {
                "local_files_only": True,
                "seq_len": 64,
                "device": "cpu",
                "pre_training": False,
            },
        )
    ]


def test_adapter_freezes_backbone_but_trains_head(tmp_path, fake_mantis):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "weights.bin").write_bytes(b"weights")
    model = adapter.Model(_args(checkpoint))
    model.train()

    assert not model.backbone.training
    assert all(not parameter.requires_grad for parameter in model.backbone.parameters())
    loss = model(torch.randn(4, 64, 2)).sum()
    loss.backward()
    assert all(parameter.grad is None for parameter in model.backbone.parameters())
    assert all(parameter.grad is not None for parameter in model.classifier.parameters())


@pytest.mark.parametrize("shape", [(2, 32, 2), (2, 64, 1), (2, 64)])
def test_adapter_rejects_implicit_resize_or_channel_changes(tmp_path, fake_mantis, shape):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "weights.bin").write_bytes(b"weights")
    model = adapter.Model(_args(checkpoint))
    with pytest.raises(ValueError):
        model(torch.randn(*shape))


def test_adapter_checks_local_path_before_optional_import(tmp_path, monkeypatch):
    def fail_import(name):
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(adapter.importlib, "import_module", fail_import)
    with pytest.raises(FileNotFoundError, match="only loads a local checkpoint"):
        adapter.Model(_args(tmp_path / "remote-looking-id"))


def test_adapter_reports_missing_optional_dependency(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "weights.bin").write_bytes(b"weights")

    def missing(name):
        raise ModuleNotFoundError("No module named 'mantis'", name="mantis")

    monkeypatch.setattr(adapter.importlib, "import_module", missing)
    with pytest.raises(RuntimeError, match="requirements-optional-mantis.txt"):
        adapter.Model(_args(checkpoint))
