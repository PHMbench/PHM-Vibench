from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch
from pytorch_lightning.callbacks import ModelCheckpoint

from src.explain_factory.p05_trace_runner import (
    export_p05_loader_trace,
    model_state_sha256,
    resolve_best_checkpoint_path,
    sha256_file,
)
from src.model_factory.X_model.TSPN_UXFD import FuzzyTraceOutput
from src.model_factory.X_model.UXFD.fuzzy.fuzzy_reasoner import FuzzyTrace


class _TraceNetwork(torch.nn.Module):
    def __init__(self, *, mutate: bool = False) -> None:
        super().__init__()
        self.consequents = torch.nn.Parameter(
            torch.tensor([[0.4, -0.2], [-0.1, 0.3]], dtype=torch.float32)
        )
        self.register_buffer(
            "centers",
            torch.tensor([[-1.0, 1.0], [-0.5, 0.5]], dtype=torch.float32),
        )
        self.mutate = mutate

    def forward_with_fuzzy_trace(self, x: torch.Tensor) -> FuzzyTraceOutput:
        if self.mutate:
            self.consequents.add_(0.01)
        reduced = x.mean(dim=1)
        membership = torch.sigmoid(
            torch.stack((reduced, -reduced), dim=-1)
        )
        firing = torch.softmax(reduced, dim=-1)
        contributions = firing.unsqueeze(-1) * self.consequents.unsqueeze(0)
        fuzzy_logits = contributions.sum(dim=1)
        non_fuzzy = torch.stack((reduced[:, 0], reduced[:, 1]), dim=-1)
        trace = FuzzyTrace(
            reduced_features=reduced,
            membership_values=membership,
            centers=self.centers,
            widths=torch.ones_like(self.centers),
            antecedent_probabilities=torch.full(
                (2, 2, 2), 0.5, dtype=torch.float32, device=x.device
            ),
            antecedent_memberships=torch.full(
                (x.shape[0], 2, 2), 0.5, dtype=torch.float32, device=x.device
            ),
            log_rule_firing=torch.log(firing),
            rule_firing=firing,
            normalized_rule_firing=firing,
            rule_consequents=self.consequents,
            rule_contributions=contributions,
            fuzzy_logits=fuzzy_logits,
            rule_mask=torch.ones(
                (x.shape[0], 2), dtype=torch.bool, device=x.device
            ),
            consequent_permutation=torch.arange(2, device=x.device),
        )
        scale = 0.5
        return FuzzyTraceOutput(
            logits=non_fuzzy + scale * fuzzy_logits,
            non_fuzzy_logits=non_fuzzy,
            fuzzy_scale=scale,
            fuzzy_trace=trace,
        )


def _batch(prefix: str, offset: int = 0) -> dict:
    return {
        "x": torch.tensor(
            [
                [[0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5]],
                [[0.5, 0.4], [0.4, 0.3], [0.3, 0.2], [0.2, 0.1]],
            ],
            dtype=torch.float32,
        ),
        "y": torch.tensor([0, 1]),
        "sample_id": [f"{prefix}-{offset}", f"{prefix}-{offset + 1}"],
        "record_id": [f"record-{prefix}", f"record-{prefix}"],
        "group_id": [f"group-{prefix}", f"group-{prefix}"],
        "window_start": torch.tensor([offset, offset + 4]),
        "window_end": torch.tensor([offset + 4, offset + 8]),
    }


def _export(tmp_path, network, loader):
    model_hash = model_state_sha256(network)
    return export_p05_loader_trace(
        tmp_path / "trace",
        network=network,
        dataloader=loader,
        config_sha256="a" * 64,
        checkpoint_sha256="b" * 64,
        model_sha256=model_hash,
        expected_window_size=4,
        require_cuda=False,
    )


def test_loader_bridge_exports_complete_trace_and_restores_mode(tmp_path) -> None:
    network = _TraceNetwork()
    network.train()
    state_hash = model_state_sha256(network)

    result = _export(tmp_path, network, [_batch("a"), _batch("b", 8)])

    assert result.status == "created"
    assert network.training is True
    assert model_state_sha256(network) == state_hash
    with np.load(result.npz_path, allow_pickle=False) as arrays:
        assert arrays["sample_id"].tolist() == ["a-0", "a-1", "b-8", "b-9"]
        assert arrays["logits"].shape == (4, 2)
        assert arrays["trace_rule_contributions"].shape == (4, 2, 2)


def test_loader_bridge_fails_before_export_if_inference_mutates_state(tmp_path) -> None:
    network = _TraceNetwork(mutate=True)

    with pytest.raises(RuntimeError, match="mutated"):
        _export(tmp_path, network, [_batch("mutation")])

    assert not (tmp_path / "trace").exists()


def test_loader_bridge_requires_provenance_float32_and_cuda(tmp_path) -> None:
    network = _TraceNetwork()
    missing = dict(_batch("missing"))
    missing.pop("group_id")
    with pytest.raises(KeyError, match="group_id"):
        _export(tmp_path / "missing", network, [missing])

    wrong_dtype = dict(_batch("dtype"))
    wrong_dtype["x"] = wrong_dtype["x"].double()
    with pytest.raises(TypeError, match="float32"):
        _export(tmp_path / "dtype", network, [wrong_dtype])

    with pytest.raises(RuntimeError, match="CUDA-resident"):
        export_p05_loader_trace(
            tmp_path / "cuda",
            network=network,
            dataloader=[_batch("cuda")],
            config_sha256="a" * 64,
            checkpoint_sha256="b" * 64,
            model_sha256=model_state_sha256(network),
            expected_window_size=4,
            require_cuda=True,
        )


def test_model_and_file_hashes_change_with_content(tmp_path) -> None:
    first = _TraceNetwork()
    second = _TraceNetwork()
    with torch.no_grad():
        second.consequents[0, 0] += 1.0
    assert model_state_sha256(first) != model_state_sha256(second)

    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"one")
    first_hash = sha256_file(artifact)
    artifact.write_bytes(b"two")
    assert sha256_file(artifact) != first_hash


def test_best_checkpoint_resolver_requires_one_real_selected_file(tmp_path) -> None:
    checkpoint = tmp_path / "best.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    callback = ModelCheckpoint(dirpath=str(tmp_path))
    callback.best_model_path = str(checkpoint)
    trainer = type("Trainer", (), {"callbacks": [callback]})()

    assert resolve_best_checkpoint_path(trainer) == checkpoint.resolve()

    with pytest.raises(RuntimeError, match="exactly one"):
        resolve_best_checkpoint_path(type("Trainer", (), {"callbacks": []})())
