from types import SimpleNamespace

import pandas as pd
import pytest
import torch
import torch.nn as nn

from src.data_factory.data_utils import MetadataAccessor
from src.model_factory.ISFM.M_01_ISFM import Model as ISFMModel
from src.task_factory.Components.ppt_time_order import (
    PPTTimeOrderConfig,
    PPTTimeOrderLoss,
    PatchPermutationBank,
)
from src.task_factory.task.pretrain.ppt_time_order import PptTimeOrderTask


def test_permutation_bank_is_seeded_nonidentity_and_batch_deterministic():
    first = PatchPermutationBank(num_patches=12, swaps=3, bank_size=16, seed=9)
    second = PatchPermutationBank(num_patches=12, swaps=3, bank_size=16, seed=9)
    identity = torch.arange(12).expand(16, -1)
    sequence = torch.arange(2 * 12 * 3).reshape(2, 12, 3)

    assert torch.equal(first.indices, second.indices)
    assert torch.all(torch.any(first.indices != identity, dim=1))
    assert torch.equal(first.permute(sequence, offset=5), second.permute(sequence, offset=5))


def test_ppt_loss_has_finite_terms_and_backpropagates_to_sequence():
    objective = PPTTimeOrderLoss(
        PPTTimeOrderConfig(
            num_patches=8,
            embedding_dim=6,
            weak_swaps=1,
            strong_swaps=4,
            bank_size=32,
            seed=3,
        )
    )
    sequence = torch.randn(5, 8, 6, requires_grad=True)
    loss, stats = objective(sequence, offset=7)

    assert torch.isfinite(loss)
    assert set(stats) == {
        "loss",
        "consistency_loss",
        "contrastive_loss",
        "positive_similarity",
        "negative_similarity",
    }
    loss.backward()
    assert sequence.grad is not None
    assert torch.isfinite(sequence.grad).all()


def test_ppt_rejects_non_strong_permutation_contract():
    with pytest.raises(ValueError, match="strong_swaps"):
        PPTTimeOrderLoss(
            PPTTimeOrderConfig(
                num_patches=8,
                embedding_dim=4,
                weak_swaps=2,
                strong_swaps=2,
            )
        )


class _SequenceNetwork(nn.Module):
    def __init__(self, num_patches=8, embedding_dim=6):
        super().__init__()
        self.num_patches = num_patches
        self.projection = nn.Linear(1, embedding_dim)

    def encode_sequence(self, x, file_id=None):
        del file_id
        patches = x.reshape(x.shape[0], self.num_patches, -1, 1).mean(dim=2)
        return self.projection(patches)


def _task_args(**model_overrides):
    model = {
        "embedding": "E_03_Patch",
        "backbone": "B_08_PatchTST",
        "input_dim": 1,
        "window_size": 64,
        "patch_size_L": 8,
        "num_patches": 8,
        "output_dim": 6,
    }
    model.update(model_overrides)
    return (
        SimpleNamespace(window_size=64),
        SimpleNamespace(**model),
        SimpleNamespace(
            weak_swaps=1,
            strong_swaps=4,
            permutation_bank_size=16,
            permutation_seed=2,
            temperature=0.1,
            consistency_weight=1.0,
            contrastive_weight=1.0,
            optimizer="adamw",
            lr=1e-3,
            weight_decay=0.0,
        ),
    )


def test_task_runs_univariate_sequence_contract(monkeypatch):
    args_data, args_model, args_task = _task_args()
    task = PptTimeOrderTask(
        _SequenceNetwork(),
        args_data,
        args_model,
        args_task,
        SimpleNamespace(),
        SimpleNamespace(),
        None,
    )
    monkeypatch.setattr(task, "log_dict", lambda *args, **kwargs: None)
    batch = {"x": torch.randn(4, 64, 1), "file_id": torch.arange(4)}
    loss = task.training_step(batch, batch_idx=0)

    assert loss.ndim == 0
    loss.backward()
    assert task.network.projection.weight.grad is not None


def test_task_rejects_multichannel_model_contract():
    args_data, args_model, args_task = _task_args(input_dim=2)
    with pytest.raises(ValueError, match="one input channel"):
        PptTimeOrderTask(
            _SequenceNetwork(),
            args_data,
            args_model,
            args_task,
            SimpleNamespace(),
            SimpleNamespace(),
            None,
        )


def test_isfm_encode_sequence_exposes_pre_head_backbone_output():
    model = ISFMModel.__new__(ISFMModel)
    nn.Module.__init__(model)
    model.args_m = SimpleNamespace(embedding="E_03_Patch")
    model.embedding = nn.Identity()
    model.backbone = nn.Sequential(nn.Linear(5, 7), nn.GELU())
    x = torch.randn(2, 3, 5)

    encoded = model.encode_sequence(x)

    assert encoded.shape == (2, 3, 7)
    assert model.shape == x.shape


def test_real_isfm_patchtst_ppt_assembly_backpropagates(monkeypatch):
    metadata = MetadataAccessor(
        pd.DataFrame(
            [
                {
                    "Id": 1,
                    "Dataset_id": 0,
                    "Label": 0,
                    "Sample_rate": 1000,
                }
            ]
        )
    )
    args_model = SimpleNamespace(
        embedding="E_03_Patch",
        backbone="B_08_PatchTST",
        task_head="H_01_Linear_cla",
        input_dim=1,
        window_size=64,
        patch_size_L=8,
        patch_size_C=1,
        num_patches=8,
        output_dim=6,
        factor=5,
        dropout=0.0,
        num_heads=2,
        d_ff=12,
        e_layers=1,
        activation="gelu",
    )
    args_data, _, args_task = _task_args()
    network = ISFMModel(args_model, metadata)
    task = PptTimeOrderTask(
        network,
        args_data,
        args_model,
        args_task,
        SimpleNamespace(),
        SimpleNamespace(),
        metadata,
    )
    monkeypatch.setattr(task, "log_dict", lambda *args, **kwargs: None)

    loss = task.training_step(
        {"x": torch.randn(3, 64, 1), "file_id": torch.ones(3, dtype=torch.long)},
        batch_idx=0,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert network.embedding.proj[0].weight.grad is not None
    assert next(network.backbone.parameters()).grad is not None
