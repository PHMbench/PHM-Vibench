#!/usr/bin/env python3
"""Targeted contract tests for the P08 prompt-model path."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import pandas as pd


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.append(str(repo_root))

from src.model_factory.ISFM.task_head.H_11_Unified_cla import H_11_Unified_cla
from src.data_factory.ID.Id_searcher import apply_label_ontology
from src.model_factory.ISFM_Prompt.M_02_ISFM_Prompt import Model as M_02_ISFM_Prompt
from src.model_factory.ISFM_Prompt.embedding.HSE_prompt import HSE_prompt


class MockMetadata:
    """Two different dataset identities with the same measurable sample rate."""

    _rows = {
        100: {"Dataset_id": 1, "Sample_rate": 1000.0},
        200: {"Dataset_id": 19, "Sample_rate": 1000.0},
    }

    def __getitem__(self, key):
        return self._rows[int(key)]


def _make_hse_args() -> SimpleNamespace:
    return SimpleNamespace(
        patch_size_L=16,
        physical_patch_duration_s=0.01,
        physical_patch_points=16,
        patch_size_C=1,
        num_patches=8,
        output_dim=16,
        shared_band_hz=400.0,
        use_prompt=True,
        prompt_dim=8,
        prompt_combination="add",
        prompt_reference_fs_hz=1000.0,
        prompt_reference_duration_s=0.01,
        freeze_prompts_in_finetuning=False,
        dropout=0.0,
    )


def _make_model_args() -> SimpleNamespace:
    args = _make_hse_args()
    args.embedding = "HSE_prompt"
    args.backbone = "B_04_Dlinear"
    args.task_head = "H_11_Unified_cla"
    args.training_stage = "pretrain"
    args.num_classes = 4
    args.unified_num_classes = 4
    return args


def test_physical_duration_and_bands(device: torch.device) -> None:
    args = _make_hse_args()
    model = HSE_prompt(args).to(device).eval()
    signal = torch.randn(3, 128, 1, device=device)
    fs = torch.tensor([1000.0, 1500.0, 1050.0], device=device)

    with torch.no_grad():
        output = model(signal, fs)

    assert output.shape == (3, args.num_patches, args.output_dim)
    assert torch.isfinite(output).all()
    assert model.last_raw_patch_points.tolist() == [10, 15, 11]
    assert model.last_band_fractions.shape == (3, args.num_patches, 2)
    assert torch.isfinite(model.last_band_fractions).all()
    band_sums = model.last_band_fractions.sum(dim=-1)
    assert torch.allclose(band_sums, torch.ones_like(band_sums), atol=1e-5)
    assert model.last_prompt_features.shape == (3, 4)


def test_unique_patch_starts(device: torch.device) -> None:
    args = _make_hse_args()
    args.physical_patch_duration_s = 0.015
    args.physical_patch_points = 256
    args.num_patches = 32
    model = HSE_prompt(args).to(device).eval()
    signal = torch.randn(1, 240, 1, device=device)

    with torch.no_grad():
        output = model(signal, torch.tensor([12000.0], device=device))

    assert output.shape == (1, 32, args.output_dim)
    assert model.last_raw_patch_points.tolist() == [180]
    starts = model.last_patch_starts[0]
    assert starts.tolist() == sorted(starts.tolist())
    assert torch.unique(starts).numel() == args.num_patches
    assert starts[0].item() == 0 and starts[-1].item() == 60

    impossible_args = _make_hse_args()
    impossible_args.physical_patch_duration_s = 0.015
    impossible_args.num_patches = 62
    impossible = HSE_prompt(impossible_args).to(device).eval()
    try:
        impossible(signal, torch.tensor([12000.0], device=device))
    except ValueError as exc:
        assert "unique physical-patch starts are impossible" in str(exc)
    else:
        raise AssertionError("non-unique physical-patch starts must be rejected")


def test_factorial_runtime_switches(device: torch.device) -> None:
    args = _make_hse_args()
    args.use_physical_duration = False
    args.fixed_raw_token_points = 12
    args.use_band_projection = False
    args.use_prompt = False
    model = HSE_prompt(args).to(device).eval()
    signal = torch.randn(2, 128, 1, device=device)

    with torch.no_grad():
        output = model(signal, torch.tensor([1000.0, 1500.0], device=device))

    assert output.shape == (2, args.num_patches, args.output_dim)
    assert model.last_raw_patch_points.tolist() == [12, 12]
    assert torch.count_nonzero(model.last_band_fractions).item() == 0


def test_uniform_rate_vectorization(device: torch.device) -> None:
    torch.manual_seed(13)
    model = HSE_prompt(_make_hse_args()).to(device).eval()
    signal = torch.randn(2, 128, 1, device=device)
    fs = torch.tensor([1000.0, 1000.0], device=device)
    with torch.no_grad():
        batch_output = model(signal, fs)
        individual_output = torch.cat(
            [
                model(signal[index : index + 1], fs[index : index + 1])
                for index in range(2)
            ],
            dim=0,
        )
    assert torch.allclose(batch_output, individual_output, atol=1e-6, rtol=1e-5)


def test_scalar_sampling_rate(device: torch.device) -> None:
    model = HSE_prompt(_make_hse_args()).to(device).eval()
    signal = torch.randn(2, 128, 1, device=device)
    with torch.no_grad():
        output = model(signal, 1000.0)
    assert output.shape == (2, 8, 16)
    assert model.last_raw_patch_points.tolist() == [10, 10]


def test_dataset_id_is_rejected(device: torch.device) -> None:
    model = HSE_prompt(_make_hse_args()).to(device).eval()
    signal = torch.randn(2, 128, 1, device=device)
    try:
        model(signal, 1000.0, dataset_ids=torch.tensor([1, 19], device=device))
    except ValueError as exc:
        assert "dataset_ids are forbidden" in str(exc)
    else:
        raise AssertionError("dataset_ids must be rejected, not ignored")


def test_unified_head_rejects_system_selector(device: torch.device) -> None:
    args = _make_model_args()
    head = H_11_Unified_cla(args).to(device).eval()
    features = torch.randn(2, 8, args.output_dim, device=device)
    assert head(features).shape == (2, args.unified_num_classes)
    try:
        head(features, system_id=1)
    except ValueError as exc:
        assert "system_id is forbidden" in str(exc)
    else:
        raise AssertionError("system_id must not select a classification head")


def test_shared_label_ontology() -> None:
    frame = pd.DataFrame(
        {
            "Dataset_id": [1, 12, 13, 19, 19],
            "Label": [2, 2, 2, 7, 6],
        }
    )
    ontology = SimpleNamespace(
        num_classes=4,
        mappings={
            "1": {"2": 2},
            "12": {"2": 3},
            "13": {"2": 3},
            "19": {"7": 1},
        },
        excluded_labels={"19": [6]},
    )
    mapped = apply_label_ontology(
        frame, SimpleNamespace(label_ontology=ontology)
    )
    assert mapped["Raw_Label"].tolist() == [2, 2, 2, 7]
    assert mapped["Label"].tolist() == [2, 3, 3, 1]


def test_end_to_end_dataset_identity_invariance(device: torch.device) -> None:
    torch.manual_seed(7)
    args = _make_model_args()
    model = M_02_ISFM_Prompt(args, metadata=MockMetadata()).to(device).eval()
    signal = torch.randn(4, 128, 1, device=device)

    with torch.no_grad():
        logits_a, features_a = model(
            signal, file_id=100, task_id="classification", return_feature=True
        )
        logits_b, features_b = model(
            signal, file_id=200, task_id="classification", return_feature=True
        )

    assert logits_a.shape == (4, args.unified_num_classes)
    assert torch.equal(logits_a, logits_b)
    assert torch.equal(features_a, features_b)
    info = model.get_model_info()
    assert info["prompt_config"]["dataset_identity_consumed"] is False


def test_explicit_per_sample_sampling_rate(device: torch.device) -> None:
    model = M_02_ISFM_Prompt(_make_model_args(), metadata=None).to(device).eval()
    signal = torch.randn(2, 128, 1, device=device)
    with torch.no_grad():
        logits = model(
            signal,
            task_id="classification",
            sampling_rate_hz=torch.tensor([1000.0, 1500.0], device=device),
        )
    assert logits.shape == (2, 4)
    try:
        model(signal, task_id="classification", sampling_rate_hz=[1000.0])
    except ValueError as exc:
        assert "length must equal batch size" in str(exc)
    else:
        raise AssertionError("sampling-rate broadcast must fail on the evidence path")


def main() -> int:
    # The evidence protocol is single-GPU, but this contract test defaults to CPU
    # so it cannot accidentally occupy a forbidden device.
    device = torch.device("cpu")
    test_physical_duration_and_bands(device)
    print("PASS physical-duration extraction and Nyquist bands")
    test_unique_patch_starts(device)
    print("PASS unique physical-patch starts")
    test_factorial_runtime_switches(device)
    print("PASS factorial runtime switches")
    test_uniform_rate_vectorization(device)
    print("PASS uniform-rate vectorization")
    test_scalar_sampling_rate(device)
    print("PASS scalar sampling rate")
    test_dataset_id_is_rejected(device)
    print("PASS dataset-ID rejection")
    test_unified_head_rejects_system_selector(device)
    print("PASS unified-head selector rejection")
    test_shared_label_ontology()
    print("PASS shared label ontology")
    test_end_to_end_dataset_identity_invariance(device)
    print("PASS end-to-end dataset-identity invariance")
    test_explicit_per_sample_sampling_rate(device)
    print("PASS explicit per-sample sampling rates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
