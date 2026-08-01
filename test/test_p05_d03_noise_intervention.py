from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import pytest
import torch

from src.explain_factory import p05_d03_noise_intervention as d03_module
from src.explain_factory.p05_d03_noise_intervention import (
    DRAWS_PER_LEVEL,
    P05D03Provenance,
    SNR_LEVELS_DB,
    TOTAL_NOISE_DRAWS,
    p05_d03_noise_seed,
    run_p05_d03_noise_interventions,
    run_p05_d03_noise_interventions_from_loader,
    verify_p05_d03_artifact,
)
from src.explain_factory.p05_trace_runner import model_state_sha256


HASHES = {
    "config_sha256": "1" * 64,
    "code_sha256": "2" * 64,
    "checkpoint_sha256": "3" * 64,
    "run_contract_sha256": "4" * 64,
    "source_metadata_sha256": "5" * 64,
    "derived_metadata_sha256": "6" * 64,
    "cache_manifest_sha256": "7" * 64,
    "split_manifest_sha256": "8" * 64,
    "normalization_sha256": "9" * 64,
}


@dataclass(frozen=True)
class _Trace:
    reduced_features: torch.Tensor
    normalized_rule_firing: torch.Tensor
    rule_contributions: torch.Tensor
    fuzzy_logits: torch.Tensor
    rule_mask: torch.Tensor
    consequent_permutation: torch.Tensor


@dataclass(frozen=True)
class _Output:
    logits: torch.Tensor
    non_fuzzy_logits: torch.Tensor
    fuzzy_scale: float
    fuzzy_trace: _Trace


class _NoiseTraceNetwork(torch.nn.Module):
    def __init__(self, *, mutate_state: bool = False) -> None:
        super().__init__()
        rule = torch.arange(10, dtype=torch.float32)
        self.consequents = torch.nn.Parameter(
            torch.stack(
                (
                    0.4 * torch.cos(rule * 0.31) + 0.02 * rule,
                    0.5 * torch.sin(rule * 0.43) - 0.01 * rule,
                ),
                dim=1,
            )
        )
        self.register_buffer(
            "firing_weight",
            torch.arange(80, dtype=torch.float32).reshape(8, 10) * 0.004 - 0.12,
        )
        self.register_buffer(
            "classifier",
            torch.tensor(
                [
                    [0.3, -0.2],
                    [-0.1, 0.4],
                    [0.2, 0.1],
                    [-0.3, 0.2],
                    [0.1, 0.3],
                    [0.2, -0.1],
                    [-0.2, 0.1],
                    [0.05, -0.15],
                ],
                dtype=torch.float32,
            ),
        )
        self.register_buffer("mutable", torch.zeros(1, dtype=torch.float32))
        self.mutate_state = mutate_state
        self.calls = 0
        self.batch_sizes: list[int] = []

    def forward_with_fuzzy_trace(self, x: torch.Tensor) -> _Output:
        self.calls += 1
        self.batch_sizes.append(int(x.shape[0]))
        if self.mutate_state:
            self.mutable.add_(1.0)
        mean = x.mean(dim=1)
        std = x.std(dim=1, unbiased=False)
        rms = x.square().mean(dim=1).sqrt()
        maximum = x.amax(dim=1)
        features = torch.cat((mean, std, rms, maximum), dim=1)
        firing = torch.softmax(features @ self.firing_weight, dim=1)
        contributions = firing.unsqueeze(-1) * self.consequents.unsqueeze(0)
        fuzzy_logits = contributions.sum(dim=1)
        non_fuzzy_logits = features @ self.classifier
        trace = _Trace(
            reduced_features=features,
            normalized_rule_firing=firing,
            rule_contributions=contributions,
            fuzzy_logits=fuzzy_logits,
            rule_mask=torch.ones(
                (x.shape[0], 10), dtype=torch.bool, device=x.device
            ),
            consequent_permutation=torch.arange(10, device=x.device),
        )
        return _Output(
            logits=non_fuzzy_logits + 0.5 * fuzzy_logits,
            non_fuzzy_logits=non_fuzzy_logits,
            fuzzy_scale=0.5,
            fuzzy_trace=trace,
        )


def _batch() -> dict[str, object]:
    return {
        "x": torch.tensor(
            [
                [[0.4, 0.1], [0.2, 0.3], [0.8, -0.1], [0.6, 0.5]],
                [[-0.2, 0.7], [0.1, 0.6], [0.3, 0.2], [0.5, -0.4]],
                [[0.9, -0.3], [0.7, 0.4], [0.5, 0.2], [0.1, 0.8]],
                [[-0.4, 0.2], [0.6, 0.9], [0.2, -0.5], [0.7, 0.3]],
            ],
            dtype=torch.float32,
        ),
        "y": torch.tensor([0, 1, 0, 1]),
        "sample_id": ["r3:12:16", "r1:4:8", "r4:16:20", "r2:8:12"],
        "record_id": ["r3", "r1", "r4", "r2"],
        "group_id": ["b", "a", "b", "a"],
        "window_start": torch.tensor([12, 4, 16, 8]),
        "window_end": torch.tensor([16, 8, 20, 12]),
    }


def _provenance(
    network: torch.nn.Module,
    *,
    cuda_identity: bool = False,
    split: str = "test",
    model_seed: int = 42,
) -> P05D03Provenance:
    return P05D03Provenance(
        dataset="XJTU",
        split=split,
        model_seed=model_seed,
        model_sha256=model_state_sha256(network),
        physical_gpu_index=0 if cuda_identity else None,
        device_uuid="GPU-test-uuid" if cuda_identity else None,
        **HASHES,
    )


def _run(tmp_path, network: _NoiseTraceNetwork, name: str = "d03"):
    batch = _batch()
    return run_p05_d03_noise_interventions(
        tmp_path / name,
        network=network,
        batch=batch,
        provenance=_provenance(network),
        expected_sample_ids=batch["sample_id"],
        phase="budget_retained_secondary",
        budget_retained=True,
        expected_window_size=4,
        require_cuda=False,
    )


def test_d03_runs_exact_protocol_and_writes_unadjudicated_artifact(tmp_path) -> None:
    network = _NoiseTraceNetwork()
    network.train()
    state_before = model_state_sha256(network)

    result = _run(tmp_path, network)

    assert result.status == "created"
    assert network.training is True
    assert network.calls == 1 + TOTAL_NOISE_DRAWS
    assert model_state_sha256(network) == state_before
    assert result.timing["performance_claim_allowed"] is False
    assert result.timing["total_seconds"] >= 0.0
    manifest = verify_p05_d03_artifact(result.artifact_dir)
    assert manifest["conclusion_control"] == {
        "claim_decisions": "not_performed",
        "confirmatory_sign_tests": "not_performed",
        "performance_claim": False,
        "scientific_status": "computed_unadjudicated",
        "scope": "budget_conditional_secondary_P05_D03_only",
    }
    assert manifest["execution"] == {
        "actual_forward_calls": 1 + TOTAL_NOISE_DRAWS,
        "budget_retained": True,
        "chunk_count": 1,
        "chunk_size": 256,
        "device_class": "cpu_test_only",
        "phase": "budget_retained_secondary",
    }
    assert manifest["model_state"]["before_sha256"] == state_before
    assert manifest["model_state"]["after_sha256"] == state_before

    with np.load(result.arrays_path, allow_pickle=False) as arrays:
        assert arrays["sample_id"].tolist() == [
            "r1:4:8",
            "r2:8:12",
            "r3:12:16",
            "r4:16:20",
        ]
        assert arrays["noise_seed"].shape == (4, 2)
        assert arrays["noise_sha256"].shape == (4, 2, DRAWS_PER_LEVEL)
        assert arrays["noisy_logits"].shape == (4, 2, DRAWS_PER_LEVEL, 2)
        assert arrays["noisy_logits"].dtype == np.dtype("float32")
        assert arrays["noisy_normalized_rule_firing"].shape == (
            4,
            2,
            DRAWS_PER_LEVEL,
            10,
        )
        assert arrays["top_rule_agreement"].shape == (4, 2, DRAWS_PER_LEVEL)
        assert arrays["top3_jaccard"].dtype == np.dtype("float64")
        assert np.all((arrays["top3_jaccard"] >= 0.0) & (arrays["top3_jaccard"] <= 1.0))
        assert np.all(arrays["firing_vector_jsd"] >= 0.0)
        assert np.all(
            (arrays["attribution_rank_tau"] >= -1.0)
            & (arrays["attribution_rank_tau"] <= 1.0)
        )
        assert arrays["snr_db"].tolist() == list(SNR_LEVELS_DB)
        median_realized = np.median(arrays["realized_snr_db"], axis=(0, 2))
        np.testing.assert_allclose(median_realized, SNR_LEVELS_DB, atol=2.0, rtol=0.0)


def test_d03_seed_and_full_arrays_are_deterministic(tmp_path) -> None:
    expected = int.from_bytes(
        hashlib.sha256(
            b"P05-stability|XJTU|test|42|r1:4:8|30"
        ).digest()[:8],
        "big",
        signed=False,
    )
    assert p05_d03_noise_seed(
        dataset="XJTU",
        split="test",
        model_seed=42,
        sample_id="r1:4:8",
        snr_db=30,
    ) == expected

    first_network = _NoiseTraceNetwork()
    first = _run(tmp_path, first_network, "first")
    second_network = _NoiseTraceNetwork()
    batch = _batch()
    second = run_p05_d03_noise_interventions(
        tmp_path / "second",
        network=second_network,
        batch=batch,
        provenance=_provenance(second_network),
        expected_sample_ids=batch["sample_id"],
        phase="budget_retained_secondary",
        budget_retained=True,
        expected_window_size=4,
        require_cuda=False,
        chunk_size=1,
    )
    with np.load(first.arrays_path, allow_pickle=False) as left, np.load(
        second.arrays_path, allow_pickle=False
    ) as right:
        assert set(left.files) == set(right.files)
        for name in left.files:
            np.testing.assert_array_equal(left[name], right[name])
    assert first.semantic_sha256 != second.semantic_sha256
    assert first_network.batch_sizes == [4] * (1 + TOTAL_NOISE_DRAWS)
    assert second_network.batch_sizes == [1] * (4 * (1 + TOTAL_NOISE_DRAWS))


def test_d03_rejects_seed_collision_across_chunk_boundaries(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        d03_module,
        "p05_d03_noise_seed",
        lambda **kwargs: int(kwargs["snr_db"]),
    )
    network = _NoiseTraceNetwork()
    batch = _batch()
    target = tmp_path / "seed-collision"

    with pytest.raises(RuntimeError, match="collision detected across chunks"):
        run_p05_d03_noise_interventions(
            target,
            network=network,
            batch=batch,
            provenance=_provenance(network),
            expected_sample_ids=batch["sample_id"],
            phase="budget_retained_secondary",
            budget_retained=True,
            expected_window_size=4,
            require_cuda=False,
            chunk_size=1,
        )

    assert network.calls == 4 * (1 + TOTAL_NOISE_DRAWS)
    assert not target.exists()


def test_d03_is_atomic_create_only_and_detects_tampering(tmp_path) -> None:
    network = _NoiseTraceNetwork()
    result = _run(tmp_path, network)
    calls = network.calls
    with pytest.raises(FileExistsError, match="create-only"):
        _run(tmp_path, network)
    assert network.calls == calls

    original_text = result.manifest_path.read_text(encoding="utf-8")
    duplicate_key = original_text.replace(
        "{\n",
        '{\n  "schema_version": 2,\n',
        1,
    )
    result.manifest_path.write_text(duplicate_key, encoding="utf-8")
    with pytest.raises(ValueError, match="invalid strict JSON"):
        verify_p05_d03_artifact(result.artifact_dir)

    for constant in ("NaN", "Infinity", "-Infinity"):
        nonfinite = original_text.replace('"sample_count": 4', f'"sample_count": {constant}')
        assert nonfinite != original_text
        result.manifest_path.write_text(nonfinite, encoding="utf-8")
        with pytest.raises(ValueError, match="invalid strict JSON"):
            verify_p05_d03_artifact(result.artifact_dir)

    manifest = json.loads(original_text)
    manifest["conclusion_control"]["claim_decisions"] = "pass"
    result.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="semantic hash"):
        verify_p05_d03_artifact(result.artifact_dir)


def test_d03_verifier_rejects_self_consistently_rehashed_seed_collision(
    tmp_path,
) -> None:
    result = _run(tmp_path, _NoiseTraceNetwork())
    with np.load(result.arrays_path, allow_pickle=False) as archive:
        arrays = {
            name: np.array(archive[name], copy=True, order="C")
            for name in archive.files
        }
    arrays["noise_seed"][1, 0] = arrays["noise_seed"][0, 0]
    with result.arrays_path.open("wb") as handle:
        np.savez(handle, **{name: arrays[name] for name in sorted(arrays)})

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    manifest["arrays"] = d03_module._array_descriptors(arrays)
    manifest["content"]["npz_sha256"] = d03_module._sha256_file(
        result.arrays_path
    )
    semantic_manifest = {
        name: value for name, value in manifest.items() if name != "content"
    }
    manifest["content"]["semantic_sha256"] = hashlib.sha256(
        d03_module._canonical_json_bytes(semantic_manifest)
    ).hexdigest()
    result.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="not globally unique"):
        verify_p05_d03_artifact(result.artifact_dir)


def test_d03_fails_closed_on_cuda_budget_identity_and_stable_id(tmp_path) -> None:
    network = _NoiseTraceNetwork()
    with pytest.raises(ValueError, match="frozen partition"):
        run_p05_d03_noise_interventions(
            tmp_path / "short-partition",
            network=network,
            batch=_batch(),
            provenance=_provenance(network, cuda_identity=True),
            expected_sample_ids=_batch()["sample_id"],
            phase="budget_retained_secondary",
            budget_retained=True,
            expected_window_size=4,
        )
    with pytest.raises(RuntimeError, match="requires a CUDA model"):
        run_p05_d03_noise_interventions(
            tmp_path / "cuda",
            network=network,
            batch=_batch(),
            provenance=_provenance(network, cuda_identity=True),
            expected_sample_ids=[f"sample-{index}" for index in range(6647 * 4)],
            phase="budget_retained_secondary",
            budget_retained=True,
            expected_window_size=4,
        )
    assert network.calls == 0

    with pytest.raises(RuntimeError, match="budget gate"):
        run_p05_d03_noise_interventions(
            tmp_path / "budget",
            network=network,
            batch=_batch(),
            provenance=_provenance(network),
            expected_sample_ids=_batch()["sample_id"],
            phase="budget_retained_secondary",
            budget_retained=False,
            expected_window_size=4,
            require_cuda=False,
        )
    assert network.calls == 0

    unstable = dict(_batch())
    unstable["sample_id"] = ["wrong", "r1:4:8", "r4:16:20", "r2:8:12"]
    with pytest.raises(ValueError, match=r"sample_id\[0\]"):
        run_p05_d03_noise_interventions(
            tmp_path / "unstable",
            network=network,
            batch=unstable,
            provenance=_provenance(network),
            expected_sample_ids=_batch()["sample_id"],
            phase="budget_retained_secondary",
            budget_retained=True,
            expected_window_size=4,
            require_cuda=False,
        )
    assert network.calls == 0


def test_d03_failure_retains_training_mode_and_does_not_publish(tmp_path) -> None:
    network = _NoiseTraceNetwork(mutate_state=True)
    network.train()
    target = tmp_path / "mutated"

    with pytest.raises(RuntimeError, match="mutated the checkpoint/model state"):
        run_p05_d03_noise_interventions(
            target,
            network=network,
            batch=_batch(),
            provenance=_provenance(network),
            expected_sample_ids=_batch()["sample_id"],
            phase="budget_retained_secondary",
            budget_retained=True,
            expected_window_size=4,
            require_cuda=False,
        )

    assert network.training is True
    assert not target.exists()


def _large_batch(count: int) -> dict[str, object]:
    base = torch.arange(count * 8, dtype=torch.float32).reshape(count, 4, 2)
    x = 0.05 + (base.remainder(29.0) / 17.0)
    record_id = [f"record-{index:04d}" for index in range(count)]
    window_start = torch.arange(count, dtype=torch.int64) * 4
    window_end = window_start + 4
    return {
        "x": x,
        "y": torch.arange(count, dtype=torch.int64).remainder(2),
        "sample_id": [
            f"{record_id[index]}:{int(window_start[index])}:{int(window_end[index])}"
            for index in range(count)
        ],
        "record_id": record_id,
        "group_id": [f"bearing-{index % 5}" for index in range(count)],
        "window_start": window_start,
        "window_end": window_end,
    }


def _slice_batch(batch: dict[str, object], start: int, stop: int) -> dict[str, object]:
    result: dict[str, object] = {}
    for name, value in batch.items():
        if torch.is_tensor(value):
            result[name] = value[start:stop]
        else:
            result[name] = value[start:stop]
    return result


def test_loader_runner_streams_large_partition_in_bounded_chunks(tmp_path) -> None:
    full = _large_batch(257)
    network = _NoiseTraceNetwork()
    later = _slice_batch(full, 128, 257)
    earlier = _slice_batch(full, 0, 128)
    later["sample_weight"] = torch.ones(129, dtype=torch.float64)
    earlier["window_index"] = torch.arange(128, dtype=torch.int64)
    result = run_p05_d03_noise_interventions_from_loader(
        tmp_path / "streamed",
        network=network,
        batches=[later, earlier],
        provenance=_provenance(network),
        expected_sample_ids=full["sample_id"],
        phase="budget_retained_secondary",
        budget_retained=True,
        expected_window_size=4,
        require_cuda=False,
        chunk_size=64,
    )

    manifest = verify_p05_d03_artifact(result.artifact_dir)
    assert manifest["execution"]["chunk_size"] == 64
    assert manifest["execution"]["chunk_count"] == 5
    assert manifest["execution"]["actual_forward_calls"] == 5 * (
        1 + TOTAL_NOISE_DRAWS
    )
    assert manifest["partition_coverage"]["coverage"] == "exact"
    assert manifest["partition_coverage"]["expected_sample_count"] == 257
    assert manifest["partition_coverage"]["selected_sample_count"] == 257
    assert max(network.batch_sizes) <= 64
    assert len(network.batch_sizes) == 5 * (1 + TOTAL_NOISE_DRAWS)
    with np.load(result.arrays_path, allow_pickle=False) as arrays:
        assert arrays["sample_id"].tolist() == sorted(full["sample_id"])


def test_loader_runner_rejects_partial_partition_without_publishing(tmp_path) -> None:
    full = _batch()
    network = _NoiseTraceNetwork()
    target = tmp_path / "partial"

    with pytest.raises(ValueError, match="coverage differs"):
        run_p05_d03_noise_interventions_from_loader(
            target,
            network=network,
            batches=[_slice_batch(full, 0, 2)],
            provenance=_provenance(network),
            expected_sample_ids=full["sample_id"],
            phase="budget_retained_secondary",
            budget_retained=True,
            expected_window_size=4,
            require_cuda=False,
            chunk_size=2,
        )

    assert network.calls == 1 + TOTAL_NOISE_DRAWS
    assert not target.exists()


def test_loader_runner_preserves_pilot_first_256_stable_id_selection(tmp_path) -> None:
    full = _large_batch(257)
    network = _NoiseTraceNetwork()
    with pytest.raises(ValueError, match="exact batched evaluator"):
        run_p05_d03_noise_interventions_from_loader(
            tmp_path / "pilot-chunked",
            network=network,
            batches=[full],
            provenance=_provenance(
                network,
                split="validation",
                model_seed=20260801,
            ),
            expected_sample_ids=full["sample_id"],
            phase="pilot_benchmark",
            budget_retained=None,
            expected_window_size=4,
            require_cuda=False,
            chunk_size=64,
        )
    assert network.calls == 0

    result = run_p05_d03_noise_interventions_from_loader(
        tmp_path / "pilot",
        network=network,
        batches=[_slice_batch(full, 200, 257), _slice_batch(full, 0, 200)],
        provenance=_provenance(
            network,
            split="validation",
            model_seed=20260801,
        ),
        expected_sample_ids=full["sample_id"],
        phase="pilot_benchmark",
        budget_retained=None,
        expected_window_size=4,
        require_cuda=False,
        chunk_size=256,
    )

    manifest = verify_p05_d03_artifact(result.artifact_dir)
    assert manifest["sample_count"] == 256
    assert manifest["input_binding"]["input_count"] == 257
    assert manifest["partition_coverage"]["expected_sample_count"] == 257
    assert manifest["partition_coverage"]["selected_sample_count"] == 256
    assert manifest["execution"]["chunk_size"] == 256
    assert manifest["execution"]["chunk_count"] == 1
    assert manifest["execution"]["actual_forward_calls"] == 33
    assert network.batch_sizes == [256] * 33
    with np.load(result.arrays_path, allow_pickle=False) as arrays:
        assert arrays["sample_id"].tolist() == sorted(full["sample_id"])[:256]
