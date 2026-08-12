from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
import pytest
import torch

import src.explain_factory.p05_trace_diagnostics as diagnostics
from src.explain_factory.p05_trace_diagnostics import (
    create_p05_d01_d02_trace_diagnostics,
)
from src.explain_factory.p05_trace_export import (
    P05TraceBatch,
    export_p05_trace_package,
)
from src.model_factory.X_model.UXFD.fuzzy.fuzzy_reasoner import FuzzyTrace


CONFIG_HASH = "a" * 64
CHECKPOINT_HASH = "b" * 64
MODEL_HASH = "c" * 64
FIRING = torch.tensor(
    [
        [0.40, 0.40, 0.11, 0.09],
        [0.09, 0.21, 0.30, 0.40],
        [0.25, 0.25, 0.25, 0.25],
        [0.70, 0.08, 0.09, 0.13],
        [0.05, 0.15, 0.40, 0.40],
        [0.09, 0.21, 0.30, 0.40],
    ],
    dtype=torch.float32,
)


def _trace_batch(labels: list[int] | None = None) -> P05TraceBatch:
    sample_count, rule_count = FIRING.shape
    feature_count = 2
    membership_count = 2
    class_count = 3
    consequents = torch.tensor(
        [
            [0.20, -0.10, 0.30],
            [-0.40, 0.50, 0.10],
            [0.15, 0.20, -0.30],
            [0.05, -0.25, 0.40],
        ],
        dtype=torch.float32,
    )
    contributions = FIRING.unsqueeze(-1) * consequents.unsqueeze(0)
    fuzzy_logits = contributions.sum(dim=1)
    non_fuzzy_logits = torch.linspace(
        -0.4,
        1.2,
        steps=sample_count * class_count,
        dtype=torch.float32,
    ).reshape(sample_count, class_count)
    logits = non_fuzzy_logits + 0.5 * fuzzy_logits
    trace = FuzzyTrace(
        reduced_features=torch.linspace(
            -0.5,
            0.5,
            steps=sample_count * feature_count,
            dtype=torch.float32,
        ).reshape(sample_count, feature_count),
        membership_values=torch.full(
            (sample_count, feature_count, membership_count),
            0.5,
            dtype=torch.float32,
        ),
        centers=torch.tensor([[-1.0, 1.0], [-0.5, 0.5]]),
        widths=torch.tensor([[0.7, 0.8], [0.9, 1.0]]),
        antecedent_probabilities=torch.full(
            (rule_count, feature_count, membership_count),
            0.5,
            dtype=torch.float32,
        ),
        antecedent_memberships=torch.full(
            (sample_count, rule_count, feature_count),
            0.6,
            dtype=torch.float32,
        ),
        log_rule_firing=torch.log(FIRING),
        rule_firing=FIRING,
        normalized_rule_firing=FIRING,
        rule_consequents=consequents,
        rule_contributions=contributions,
        fuzzy_logits=fuzzy_logits,
        rule_mask=torch.ones((sample_count, rule_count), dtype=torch.bool),
        consequent_permutation=torch.arange(rule_count),
    )
    if labels is None:
        labels = [0, 0, 1, 1, 2, 2]
    return P05TraceBatch(
        sample_id=[f"sample-{index:02d}" for index in range(sample_count)],
        record_id=[f"record-{index:02d}" for index in range(sample_count)],
        group_id=[f"class-{label}" for label in labels],
        window_start=torch.arange(sample_count, dtype=torch.int64) * 10,
        window_end=torch.arange(sample_count, dtype=torch.int64) * 10 + 10,
        y=torch.tensor(labels, dtype=torch.int64),
        logits=logits,
        non_fuzzy_logits=non_fuzzy_logits,
        fuzzy_scale=0.5,
        fuzzy_trace=trace,
    )


def _export_trace(path: Path, *, labels: list[int] | None = None):
    return export_p05_trace_package(
        path,
        [_trace_batch(labels)],
        config_sha256=CONFIG_HASH,
        checkpoint_sha256=CHECKPOINT_HASH,
        model_sha256=MODEL_HASH,
    )


def _binding(path: Path) -> dict[str, str]:
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    return {
        "expected_trace_semantic_sha256": manifest["content"]["semantic_sha256"],
        "expected_config_sha256": CONFIG_HASH,
        "expected_checkpoint_sha256": CHECKPOINT_HASH,
        "expected_model_sha256": MODEL_HASH,
    }


def _create(artifact: Path, trace: Path):
    return create_p05_d01_d02_trace_diagnostics(
        artifact,
        trace_package=trace,
        **_binding(trace),
    )


def _reseal_trace(
    trace: Path,
    mutation: Callable[[dict[str, np.ndarray]], None],
) -> None:
    arrays_path = trace / "trace_arrays.npz"
    with np.load(arrays_path, allow_pickle=False) as archive:
        arrays = {
            name: np.array(archive[name], copy=True, order="C")
            for name in archive.files
        }
    mutation(arrays)
    with arrays_path.open("wb") as handle:
        np.savez(handle, **{name: arrays[name] for name in sorted(arrays)})

    manifest_path = trace / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, array in arrays.items():
        descriptor = manifest["arrays"][name]
        descriptor["dtype"] = array.dtype.str
        descriptor["shape"] = [int(size) for size in array.shape]
        descriptor["sha256"] = diagnostics._array_sha256(array)
    manifest["content"]["npz_sha256"] = diagnostics._sha256_file(arrays_path)
    semantic = {name: value for name, value in manifest.items() if name != "content"}
    manifest["content"]["semantic_sha256"] = diagnostics._sha256_bytes(
        diagnostics._canonical_json_bytes(semantic)
    )
    manifest_path.write_bytes(diagnostics._pretty_json_bytes(manifest))


def test_d01_d02_artifact_is_hash_bound_create_only_and_unadjudicated(
    tmp_path,
) -> None:
    trace = _export_trace(tmp_path / "trace")
    artifact = tmp_path / "diagnostics"

    created = _create(artifact, trace.package_dir)
    arrays_before = created.arrays_path.read_bytes()
    manifest_before = created.manifest_path.read_bytes()
    reused = _create(artifact, trace.package_dir)

    assert created.status == "created"
    assert reused.status == "reused"
    assert reused.arrays_path.read_bytes() == arrays_before
    assert reused.manifest_path.read_bytes() == manifest_before
    manifest = json.loads(manifest_before)
    assert manifest["schema_name"] == "p05.d01_d02_trace_diagnostics"
    assert manifest["source_trace"] == {
        "checkpoint_sha256": CHECKPOINT_HASH,
        "config_sha256": CONFIG_HASH,
        "manifest_sha256": trace.manifest_sha256,
        "model_sha256": MODEL_HASH,
        "npz_sha256": trace.npz_sha256,
        "schema_name": "p05.complete_fuzzy_trace",
        "schema_version": 1,
        "semantic_sha256": trace.semantic_sha256,
    }
    assert manifest["conclusion_control"] == {
        "claim_decisions": "not_performed",
        "confirmatory_sign_tests": "not_performed",
        "scientific_status": "computed_unadjudicated",
        "scope": "mandatory_P05_D01_D02_trace_diagnostics_only",
    }
    assert manifest["protocol"]["tie_break"] == (
        "descending firing, then lower rule index"
    )
    assert manifest["protocol"]["P05-D02"]["threshold_operator"] == ">"

    with np.load(created.arrays_path, allow_pickle=False) as arrays:
        firing = FIRING.numpy().astype(np.float64)
        firing /= firing.sum(axis=1, keepdims=True)
        entropy = -(firing * np.log(firing)).sum(axis=1)
        np.testing.assert_allclose(
            arrays["d01_effective_rule_count"],
            np.exp(entropy),
            rtol=0.0,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            arrays["d01_top1_mass"],
            [0.40, 0.40, 0.25, 0.70, 0.40, 0.40],
            atol=1.0e-7,
        )
        np.testing.assert_allclose(
            arrays["d01_top3_mass"],
            [0.91, 0.91, 0.75, 0.92, 0.95, 0.91],
            atol=1.0e-7,
        )
        assert arrays["d01_gini"][2] == pytest.approx(0.0, abs=1.0e-15)
        assert arrays["d01_top_rule_index"].tolist() == [0, 3, 0, 0, 2, 3]
        assert arrays["d01_top3_rule_indices"].tolist() == [
            [0, 1, 2],
            [3, 2, 1],
            [0, 1, 2],
            [0, 3, 2],
            [2, 3, 1],
            [3, 2, 1],
        ]
        np.testing.assert_allclose(
            arrays["d02_overall_ever_top_ranked_coverage"],
            0.75,
        )
        np.testing.assert_allclose(
            arrays["d02_overall_firing_gt_0_10_coverage"],
            1.0,
        )
        np.testing.assert_allclose(
            arrays["d02_overall_appearing_top3_coverage"],
            1.0,
        )
        np.testing.assert_allclose(
            arrays["d02_by_class_ever_top_ranked_coverage"],
            [0.50, 0.25, 0.50],
        )
        np.testing.assert_allclose(
            arrays["d02_by_class_firing_gt_0_10_coverage"],
            [1.00, 1.00, 0.75],
        )
        np.testing.assert_allclose(
            arrays["d02_by_class_appearing_top3_coverage"],
            [1.00, 1.00, 0.75],
        )
        assert arrays["protocol_class_sample_count"].tolist() == [2, 2, 2]


def test_expected_hash_mismatch_fails_without_artifact(tmp_path) -> None:
    trace = _export_trace(tmp_path / "trace")
    artifact = tmp_path / "diagnostics"
    binding = _binding(trace.package_dir)
    binding["expected_model_sha256"] = "d" * 64

    with pytest.raises(ValueError, match="model_sha256.*expected binding"):
        create_p05_d01_d02_trace_diagnostics(
            artifact,
            trace_package=trace.package_dir,
            **binding,
        )

    assert not artifact.exists()


def test_missing_protocol_class_fails_without_artifact(tmp_path) -> None:
    trace = _export_trace(
        tmp_path / "trace",
        labels=[0, 0, 1, 1, 0, 1],
    )
    artifact = tmp_path / "diagnostics"

    with pytest.raises(ValueError, match=r"missing protocol classes: \[2\]"):
        _create(artifact, trace.package_dir)

    assert not artifact.exists()


def _duplicate_sample_id(arrays: dict[str, np.ndarray]) -> None:
    arrays["sample_id"][1] = arrays["sample_id"][0]


def _nonnormal_firing(arrays: dict[str, np.ndarray]) -> None:
    arrays["trace_normalized_rule_firing"][0] *= 1.2


def _nonfinite_firing(arrays: dict[str, np.ndarray]) -> None:
    arrays["trace_normalized_rule_firing"][0, 0] = np.nan


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (_duplicate_sample_id, "sample_id values must be unique"),
        (_nonnormal_firing, "rows do not sum to one"),
        (_nonfinite_firing, "non-finite"),
    ],
)
def test_resealed_invalid_trace_is_rejected_without_artifact(
    tmp_path,
    mutation,
    message,
) -> None:
    trace = _export_trace(tmp_path / "trace")
    _reseal_trace(trace.package_dir, mutation)
    artifact = tmp_path / "diagnostics"

    with pytest.raises((ValueError, FloatingPointError), match=message):
        _create(artifact, trace.package_dir)

    assert not artifact.exists()


def test_existing_artifact_conflict_preserves_original_bytes(tmp_path) -> None:
    trace = _export_trace(tmp_path / "trace")
    artifact = tmp_path / "diagnostics"
    created = _create(artifact, trace.package_dir)
    arrays_before = created.arrays_path.read_bytes()
    manifest_before = created.manifest_path.read_bytes()
    (artifact / "unexpected.txt").write_text("conflict", encoding="utf-8")

    with pytest.raises(FileExistsError, match="unexpected or incomplete"):
        _create(artifact, trace.package_dir)

    assert created.arrays_path.read_bytes() == arrays_before
    assert created.manifest_path.read_bytes() == manifest_before


def test_atomic_write_failure_removes_staging_directory(tmp_path, monkeypatch) -> None:
    trace = _export_trace(tmp_path / "trace")
    artifact = tmp_path / "diagnostics"

    def _fail_save(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("synthetic diagnostic write failure")

    monkeypatch.setattr(diagnostics.np, "savez", _fail_save)
    with pytest.raises(RuntimeError, match="synthetic diagnostic write failure"):
        _create(artifact, trace.package_dir)

    assert not artifact.exists()
    assert not list(tmp_path.glob(".diagnostics.*.tmp"))
