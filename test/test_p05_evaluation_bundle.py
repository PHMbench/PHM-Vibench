from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import src.explain_factory.p05_evaluation_bundle as evaluation_bundle
import src.Pipeline_05_Explainable_Fault_Diagnosis as pipeline_module
from src.explain_factory.p05_evaluation_bundle import (
    P05EvaluationFrozenParameters,
    create_p05_c2_c3_evaluation_bundle,
)
from src.explain_factory.p05_trace_export import (
    export_p05_trace_package,
)
from src.explain_factory.p05_intervention_runner import (
    P05InterventionProvenance,
    run_p05_same_checkpoint_interventions,
)
from src.explain_factory.p05_trace_runner import (
    export_p05_loader_trace,
    model_state_sha256,
)
from src.model_factory.X_model.UXFD.fuzzy.fuzzy_reasoner import (
    FuzzyConfig,
    FuzzyReasoner,
    FuzzyTrace,
)


CONFIG_HASH = "a" * 64
CHECKPOINT_HASH = "b" * 64
@dataclass(frozen=True)
class _Output:
    logits: torch.Tensor
    non_fuzzy_logits: torch.Tensor
    fuzzy_scale: float
    fuzzy_trace: FuzzyTrace


class _BundleNetwork(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        with torch.random.fork_rng():
            torch.manual_seed(17)
            self.non_fuzzy = torch.nn.Linear(2, 2)
            self.reasoner = FuzzyReasoner(
                dim_in=2,
                num_classes=2,
                cfg=FuzzyConfig(
                    num_fuzzy_features=2,
                    num_membership_functions=2,
                    num_rules=10,
                    logit_scale=0.5,
                ),
            )

    def forward_with_fuzzy_trace(
        self,
        x: torch.Tensor,
        *,
        rule_mask: torch.Tensor | None = None,
        consequent_permutation: torch.Tensor | None = None,
    ) -> _Output:
        features = x.mean(dim=1)
        non_fuzzy_logits = self.non_fuzzy(features)
        trace = self.reasoner.forward_with_trace(
            features,
            rule_mask=rule_mask,
            consequent_permutation=consequent_permutation,
        )
        return _Output(
            logits=non_fuzzy_logits + 0.5 * trace.fuzzy_logits,
            non_fuzzy_logits=non_fuzzy_logits,
            fuzzy_scale=0.5,
            fuzzy_trace=trace,
        )


def _stable_batch(network: _BundleNetwork, prefix: str, *, phase: float) -> dict:
    sample_count = 40
    index = torch.arange(sample_count, dtype=torch.float32)
    step = torch.arange(4, dtype=torch.float32)
    first = torch.sin(index[:, None] * 0.19 + step[None, :] * 0.11 + phase)
    second = torch.cos(index[:, None] * 0.23 - step[None, :] * 0.07 + phase)
    x = torch.stack((first, second), dim=-1)
    with torch.no_grad():
        predictions = network.forward_with_fuzzy_trace(x).logits.argmax(dim=1)
    labels = predictions.clone()
    labels[::4] = 1 - labels[::4]
    return {
        "x": x,
        "y": labels,
        "sample_id": [f"{prefix}-sample-{item:03d}" for item in range(sample_count)],
        "record_id": [f"{prefix}-record-{item // 20}" for item in range(sample_count)],
        "group_id": [f"{prefix}-bearing-{item // 20}" for item in range(sample_count)],
        "window_start": torch.arange(sample_count, dtype=torch.int64) * 32,
        "window_end": torch.arange(sample_count, dtype=torch.int64) * 32 + 128,
    }


@pytest.fixture(scope="module")
def trace_pair(tmp_path_factory):
    root = tmp_path_factory.mktemp("p05-actual-bundle")
    network = _BundleNetwork()
    model_hash = model_state_sha256(network)
    validation_batch = _stable_batch(network, "validation", phase=0.0)
    evaluation_batch = _stable_batch(network, "evaluation", phase=0.6)
    validation = export_p05_loader_trace(
        root / "validation-trace",
        network=network,
        dataloader=[validation_batch],
        config_sha256=CONFIG_HASH,
        checkpoint_sha256=CHECKPOINT_HASH,
        model_sha256=model_hash,
        expected_window_size=4,
        require_cuda=False,
    )
    actual = run_p05_same_checkpoint_interventions(
        network=network,
        batch=evaluation_batch,
        provenance=P05InterventionProvenance(
            dataset="XJTU",
            split="test",
            model_seed=42,
            config_sha256=CONFIG_HASH,
            checkpoint_sha256=CHECKPOINT_HASH,
            model_sha256=model_hash,
        ),
        expected_window_size=4,
        require_cuda=False,
    )
    evaluation = export_p05_trace_package(
        root / "evaluation-trace",
        [actual.as_trace_batch()],
        config_sha256=CONFIG_HASH,
        checkpoint_sha256=CHECKPOINT_HASH,
        model_sha256=model_hash,
    )
    frozen = P05EvaluationFrozenParameters(
        dataset="XJTU",
        model_seed=42,
        validation_trace_semantic_sha256=validation.semantic_sha256,
        evaluation_trace_semantic_sha256=evaluation.semantic_sha256,
    )
    return validation, evaluation, frozen, [actual]


def _create(bundle_dir, trace_pair, *, frozen=None):
    validation, evaluation, registered, actual = trace_pair
    return create_p05_c2_c3_evaluation_bundle(
        bundle_dir,
        validation_trace_package=validation.package_dir,
        evaluation_trace_package=evaluation.package_dir,
        actual_intervention_results=actual,
        frozen=registered if frozen is None else frozen,
    )


def test_bundle_is_hashed_create_only_reusable_and_unadjudicated(
    tmp_path, trace_pair
) -> None:
    target = tmp_path / "evaluation-bundle"
    created = _create(target, trace_pair)
    bytes_before = {path.name: path.read_bytes() for path in target.iterdir()}
    reused = _create(target, trace_pair)

    assert created.status == "created"
    assert reused.status == "reused"
    assert {path.name: path.read_bytes() for path in target.iterdir()} == bytes_before
    manifest = json.loads(created.manifest_path.read_text(encoding="utf-8"))
    c3 = json.loads(created.c3_path.read_text(encoding="utf-8"))
    validation, evaluation, frozen, actual = trace_pair
    assert manifest["schema_name"] == "p05.c2_c3_evaluation_bundle"
    assert manifest["conclusion_control"] == {
        "c2_intervention_source": "actual_same_checkpoint_forwards",
        "claim_decisions": "not_performed",
        "decisive": False,
        "operational_wording_gate": "not_evaluated",
        "predictive_cost_gate": "not_evaluated",
        "status": "computed_unadjudicated",
    }
    assert manifest["frozen_parameters"]["model_seed"] == frozen.model_seed
    assert manifest["inputs"]["validation_trace"]["semantic_sha256"] == (
        validation.semantic_sha256
    )
    assert manifest["inputs"]["evaluation_trace"]["semantic_sha256"] == (
        evaluation.semantic_sha256
    )
    assert manifest["inputs"]["actual_interventions"]["source"] == (
        "p05.actual_same_checkpoint_forward"
    )
    assert manifest["inputs"]["actual_interventions"]["chunks"][0][
        "semantic_sha256"
    ] == actual[0].semantic_sha256
    assert manifest["content"]["arrays_sha256"] == hashlib.sha256(
        created.arrays_path.read_bytes()
    ).hexdigest()
    assert manifest["content"]["c3_sha256"] == hashlib.sha256(
        created.c3_path.read_bytes()
    ).hexdigest()
    assert c3["decisive"] is False
    assert c3["claim_decisions"] == "not_performed"
    assert c3["interpretation"]["inference"] == "not_performed"
    assert c3["interpretation"]["confirmatory_sign_tests"] == "not_performed"
    assert c3["interpretation"]["cross_seed_aggregation"] == "not_performed"
    assert c3["interpretation"]["operational_wording_gate"]["evaluated"] is False
    assert c3["interpretation"]["predictive_cost_gate"]["evaluated"] is False
    assert set(c3["evaluation"]["methods"]) == {"trace", "R0", "R1", "R2", "R3"}

    with np.load(created.arrays_path, allow_pickle=False) as arrays:
        assert all(not arrays[name].dtype.hasobject for name in arrays.files)
        assert arrays["c2_deletion_logits"].shape == (40, 10, 2)
        assert arrays["c2_shuffle_permutations"].shape == (40, 32, 10)
        assert arrays["c2_shuffle_predictive_jsd"].shape == (40, 32)
        assert arrays["record_id"].dtype.kind == "U"
        assert arrays["window_start"].dtype == np.dtype("<i8")
        assert arrays["c2_shuffle_membership_invariant"].all()
        assert arrays["c2_shuffle_antecedent_invariant"].all()
        assert arrays["c2_shuffle_firing_invariant"].all()
        assert arrays["c2_actual_forward_bound"].all()
        assert arrays["c2_actual_deletion_membership_invariant_pass"].all()
        assert arrays["c2_actual_deletion_antecedent_invariant_pass"].all()
        assert arrays["c2_actual_deletion_firing_invariant_pass"].all()
        assert np.max(arrays["c2_actual_vs_offline_deletion_logits_max_abs"]) <= 1e-6
        assert np.max(arrays["c2_actual_vs_offline_shuffle_logits_max_abs"]) <= 1e-6
        assert np.isfinite(arrays["c3_score_trace"]).all()
    assert not list(tmp_path.glob(".evaluation-bundle.*.tmp"))


def test_existing_bundle_conflict_preserves_original_bytes(tmp_path, trace_pair) -> None:
    target = tmp_path / "evaluation-bundle"
    _create(target, trace_pair)
    before = {path.name: path.read_bytes() for path in target.iterdir()}
    unexpected = target / "unexpected.txt"
    unexpected.write_text("conflict", encoding="utf-8")

    with pytest.raises(FileExistsError, match="conflict"):
        _create(target, trace_pair)

    assert {
        name: (target / name).read_bytes() for name in before
    } == before
    assert unexpected.read_text(encoding="utf-8") == "conflict"


def test_missing_or_unbound_trace_fails_without_output(tmp_path, trace_pair) -> None:
    target = tmp_path / "missing-input-bundle"
    validation, evaluation, frozen, actual = trace_pair
    with pytest.raises(FileNotFoundError, match="trace package"):
        create_p05_c2_c3_evaluation_bundle(
            target,
            validation_trace_package=tmp_path / "absent-trace",
            evaluation_trace_package=evaluation.package_dir,
            actual_intervention_results=actual,
            frozen=frozen,
        )
    assert not target.exists()

    mismatched = replace(frozen, evaluation_trace_semantic_sha256="f" * 64)
    with pytest.raises(ValueError, match="semantic hash does not match"):
        create_p05_c2_c3_evaluation_bundle(
            target,
            validation_trace_package=validation.package_dir,
            evaluation_trace_package=evaluation.package_dir,
            actual_intervention_results=actual,
            frozen=mismatched,
        )
    assert not target.exists()

    with pytest.raises(ValueError, match="actual same-checkpoint"):
        create_p05_c2_c3_evaluation_bundle(
            target,
            validation_trace_package=validation.package_dir,
            evaluation_trace_package=evaluation.package_dir,
            actual_intervention_results=[],
            frozen=frozen,
        )
    assert not target.exists()


def test_tampered_trace_hash_fails_without_output(tmp_path, trace_pair) -> None:
    target = tmp_path / "tampered-input-bundle"
    evaluation_npz = trace_pair[1].npz_path
    original_bytes = evaluation_npz.read_bytes()
    evaluation_npz.write_bytes(original_bytes + b"tamper")
    try:
        with pytest.raises(ValueError, match="NPZ hash"):
            _create(target, trace_pair)
    finally:
        evaluation_npz.write_bytes(original_bytes)

    assert not target.exists()


def test_atomic_write_failure_removes_staging_directory(
    tmp_path, trace_pair, monkeypatch
) -> None:
    target = tmp_path / "atomic-failure-bundle"

    def fail_savez(*args, **kwargs):
        raise RuntimeError("injected NPZ failure")

    monkeypatch.setattr(evaluation_bundle.np, "savez", fail_savez)
    with pytest.raises(RuntimeError, match="injected NPZ failure"):
        _create(target, trace_pair)

    assert not target.exists()
    assert not list(tmp_path.glob(".atomic-failure-bundle.*.tmp"))


def test_pipeline_decisive_m_binds_actual_forwards_before_bundle(
    tmp_path, monkeypatch
) -> None:
    config_path = tmp_path / "config_snapshot.yaml"
    checkpoint_path = tmp_path / "best.ckpt"
    config_path.write_bytes(b"config")
    checkpoint_path.write_bytes(b"checkpoint")
    network = torch.nn.Linear(2, 2)
    stable_batches = []
    for batch_index in range(2):
        stable_batches.append(
            {
                "x": torch.zeros((1, 4, 2), dtype=torch.float32),
                "y": torch.tensor([batch_index % 2]),
                "sample_id": [f"test-{batch_index}"],
                "record_id": [f"record-{batch_index}"],
                "group_id": [f"bearing-{batch_index}"],
                "window_start": torch.tensor([batch_index * 4]),
                "window_end": torch.tensor([batch_index * 4 + 4]),
                "ignored_metadata": ["not forwarded"],
            }
        )

    class _Factory:
        def get_dataloader(self, partition):
            return stable_batches if partition == "test" else [stable_batches[0]]

    export_calls = []

    def fake_trace_export(package_dir, **kwargs):
        partition = Path(package_dir).name
        export_calls.append((partition, kwargs))
        semantic = ("d" if partition == "val" else "e") * 64
        return SimpleNamespace(
            package_dir=Path(package_dir),
            manifest_path=Path(package_dir) / "manifest.json",
            manifest_sha256=("1" if partition == "val" else "2") * 64,
            semantic_sha256=semantic,
            status="created",
        )

    actual_calls = []

    def fake_actual_runner(**kwargs):
        actual_calls.append(kwargs)
        return SimpleNamespace(chunk=len(actual_calls))

    bundle_calls = []

    def fake_bundle(bundle_dir, **kwargs):
        bundle_calls.append((Path(bundle_dir), kwargs))
        return SimpleNamespace(
            arrays_sha256="3" * 64,
            manifest_path=Path(bundle_dir) / "manifest.json",
            manifest_sha256="4" * 64,
            semantic_sha256="5" * 64,
            status="created",
        )

    monkeypatch.setattr(pipeline_module, "export_p05_loader_trace", fake_trace_export)
    monkeypatch.setattr(
        pipeline_module,
        "run_p05_same_checkpoint_interventions",
        fake_actual_runner,
    )
    monkeypatch.setattr(
        pipeline_module,
        "create_p05_c2_c3_evaluation_bundle",
        fake_bundle,
    )
    contract = SimpleNamespace(
        arm_id="P05-M",
        dataset="XJTU",
        phase="decisive",
        seed=42,
    )
    records, evaluation = pipeline_module._export_registered_p05_traces(
        task=SimpleNamespace(network=network),
        data_factory=_Factory(),
        run_path=tmp_path / "run",
        config_snapshot_path=config_path,
        checkpoint_path=checkpoint_path,
        execution_stage="fit_validate_test",
        expected_window_size=4,
        experiment_contract=contract,
    )

    assert [name for name, _ in export_calls] == ["val", "test"]
    assert set(records) == {"val", "test"}
    assert len(actual_calls) == 2
    assert all(call["require_cuda"] is True for call in actual_calls)
    assert all(set(call["batch"]) == {
        "x",
        "y",
        "sample_id",
        "record_id",
        "group_id",
        "window_start",
        "window_end",
    } for call in actual_calls)
    assert all(call["provenance"].dataset == "XJTU" for call in actual_calls)
    assert len(bundle_calls) == 1
    _, bundle_kwargs = bundle_calls[0]
    assert bundle_kwargs["actual_intervention_results"] == [
        SimpleNamespace(chunk=1),
        SimpleNamespace(chunk=2),
    ]
    assert bundle_kwargs["frozen"].model_seed == 42
    assert bundle_kwargs["frozen"].validation_trace_semantic_sha256 == "d" * 64
    assert bundle_kwargs["frozen"].evaluation_trace_semantic_sha256 == "e" * 64
    assert evaluation["scientific_status"] == "computed_unadjudicated"
