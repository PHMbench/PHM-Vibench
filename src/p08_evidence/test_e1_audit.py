"""CPU-only negative-path tests for the independent P08 E1 artifact audit."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import hmac
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from src.p08_evidence import e1_audit
from src.p08_evidence.e1_data import (
    CLASS_IDS,
    EVALUATION_RATES_HZ,
    GENERATOR_VERSION,
    PROTOCOL_ID,
    canonical_json_sha256,
    split_underlying_ids,
)
from src.p08_evidence.runtime import (
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(
            canonical_json_bytes(row).decode("utf-8") + "\n" for row in rows
        ),
        encoding="utf-8",
    )


def _write_parquet(path: Path, rows: list[dict[str, object]]) -> None:
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd", version="2.6")


def _rewrite_artifact_manifest(root: Path) -> dict[str, str]:
    manifest = root / "artifact_manifest.sha256"
    if manifest.exists():
        manifest.unlink()
    entries = {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
    manifest.write_text(
        "".join(f"{digest}  {relative}\n" for relative, digest in entries.items()),
        encoding="utf-8",
    )
    return entries


def _time(base: datetime, seconds: int) -> str:
    return (base + timedelta(seconds=seconds)).isoformat(timespec="microseconds")


def _signal_handle(class_id: int, underlying_id: int) -> str:
    return canonical_json_sha256(
        {
            "generator_version": GENERATOR_VERSION,
            "class_id": class_id,
            "underlying_id": underlying_id,
        }
    )


def _build_complete_fixture(root: Path) -> dict[str, object]:
    base = datetime(2026, 8, 1, tzinfo=timezone.utc)
    train_pairs: set[tuple[int, int]] = set()
    validation_pairs: set[tuple[int, int]] = set()
    test_pairs: set[tuple[int, int]] = set()
    by_class: dict[str, dict[str, list[int]]] = {}
    for class_id in CLASS_IDS:
        split = split_underlying_ids(class_id)
        by_class[str(class_id)] = {
            "train": list(split["train"]),
            "validation": list(split["validation"]),
        }
        train_pairs.update((class_id, value) for value in split["train"])
        validation_pairs.update((class_id, value) for value in split["validation"])
        test_pairs.update((class_id, value) for value in split["test"])
    token_salt = bytes.fromhex("ab" * 32)
    preliminary_mapping = []
    for class_id, underlying_id in test_pairs:
        source_handle = _signal_handle(class_id, underlying_id)
        target_handle = hmac.new(
            token_salt, source_handle.encode("utf-8"), sha256
        ).hexdigest()
        preliminary_mapping.append(
            {
                "target_handle": target_handle,
                "class_id": class_id,
                "source_signal_handle": source_handle,
                "source_underlying_id": underlying_id,
            }
        )
    preliminary_mapping.sort(key=lambda row: str(row["target_handle"]))
    sealed_entries = [
        dict(row, opaque_signal_index=index)
        for index, row in enumerate(preliminary_mapping)
    ]
    sealed_by_handle = {
        str(row["target_handle"]): row for row in sealed_entries
    }
    target_handles = sorted(sealed_by_handle)
    frozen_test_pair_hash = e1_audit.partition_id_set_sha256(tuple(test_pairs))
    mapping_commitment = sha256_bytes(canonical_json_bytes(sealed_entries))
    target_handle_set_hash = sha256_bytes(canonical_json_bytes(target_handles))
    fold = {
        "protocol_id": PROTOCOL_ID,
        "experiment_id": "P08-E1",
        "training_and_validation_underlying_ids_by_class": by_class,
        "target_test": {
            "state": "sealed_before_checkpoint_finalization",
            "underlying_signal_count": len(test_pairs),
            "frozen_test_pair_set_sha256": frozen_test_pair_hash,
            "labels_or_class_counts_visible": False,
        },
        "rate_copies_stay_with_underlying_split": True,
        "evaluation_rates_hz": list(EVALUATION_RATES_HZ),
    }
    _write_json(root / "fold_manifest.json", fold)

    partition_hashes = {
        "train": e1_audit.partition_id_set_sha256(tuple(train_pairs)),
        "validation": e1_audit.partition_id_set_sha256(tuple(validation_pairs)),
        "test": e1_audit.partition_id_set_sha256(tuple(test_pairs)),
    }
    _write_json(
        root / "partition_disjointness.json",
        {
            "status": "pass",
            "unit": "class_id_plus_underlying_id_before_rate_copy_generation",
            "counts": {
                "train": len(train_pairs),
                "validation": len(validation_pairs),
                "test": len(test_pairs),
            },
            "overlap_counts": {
                "train_vs_validation": 0,
                "train_vs_test": 0,
                "validation_vs_test": 0,
            },
            e1_audit.PARTITION_HASH_FIELD: partition_hashes,
            "all_rate_copies_inherit_underlying_split": True,
        },
    )

    source = e1_audit._independent_source_recompute(fold)
    validation_hash = "1" * 64
    pretest = {
        "protocol_id": PROTOCOL_ID,
        "generator_version": GENERATOR_VERSION,
        "target_state": "sealed",
        "train": {
            "bank_sha256": source["train_bank_sha256"],
            "rate_copy_count": source["rate_copy_count"],
            "sample_count": source["sample_count"],
        },
        "validation": {
            "bank_sha256": validation_hash,
            "rate_copy_count": len(validation_pairs) * len(EVALUATION_RATES_HZ),
            "sample_count": 1,
        },
    }
    _write_json(root / "data_manifest_pretest.json", pretest)
    _write_json(
        root / "loader_partition_log.json",
        {
            "training_process_visible_splits": ["train", "validation"],
            "target_dataset_object_count": 0,
            "target_label_table_count": 0,
            "train_rate_copy_count": source["rate_copy_count"],
            "validation_rate_copy_count": len(validation_pairs)
            * len(EVALUATION_RATES_HZ),
            "train_bank_sha256": source["train_bank_sha256"],
            "validation_bank_sha256": validation_hash,
            "status": "pass",
        },
    )
    _write_json(root / "normalization.json", source["normalization"])
    _write_json(
        root / "normalization_recompute.json",
        {
            "status": "pass",
            "original": source["normalization"],
            "recomputed": source["normalization"],
            "exact_mapping_equality": True,
            "regenerated_train_bank_sha256": source["train_bank_sha256"],
        },
    )
    _write_json(
        root / "source_sampling_rate_table.json",
        {
            "status": "pass",
            "scope": "analytic_train_split_only",
            "rate_copy_counts_by_hz": source["rate_counts"],
            "stored_shared_cutoff_hz": 6000.0,
            "recomputed_shared_cutoff_hz": 6000.0,
        },
    )
    (root / "resolved_config.yaml").write_text(
        yaml.safe_dump(
            {
                "base_config": {
                    "protocol": {
                        "id": PROTOCOL_ID,
                        "source_path": e1_audit.PROTOCOL_SOURCE_PATH,
                        "source_sha256": e1_audit.PROTOCOL_SOURCE_SHA256,
                    },
                    "data": {"generator": {"source_shared_band_hz": 6000.0}},
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    contract = {
        "dataset_id_prompt_rejected": {
            "rejected": True,
            "exception_type": "ValueError",
            "exception_message": "dataset identity is forbidden",
            "forbidden_argument": "dataset_ids",
            "batch_size": 2,
        },
        "system_selected_head_rejected": {
            "rejected": True,
            "exception_type": "ValueError",
            "exception_message": "system-selected heads are forbidden",
            "forbidden_argument": "system_id",
            "batch_size": 2,
        },
        "sampling_rate_length_mismatch_rejected": {
            "rejected": True,
            "exception_type": "ValueError",
            "exception_message": "sampling rate vector length differs from batch",
            "forbidden_argument": "sampling_rate_hz",
            "batch_size": 2,
            "metadata_count": 1,
        },
    }
    _write_json(root / "contract_checks.json", contract)
    _write_jsonl(
        root / "epoch_log.jsonl",
        [{"candidate_id": "DN-10ms", "stage": "finetune", "epoch": 1}],
    )
    _write_jsonl(
        root / "collation_assertion_log.jsonl",
        [
            {
                "candidate_id": "DN-10ms",
                "stage": "finetune",
                "epoch": 1,
                "batch_count": 6,
                "batch_original_rate_homogeneous": True,
                "rate_batch_counts": {
                    str(rate): 1 for rate in EVALUATION_RATES_HZ
                },
                "class_example_counts": {str(class_id): 96 for class_id in CLASS_IDS},
                "metadata_length_mismatch_count": 0,
                "sampling_rate_scalar_broadcast_count": 0,
            }
        ],
    )
    selection_rows = [
        {
            "protocol_id": PROTOCOL_ID,
            "experiment_id": "P08-E1",
            "arm_id": "P08-DN",
            "model_seed": 42,
            "candidate_id": "DN-10ms",
            "validation_balanced_accuracy_equal_rates_then_classes": 0.75,
            "selected": True,
            "selection_criterion": "validation_balanced_accuracy_equal_rates_then_classes",
            "completed_at_utc": _time(base, 1),
        }
    ]
    _write_jsonl(root / "selection_trace.jsonl", selection_rows)
    (root / "selected.ckpt").write_bytes(b"independent-audit-checkpoint")
    checkpoint_digest = sha256_file(root / "selected.ckpt")
    (root / "checkpoint.sha256").write_text(
        checkpoint_digest + "\n", encoding="utf-8"
    )

    entries: list[dict[str, object]] = []
    predictions: list[dict[str, object]] = []
    for handle in target_handles:
        sealed_entry = sealed_by_handle[handle]
        class_id = int(sealed_entry["class_id"])
        opaque_signal_index = int(sealed_entry["opaque_signal_index"])
        for rate in EVALUATION_RATES_HZ:
            entries.append(
                {
                    "signal_handle": handle,
                    "opaque_signal_index": opaque_signal_index,
                    "original_rate_hz": rate,
                    "model_rate_numerator_hz": rate,
                    "model_rate_denominator": 1,
                    "sample_count": rate // 50,
                }
            )
            row: dict[str, object] = {
                "protocol_id": PROTOCOL_ID,
                "experiment_id": "P08-E1",
                "arm_id": "P08-DN",
                "model_seed": 42,
                "signal_handle": handle,
                "opaque_signal_index": opaque_signal_index,
                "original_rate_hz": rate,
                "model_rate_numerator_hz": rate,
                "model_rate_denominator": 1,
                "predicted_class": class_id,
            }
            for supported_class in CLASS_IDS:
                row[f"p_class_{supported_class}"] = float(
                    supported_class == class_id
                )
            predictions.append(row)
    target = {
        "protocol_id": PROTOCOL_ID,
        "experiment_id": "P08-E1",
        "arm_id": "P08-DN",
        "model_seed": 42,
        "unsealed_after": [
            "selection_trace_finalized",
            "checkpoint_sha256_written",
        ],
        "checkpoint_sha256": checkpoint_digest,
        "broker_manifest_sha256": "3" * 64,
        "shared_native_payload_sha256": "4" * 64,
        "normalization_sha256": sha256_file(root / "normalization.json"),
        "labels_present": False,
        "source_identity_present": False,
        "target_handle_set_sha256": target_handle_set_hash,
        "frozen_test_pair_set_sha256": frozen_test_pair_hash,
        "mapping_commitment_sha256": mapping_commitment,
        "entries": entries,
        "written_at_utc": _time(base, 5),
    }
    _write_json(root / "target_eval_manifest.json", target)
    _write_json(
        root / "target_decode_log.json",
        {
            "protocol_id": PROTOCOL_ID,
            "experiment_id": "P08-E1",
            "status": "running",
            "arm_id": "P08-DN",
            "model_seed": 42,
            "checkpoint_sha256": checkpoint_digest,
            "normalization_sha256": sha256_file(root / "normalization.json"),
            "target_eval_manifest_sha256": sha256_file(
                root / "target_eval_manifest.json"
            ),
            "broker_manifest_sha256": "3" * 64,
            "shared_native_payload_sha256": "4" * 64,
            "selected_arm_spec": {"arm_id": "P08-DN"},
            "labels_present": False,
            "source_identity_present": False,
            "target_handle_set_sha256": target_handle_set_hash,
            "frozen_test_pair_set_sha256": frozen_test_pair_hash,
            "mapping_commitment_sha256": mapping_commitment,
            "decode_started_at_utc": _time(base, 3),
            "decode_completed_at_utc": _time(base, 4),
        },
    )
    _write_parquet(
        root / "window_predictions.parquet",
        [dict(row, window_index=0) for row in predictions],
    )
    _write_parquet(root / "record_predictions.parquet", predictions)
    prediction_digest = sha256_file(root / "record_predictions.parquet")
    (root / "prediction.sha256").write_text(
        prediction_digest + "\n", encoding="utf-8"
    )
    sealed_payload = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "experiment_id": "P08-E1",
        "status": "sealed",
        "tokenization": "HMAC-SHA256",
        "token_salt_hex": token_salt.hex(),
        "token_salt_visibility": "sealed_scorer_only",
        "entry_count": len(sealed_entries),
        "entries": sealed_entries,
        "created_at_utc": _time(base, 3),
    }
    _write_json(
        root / e1_audit.SEALED_LABEL_COPY_NAME,
        sealed_payload,
    )
    sealed_copy_digest = sha256_file(root / e1_audit.SEALED_LABEL_COPY_NAME)
    scored = [
        dict(
            row,
            class_id=sealed_by_handle[str(row["signal_handle"])]["class_id"],
        )
        for row in predictions
    ]
    _write_parquet(root / "scored_records.parquet", scored)
    scored_digest = sha256_file(root / "scored_records.parquet")
    _write_json(
        root / "metrics.json",
        {
            "protocol_id": PROTOCOL_ID,
            "experiment_id": "P08-E1",
            "arm_id": "P08-DN",
            "model_seed": 42,
            "prediction_sha256_before_label_join": prediction_digest,
            "scored_records_sha256": scored_digest,
            "sealed_label_table_sha256": sealed_copy_digest,
        },
    )
    _write_json(
        root / "scorer_join_log.json",
        {
            "protocol_id": PROTOCOL_ID,
            "experiment_id": "P08-E1",
            "status": "running",
            "phase": "scored_pending_final_audit",
            "arm_id": "P08-DN",
            "model_seed": 42,
            "checkpoint_sha256": checkpoint_digest,
            "prediction_sha256_before_label_join": prediction_digest,
            "prediction_written_at_utc": _time(base, 6),
            "all_four_prediction_hashes_verified_before_label_open": True,
            "all_prediction_hashes_gate_completed_at_utc": _time(base, 7),
            "sealed_label_table_opened_at_utc": _time(base, 8),
            "sealed_label_table_after_prediction_hashes_sha256": sealed_copy_digest,
            "sealed_label_table_copied_at_utc": _time(base, 9),
            "scorer_joined_at_utc": _time(base, 10),
            "scorer_completed_at_utc": _time(base, 11),
            "scored_records_sha256": scored_digest,
            "metrics_sha256": sha256_file(root / "metrics.json"),
        },
    )
    command = (
        "conda run -n LQ_signal env CUDA_VISIBLE_DEVICES=0 "
        "python -m src.p08_evidence.e1_runner run-seed --seed 42"
    )
    (root / "command.txt").write_text(command + "\n", encoding="utf-8")
    environment_document = {
        "schema": "p08.environment-snapshot/v1",
        "environment": {
            "name": "LQ_signal",
            "prefix_disclosure": "redacted; all recorded paths are prefix-relative",
        },
        "loaded_modules": [
            {
                "module": module,
                "loaded_path": f"lib/python/site-packages/{module}/__init__.py",
                "sha256": f"{index + 5:x}" * 64,
            }
            for index, module in enumerate(("numpy", "pyarrow", "scipy", "torch"))
        ],
        "privacy_contract": {
            "absolute_prefix_recorded": False,
            "channel_or_package_urls_recorded": False,
            "environment_variables_recorded": False,
            "host_or_user_identifiers_recorded": False,
            "timestamps_recorded": False,
        },
        "counts": {"loaded_modules": 4},
    }
    _write_json(root / "environment.yml", environment_document)
    protocol_source = (
        Path(__file__).resolve().parents[4]
        / "paper/experiments/config_bridge.yaml"
    )
    protocol_bytes = protocol_source.read_bytes()
    if sha256_bytes(protocol_bytes) != e1_audit.PROTOCOL_SOURCE_SHA256:
        raise RuntimeError("test fixture protocol source hash changed")
    protocol_snapshot = root / e1_audit.PROTOCOL_SNAPSHOT_PATH
    protocol_snapshot.parent.mkdir(parents=True, exist_ok=True)
    protocol_snapshot.write_bytes(protocol_bytes)
    source_manifest_base = {
        "files": [
            {
                "path": e1_audit.PROTOCOL_SOURCE_PATH,
                "bytes": len(protocol_bytes),
                "sha256": e1_audit.PROTOCOL_SOURCE_SHA256,
            }
        ]
    }
    source_manifest_digest = sha256_bytes(
        canonical_json_bytes(source_manifest_base)
    )
    _write_json(
        root / "source_manifest.json",
        {
            **source_manifest_base,
            "source_manifest_sha256": source_manifest_digest,
        },
    )
    _write_json(
        root / "training_input_schema.json",
        {
            "allowed_payloads": [
                "source_train_signal",
                "source_train_label",
                "source_validation_signal",
                "source_validation_label",
            ],
            "model_input_fields": ["signal", "sampling_rate_hz"],
            "forbidden_fields": [
                "dataset_id",
                "system_id",
                "target_signal",
                "target_label",
            ],
            "target_object_constructed": False,
        },
    )
    preflight = {
        "status": "pass",
        "mode": "cuda",
        "physical_gpu_indices": [0],
        "visible_to_physical_gpu_map": {"0": 0},
        "cuda_visible_devices": "0",
        "cuda_device_count": 1,
        "cuda_device_names": ["fixture GPU"],
        "world_size": 1,
        "local_world_size": 1,
        "trainer_strategy": "auto",
        "multi_gpu": False,
    }
    _write_json(
        root / "provenance.json",
        {
            "protocol_id": PROTOCOL_ID,
            "experiment_id": "P08-E1",
            "arm_id": "P08-DN",
            "model_seed": 42,
            "mode": "formal_evidence",
            "command": command,
            "conda_environment": "LQ_signal",
            "git_commit": "a" * 40,
            "config_sha256": "b" * 64,
            "data_sha256": "c" * 64,
            "gpu_preflight": preflight,
            "evaluation_gpu_preflight": preflight,
            "environment_yml_sha256": sha256_file(root / "environment.yml"),
            "source_manifest_sha256": source_manifest_digest,
            "protocol_source_sha256": e1_audit.PROTOCOL_SOURCE_SHA256,
            "checkpoint_sha256": checkpoint_digest,
            "checkpoint_written_at_utc": _time(base, 2),
            "completed_at_utc": _time(base, 12),
        },
    )
    _write_json(
        root / "run_status.json",
        {
            "status": "completed",
            "mode": "formal_evidence",
            "protocol_id": PROTOCOL_ID,
            "protocol_source_sha256": e1_audit.PROTOCOL_SOURCE_SHA256,
            "experiment_id": "P08-E1",
            "arm_id": "P08-DN",
            "model_seed": 42,
            "selected_candidate_id": "DN-10ms",
            "checkpoint_sha256": checkpoint_digest,
            "metrics_sha256": sha256_file(root / "metrics.json"),
            "completed_at_utc": _time(base, 12),
        },
    )
    (root / "stdout.log").write_text("fixture completed\n", encoding="utf-8")
    (root / "stderr.log").write_text("", encoding="utf-8")

    # Establish a strict filesystem event order independently of JSON claims.
    epoch_ns = 1_800_000_000_000_000_000
    ordered = (
        "selection_trace.jsonl",
        "selected.ckpt",
        "checkpoint.sha256",
        "target_eval_manifest.json",
        "target_decode_log.json",
        "window_predictions.parquet",
        "record_predictions.parquet",
        "prediction.sha256",
        e1_audit.SEALED_LABEL_COPY_NAME,
        "scored_records.parquet",
        "metrics.json",
        "scorer_join_log.json",
    )
    for offset, relative in enumerate(ordered, start=1):
        timestamp = epoch_ns + offset * 1_000_000_000
        os.utime(root / relative, ns=(timestamp, timestamp))
    digests = _rewrite_artifact_manifest(root)
    return {"source": source, "artifact_digests": digests}


class TestIndependentE1Audit(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._class_tmp = tempfile.TemporaryDirectory(prefix="p08-e1-audit-")
        cls.template = Path(cls._class_tmp.name) / "template"
        cls.template.mkdir()
        fixture = _build_complete_fixture(cls.template)
        cls.source = fixture["source"]

    @classmethod
    def tearDownClass(cls) -> None:
        cls._class_tmp.cleanup()

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="p08-e1-audit-case-")
        self.root = Path(self._tmp.name) / "run"
        shutil.copytree(self.template, self.root)
        self.digests = _rewrite_artifact_manifest(self.root)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _audit_with_cached_source(self) -> dict[str, object]:
        with mock.patch.object(
            e1_audit,
            "_independent_source_recompute",
            return_value=deepcopy(self.source),
        ):
            return e1_audit.audit_run_artifacts(
                self.root, artifact_digests=self.digests
            )

    def _set_pending_status(self, *, phase: str | None) -> None:
        status = json.loads(
            (self.root / "run_status.json").read_text(encoding="utf-8")
        )
        status["status"] = "running"
        if phase is None:
            status.pop("phase", None)
        else:
            status["phase"] = phase
        _write_json(self.root / "run_status.json", status)
        self.digests = _rewrite_artifact_manifest(self.root)

    def _audit_pending_with_cached_source(self) -> dict[str, object]:
        with mock.patch.object(
            e1_audit,
            "_independent_source_recompute",
            return_value=deepcopy(self.source),
        ):
            return e1_audit.audit_run_artifacts(
                self.root,
                artifact_digests=self.digests,
                expected_run_state="scored_pending_final_audit",
            )

    def test_complete_materialized_fixture_passes_all_l01_l11(self) -> None:
        result = e1_audit.audit_run_artifacts(
            self.root, artifact_digests=self.digests
        )
        self.assertEqual(result["status"], "pass", result)
        self.assertEqual(result["audited_run_state"], "completed")
        self.assertEqual([item["item_id"] for item in result["items"]], [f"L{value:02d}" for value in range(1, 12)])
        self.assertTrue(all(item["status"] == "pass" for item in result["items"]))

    def test_exact_scored_pending_final_audit_state_passes_pending_audit(self) -> None:
        self._set_pending_status(phase="scored_pending_final_audit")
        result = self._audit_pending_with_cached_source()
        self.assertEqual(result["status"], "pass", result)
        self.assertEqual(
            result["audited_run_state"], "scored_pending_final_audit"
        )
        by_id = {item["item_id"]: item for item in result["items"]}
        self.assertEqual(by_id["L11"]["status"], "pass")
        self.assertEqual(
            by_id["L11"]["observed_value"]["audited_run_state"],
            "scored_pending_final_audit",
        )

    def test_bare_running_or_wrong_pending_phase_fails_closed(self) -> None:
        for phase in (None, "checkpoint_finalized_source_only"):
            with self.subTest(phase=phase):
                self._set_pending_status(phase=phase)
                result = self._audit_pending_with_cached_source()
                by_id = {item["item_id"]: item for item in result["items"]}
                self.assertEqual(result["status"], "fail")
                self.assertEqual(by_id["L11"]["status"], "fail")
                self.assertIn(
                    "exact scored_pending_final_audit state",
                    str(by_id["L11"]["observed_value"]),
                )

    def test_default_completed_audit_rejects_pending_state(self) -> None:
        self._set_pending_status(phase="scored_pending_final_audit")
        result = self._audit_with_cached_source()
        by_id = {item["item_id"]: item for item in result["items"]}
        self.assertEqual(result["audited_run_state"], "completed")
        self.assertEqual(result["status"], "fail")
        self.assertEqual(by_id["L11"]["status"], "fail")

    def test_unknown_expected_run_state_fails_before_artifact_acceptance(self) -> None:
        result = e1_audit.audit_run_artifacts(
            self.root,
            artifact_digests=self.digests,
            expected_run_state="running",  # type: ignore[arg-type]
        )
        self.assertEqual(result["status"], "fail")
        self.assertIn(
            "expected_run_state",
            str(result["artifact_integrity"]["errors"]),
        )

    def test_empty_artifact_digests_fail_closed(self) -> None:
        result = e1_audit.audit_run_artifacts(self.root, artifact_digests={})
        self.assertEqual(result["status"], "fail")
        self.assertIn("non-empty", str(result["artifact_integrity"]["errors"]))

    def test_missing_required_file_and_missing_hash_fail_closed(self) -> None:
        (self.root / "normalization.json").unlink()
        result = e1_audit.audit_run_artifacts(
            self.root, artifact_digests=self.digests
        )
        self.assertEqual(result["status"], "fail")
        self.assertIn("normalization.json", str(result["artifact_integrity"]["errors"]))

        self.digests = _rewrite_artifact_manifest(self.root)
        self.digests.pop("metrics.json")
        result = e1_audit.audit_run_artifacts(
            self.root, artifact_digests=self.digests
        )
        self.assertEqual(result["status"], "fail")
        self.assertIn("metrics.json", str(result["artifact_integrity"]["errors"]))

    def test_tampered_hashed_artifact_fails_before_semantic_checks(self) -> None:
        with (self.root / "normalization.json").open("a", encoding="utf-8") as handle:
            handle.write(" ")
        result = e1_audit.audit_run_artifacts(
            self.root, artifact_digests=self.digests
        )
        self.assertEqual(result["status"], "fail")
        self.assertIn("hash mismatch", str(result["artifact_integrity"]["errors"]))

    def test_bare_boolean_constant_self_attestation_is_rejected(self) -> None:
        _write_json(
            self.root / "contract_checks.json",
            {
                "dataset_id_prompt_rejected": True,
                "system_selected_head_rejected": True,
                "sampling_rate_length_mismatch_rejected": True,
            },
        )
        self.digests = _rewrite_artifact_manifest(self.root)
        result = self._audit_with_cached_source()
        by_id = {item["item_id"]: item for item in result["items"]}
        self.assertEqual(result["status"], "fail")
        self.assertEqual(by_id["L06"]["status"], "fail")
        self.assertEqual(by_id["L07"]["status"], "fail")
        self.assertEqual(by_id["L08"]["status"], "fail")
        self.assertIn("structured exception log", str(by_id["L06"]["observed_value"]))

    def test_constant_pass_partition_with_wrong_hashes_is_rejected(self) -> None:
        partition = json.loads(
            (self.root / "partition_disjointness.json").read_text(encoding="utf-8")
        )
        partition["status"] = "pass"
        partition[e1_audit.PARTITION_HASH_FIELD] = {
            "train": "0" * 64,
            "validation": "0" * 64,
            "test": "0" * 64,
        }
        _write_json(self.root / "partition_disjointness.json", partition)
        self.digests = _rewrite_artifact_manifest(self.root)
        result = self._audit_with_cached_source()
        by_id = {item["item_id"]: item for item in result["items"]}
        self.assertEqual(by_id["L01"]["status"], "fail")
        self.assertIn("hashes differ", str(by_id["L01"]["observed_value"]))

    def test_selection_false_test_flag_is_still_a_forbidden_test_field(self) -> None:
        rows = [
            json.loads(line)
            for line in (self.root / "selection_trace.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line
        ]
        rows[0]["test_metric_present"] = False
        _write_jsonl(self.root / "selection_trace.jsonl", rows)
        # Preserve a valid early mtime after rewriting the trace.
        timestamp = 1_800_000_001_000_000_000
        os.utime(self.root / "selection_trace.jsonl", ns=(timestamp, timestamp))
        self.digests = _rewrite_artifact_manifest(self.root)
        result = self._audit_with_cached_source()
        by_id = {item["item_id"]: item for item in result["items"]}
        self.assertEqual(by_id["L05"]["status"], "fail")
        self.assertIn("test_metric_present", str(by_id["L05"]["observed_value"]))

    def test_unlabeled_prediction_parquet_with_class_id_is_rejected(self) -> None:
        table = pq.read_table(self.root / "record_predictions.parquet")
        table = table.append_column(
            "class_id", pq.read_table(self.root / "scored_records.parquet")["class_id"]
        )
        pq.write_table(table, self.root / "record_predictions.parquet", compression="zstd")
        timestamp = 1_800_000_006_000_000_000
        os.utime(self.root / "record_predictions.parquet", ns=(timestamp, timestamp))
        self.digests = _rewrite_artifact_manifest(self.root)
        result = self._audit_with_cached_source()
        by_id = {item["item_id"]: item for item in result["items"]}
        self.assertEqual(by_id["L11"]["status"], "fail")
        self.assertIn("class_id", str(by_id["L11"]["observed_value"]))

    def test_evaluator_target_manifest_leaking_source_id_is_rejected(self) -> None:
        target = json.loads(
            (self.root / "target_eval_manifest.json").read_text(encoding="utf-8")
        )
        target["entries"][0]["source_underlying_id"] = 7
        _write_json(self.root / "target_eval_manifest.json", target)
        timestamp = 1_800_000_004_000_000_000
        os.utime(self.root / "target_eval_manifest.json", ns=(timestamp, timestamp))
        self.digests = _rewrite_artifact_manifest(self.root)
        result = self._audit_with_cached_source()
        by_id = {item["item_id"]: item for item in result["items"]}
        self.assertEqual(by_id["L01"]["status"], "fail")
        self.assertIn("source/label fields", str(by_id["L01"]["observed_value"]))

    def test_tampered_hmac_mapping_is_rejected_even_when_rehashed(self) -> None:
        sealed_path = self.root / e1_audit.SEALED_LABEL_COPY_NAME
        sealed = json.loads(sealed_path.read_text(encoding="utf-8"))
        sealed["entries"][0]["target_handle"] = "0" * 64
        _write_json(sealed_path, sealed)
        timestamp = 1_800_000_009_000_000_000
        os.utime(sealed_path, ns=(timestamp, timestamp))
        self.digests = _rewrite_artifact_manifest(self.root)
        result = self._audit_with_cached_source()
        by_id = {item["item_id"]: item for item in result["items"]}
        self.assertEqual(by_id["L01"]["status"], "fail")
        self.assertIn("HMAC target handle", str(by_id["L01"]["observed_value"]))

    def test_environment_privacy_tamper_fails_even_with_updated_file_hash(self) -> None:
        environment_path = self.root / "environment.yml"
        environment = json.loads(environment_path.read_text(encoding="utf-8"))
        environment["privacy_contract"]["timestamps_recorded"] = True
        _write_json(environment_path, environment)
        provenance_path = self.root / "provenance.json"
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        provenance["environment_yml_sha256"] = sha256_file(environment_path)
        _write_json(provenance_path, provenance)
        self.digests = _rewrite_artifact_manifest(self.root)
        result = self._audit_with_cached_source()
        by_id = {item["item_id"]: item for item in result["items"]}
        self.assertEqual(by_id["L10"]["status"], "fail")
        self.assertIn("privacy contract", str(by_id["L10"]["observed_value"]))

    def test_superseded_v1_protocol_string_fails_with_valid_artifact_hashes(self) -> None:
        resolved_path = self.root / "resolved_config.yaml"
        resolved = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
        resolved["base_config"]["protocol"]["id"] = "P08-LOSO-v1"
        resolved_path.write_text(
            yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
        )
        self.digests = _rewrite_artifact_manifest(self.root)
        result = self._audit_with_cached_source()
        by_id = {item["item_id"]: item for item in result["items"]}
        self.assertEqual(by_id["L10"]["status"], "fail")
        self.assertIn("not P08-LOSO-v1.1", str(by_id["L10"]["observed_value"]))


if __name__ == "__main__":
    unittest.main()
