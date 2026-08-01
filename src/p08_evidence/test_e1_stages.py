"""CPU-only tests for the formal P08 E1 post-checkpoint stages."""

from __future__ import annotations

from hashlib import sha256
import io
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from src.p08_evidence.e1_data import (
    EVALUATION_RATES_HZ,
    GENERATOR_VERSION,
    PROTOCOL_ID,
    canonical_json_sha256,
    samples_sha256,
    split_underlying_ids,
)
from src.p08_evidence.e1_model import arm_spec
from src.p08_evidence.environment import snapshot_text
from src.p08_evidence import e1_runner as runner
from src.p08_evidence.e1_stages import (
    ARMS,
    BROKER_MANIFEST_NAME,
    PAYLOAD_NAME,
    SEALED_LABEL_NAME,
    SEEDS,
    NativeUnlabeledRecord,
    _canonical_stage_launch_command,
    _evaluate_seed_core,
    _prepare_payload_for_spec,
    _verify_source_campaign,
    _validate_launch_command,
    evaluate_seed,
    finalize_seed,
    prepare_target,
    score_seed,
)
from src.p08_evidence.runtime import DevicePreflightRecord


def _cpu_preflight() -> DevicePreflightRecord:
    return DevicePreflightRecord(
        status="pass",
        mode="cpu",
        physical_gpu_indices=(),
        visible_to_physical_gpu_map={},
        cuda_visible_devices="",
        cuda_device_count=0,
        cuda_device_names=(),
        world_size=1,
        local_world_size=1,
        trainer_strategy="auto",
        multi_gpu=False,
    )


def _gpu_preflight(physical: int) -> DevicePreflightRecord:
    return DevicePreflightRecord(
        status="pass",
        mode="cuda",
        physical_gpu_indices=(physical,),
        visible_to_physical_gpu_map={"0": physical},
        cuda_visible_devices=str(physical),
        cuda_device_count=1,
        cuda_device_names=("test-gpu",),
        world_size=1,
        local_world_size=1,
        trainer_strategy="auto",
        multi_gpu=False,
    )


def _normalization() -> runner.NormalizationRecord:
    base = {
        "ordered_input_hash": "1" * 64,
        "sample_count": 10_000,
        "mean": 0.25,
        "standard_deviation": 2.0,
        "algorithm": "deterministic_float64_welford_population_ddof_0",
        "dtype": "float64_fit_float64_apply_then_float32_cast",
        "iteration_order": (
            "class_id_sorted",
            "underlying_id_sorted",
            "exact_sampling_rate_hz_sorted",
            "sample_index_ascending",
        ),
    }
    return runner.NormalizationRecord(
        **base, canonical_json_sha256=canonical_json_sha256(base)
    )


def _selection_rows(arm_id: str, seed: int) -> list[dict[str, object]]:
    if arm_id == "P08-DN":
        candidates = [
            ("DN-5ms", arm_spec(arm_id, duration_ms=5.0)),
            ("DN-10ms", arm_spec(arm_id, duration_ms=10.0)),
            ("DN-15ms", arm_spec(arm_id, duration_ms=15.0)),
        ]
        selected_id = "DN-5ms"
    elif arm_id == "P08-M":
        candidates = [("M-reuse-DN-5ms", arm_spec(arm_id, duration_ms=5.0))]
        selected_id = "M-reuse-DN-5ms"
    elif arm_id == "P08-BG":
        candidates = [
            (
                "BG-51200over3Hz",
                arm_spec(
                    arm_id,
                    global_resample_numerator_hz=51_200,
                    global_resample_denominator=3,
                ),
            ),
            (
                "BG-25600Hz",
                arm_spec(
                    arm_id,
                    global_resample_numerator_hz=25_600,
                    global_resample_denominator=1,
                ),
            ),
            (
                "BG-51200Hz",
                arm_spec(
                    arm_id,
                    global_resample_numerator_hz=51_200,
                    global_resample_denominator=1,
                ),
            ),
        ]
        selected_id = "BG-25600Hz"
    else:
        candidates = [("NC-fixed-128-points", arm_spec(arm_id))]
        selected_id = "NC-fixed-128-points"
    rows: list[dict[str, object]] = []
    for candidate_id, spec in candidates:
        row: dict[str, object] = {
            "protocol_id": PROTOCOL_ID,
            "experiment_id": "P08-E1",
            "arm_id": arm_id,
            "model_seed": seed,
            "candidate_id": candidate_id,
            "arm_spec": spec.to_dict(),
            "selected": candidate_id == selected_id,
        }
        if arm_id == "P08-M":
            row["representation_reuse_source"] = "DN-5ms"
            row["additional_representation_selection_trials"] = 0
        rows.append(row)
    return rows


def _write_source_campaign(root: Path) -> None:
    normalization = _normalization()
    protocol_source_digest = "a" * 64
    source_manifest = runner._source_manifest(runner.DEFAULT_CONFIG)
    source_manifest_text = (
        json.dumps(
            source_manifest,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )
    environment_text = snapshot_text()
    environment_digest = sha256(environment_text.encode("utf-8")).hexdigest()
    for seed in SEEDS:
        for arm_id in ARMS:
            run_id = f"P08-E1-{arm_id}-seed{seed}"
            run_root = root / run_id
            run_root.mkdir(parents=True)
            selection = _selection_rows(arm_id, seed)
            selected = next(row for row in selection if row["selected"] is True)
            checkpoint_buffer = io.BytesIO()
            torch.save(
                {
                    "protocol_id": PROTOCOL_ID,
                    "experiment_id": "P08-E1",
                    "arm_id": arm_id,
                    "model_seed": seed,
                    "candidate_id": selected["candidate_id"],
                    "arm_spec": selected["arm_spec"],
                    "validation_score": 0.5,
                    "finetune_best_epoch": 1,
                    "state_dict": {"fixture": torch.tensor([1.0])},
                },
                checkpoint_buffer,
            )
            checkpoint = checkpoint_buffer.getvalue()
            checkpoint_digest = sha256(checkpoint).hexdigest()
            (run_root / "selected.ckpt").write_bytes(checkpoint)
            (run_root / "checkpoint.sha256").write_text(
                checkpoint_digest + "\n", encoding="ascii"
            )
            (run_root / "selection_trace.jsonl").write_text(
                "".join(
                    json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                    for row in selection
                ),
                encoding="utf-8",
            )
            (run_root / "normalization.json").write_text(
                json.dumps(normalization.to_dict(), sort_keys=True), encoding="utf-8"
            )
            (run_root / "source_manifest.json").write_text(
                source_manifest_text, encoding="utf-8"
            )
            (run_root / "environment.yml").write_text(
                environment_text, encoding="utf-8"
            )
            (run_root / "resolved_config.yaml").write_text(
                json.dumps(
                    {
                        "base_config": {
                            "protocol": {
                                "source_sha256": protocol_source_digest
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            status = {
                "status": "running",
                "phase": "checkpoint_finalized_source_only",
                "mode": "formal_evidence",
                "protocol_id": PROTOCOL_ID,
                "protocol_source_sha256": protocol_source_digest,
                "experiment_id": "P08-E1",
                "arm_id": arm_id,
                "model_seed": seed,
                "selected_candidate_id": selected["candidate_id"],
                "checkpoint_sha256": checkpoint_digest,
                "target_object_constructed": False,
            }
            (run_root / "run_status.json").write_text(
                json.dumps(status, sort_keys=True), encoding="utf-8"
            )
            provenance = {
                "protocol_id": PROTOCOL_ID,
                "protocol_source_sha256": protocol_source_digest,
                "experiment_id": "P08-E1",
                "arm_id": arm_id,
                "model_seed": seed,
                "mode": "formal_evidence_source_only_training",
                "target_object_constructed": False,
                "checkpoint_sha256": checkpoint_digest,
                "checkpoint_written_at_utc": "2026-01-01T00:00:00+00:00",
                "command": (
                    "conda run -n LQ_signal env CUDA_VISIBLE_DEVICES=0 "
                    "python -m src.p08_evidence.e1_runner run-seed --seed "
                    f"{seed}"
                ),
                "conda_environment": "LQ_signal",
                "gpu_preflight": _gpu_preflight(0).to_dict(),
                "source_manifest_sha256": source_manifest[
                    "source_manifest_sha256"
                ],
                "environment_yml_sha256": environment_digest,
            }
            (run_root / "provenance.json").write_text(
                json.dumps(provenance, sort_keys=True), encoding="utf-8"
            )


def _fake_target_records() -> list[runner.RawRecord]:
    records: list[runner.RawRecord] = []
    lengths = {
        12_000: 3,
        20_480: 4,
        25_600: 5,
        48_000: 10,
        50_000: 10,
        200_000: 40,
    }
    for class_id in range(4):
        for underlying_id in split_underlying_ids(class_id)["test"]:
            source_handle = canonical_json_sha256(
                {
                    "generator_version": GENERATOR_VERSION,
                    "class_id": class_id,
                    "underlying_id": underlying_id,
                }
            )
            for rate in EVALUATION_RATES_HZ:
                count = lengths[rate]
                phase = class_id + underlying_id / 100.0 + rate / 1_000_000.0
                samples = np.linspace(phase, phase + 0.5, count, dtype=np.float64)
                records.append(
                    runner.RawRecord(
                        class_id=class_id,
                        underlying_id=underlying_id,
                        split="test",
                        original_rate_hz=rate,
                        signal_handle=source_handle,
                        samples=samples,
                        sample_sha256=samples_sha256(samples),
                    )
                )
    return records


class E1FormalStageTests(unittest.TestCase):
    def test_canonical_stage_command_rejects_seed_path_and_control_tampering(self) -> None:
        run_root = Path("/tmp/p08-stage-command-runs")
        broker_root = Path("/tmp/p08-stage-command-broker")
        command = _canonical_stage_launch_command(
            stage="evaluate-seed",
            seed=42,
            run_root=run_root,
            broker_root=broker_root,
        )
        self.assertEqual(
            _validate_launch_command(
                command,
                expected_stage="evaluate-seed",
                seed=42,
                run_root=run_root,
                broker_root=broker_root,
            ),
            command,
        )
        for tampered in (
            command.replace("--seed 42", "--seed 123"),
            command.replace(str(run_root), "/tmp/other-run-root"),
            command + " ';' echo unsafe",
        ):
            with self.assertRaises(ValueError):
                _validate_launch_command(
                    tampered,
                    expected_stage="evaluate-seed",
                    seed=42,
                    run_root=run_root,
                    broker_root=broker_root,
                )

    def test_prepare_target_gates_twenty_runs_and_physically_separates_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_root = root / "runs"
            run_root.mkdir()
            _write_source_campaign(run_root)
            broker_root = root / "broker"
            sealed_root = root / "sealed"
            with (
                patch(
                    "src.p08_evidence.e1_stages.strict_single_gpu_preflight",
                    return_value=_cpu_preflight(),
                ),
                patch(
                    "src.p08_evidence.e1_stages.runner._load_raw_records",
                    return_value=_fake_target_records(),
                ),
                patch(
                    "src.p08_evidence.e1_stages.secrets.token_bytes",
                    return_value=b"t" * 32,
                ),
            ):
                result = prepare_target(
                    run_root=run_root,
                    broker_root=broker_root,
                    sealed_root=sealed_root,
                    launch_command=None,
                )

            self.assertEqual(result["status"], "running")
            self.assertEqual(result["source_run_count"], 20)
            self.assertTrue((broker_root / PAYLOAD_NAME).is_file())
            self.assertFalse((broker_root / SEALED_LABEL_NAME).exists())
            self.assertTrue((sealed_root / SEALED_LABEL_NAME).is_file())
            with np.load(broker_root / PAYLOAD_NAME, allow_pickle=False) as archive:
                self.assertFalse(
                    any("label" in name or "class" in name for name in archive.files)
                )
                self.assertEqual(len(archive["signal_handles"]), 1_224)
                self.assertEqual(len(np.unique(archive["signal_handles"])), 204)
            broker_text = (broker_root / BROKER_MANIFEST_NAME).read_text()
            self.assertNotIn(str(sealed_root), broker_text)
            self.assertIn("sealed_label_location_disclosed_to_evaluator", broker_text)
            sealed = json.loads((sealed_root / SEALED_LABEL_NAME).read_text())
            self.assertEqual(len(sealed["entries"]), 204)
            self.assertEqual(
                {entry["class_id"] for entry in sealed["entries"]}, {0, 1, 2, 3}
            )
            for seed in SEEDS:
                for arm_id in ARMS:
                    status = json.loads(
                        (run_root / f"P08-E1-{arm_id}-seed{seed}" / "run_status.json").read_text()
                    )
                    self.assertEqual(status["status"], "running")
            with self.assertRaises(FileExistsError):
                prepare_target(
                    run_root=run_root,
                    broker_root=broker_root,
                    sealed_root=sealed_root,
                    launch_command=None,
                )

    def test_source_gate_fails_before_target_decode_on_checkpoint_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_root = Path(temporary) / "runs"
            run_root.mkdir()
            _write_source_campaign(run_root)
            tampered = run_root / "P08-E1-P08-BG-seed456" / "selected.ckpt"
            tampered.write_bytes(b"tampered")
            with self.assertRaisesRegex(RuntimeError, "checkpoint hash"):
                _verify_source_campaign(run_root)

    def test_source_gate_rejects_post_training_source_manifest_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_root = Path(temporary) / "runs"
            run_root.mkdir()
            _write_source_campaign(run_root)
            manifest = run_root / "P08-E1-P08-DN-seed42" / "source_manifest.json"
            value = json.loads(manifest.read_text())
            value["files"][0]["sha256"] = "0" * 64
            manifest.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "self-hash|differs"):
                _verify_source_campaign(run_root)

    def test_non_evidence_temporary_fixture_walks_all_four_stage_boundaries(self) -> None:
        """Schema/lifecycle integration only; synthetic outputs are never evidence."""

        def fixture_prepare(
            records: list[NativeUnlabeledRecord], spec: object
        ) -> list[runner.UnlabeledInferenceRecord]:
            result = []
            for record in records:
                if getattr(spec, "arm_id") == "P08-BG":
                    numerator = int(getattr(spec, "global_resample_numerator_hz"))
                    denominator = int(getattr(spec, "global_resample_denominator"))
                else:
                    numerator, denominator = record.original_rate_hz, 1
                values = np.asarray(record.samples, dtype=np.float32)
                values.setflags(write=False)
                result.append(
                    runner.UnlabeledInferenceRecord(
                        underlying_id=record.opaque_signal_index,
                        original_rate_hz=record.original_rate_hz,
                        signal_handle=record.signal_handle,
                        model_rate_numerator_hz=numerator,
                        model_rate_denominator=denominator,
                        samples=values,
                    )
                )
            return result

        def fixture_inference(
            fit: runner.FitResult,
            records: list[runner.UnlabeledInferenceRecord],
            seed: int,
            device: torch.device,
        ) -> list[dict[str, object]]:
            del device
            rows: list[dict[str, object]] = []
            for record in records:
                predicted = int(record.underlying_id % 4)
                probabilities = np.full(4, 0.1, dtype=np.float64)
                probabilities[predicted] = 0.7
                row: dict[str, object] = {
                    "protocol_id": PROTOCOL_ID,
                    "experiment_id": "P08-E1",
                    "arm_id": fit.candidate.spec.arm_id,
                    "model_seed": seed,
                    "signal_handle": record.signal_handle,
                    "underlying_id": record.underlying_id,
                    "original_rate_hz": record.original_rate_hz,
                    "model_rate_numerator_hz": record.model_rate_numerator_hz,
                    "model_rate_denominator": record.model_rate_denominator,
                    "predicted_class": predicted,
                }
                for class_id, value in enumerate(probabilities):
                    row[f"p_class_{class_id}"] = float(value)
                for feature_index in range(128):
                    row[f"feature_{feature_index:03d}"] = float(
                        record.underlying_id / 204.0
                        + record.original_rate_hz / 1_000_000.0
                    )
                rows.append(row)
            return rows

        audit_calls: list[str] = []

        def fixture_audit(
            run_root: Path,
            *,
            artifact_digests: dict[str, str],
            expected_run_state: str,
        ) -> dict[str, object]:
            self.assertTrue(artifact_digests)
            self.assertEqual(
                set(artifact_digests),
                {
                    path.relative_to(run_root).as_posix()
                    for path in run_root.rglob("*")
                    if path.is_file() and path.name != "artifact_manifest.sha256"
                },
            )
            audit_calls.append(expected_run_state)
            return {
                "protocol_id": PROTOCOL_ID,
                "experiment_id": "P08-E1",
                "status": "pass",
                "audited_run_state": expected_run_state,
                "artifact_integrity": {"status": "pass", "errors": []},
                "items": [
                    {"item_id": f"L{index:02d}", "status": "pass"}
                    for index in range(1, 12)
                ],
            }

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_root = root / "runs"
            run_root.mkdir()
            _write_source_campaign(run_root)
            broker_root = root / "broker"
            sealed_root = root / "sealed"
            with (
                patch(
                    "src.p08_evidence.e1_stages.strict_single_gpu_preflight",
                    return_value=_cpu_preflight(),
                ),
                patch(
                    "src.p08_evidence.e1_stages.runner._load_raw_records",
                    return_value=_fake_target_records(),
                ),
                patch(
                    "src.p08_evidence.e1_stages.secrets.token_bytes",
                    return_value=b"s" * 32,
                ),
            ):
                prepare_target(
                    run_root=run_root,
                    broker_root=broker_root,
                    sealed_root=sealed_root,
                    launch_command=None,
                )
            with patch(
                "src.p08_evidence.e1_stages._prepare_payload_for_spec",
                side_effect=fixture_prepare,
            ):
                evaluated = _evaluate_seed_core(
                    seed=42,
                    run_root=run_root,
                    broker_root=broker_root,
                    launch_command=(
                        "conda run -n LQ_signal env CUDA_VISIBLE_DEVICES=0 "
                        "python -m src.p08_evidence.e1_stages evaluate-seed --seed 42"
                    ),
                    preflight=_gpu_preflight(0),
                    device=torch.device("cpu"),
                    inference=fixture_inference,
                )
            self.assertEqual(evaluated["status"], "running")
            for arm_id in ARMS:
                import pyarrow.parquet as pq

                schema = pq.read_schema(
                    run_root / f"P08-E1-{arm_id}-seed42" / "record_predictions.parquet"
                )
                self.assertIn("opaque_signal_index", schema.names)
                self.assertNotIn("underlying_id", schema.names)
                self.assertNotIn("class_id", schema.names)

            with patch(
                "src.p08_evidence.e1_stages.strict_single_gpu_preflight",
                return_value=_cpu_preflight(),
            ):
                scored = score_seed(
                    seed=42,
                    run_root=run_root,
                    broker_root=broker_root,
                    sealed_root=sealed_root,
                    launch_command=None,
                )
                finalized = finalize_seed(
                    seed=42,
                    run_root=run_root,
                    launch_command=None,
                    audit_function=fixture_audit,
                )
            self.assertEqual(scored["status"], "running")
            self.assertEqual(finalized["status"], "completed")
            self.assertEqual(
                audit_calls,
                [
                    "scored_pending_final_audit",
                    "scored_pending_final_audit",
                    "completed",
                ]
                * 4,
            )
            for arm_id in ARMS:
                path = run_root / f"P08-E1-{arm_id}-seed42"
                status = json.loads((path / "run_status.json").read_text())
                self.assertEqual(status["status"], "completed")
                self.assertTrue((path / "leakage_audit.json").is_file())
                self.assertTrue(
                    (path / "sealed_label_table_after_prediction_hashes.json").is_file()
                )

    def test_bg_evaluation_transform_preserves_exact_selected_rational(self) -> None:
        records = []
        for index, rate in enumerate(EVALUATION_RATES_HZ):
            point_count = int(round(rate * 0.02))
            samples = np.sin(
                2.0 * np.pi * 200.0 * np.arange(point_count, dtype=np.float64) / rate
            )
            records.append(
                NativeUnlabeledRecord(
                    opaque_signal_index=index,
                    original_rate_hz=rate,
                    signal_handle=sha256(f"target:{index}".encode()).hexdigest(),
                    samples=samples,
                )
            )
        spec = arm_spec(
            "P08-BG",
            global_resample_numerator_hz=51_200,
            global_resample_denominator=3,
        )
        prepared = _prepare_payload_for_spec(records, spec)
        self.assertEqual({record.samples.size for record in prepared}, {341})
        self.assertEqual(
            {
                (record.model_rate_numerator_hz, record.model_rate_denominator)
                for record in prepared
            },
            {(51_200, 3)},
        )
        self.assertNotIn("sealed_root", inspect.signature(evaluate_seed).parameters)

    def test_gpu2_is_rejected_before_evaluator_can_read_broker(self) -> None:
        broker_loader = Mock()
        with (
            patch(
                "src.p08_evidence.e1_stages.strict_single_gpu_preflight",
                return_value=_gpu_preflight(2),
            ),
            patch(
                "src.p08_evidence.e1_stages._load_broker_manifest", broker_loader
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "forbids GPU 2"):
                evaluate_seed(
                    seed=42,
                    run_root=Path("/tmp/not-read-runs"),
                    broker_root=Path("/tmp/not-read-broker"),
                    launch_command=None,
                )
        broker_loader.assert_not_called()

    def test_scorer_never_opens_labels_when_prediction_gate_fails(self) -> None:
        label_loader = Mock()
        broker = {
            "sealed_label_table_sha256": "a" * 64,
            "payload_sha256": "b" * 64,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            broker_root = root / "broker"
            sealed_root = root / "sealed"
            with (
                patch(
                    "src.p08_evidence.e1_stages.strict_single_gpu_preflight",
                    return_value=_cpu_preflight(),
                ),
                patch(
                    "src.p08_evidence.e1_stages._load_broker_manifest",
                    return_value=(broker, "c" * 64),
                ),
                patch(
                    "src.p08_evidence.e1_stages._verify_all_prediction_hashes_before_labels",
                    side_effect=RuntimeError("prediction hash gate failed"),
                ),
                patch(
                    "src.p08_evidence.e1_stages._load_sealed_labels", label_loader
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "prediction hash gate"):
                    score_seed(
                        seed=42,
                        run_root=root / "runs",
                        broker_root=broker_root,
                        sealed_root=sealed_root,
                        launch_command=None,
                    )
        label_loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
