"""CPU unit tests for the standalone P08 E1 runner boundaries."""

from __future__ import annotations

from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from src.p08_evidence.e1_data import generate_rate_copies
from src.p08_evidence.e1_model import arm_spec
from src.p08_evidence.e1_runner import (
    DEFAULT_CONFIG,
    Candidate,
    FitResult,
    NormalizationRecord,
    RawRecord,
    _bg_candidates,
    _canonical_formal_launch_command,
    _half_up_duration_points,
    _load_config,
    _m_reuse_candidate,
    _prepare_records,
    _select_candidate,
    _test_payload,
    _training_batches,
    _validate_formal_launch_command,
)


def _normalization() -> NormalizationRecord:
    return NormalizationRecord(
        ordered_input_hash="0" * 64,
        sample_count=1,
        mean=0.0,
        standard_deviation=1.0,
        algorithm="test",
        dtype="float64",
        iteration_order=("test",),
        canonical_json_sha256="1" * 64,
    )


def _raw_records() -> list[RawRecord]:
    result = []
    for class_id in range(4):
        for copy in generate_rate_copies(class_id, 0):
            result.append(
                RawRecord(
                    class_id=class_id,
                    underlying_id=0,
                    split=copy.split,
                    original_rate_hz=copy.sample_rate_hz,
                    signal_handle=f"signal-{class_id}",
                    samples=copy.samples,
                    sample_sha256=copy.sample_sha256,
                )
            )
    return sorted(
        result,
        key=lambda record: (
            record.class_id,
            record.underlying_id,
            record.original_rate_hz,
        ),
    )


def _fit_result(candidate: Candidate, score: float) -> FitResult:
    return FitResult(
        candidate=candidate,
        state_dict={"value": torch.tensor([score])},
        validation_score=score,
        validation_by_rate={},
        pretrain_best_epoch=1,
        pretrain_best_validation_score=score,
        finetune_best_epoch=1,
        epoch_rows=[],
        elapsed_seconds=1.0,
        total_parameters=1,
        trainable_parameters=1,
    )


class E1RunnerTests(unittest.TestCase):
    def test_formal_command_is_canonical_and_tamper_evident(self) -> None:
        output_root = DEFAULT_CONFIG.parent / "evidence-output"
        with patch.dict(
            "os.environ",
            {
                "CUDA_VISIBLE_DEVICES": "0",
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "MPLCONFIGDIR": "/tmp/p08-mpl",
            },
            clear=False,
        ):
            command = _canonical_formal_launch_command(
                seed=42,
                config_path=DEFAULT_CONFIG,
                output_root=output_root,
            )
            self.assertTrue(command.startswith("conda run -n LQ_signal "))
            self.assertEqual(
                _validate_formal_launch_command(
                    command,
                    seed=42,
                    config_path=DEFAULT_CONFIG,
                    output_root=output_root,
                ),
                command,
            )
            with self.assertRaisesRegex(ValueError, "canonical process contract"):
                _validate_formal_launch_command(
                    command.replace("--seed 42", "--seed 123"),
                    seed=42,
                    config_path=DEFAULT_CONFIG,
                    output_root=output_root,
                )

    def test_resolved_config_hash_and_fit_grid_are_locked(self) -> None:
        config, digest = _load_config(Path(DEFAULT_CONFIG), require_approved=False)
        self.assertEqual(len(digest), 64)
        self.assertEqual(config["candidate_selection"]["total_fits_per_seed"], 8)
        self.assertEqual(len(_bg_candidates(config)), 3)
        approved, approved_digest = _load_config(
            Path(DEFAULT_CONFIG), require_approved=True
        )
        self.assertEqual(approved["protocol"]["id"], "P08-LOSO-v1.1")
        self.assertEqual(approved_digest, digest)

    def test_bg_uses_exact_rationals_and_frozen_center_crop_lengths(self) -> None:
        records = _raw_records()
        for numerator, denominator, expected_points in (
            (51_200, 3, 341),
            (25_600, 1, 512),
            (51_200, 1, 1024),
        ):
            self.assertEqual(
                _half_up_duration_points(numerator, denominator), expected_points
            )
            spec = arm_spec(
                "P08-BG",
                global_resample_numerator_hz=numerator,
                global_resample_denominator=denominator,
            )
            prepared = _prepare_records(records, _normalization(), spec)
            self.assertEqual({record.samples.size for record in prepared}, {expected_points})
            self.assertEqual(
                {
                    (
                        record.model_rate_numerator_hz,
                        record.model_rate_denominator,
                    )
                    for record in prepared
                },
                {(numerator, denominator)},
            )
            for record in prepared:
                audit = record.preprocessing
                self.assertEqual(
                    audit["crop_stop"] - audit["crop_start"], expected_points
                )
                self.assertGreaterEqual(audit["resampled_points_before_crop"], expected_points)

    def test_rate_and_class_balanced_sampler_and_label_seal(self) -> None:
        spec = arm_spec("P08-NC")
        prepared = _prepare_records(_raw_records(), _normalization(), spec)
        batches = list(
            _training_batches(
                prepared,
                seed=42,
                stage_index=0,
                epoch=1,
                batch_size=64,
                batches_per_rate=2,
            )
        )
        self.assertEqual(len(batches), 12)
        self.assertEqual(
            {rate: sum(batch[0].original_rate_hz == rate for batch in batches) for rate in (12_000, 20_480, 25_600, 48_000, 50_000, 200_000)},
            {12_000: 2, 20_480: 2, 25_600: 2, 48_000: 2, 50_000: 2, 200_000: 2},
        )
        for batch in batches:
            self.assertEqual(
                [sum(record.class_id == class_id for record in batch) for class_id in range(4)],
                [16, 16, 16, 16],
            )
        payload, labels = _test_payload(prepared)
        self.assertEqual(len(labels), 4)
        self.assertFalse(hasattr(payload[0], "class_id"))

    def test_candidate_ties_and_method_reuse_are_fail_closed(self) -> None:
        short = Candidate(
            "DN-5ms",
            arm_spec("P08-DN", duration_ms=5.0),
            5,
            1,
            5,
        )
        long = Candidate(
            "DN-15ms",
            arm_spec("P08-DN", duration_ms=15.0),
            15,
            1,
            15,
        )
        selected = _select_candidate([_fit_result(long, 0.75), _fit_result(short, 0.75)])
        self.assertEqual(selected.candidate.candidate_id, "DN-5ms")
        method = _m_reuse_candidate(selected)
        self.assertEqual(method.numeric_value, selected.candidate.numeric_value)
        self.assertEqual(method.spec.arm_id, "P08-M")


if __name__ == "__main__":
    unittest.main()
