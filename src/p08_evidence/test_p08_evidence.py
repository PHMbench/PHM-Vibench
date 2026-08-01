from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.p08_evidence.metrics import (
    DEFAULT_E1_RATES_HZ,
    DEFAULT_E1_SEEDS,
    E1Predictions,
    aggregate_probabilities_by_group,
    bootstrap_e1_paired_contrast,
    e1_prediction_consistency,
    e1_representation_distance,
    e1_worst_rate_balanced_accuracy,
    jensen_shannon_divergence,
    record_classification_metrics,
)
from src.p08_evidence.runtime import EvidenceWriter, strict_single_gpu_preflight


def _e1_tables() -> tuple[E1Predictions, E1Predictions]:
    labels: list[int] = []
    signal_ids: list[int] = []
    seeds: list[int] = []
    rates: list[int] = []
    mechanism_probabilities: list[np.ndarray] = []
    baseline_probabilities: list[np.ndarray] = []
    for class_id in range(4):
        for underlying_id in range(2):
            for seed in DEFAULT_E1_SEEDS:
                for rate_index, rate in enumerate(DEFAULT_E1_RATES_HZ):
                    labels.append(class_id)
                    signal_ids.append(underlying_id)
                    seeds.append(seed)
                    rates.append(rate)
                    correct = np.full(4, 0.02, dtype=np.float64)
                    correct[class_id] = 0.94
                    mechanism_probabilities.append(correct)
                    if rate_index == 0:
                        wrong = np.full(4, 0.02, dtype=np.float64)
                        wrong[(class_id + 1) % 4] = 0.94
                        baseline_probabilities.append(wrong)
                    else:
                        baseline_probabilities.append(correct)
    columns = {
        "labels": labels,
        "signal_ids": signal_ids,
        "model_seeds": seeds,
        "rates_hz": rates,
    }
    return (
        E1Predictions.from_columns(
            probabilities=np.stack(mechanism_probabilities), **columns
        ),
        E1Predictions.from_columns(
            probabilities=np.stack(baseline_probabilities), **columns
        ),
    )


class DevicePreflightTests(unittest.TestCase):
    def test_accepts_one_allowed_physical_gpu_and_records_mapping(self) -> None:
        result = strict_single_gpu_preflight(
            trainer_strategy="auto",
            environment={
                "CUDA_VISIBLE_DEVICES": "3",
                "WORLD_SIZE": "1",
                "LOCAL_WORLD_SIZE": "1",
            },
            cuda_device_count=1,
            cuda_device_names=["NVIDIA GeForce RTX 4090"],
        )
        self.assertEqual(result.physical_gpu_indices, (3,))
        self.assertEqual(result.visible_to_physical_gpu_map, {"0": 3})
        self.assertFalse(result.multi_gpu)

    def test_rejects_forbidden_multi_gpu_and_ddp_states(self) -> None:
        cases = [
            ({"CUDA_VISIBLE_DEVICES": "2"}, "physical GPU index 2"),
            ({"CUDA_VISIBLE_DEVICES": "0,1"}, "exactly one"),
            (
                {"CUDA_VISIBLE_DEVICES": "0", "WORLD_SIZE": "2"},
                "forbids multi-process",
            ),
        ]
        for environment, message in cases:
            with self.subTest(environment=environment), self.assertRaisesRegex(
                RuntimeError, message
            ):
                strict_single_gpu_preflight(
                    environment=environment,
                    cuda_device_count=1,
                )
        with self.assertRaisesRegex(RuntimeError, "distributed trainer strategy"):
            strict_single_gpu_preflight(
                trainer_strategy="ddp",
                environment={"CUDA_VISIBLE_DEVICES": "0"},
                cuda_device_count=1,
            )

    def test_requires_an_explicit_auditable_mask(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "must be set explicitly"):
            strict_single_gpu_preflight(environment={}, cuda_device_count=1)


class ArtifactWriterTests(unittest.TestCase):
    def test_writes_atomic_artifacts_and_self_excluding_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            writer = EvidenceWriter(temporary_directory)
            path, digest = writer.write_json("metrics.json", {"value": 0.5})
            self.assertTrue(path.is_file())
            self.assertEqual(len(digest), 64)
            with self.assertRaises(FileExistsError):
                writer.write_json("metrics.json", {"value": 0.6})

            manifest_path, _ = writer.write_sha256_manifest()
            manifest = manifest_path.read_text(encoding="utf-8")
            self.assertIn("  metrics.json\n", manifest)
            self.assertNotIn("artifact_manifest.sha256", manifest)
            self.assertFalse(any(path.parent.glob("*.tmp")))

    def test_provenance_is_validated_and_cannot_escape_run_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            writer = EvidenceWriter(temporary_directory)
            preflight = strict_single_gpu_preflight(
                environment={"CUDA_VISIBLE_DEVICES": "0"},
                cuda_device_count=1,
            )
            provenance = {
                "command": "conda run -n LQ_signal python -m src.p08_runner",
                "conda_environment": "LQ_signal",
                "git_commit": "deadbeef",
                "config_sha256": "a" * 64,
                "data_sha256": "b" * 64,
                "gpu_preflight": preflight,
            }
            path, _ = writer.write_provenance(provenance)
            stored = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(stored["gpu_preflight"]["physical_gpu_indices"], [0])
            with self.assertRaisesRegex(ValueError, "stay below run root"):
                writer.write_text("../escape.txt", "no")


class RecordMetricTests(unittest.TestCase):
    def test_aggregates_windows_and_scores_full_four_class_probabilities(self) -> None:
        windows = np.asarray(
            [
                [0.8, 0.1, 0.05, 0.05],
                [0.6, 0.2, 0.1, 0.1],
                [0.1, 0.7, 0.1, 0.1],
                [0.1, 0.5, 0.3, 0.1],
            ]
        )
        grouped = aggregate_probabilities_by_group(
            probabilities=windows,
            group_ids=["r0", "r0", "r1", "r1"],
            labels=[0, 0, 1, 1],
        )
        np.testing.assert_allclose(
            grouped["probabilities"],
            [[0.7, 0.15, 0.075, 0.075], [0.1, 0.6, 0.2, 0.1]],
        )
        metrics = record_classification_metrics(
            probabilities=grouped["probabilities"],
            labels=grouped["labels"],
            supported_classes=[0, 1],
        )
        self.assertEqual(metrics["balanced_accuracy"], 1.0)
        self.assertIsNone(metrics["four_class_macro_f1"])
        self.assertFalse(metrics["renormalized_over_supported_classes"])

    def test_rejects_logits_and_missing_prespecified_class(self) -> None:
        with self.assertRaisesRegex(ValueError, "probabilities"):
            record_classification_metrics(
                probabilities=[[2.0, 1.0, 0.0, 0.0]],
                labels=[0],
                supported_classes=[0],
            )
        with self.assertRaisesRegex(ValueError, "has no true observations"):
            record_classification_metrics(
                probabilities=[[0.9, 0.05, 0.03, 0.02]],
                labels=[0],
                supported_classes=[0, 1],
            )


class E1MetricTests(unittest.TestCase):
    def test_jsd_is_divergence_with_natural_log(self) -> None:
        value = jensen_shannon_divergence([1, 0, 0, 0], [0, 1, 0, 0])
        self.assertAlmostEqual(value, math.log(2.0), places=12)
        self.assertEqual(
            jensen_shannon_divergence([0.7, 0.1, 0.1, 0.1], [0.7, 0.1, 0.1, 0.1]),
            0.0,
        )

    def test_e1_metrics_obey_rate_and_signal_hierarchy(self) -> None:
        mechanism, baseline = _e1_tables()
        mechanism_consistency = e1_prediction_consistency(mechanism)
        baseline_consistency = e1_prediction_consistency(baseline)
        self.assertAlmostEqual(
            mechanism_consistency["mean_jsd_to_within_signal_centroid"],
            0.0,
            places=15,
        )
        self.assertGreater(
            baseline_consistency["mean_jsd_to_within_signal_centroid"], 0.0
        )
        self.assertEqual(
            e1_worst_rate_balanced_accuracy(mechanism)[
                "worst_rate_balanced_accuracy"
            ],
            1.0,
        )
        self.assertEqual(
            e1_worst_rate_balanced_accuracy(baseline)[
                "worst_rate_balanced_accuracy"
            ],
            0.0,
        )

    def test_representation_distance_uses_15_unordered_pairs(self) -> None:
        mechanism, _ = _e1_tables()
        identical = np.zeros((len(mechanism.labels), 2, 6), dtype=np.float64)
        rate_specific = np.zeros_like(identical)
        for row, rate in enumerate(mechanism.rates_hz):
            rate_index = DEFAULT_E1_RATES_HZ.index(int(rate))
            identical[row, :, 0] = 1.0
            rate_specific[row, :, rate_index] = 1.0
        identical_result = e1_representation_distance(mechanism, identical)
        rate_result = e1_representation_distance(mechanism, rate_specific)
        self.assertEqual(identical_result["rate_pair_count"], 15)
        self.assertAlmostEqual(
            identical_result["mean_rate_pair_cosine_distance"], 0.0, places=15
        )
        self.assertAlmostEqual(
            rate_result["mean_rate_pair_cosine_distance"], 1.0, places=15
        )

    def test_paired_bootstrap_is_deterministic_and_keeps_rate_copies_together(self) -> None:
        mechanism, baseline = _e1_tables()
        first = bootstrap_e1_paired_contrast(
            mechanism=mechanism,
            baseline=baseline,
            replicates=50,
            bootstrap_seed=20260801,
            include_samples=True,
        )
        second = bootstrap_e1_paired_contrast(
            mechanism=mechanism,
            baseline=baseline,
            replicates=50,
            bootstrap_seed=20260801,
            include_samples=True,
        )
        np.testing.assert_array_equal(
            first["samples"]["worst_rate_balanced_accuracy_effect"],
            second["samples"]["worst_rate_balanced_accuracy_effect"],
        )
        np.testing.assert_array_equal(
            first["samples"]["jsd_reduction_effect"],
            second["samples"]["jsd_reduction_effect"],
        )
        self.assertTrue(first["gate"]["c1_supported"])
        self.assertFalse(first["bootstrap"]["rate_copies_independent"])

    def test_bootstrap_one_replicate_matches_manual_crossed_resample(self) -> None:
        mechanism, baseline = _e1_tables()
        mechanism_probabilities = mechanism.probabilities.copy()
        baseline_probabilities = baseline.probabilities.copy()
        for row, (label, signal_id, seed, rate) in enumerate(
            zip(
                mechanism.labels,
                mechanism.signal_ids,
                mechanism.model_seeds,
                mechanism.rates_hz,
                strict=True,
            )
        ):
            if signal_id == 1 and seed == 999 and rate == 12000:
                mechanism_probabilities[row] = 0.02
                mechanism_probabilities[row, (int(label) + 1) % 4] = 0.94
            if signal_id == 0 and seed == 42 and rate == 20480:
                baseline_probabilities[row] = 0.02
                baseline_probabilities[row, (int(label) + 2) % 4] = 0.94
        mechanism = E1Predictions.from_columns(
            probabilities=mechanism_probabilities,
            labels=mechanism.labels,
            signal_ids=mechanism.signal_ids,
            model_seeds=mechanism.model_seeds,
            rates_hz=mechanism.rates_hz,
        )
        baseline = E1Predictions.from_columns(
            probabilities=baseline_probabilities,
            labels=baseline.labels,
            signal_ids=baseline.signal_ids,
            model_seeds=baseline.model_seeds,
            rates_hz=baseline.rates_hz,
        )
        result = bootstrap_e1_paired_contrast(
            mechanism=mechanism,
            baseline=baseline,
            replicates=1,
            bootstrap_seed=73,
            include_samples=True,
        )

        rng = np.random.Generator(np.random.PCG64(73))
        seed_draws = rng.integers(0, 5, size=(1, 5), dtype=np.int64)[0]
        signal_draws = [
            rng.integers(0, 2, size=(1, 2), dtype=np.int64)[0]
            for _ in range(4)
        ]

        def manual_summary(table: E1Predictions) -> tuple[float, float]:
            rate_balanced_accuracy = []
            divergences = []
            for class_id in range(4):
                class_divergences = []
                for seed_index in seed_draws:
                    for signal_id in signal_draws[class_id]:
                        row_probabilities = []
                        for rate in DEFAULT_E1_RATES_HZ:
                            row = np.flatnonzero(
                                (table.labels == class_id)
                                & (table.signal_ids == int(signal_id))
                                & (table.model_seeds == DEFAULT_E1_SEEDS[seed_index])
                                & (table.rates_hz == rate)
                            )
                            self.assertEqual(len(row), 1)
                            row_probabilities.append(table.probabilities[row[0]])
                        centroid = np.mean(row_probabilities, axis=0)
                        class_divergences.append(
                            np.mean(
                                [
                                    jensen_shannon_divergence(value, centroid)
                                    for value in row_probabilities
                                ]
                            )
                        )
                divergences.append(float(np.mean(class_divergences)))
            for rate in DEFAULT_E1_RATES_HZ:
                class_recalls = []
                for class_id in range(4):
                    correctness = []
                    for seed_index in seed_draws:
                        for signal_id in signal_draws[class_id]:
                            row = np.flatnonzero(
                                (table.labels == class_id)
                                & (table.signal_ids == int(signal_id))
                                & (table.model_seeds == DEFAULT_E1_SEEDS[seed_index])
                                & (table.rates_hz == rate)
                            )[0]
                            correctness.append(
                                int(np.argmax(table.probabilities[row]) == class_id)
                            )
                    class_recalls.append(float(np.mean(correctness)))
                rate_balanced_accuracy.append(float(np.mean(class_recalls)))
            return min(rate_balanced_accuracy), float(np.mean(divergences))

        mechanism_worst, mechanism_jsd = manual_summary(mechanism)
        baseline_worst, baseline_jsd = manual_summary(baseline)
        self.assertAlmostEqual(
            result["samples"]["worst_rate_balanced_accuracy_effect"][0],
            mechanism_worst - baseline_worst,
            places=14,
        )
        self.assertAlmostEqual(
            result["samples"]["jsd_reduction_effect"][0],
            baseline_jsd - mechanism_jsd,
            places=14,
        )

    def test_incomplete_rate_copy_grid_fails_closed(self) -> None:
        mechanism, _ = _e1_tables()
        incomplete = E1Predictions.from_columns(
            probabilities=mechanism.probabilities[:-1],
            labels=mechanism.labels[:-1],
            signal_ids=mechanism.signal_ids[:-1],
            model_seeds=mechanism.model_seeds[:-1],
            rates_hz=mechanism.rates_hz[:-1],
        )
        with self.assertRaisesRegex(ValueError, "incomplete E1 Cartesian grid"):
            e1_prediction_consistency(incomplete)


if __name__ == "__main__":
    unittest.main()
