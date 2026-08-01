"""Focused CPU contract tests for the frozen P08 E1 analytic signal bank."""

from __future__ import annotations

import unittest

import numpy as np

from src.p08_evidence.e1_data import (
    CLASS_IDS,
    EVALUATION_RATES_HZ,
    NATIVE_POINTS,
    NATIVE_RATE_HZ,
    SIGNALS_PER_CLASS,
    SPLIT_COUNTS,
    build_bank_manifest,
    expected_rate_points,
    generate_native_signal,
    generate_rate_copies,
    iter_rate_copies,
    protocol_manifest,
    split_for_underlying,
    split_underlying_ids,
)


class E1DataContractTests(unittest.TestCase):
    def test_seeded_splits_are_complete_disjoint_and_stable(self) -> None:
        for class_id in CLASS_IDS:
            splits = split_underlying_ids(class_id)
            self.assertEqual(
                {name: len(ids) for name, ids in splits.items()}, SPLIT_COUNTS
            )
            flat = [underlying_id for ids in splits.values() for underlying_id in ids]
            self.assertEqual(len(flat), SIGNALS_PER_CLASS)
            self.assertEqual(set(flat), set(range(SIGNALS_PER_CLASS)))
            for split_name, ids in splits.items():
                for underlying_id in ids:
                    self.assertEqual(
                        split_for_underlying(class_id, underlying_id), split_name
                    )

        self.assertEqual(
            protocol_manifest()["split_underlying_ids_sha256"],
            "76408c09d111b38f6e9cd403bcb8af41f6f83e854446de1e8b214a8b41d60e6e",
        )

    def test_native_generation_is_deterministic_and_obeys_signal_contract(self) -> None:
        normal = generate_native_signal(0, 7)
        self.assertEqual(normal.impulse_indices, ())
        self.assertIsNone(normal.fault_amplitude)

        for class_id in (1, 2, 3):
            fault = generate_native_signal(class_id, 7)
            self.assertGreater(len(fault.impulse_indices), 0)
            self.assertEqual(tuple(sorted(fault.impulse_indices)), fault.impulse_indices)
            self.assertTrue(all(0 <= index < NATIVE_POINTS for index in fault.impulse_indices))

        signal = generate_native_signal(2, 7)
        replay = generate_native_signal(2, 7)
        self.assertEqual(signal.sample_rate_hz, NATIVE_RATE_HZ)
        self.assertEqual(signal.samples.shape, (NATIVE_POINTS,))
        self.assertEqual(signal.samples.dtype, np.dtype("<f8"))
        self.assertFalse(signal.samples.flags.writeable)
        self.assertFalse(signal.clean_samples.flags.writeable)
        np.testing.assert_array_equal(signal.samples, replay.samples)
        self.assertEqual(signal.signal_sha256, replay.signal_sha256)
        self.assertNotEqual(
            signal.signal_sha256, generate_native_signal(2, 8).signal_sha256
        )
        self.assertAlmostEqual(
            float(np.sqrt(np.mean(np.square(signal.clean_samples)))), 1.0, places=14
        )
        noise = signal.samples - signal.clean_samples
        self.assertAlmostEqual(float(noise.mean()), 0.0, places=14)
        realized_snr_db = 10.0 * np.log10(
            np.mean(np.square(signal.clean_samples)) / np.mean(np.square(noise))
        )
        self.assertAlmostEqual(float(realized_snr_db), signal.snr_db, places=11)

        # These reference hashes intentionally make a generator or draw-order
        # change visible; NumPy/SciPy versions are retained in the manifest.
        self.assertEqual(
            signal.clean_sha256,
            "5fd92bfc2798940c4c1c8764c3af296457af698b6a38fb11dd5bf281ea6967de",
        )
        self.assertEqual(
            signal.signal_sha256,
            "4160a3f945fab65bd6dc88dd124770fbd96de2ca1f1624971d1843c92de46ad1",
        )

    def test_six_exact_polyphase_rate_copies_share_one_split(self) -> None:
        copies = generate_rate_copies(2, 7)
        self.assertEqual(tuple(copy.sample_rate_hz for copy in copies), EVALUATION_RATES_HZ)
        self.assertEqual(
            tuple(copy.samples.size for copy in copies),
            tuple(expected_rate_points(rate) for rate in EVALUATION_RATES_HZ),
        )
        self.assertEqual(
            tuple(copy.samples.size for copy in copies), (240, 410, 512, 960, 1000, 4000)
        )
        self.assertEqual({copy.split for copy in copies}, {"train"})
        self.assertEqual(len({copy.native_signal_sha256 for copy in copies}), 1)
        self.assertTrue(all(not copy.samples.flags.writeable for copy in copies))
        self.assertEqual(copies[-1].sample_sha256, copies[-1].native_signal_sha256)
        self.assertEqual(
            copies[1].sample_sha256,
            "4b1c408d47cc229e0df3eea71befdf00115ee73d032458f43ebec6813efb4750",
        )

    def test_split_filter_keeps_all_rate_copies_with_underlying_signal(self) -> None:
        test_copies = list(iter_rate_copies(split="test", rates_hz=(12_000,)))
        self.assertEqual(len(test_copies), len(CLASS_IDS) * SPLIT_COUNTS["test"])
        self.assertTrue(all(copy.split == "test" for copy in test_copies))
        self.assertEqual(
            {(copy.class_id, copy.underlying_id) for copy in test_copies},
            {
                (class_id, underlying_id)
                for class_id in CLASS_IDS
                for underlying_id in split_underlying_ids(class_id)["test"]
            },
        )

    def test_deterministic_bank_hash_covers_order_metadata_and_samples(self) -> None:
        manifest = build_bank_manifest(
            split="test", rates_hz=(12_000, 200_000), include_record_hashes=False
        )
        self.assertEqual(manifest["rate_copy_count"], 4 * 51 * 2)
        self.assertEqual(manifest["total_sample_count"], 4 * 51 * (240 + 4_000))
        self.assertEqual(
            manifest["bank_sha256"],
            "4b5b2eac7f1fa6522276ec6ef4ad70dfbe669299b640761f0d1addb5ae86f546",
        )

    def test_invalid_identifiers_and_rates_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            generate_native_signal(4, 0)
        with self.assertRaises(ValueError):
            generate_native_signal(0, SIGNALS_PER_CLASS)
        with self.assertRaises(ValueError):
            expected_rate_points(44_100)
        with self.assertRaises(ValueError):
            list(iter_rate_copies(split="holdout"))  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            list(iter_rate_copies(rates_hz=(12_000, 12_000)))


if __name__ == "__main__":
    unittest.main()
