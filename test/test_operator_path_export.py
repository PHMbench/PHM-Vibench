"""Temporary-file checks for explicit PHMFactory export; no industrial data."""
from __future__ import annotations

from collections import Counter
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from src.data_factory import operator_path_export as export_data


def load_folds(manifest: Path) -> list[dict]:
    return [fold | {"path": manifest.parent / fold["path"]}
            for fold in json.loads(manifest.read_text())["folds"]]


class ExportDataTests(unittest.TestCase):
    def fixture(self, root: Path, reader: str = "npy") -> tuple[Path, dict, dict]:
        records, signals = [], {}
        for domain, split in (("source_A", "train"), ("source_B", "train"),
                              ("source_val", "val"), ("heldout", "test")):
            for label in (0, 1):
                index = len(records)
                path = root / f"record{index}.{'npy' if reader == 'npy' else 'mat'}"
                signal = np.arange(128, dtype=np.float32).reshape(64, 2) + 1000 * index
                signals[path.resolve()] = signal
                if reader == "npy":
                    np.save(path, signal)
                else:
                    path.write_bytes(b"reader-test-placeholder")
                records.append(dict(path=path.name, label=label, unit_id=f"unit{index}",
                                    domain=domain, split=split))
        spec = {"folds": [dict(dataset="PU", fold="heldout", reader=reader,
                              window_size=32, stride=16, channels=[1], sampling_rate=64000,
                              source_version="fixture-v1", provenance_notes="Explicit temporary record mapping",
                              records=records)]}
        path = root / "spec.json"
        path.write_text(json.dumps(spec), encoding="utf-8")
        return path, spec, signals

    def test_npy_export_preserves_samples_and_enters_the_real_data_check(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec_path, spec, signals = self.fixture(root)
            before = {path: path.read_bytes() for path in signals}
            output = root / "export"
            manifest = export_data.export_spec(spec_path, output)
            self.assertEqual(manifest, output / "manifest.json")
            folds = load_folds(manifest)
            self.assertEqual([(fold["dataset"], fold["fold"]) for fold in folds], [("PU", "heldout")])
            with np.load(folds[0]["path"], allow_pickle=False) as archive:
                data = dict(archive)
            export_data.validate_data(data, domain_holdout=True)
            self.assertEqual(data["x"].shape, (24, 32, 1))
            expected = np.stack([signal[start:start + 32, [1]]
                                 for signal in signals.values() for start in (0, 16, 32)])
            np.testing.assert_array_equal(data["x"], expected)
            np.testing.assert_array_equal(data["y"], np.repeat([record["label"] for record in spec["folds"][0]["records"]], 3))
            self.assertEqual(json.loads((output / "spec.json").read_text()), spec)
            resolved = json.loads((output / "resolved_spec.json").read_text())
            self.assertTrue(all(Path(record["path"]).is_absolute() for record in resolved["folds"][0]["records"]))
            with (output / "window_records.csv").open(newline="", encoding="utf-8") as stream:
                provenance = list(csv.DictReader(stream))
            self.assertEqual([(row["path"], int(row["start"])) for row in provenance],
                             [(str(path), start) for path in signals for start in (0, 16, 32)])
            self.assertEqual([int(row["window"]) for row in provenance], list(range(24)))
            self.assertTrue(all(int(row["stop"]) - int(row["start"]) == 32 for row in provenance))
            self.assertEqual(before, {path: path.read_bytes() for path in signals})


    def test_pu_reader_entry_is_reused_once_per_record_with_unmodified_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec_path, _, signals = self.fixture(root, reader="RM_027_PU")
            original = {path: signal.copy() for path, signal in signals.items()}
            calls = []

            def reader(name, path):
                self.assertEqual(name, "RM_027_PU")
                resolved = Path(path).resolve()
                calls.append(resolved)
                return signals[resolved]

            with patch.object(export_data, "read_signal", side_effect=reader):
                manifest = export_data.export_spec(spec_path, root / "export")
            self.assertEqual(Counter(calls), Counter(signals.keys()))
            folds = load_folds(manifest)
            with np.load(folds[0]["path"], allow_pickle=False) as archive:
                expected = np.stack([signal[start:start + 32, [1]]
                                     for signal in original.values() for start in (0, 16, 32)])
                np.testing.assert_array_equal(archive["x"], expected)
            for path in signals:
                np.testing.assert_array_equal(signals[path], original[path])

    def test_existing_pu_module_read_function_receives_the_exact_record_path(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "record.mat"
            path.write_bytes(b"reader-test-placeholder")
            signal = np.arange(128, dtype=np.float32).reshape(64, 2)
            module = Mock()
            module.read.return_value = signal
            with patch.object(export_data.importlib, "import_module", return_value=module) as imported:
                result = export_data.read_signal("RM_027_PU", path)
            imported.assert_called_once_with("src.data_factory.reader.RM_027_PU")
            module.read.assert_called_once_with(str(path))
            np.testing.assert_array_equal(result, signal)

    def test_metadata_leakage_is_rejected_before_any_reader_call(self):
        for defect in ("duplicate_path", "unit_partition", "unit_label", "domain_partition"):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                spec_path, spec, _ = self.fixture(root)
                records = spec["folds"][0]["records"]
                if defect == "duplicate_path":
                    records[1]["path"] = records[0]["path"]
                elif defect == "unit_partition":
                    records[6]["unit_id"] = records[0]["unit_id"]
                elif defect == "unit_label":
                    records[1]["unit_id"] = records[0]["unit_id"]
                else:
                    records[6]["domain"] = records[0]["domain"]
                spec_path.write_text(json.dumps(spec), encoding="utf-8")
                with patch.object(export_data, "read_signal") as read:
                    with self.assertRaises(ValueError):
                        export_data.export_spec(spec_path, root / "export")
                    read.assert_not_called()

    def test_existing_output_and_raw_records_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec_path, _, signals = self.fixture(root)
            original = {path: path.read_bytes() for path in signals}
            output = root / "export"
            output.mkdir()
            marker = output / "keep.txt"
            marker.write_text("existing output", encoding="utf-8")
            with patch.object(export_data, "read_signal") as read:
                with self.assertRaises(FileExistsError):
                    export_data.export_spec(spec_path, output)
                read.assert_not_called()
            self.assertEqual({path.name for path in output.iterdir()}, {"keep.txt"})
            self.assertEqual(marker.read_text(), "existing output")
            self.assertEqual(original, {path: path.read_bytes() for path in signals})

    def test_invalid_signal_or_channel_stops_export_and_retains_failure(self):
        for defect in ("too_short", "nonfinite", "channel_out_of_range"):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                spec_path, spec, signals = self.fixture(root)
                first = next(iter(signals))
                if defect == "too_short":
                    np.save(first, np.zeros((16, 2), dtype=np.float32))
                elif defect == "nonfinite":
                    invalid = signals[first].copy()
                    invalid[0, 0] = np.nan
                    np.save(first, invalid)
                else:
                    spec["folds"][0]["channels"] = [2]
                    spec_path.write_text(json.dumps(spec), encoding="utf-8")
                original = {path: path.read_bytes() for path in signals}
                output = root / "export"
                with self.assertRaises(ValueError):
                    export_data.export_spec(spec_path, output)
                self.assertEqual(json.loads((output / "failure.json").read_text())["status"], "failed")
                self.assertTrue((output / "error.log").is_file())
                self.assertFalse((output / "manifest.json").exists())
                self.assertFalse((output / "status.json").exists())
                self.assertEqual(original, {path: path.read_bytes() for path in signals})

    def test_invalid_or_unavailable_reader_is_rejected_before_reading(self):
        for reader in ("../npy", "RM_999_NOT_AN_EXISTING_READER"):
            with self.subTest(reader=reader), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                spec_path, spec, _ = self.fixture(root)
                spec["folds"][0]["reader"] = reader
                spec_path.write_text(json.dumps(spec), encoding="utf-8")
                with patch.object(export_data, "read_signal") as read:
                    with self.assertRaises((ValueError, FileNotFoundError)):
                        export_data.export_spec(spec_path, root / "export")
                    read.assert_not_called()

    def test_npy_reader_requires_two_dimensions_and_never_transposes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "signal.npy"
            np.save(path, np.arange(64, dtype=np.float32))
            with self.assertRaises(ValueError):
                export_data.read_signal("npy", path)
            signal = np.arange(256, dtype=np.float32).reshape(4, 64)
            np.save(path, signal)
            loaded = export_data.read_signal("npy", path)
            self.assertEqual(loaded.shape, (4, 64))
            np.testing.assert_array_equal(loaded, signal)


    def test_source_export_preserves_exact_train_val_arrays_and_required_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec_path, spec, signals = self.fixture(root)
            records = spec["folds"][0]["records"]
            spec["folds"][0]["records"] = [record for record in records if record["split"] != "test"]
            spec_path.write_text(json.dumps(spec), encoding="utf-8")
            output = root / "source"
            manifest = export_data.export_source(spec_path, output)
            with np.load(load_folds(manifest)[0]["path"], allow_pickle=False) as archive:
                data = dict(archive)
            self.assertEqual(set(data), {"x", "y", "unit_id", "domain", "split", "sampling_rate"})
            self.assertEqual(set(data["split"]), {"train", "val"})
            self.assertEqual(data["x"].shape, (18, 32, 1))
            expected = np.stack([signals[(root / record["path"]).resolve()][start:start + 32, [1]]
                                 for record in spec["folds"][0]["records"] for start in (0, 16, 32)])
            np.testing.assert_array_equal(data["x"], expected)
            self.assertEqual(data["x"].dtype, np.float32)
            export_data.validate_data(data, True, required_splits=("train", "val"))
            with self.assertRaisesRegex(ValueError, "Explicit partitions"):
                export_data.validate_data(data, True)
            self.assertTrue(json.loads((output / "resolved_spec.json").read_text())["source_only"])
            with (output / "window_records.csv").open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 18)
            self.assertEqual({row["split"] for row in rows}, {"train", "val"})

    def test_source_only_rejects_target_in_any_fold_before_any_waveform_reader(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec_path, spec, _ = self.fixture(root)
            source_fold = spec["folds"][0] | {"fold": "source"}
            source_fold["records"] = [record for record in source_fold["records"] if record["split"] != "test"]
            spec["folds"].insert(0, source_fold)
            spec_path.write_text(json.dumps(spec), encoding="utf-8")
            with patch.object(export_data, "read_signal") as read:
                with self.assertRaisesRegex(ValueError, "target records are forbidden"):
                    export_data.export_source(spec_path, root / "export")
                read.assert_not_called()

    def test_default_export_still_requires_test_and_source_mode_requires_val(self):
        for source_only, retained in ((False, {"train", "val"}), (True, {"train"})):
            with self.subTest(source_only=source_only), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                spec_path, spec, _ = self.fixture(root)
                spec["folds"][0]["records"] = [record for record in spec["folds"][0]["records"]
                                                if record["split"] in retained]
                spec_path.write_text(json.dumps(spec), encoding="utf-8")
                with patch.object(export_data, "read_signal") as read:
                    with self.assertRaisesRegex(ValueError, "same class support"):
                        export_data.export_spec(spec_path, root / "export", source_only=source_only)
                    read.assert_not_called()

    def test_source_rejects_unit_or_domain_overlap_before_reading(self):
        for defect in ("unit", "domain"):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                spec_path, spec, _ = self.fixture(root)
                records = [record for record in spec["folds"][0]["records"] if record["split"] != "test"]
                if defect == "unit":
                    records[-2]["unit_id"] = records[0]["unit_id"]
                else:
                    records[-2]["domain"] = records[0]["domain"]
                spec["folds"][0]["records"] = records
                spec_path.write_text(json.dumps(spec), encoding="utf-8")
                with patch.object(export_data, "read_signal") as read:
                    with self.assertRaisesRegex(ValueError, "crosses"):
                        export_data.export_source(spec_path, root / "export")
                    read.assert_not_called()


if __name__ == "__main__":
    unittest.main()
