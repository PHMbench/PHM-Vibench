"""CPU-only tests for the deterministic P08 environment snapshot."""

from __future__ import annotations

import hashlib
import importlib.metadata as importlib_metadata
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest

from src.p08_evidence.environment import (
    EXPECTED_ENVIRONMENT,
    SNAPSHOT_SCHEMA,
    _conda_non_pypi_packages,
    _dist_info_inventory,
    snapshot_sha,
    snapshot_text,
)


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class _FakeDistribution:
    def __init__(self, path: Path, name: str, version: str) -> None:
        self._path = path
        self.metadata = {"Name": name}
        self.version = version


class EnvironmentSnapshotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.first = snapshot_text()
        cls.second = snapshot_text()
        cls.document = json.loads(cls.first)

    def test_snapshot_is_byte_stable_and_self_hashing(self) -> None:
        self.assertEqual(self.first, self.second)
        self.assertEqual(
            snapshot_sha(self.first), hashlib.sha256(self.first.encode()).hexdigest()
        )
        self.assertTrue(_SHA256.fullmatch(snapshot_sha(self.first)))

    def test_runtime_and_loaded_module_provenance_are_complete(self) -> None:
        document = self.document
        self.assertEqual(document["schema"], SNAPSHOT_SCHEMA)
        self.assertEqual(document["environment"]["name"], EXPECTED_ENVIRONMENT)
        self.assertEqual(
            set(document["runtime_versions"]),
            {
                "python",
                "torch",
                "cuda_compiled_for_torch",
                "numpy",
                "scipy",
                "pyarrow",
            },
        )
        modules = document["loaded_modules"]
        self.assertEqual([item["module"] for item in modules], sorted(("numpy", "pyarrow", "scipy", "torch")))
        for item in modules:
            self.assertFalse(Path(item["loaded_path"]).is_absolute())
            loaded = Path(sys.prefix) / item["loaded_path"]
            self.assertTrue(loaded.is_file())
            self.assertEqual(hashlib.sha256(loaded.read_bytes()).hexdigest(), item["sha256"])

    def test_actual_dist_info_is_complete_sorted_and_duplicates_are_explicit(self) -> None:
        records = self.document["python_dist_info"]
        paths = [item["metadata_path"] for item in records]
        expected_paths = {
            Path(getattr(distribution, "_path")).resolve().relative_to(
                Path(sys.prefix).resolve()
            ).as_posix()
            for distribution in importlib_metadata.distributions()
            if getattr(distribution, "_path", None) is not None
            and Path(getattr(distribution, "_path")).name.endswith(".dist-info")
        }
        self.assertEqual(set(paths), expected_paths)
        self.assertEqual(len(paths), len(set(paths)))
        sort_keys = [
            (item["normalized_name"], item["version"], item["metadata_path"])
            for item in records
        ]
        self.assertEqual(sort_keys, sorted(sort_keys))

        grouped: dict[str, list[dict[str, str]]] = {}
        for item in records:
            grouped.setdefault(item["normalized_name"], []).append(item)
        reported = {
            item["normalized_name"]: item
            for item in self.document["duplicate_python_metadata"]
        }
        expected_duplicate_names = {
            name for name, matches in grouped.items() if len(matches) > 1
        }
        self.assertEqual(set(reported), expected_duplicate_names)
        for name in expected_duplicate_names:
            self.assertEqual(reported[name]["record_count"], len(grouped[name]))
            self.assertEqual(
                [item["metadata_path"] for item in reported[name]["records"]],
                [item["metadata_path"] for item in grouped[name]],
            )

    def test_synthetic_duplicate_versions_are_both_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory)
            site = prefix / "lib/python3.10/site-packages"
            old = site / "typing_extensions-4.8.0.dist-info"
            new = site / "typing_extensions-4.14.0.dist-info"
            for path, version in ((old, "4.8.0"), (new, "4.14.0")):
                path.mkdir(parents=True)
                (path / "METADATA").write_text(
                    f"Name: typing_extensions\nVersion: {version}\n",
                    encoding="utf-8",
                )
            records, duplicates = _dist_info_inventory(
                (
                    _FakeDistribution(old, "typing_extensions", "4.8.0"),
                    _FakeDistribution(new, "typing_extensions", "4.14.0"),
                    # A repeated discovery of the same directory is not a third
                    # installation and must be de-duplicated by concrete path.
                    _FakeDistribution(old, "typing_extensions", "4.8.0"),
                ),
                prefix,
            )
        self.assertEqual(len(records), 2)
        self.assertEqual(len(duplicates), 1)
        self.assertEqual(duplicates[0]["normalized_name"], "typing-extensions")
        self.assertEqual(
            {item["version"] for item in duplicates[0]["records"]},
            {"4.8.0", "4.14.0"},
        )

    def test_conda_inventory_omits_pypi_and_all_source_urls(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory)
            root = prefix / "conda-meta"
            root.mkdir()
            (root / "safe-1.0-build_0.json").write_text(
                json.dumps(
                    {
                        "name": "safe",
                        "version": "1.0",
                        "build": "build_0",
                        "build_number": 0,
                        "subdir": "linux-64",
                        "channel": "https://user:secret@example.test/t/token/channel",
                        "url": "https://user:secret@example.test/pkg.conda?token=secret",
                    }
                ),
                encoding="utf-8",
            )
            (root / "pip-only-2.0-pypi_0.json").write_text(
                json.dumps(
                    {
                        "name": "pip-only",
                        "version": "2.0",
                        "build": "pypi_0",
                        "build_number": 0,
                        "subdir": "pypi",
                        "channel": "pypi",
                    }
                ),
                encoding="utf-8",
            )
            packages = _conda_non_pypi_packages(prefix)
        self.assertEqual([item["name"] for item in packages], ["safe"])
        serialized = json.dumps(packages, sort_keys=True)
        self.assertNotIn("secret", serialized)
        self.assertNotIn("example.test", serialized)
        self.assertNotIn("channel", serialized)
        self.assertNotIn("url", serialized)

    def test_snapshot_has_no_absolute_prefix_or_volatile_identity_fields(self) -> None:
        self.assertNotIn(str(Path(sys.prefix).resolve()), self.first)
        self.assertNotIn(str(Path.home()), self.first)
        privacy = self.document["privacy_contract"]
        self.assertTrue(all(value is False for value in privacy.values()))
        forbidden_keys = {"timestamp", "hostname", "username", "channel", "url"}

        def visit(value: object) -> None:
            if isinstance(value, dict):
                self.assertTrue(forbidden_keys.isdisjoint(value))
                for nested in value.values():
                    visit(nested)
            elif isinstance(value, list):
                for nested in value:
                    visit(nested)

        visit(self.document)


if __name__ == "__main__":
    unittest.main()
