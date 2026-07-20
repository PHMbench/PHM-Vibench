#!/usr/bin/env python3
"""Tests for the 1D-2D fusion ablation smoke runner."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts/run_fusion_ablation_smoke.py"


class TestFusionAblationSmokeRunner(unittest.TestCase):
    def test_single_condition_writes_metadata_and_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--condition",
                    "fft_only_proxy",
                    "--output",
                    tmpdir,
                    "--seed",
                    "0",
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

            run_dir = Path(tmpdir) / "fft_only_proxy" / "seed_0"
            metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
            run_meta = json.loads((run_dir / "run_meta.yaml").read_text(encoding="utf-8"))

            self.assertFalse(metrics["accepted_evidence"])
            self.assertEqual(metrics["condition_id"], "fft_only_proxy")
            self.assertGreater(metrics["spectral_path_proxy"], 0.7)
            self.assertEqual(metrics["current_root_rewrite_proxy"], 1.0)
            self.assertEqual(run_meta["metrics_path"], str(run_dir / "metrics.json"))

    def test_all_conditions_write_expected_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--condition",
                    "all",
                    "--output",
                    tmpdir,
                    "--seed",
                    "0",
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

            for condition in ("fft_only_proxy", "legacy_ablation_surface"):
                run_dir = Path(tmpdir) / condition / "seed_0"
                self.assertTrue((run_dir / "metrics.json").exists())
                self.assertTrue((run_dir / "run_meta.yaml").exists())

            legacy = json.loads(
                (
                    Path(tmpdir)
                    / "legacy_ablation_surface"
                    / "seed_0"
                    / "metrics.json"
                ).read_text(encoding="utf-8")
            )
            self.assertLess(legacy["current_root_rewrite_proxy"], 0.5)


if __name__ == "__main__":
    unittest.main()
