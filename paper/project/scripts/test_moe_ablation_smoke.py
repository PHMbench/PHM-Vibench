#!/usr/bin/env python3
"""Tests for the MoE ablation smoke runner."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts/run_moe_ablation_smoke.py"


class TestMoeAblationSmokeRunner(unittest.TestCase):
    def test_single_condition_writes_metadata_and_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--condition",
                    "no_load_balance",
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

            run_dir = Path(tmpdir) / "no_load_balance" / "seed_0"
            metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
            run_meta = json.loads((run_dir / "run_meta.yaml").read_text(encoding="utf-8"))

            self.assertFalse(metrics["accepted_evidence"])
            self.assertEqual(metrics["condition_id"], "no_load_balance")
            self.assertLess(metrics["load_balance_proxy"], 0.5)
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

            for condition in (
                "no_load_balance",
                "no_sparsity",
                "temperature_sweep",
                "remove_expert_family",
                "uniform_router",
            ):
                run_dir = Path(tmpdir) / condition / "seed_0"
                self.assertTrue((run_dir / "metrics.json").exists())
                self.assertTrue((run_dir / "run_meta.yaml").exists())

            temperature_metrics = json.loads(
                (
                    Path(tmpdir)
                    / "temperature_sweep"
                    / "seed_0"
                    / "metrics.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(len(temperature_metrics["temperature_sweep_rows"]), 3)


if __name__ == "__main__":
    unittest.main()
