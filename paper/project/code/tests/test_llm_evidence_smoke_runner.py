#!/usr/bin/env python3
"""Tests for the non-accepted LLM evidence smoke runner."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "experiments/scripts/run_llm_evidence_smoke.py"


class TestLLMEvidenceSmokeRunner(unittest.TestCase):
    def test_grounded_condition_writes_metadata_and_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--condition",
                    "grounded",
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

            run_dir = Path(tmpdir) / "grounded" / "seed_0"
            metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
            run_meta = json.loads((run_dir / "run_meta.yaml").read_text(encoding="utf-8"))

            self.assertFalse(metrics["accepted_evidence"])
            self.assertEqual(metrics["condition_id"], "grounded")
            self.assertGreater(metrics["prompt_count"], 0)
            self.assertIn("latency_p95_seconds", metrics)
            self.assertEqual(run_meta["metrics_path"], str(run_dir / "metrics.json"))
            self.assertEqual(run_meta["cuda"]["CUDA_VISIBLE_DEVICES"], "")

    def test_all_condition_writes_context_checker_and_latency_conditions(self) -> None:
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
                "grounded",
                "no_checker",
                "no_domain_context",
                "latency_short",
                "latency_long",
            ):
                run_dir = Path(tmpdir) / condition / "seed_0"
                metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
                self.assertEqual(metrics["condition_id"], condition)
                self.assertFalse(metrics["accepted_evidence"])


if __name__ == "__main__":
    unittest.main()
