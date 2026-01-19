from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


@dataclass(frozen=True)
class RunArtifacts:
    run_dir: Path
    manifest_path: Path


def discover_runs(root_dir: Path, manifests_glob: str = "**/artifacts/manifest.json") -> List[RunArtifacts]:
    manifests = sorted(root_dir.glob(manifests_glob))
    runs: List[RunArtifacts] = []
    for mp in manifests:
        try:
            run_dir = mp.parents[1]
            runs.append(RunArtifacts(run_dir=run_dir, manifest_path=mp))
        except Exception:
            continue
    return runs


def read_manifest(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def find_metrics_csv(run_dir: Path) -> Optional[Path]:
    candidates = list(sorted((run_dir / "logs").glob("**/metrics.csv")))
    return candidates[0] if candidates else None


def find_test_results(run_dir: Path) -> Optional[Path]:
    candidates = list(sorted(run_dir.glob("test_result_*.csv")))
    return candidates[0] if candidates else None


def find_predictions(run_dir: Path) -> Optional[Path]:
    p = run_dir / "artifacts" / "predictions.npz"
    if p.exists():
        return p
    candidates = list(sorted((run_dir / "artifacts").glob("predictions.*")))
    return candidates[0] if candidates else None

