from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from src.config_schema import ExperimentConfig


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts/materialize_p05_pilot_job.py"
SPEC = importlib.util.spec_from_file_location("p05_pilot_materializer", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


EXPECTED_GPU = {
    "P05-PILOT-B0-CWRU": (0, "GPU-TEST-ZERO"),
    "P05-PILOT-M-CWRU": (0, "GPU-TEST-ZERO"),
    "P05-PILOT-M-XJTU": (1, "GPU-TEST-ONE"),
    "P05-PILOT-B0-XJTU": (1, "GPU-TEST-ONE"),
}


@pytest.mark.parametrize("job_id", sorted(EXPECTED_GPU))
def test_materializer_emits_one_strict_create_only_job_package(tmp_path, job_id) -> None:
    physical_index, uuid = EXPECTED_GPU[job_id]
    result = MODULE.materialize_p05_pilot_job(
        job_id=job_id,
        gpu_uuid=uuid,
        output_package=tmp_path / job_id,
    )

    config_path = Path(result["config_path"])
    manifest_path = Path(result["manifest_path"])
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    typed = ExperimentConfig.model_validate(config, strict=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert typed.trainer.expected_gpu_uuid == uuid
    assert typed.trainer.p05_pilot_mode is True
    assert typed.environment.stage == "fit_validate_only"
    assert config["task"]["loss"] == "CE_weighted"
    assert config["task"]["p05_arm_id"] in {"P05-M", "P05-B0"}
    assert config["task"]["p05_trace_export"] is (
        config["task"]["p05_arm_id"] == "P05-M"
    )
    assert result["physical_gpu_index"] == physical_index
    assert manifest["physical_gpu_index"] == physical_index
    assert manifest["config_sha256"] == hashlib.sha256(config_path.read_bytes()).hexdigest()
    assert manifest["evidence_eligible"] is False
    assert set(Path(result["package_dir"]).iterdir()) == {config_path, manifest_path}

    with pytest.raises(FileExistsError, match="already exists"):
        MODULE.materialize_p05_pilot_job(
            job_id=job_id,
            gpu_uuid=uuid,
            output_package=tmp_path / job_id,
        )


@pytest.mark.parametrize(
    "uuid",
    ["__REQUIRED_GPU_UUID_AT_LAUNCH__", "0", "GPU-BAD UUID", ""],
)
def test_materializer_rejects_unbound_or_invalid_gpu_uuid(tmp_path, uuid) -> None:
    with pytest.raises(ValueError, match="observed printable"):
        MODULE.materialize_p05_pilot_job(
            job_id="P05-PILOT-B0-CWRU",
            gpu_uuid=uuid,
            output_package=tmp_path / "invalid",
        )
    assert not (tmp_path / "invalid").exists()


def test_materializer_rejects_unknown_job_without_writing(tmp_path) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        MODULE.materialize_p05_pilot_job(
            job_id="P05-PILOT-UNKNOWN",
            gpu_uuid="GPU-TEST-ZERO",
            output_package=tmp_path / "unknown",
        )
    assert not (tmp_path / "unknown").exists()
