from __future__ import annotations

import hashlib
import importlib.util
import json
import shlex
from copy import deepcopy
from itertools import product
from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = (
    REPO_ROOT
    / "configs"
    / "experiments"
    / "p05"
    / "protocol"
    / "cpu_arm_matrix_p05_v1.yaml"
)
SCRIPT = REPO_ROOT / "scripts" / "materialize_p05_cpu_arm_job.py"
SPEC = importlib.util.spec_from_file_location("p05_cpu_arm_materializer", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

REGISTERED_SEEDS = (42, 123, 456, 789, 1024)
COMMAND_PREFIX = ("conda", "run", "-n", "LQ_signal", "python")


def _load_matrix(path: Path = MATRIX_PATH) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _selected_templates(matrix: dict, job_id: str) -> dict[str, dict[str, str]]:
    job = next(job for job in matrix["jobs"] if job["id"] == job_id)
    arm = matrix["arms"][job["arm"]]
    dataset = matrix["datasets"][job["dataset"]]
    templates = {
        name: dataset["artifacts"][name]
        for name in arm["required_dataset_artifacts"]
    }
    if job["arm"] == "P05-B2":
        templates.update(
            {
                f"b0_{name}": job["b0_artifacts"][name]
                for name in arm["required_b0_artifacts"]
            }
        )
    return templates


def _bindings(tmp_path: Path, job_id: str) -> dict[str, str]:
    matrix = _load_matrix()
    values: dict[str, str] = {}
    artifact_dir = tmp_path / f"artifacts-{job_id}"
    artifact_dir.mkdir(parents=True)
    for name, template in sorted(_selected_templates(matrix, job_id).items()):
        artifact = artifact_dir / f"{name}.bin"
        content = f"fixture:{job_id}:{name}\n".encode()
        artifact.write_bytes(content)
        values[template["path"]] = str(artifact)
        values[template["sha256"]] = hashlib.sha256(content).hexdigest()
    return values


def test_cpu_matrix_has_exact_b2_b4_cartesian_budget_and_conda_commands() -> None:
    matrix = _load_matrix()
    MODULE._validate_matrix(matrix)

    assert matrix["budget"] == {
        "device_class": "cpu",
        "total_fit_ceiling": 12,
        "arm_fit_ceiling": {"P05-B2": 10, "P05-B4": 2},
        "ceiling_exceeded": "hard_error",
    }
    assert matrix["outputs"]["evidence_status"] == "unadjudicated"
    assert matrix["outputs"]["execution_status_on_materialization"] == (
        "not_started"
    )
    assert matrix["runtime"]["command_prefix"] == list(COMMAND_PREFIX)
    assert matrix["runtime"]["gpu_use"] == "forbidden"
    assert matrix["runtime"]["network_use"] == "forbidden"

    jobs = matrix["jobs"]
    b2_jobs = [job for job in jobs if job["arm"] == "P05-B2"]
    b4_jobs = [job for job in jobs if job["arm"] == "P05-B4"]
    assert len(jobs) == 12
    assert len(b2_jobs) == 10
    assert {
        (job["dataset"], job["b0_model_seed"]) for job in b2_jobs
    } == set(product(("CWRU", "XJTU"), REGISTERED_SEEDS))
    assert all(
        set(job["b0_artifacts"]) == {"checkpoint", "run_manifest", "predictions"}
        for job in b2_jobs
    )
    assert all(
        MODULE._PLACEHOLDER_PATTERN.fullmatch(artifact[field]) is not None
        for job in b2_jobs
        for artifact in job["b0_artifacts"].values()
        for field in ("path", "sha256")
    )

    assert len(b4_jobs) == 2
    assert {job["dataset"] for job in b4_jobs} == {"CWRU", "XJTU"}
    assert all(job["fit_identity"] == "dataset_only_no_model_seed" for job in b4_jobs)
    assert all("b0_model_seed" not in job and "b0_artifacts" not in job for job in b4_jobs)
    assert matrix["arms"]["P05-B4"]["fits_per_dataset"] == 1
    assert matrix["arms"]["P05-B4"]["model_seed_axis"] == "forbidden"
    assert matrix["arms"]["P05-B4"]["model_seed_repetition"].startswith(
        "forbidden"
    )

    for job in jobs:
        command = tuple(shlex.split(job["materialize_command"]))
        assert command[: len(COMMAND_PREFIX)] == COMMAND_PREFIX
        assert "scripts/materialize_p05_cpu_arm_job.py" in command
        assert command[command.index("--job-id") + 1] == job["id"]


@pytest.mark.parametrize(
    "job_id",
    [
        *(
            f"P05-CPU-B2-{dataset}-S{seed}"
            for dataset in ("CWRU", "XJTU")
            for seed in REGISTERED_SEEDS
        ),
        "P05-CPU-B4-CWRU",
        "P05-CPU-B4-XJTU",
    ],
)
def test_each_cpu_job_materializes_only_after_all_hashes_are_verified(
    tmp_path,
    job_id,
) -> None:
    bindings = _bindings(tmp_path, job_id)
    result = MODULE.materialize_p05_cpu_arm_job(
        job_id=job_id,
        bindings=bindings,
        output_package=tmp_path / "package",
    )
    job_path = Path(result["job_path"])
    manifest_path = Path(result["manifest_path"])
    resolved = yaml.safe_load(job_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert "__REQUIRED_" not in job_path.read_text(encoding="utf-8")
    assert result["status"] == "created_not_executed"
    assert resolved["device_class"] == "cpu"
    assert resolved["runtime"]["conda_environment"] == "LQ_signal"
    assert resolved["runtime"]["required_command_prefix"] == list(COMMAND_PREFIX)
    assert resolved["runtime"]["gpu_use"] == "forbidden"
    assert resolved["output"]["execution_status"] == "not_started"
    assert resolved["output"]["evidence_status"] == "unadjudicated"
    assert manifest["materialization_status"] == "created_not_executed"
    assert manifest["execution_status"] == "not_started"
    assert manifest["evidence_status"] == "unadjudicated"
    assert manifest["bound_placeholder_count"] == len(bindings)
    assert manifest["job_sha256"] == hashlib.sha256(job_path.read_bytes()).hexdigest()
    assert set(Path(result["package_dir"]).iterdir()) == {job_path, manifest_path}

    expected_dependency_count = 8 if result["arm"] == "P05-B2" else 4
    assert len(resolved["dependencies"]) == expected_dependency_count
    for dependency in resolved["dependencies"].values():
        path = Path(dependency["path"])
        assert path.is_absolute() and path.is_file()
        assert dependency["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert dependency["size_bytes"] == path.stat().st_size

    if result["arm"] == "P05-B2":
        assert resolved["corresponding_b0_model_seed"] in REGISTERED_SEEDS
        assert {name for name in resolved["dependencies"] if name.startswith("b0_")} == {
            "b0_checkpoint",
            "b0_run_manifest",
            "b0_predictions",
        }
    else:
        assert "corresponding_b0_model_seed" not in resolved
        assert resolved["fit_identity"] == "dataset_only_no_model_seed"
        assert resolved["model_seed_axis"] == "forbidden"
        assert resolved["model_seed_repetition"].startswith("forbidden")


def test_materialization_is_deterministic_and_create_only(tmp_path, monkeypatch) -> None:
    job_id = "P05-CPU-B2-XJTU-S456"
    bindings = _bindings(tmp_path, job_id)
    first = MODULE.materialize_p05_cpu_arm_job(
        job_id=job_id,
        bindings=bindings,
        output_package=tmp_path / "first",
    )
    second = MODULE.materialize_p05_cpu_arm_job(
        job_id=job_id,
        bindings=bindings,
        output_package=tmp_path / "second",
    )
    assert first["job_sha256"] == second["job_sha256"]
    assert first["manifest_sha256"] == second["manifest_sha256"]
    assert first["semantic_sha256"] == second["semantic_sha256"]

    manifest_before = Path(first["manifest_path"]).read_bytes()

    def fail_if_revalidated(*args, **kwargs):
        del args, kwargs
        raise AssertionError("existing target must fail before matrix validation")

    monkeypatch.setattr(MODULE, "_validate_matrix", fail_if_revalidated)
    with pytest.raises(FileExistsError, match="already exists"):
        MODULE.materialize_p05_cpu_arm_job(
            job_id=job_id,
            bindings=bindings,
            output_package=tmp_path / "first",
        )
    assert Path(first["manifest_path"]).read_bytes() == manifest_before


def test_materializer_rejects_missing_extra_unresolved_and_tampered_bindings(
    tmp_path,
) -> None:
    job_id = "P05-CPU-B4-XJTU"

    missing = _bindings(tmp_path / "missing", job_id)
    missing.pop(next(iter(missing)))
    with pytest.raises(ValueError, match="exactly replace.*missing="):
        MODULE.materialize_p05_cpu_arm_job(
            job_id=job_id,
            bindings=missing,
            output_package=tmp_path / "missing-package",
        )

    extra = _bindings(tmp_path / "extra", job_id)
    extra["__REQUIRED_UNEXPECTED_PATH__"] = "/tmp/unexpected"
    with pytest.raises(ValueError, match="unexpected="):
        MODULE.materialize_p05_cpu_arm_job(
            job_id=job_id,
            bindings=extra,
            output_package=tmp_path / "extra-package",
        )

    unresolved = _bindings(tmp_path / "unresolved", job_id)
    path_key = next(key for key in unresolved if key.endswith("_PATH__"))
    unresolved[path_key] = path_key
    with pytest.raises(ValueError, match="unresolved placeholder"):
        MODULE.materialize_p05_cpu_arm_job(
            job_id=job_id,
            bindings=unresolved,
            output_package=tmp_path / "unresolved-package",
        )

    tampered = _bindings(tmp_path / "tampered", job_id)
    bound_path = Path(next(value for key, value in tampered.items() if key.endswith("_PATH__")))
    bound_path.write_bytes(b"tampered\n")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        MODULE.materialize_p05_cpu_arm_job(
            job_id=job_id,
            bindings=tampered,
            output_package=tmp_path / "tampered-package",
        )

    assert not any(tmp_path.glob("*-package"))


@pytest.mark.parametrize("mutation", ["ceiling", "b4_seed", "command"])
def test_materializer_rejects_matrix_drift_before_writing(
    tmp_path,
    mutation,
) -> None:
    matrix = deepcopy(_load_matrix())
    if mutation == "ceiling":
        matrix["budget"]["total_fit_ceiling"] = 13
        expected = "ceiling"
    elif mutation == "b4_seed":
        b4_job = next(job for job in matrix["jobs"] if job["arm"] == "P05-B4")
        b4_job["b0_model_seed"] = 42
        expected = "forbids model-seed"
    else:
        matrix["jobs"][0]["materialize_command"] = "python unsafe.py"
        expected = "must start with conda"
    matrix_path = tmp_path / f"matrix-{mutation}.yaml"
    matrix_path.write_text(yaml.safe_dump(matrix, sort_keys=False), encoding="utf-8")
    bindings = _bindings(tmp_path / f"bindings-{mutation}", "P05-CPU-B4-CWRU")

    with pytest.raises(ValueError, match=expected):
        MODULE.materialize_p05_cpu_arm_job(
            job_id="P05-CPU-B4-CWRU",
            bindings=bindings,
            output_package=tmp_path / f"package-{mutation}",
            matrix_path=matrix_path,
        )
    assert not (tmp_path / f"package-{mutation}").exists()
