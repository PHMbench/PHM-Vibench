from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest
import yaml

from scripts.p04 import prepare_decisive_run as preparer
from scripts.p04.package_decisive_run import REQUIRED_RUN_META_FIELDS


SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prepare(
    tmp_path: Path,
    *,
    arm: str = "FULL",
    stage: str = "S2",
    seed: int = 42,
    attempt: int = 1,
    retry_reason: str | None = None,
    gpu: int = 0,
    output: Path | None = None,
    suffix: str = "",
) -> tuple[dict[str, object], Path, Path]:
    output_path = output or tmp_path / f"training-{stage}-{arm}-{seed}{suffix}"
    staging = tmp_path / f"staging-{stage}-{arm}-{seed}{suffix}"
    plan = preparer.prepare_decisive_run(
        arm=arm,
        stage=stage,
        seed=seed,
        attempt=attempt,
        retry_reason=retry_reason,
        physical_gpu=gpu,
        output_dir=output_path,
        staging_dir=staging,
    )
    return plan, output_path.resolve(), staging.resolve()


def test_s2_full_publishes_pure_resolved_config_and_nonexecuting_plan(
    tmp_path: Path,
) -> None:
    plan, output, staging = _prepare(tmp_path)

    assert not output.exists()
    assert sorted(path.name for path in staging.iterdir()) == [
        "launch_plan.json",
        "resolved_config.yaml",
    ]
    config_path = staging / "resolved_config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    persisted_plan = json.loads(
        (staging / "launch_plan.json").read_text(encoding="utf-8")
    )

    assert isinstance(config, dict)
    assert "resolved" not in config
    assert "base_configs" not in config
    assert config["environment"]["seed"] == 42
    assert config["environment"]["output_dir"] == str(output)
    assert config["data"]["cache_dir"] == str(preparer.DATA_CACHE_ROOT.resolve())
    assert config["data"]["split"]["manifest_path"] == str(
        preparer.PARTITION_MANIFEST.resolve()
    )
    assert config["data"]["split"]["manifest_sha256"] == _sha256(
        preparer.PARTITION_MANIFEST
    )
    assert config["protocol"]["stage"] == "S2"
    assert config["protocol"]["evidence_intent"] is True
    assert config["protocol"]["launch_status"] == "ready"
    assert config["trainer"]["num_epochs"] == 50
    assert config["trainer"]["early_stopping"] is True
    assert config["trainer"]["test_after_fit"] is False
    assert config["task"]["optimizer"] == "adamw"

    assert persisted_plan == plan
    assert plan["run_id"] == "P04-G050-S2-FULL-S42-A1"
    assert plan["attempt"] == 1
    assert plan["retry_reason"] is None
    assert plan["supersedes_attempt"] is None
    assert plan["execute"] is False
    assert plan["claim_eligible"] is True
    assert plan["physical_gpu_indices"] == [0]
    assert plan["multi_gpu"] is False
    assert plan["resolved_config_sha256"] == _sha256(config_path)
    assert plan["trainable_parameter_count"] == 62_069
    assert SHA256_RE.fullmatch(str(plan["trainable_parameter_signature_sha256"]))
    assert SHA256_RE.fullmatch(str(plan["git_diff_sha256"]))
    assert SHA256_RE.fullmatch(str(plan["code_artifact_sha256"]))
    assert "untracked files are excluded" in str(plan["git_diff_contract"])
    assert plan["argv"] == [
        "conda",
        "run",
        "-n",
        "LQ_signal",
        "env",
        "CUDA_VISIBLE_DEVICES=0",
        "python",
        "main.py",
        "--config",
        str(config_path),
    ]
    assert str(plan["command"]).startswith(
        "conda run -n LQ_signal env CUDA_VISIBLE_DEVICES=0 python main.py"
    )


@pytest.mark.parametrize(
    ("seed", "expected"),
    sorted(preparer.FROZEN_S2_RANDOM_ROLE_PERMUTATIONS.items()),
)
def test_s2_rand_atomically_binds_each_frozen_seed_permutation(
    tmp_path: Path, seed: int, expected: list[int]
) -> None:
    plan, _, staging = _prepare(
        tmp_path, arm="RAND", seed=seed, suffix=f"-{seed}"
    )
    config = yaml.safe_load(
        (staging / "resolved_config.yaml").read_text(encoding="utf-8")
    )

    assert plan["random_role_permutation"] == expected
    assert config["model"]["role_prior_permutation"] == expected
    assert config["protocol"]["runtime_bindings"][
        "seed_specific_random_role_permutation"
    ] == expected
    matches = [
        entry
        for entry in config["protocol"]["random_role_prior_permutations"]
        if entry["seed"] == seed
    ]
    assert matches == [{"seed": seed, "permutation": expected}]


def test_homo_preserves_capacity_and_changes_only_registered_representation(
    tmp_path: Path,
) -> None:
    plan, _, staging = _prepare(tmp_path, arm="HOMO", gpu=1)
    config = yaml.safe_load(
        (staging / "resolved_config.yaml").read_text(encoding="utf-8")
    )

    assert config["model"]["expert_representation_mode"] == "homogeneous_raw"
    assert config["model"]["role_prior_permutation"] == [0, 1, 2, 3]
    assert plan["trainable_parameter_count"] == 62_069
    assert plan["physical_gpu_indices"] == [1]
    assert plan["cuda_visible_devices"] == "1"


def test_s1_is_nonclaim_engineering_only_and_may_reference_existing_output(
    tmp_path: Path,
) -> None:
    output = tmp_path / "completed-s1-output"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("do not touch\n", encoding="utf-8")

    plan, resolved_output, staging = _prepare(
        tmp_path,
        arm="RAND",
        stage="S1",
        seed=314159,
        output=output,
    )
    config = yaml.safe_load(
        (staging / "resolved_config.yaml").read_text(encoding="utf-8")
    )

    assert resolved_output == output.resolve()
    assert sentinel.read_text(encoding="utf-8") == "do not touch\n"
    assert plan["run_id"] == "P04-G050-S1-RAND-S314159-A1"
    assert plan["claim_eligible"] is False
    assert plan["evidence_intent"] is False
    assert plan["training_output_preexisted"] is True
    assert config["protocol"]["launch_status"] == "engineering_calibration"
    assert config["trainer"]["num_epochs"] == 1
    assert config["trainer"]["early_stopping"] is False
    assert config["trainer"]["test_after_fit"] is False
    assert config["model"]["role_prior_permutation"] == [1, 2, 3, 0]
    matches = [
        entry
        for entry in config["protocol"]["random_role_prior_permutations"]
        if entry["seed"] == 314159
    ]
    assert matches == [
        {"seed": 314159, "permutation": [1, 2, 3, 0]}
    ]


def test_s1_allows_explicit_second_engineering_attempt(tmp_path: Path) -> None:
    plan, _, staging = _prepare(
        tmp_path,
        arm="FULL",
        stage="S1",
        seed=314159,
        attempt=2,
        suffix="-attempt-2",
    )
    persisted = json.loads(
        (staging / "launch_plan.json").read_text(encoding="utf-8")
    )

    assert plan["attempt"] == 2
    assert plan["run_id"] == "P04-G050-S1-FULL-S314159-A2"
    assert plan["claim_eligible"] is False
    assert persisted["attempt"] == 2
    assert persisted["run_id"] == plan["run_id"]


@pytest.mark.parametrize(
    ("stage", "seed", "gpu", "match"),
    [
        ("S2", 314159, 0, "S2 seed"),
        ("S1", 42, 0, "S1 seed"),
        ("S2", 42, 2, "physical GPU 2 is forbidden"),
    ],
)
def test_invalid_stage_seed_or_gpu_fails_before_writing(
    tmp_path: Path, stage: str, seed: int, gpu: int, match: str
) -> None:
    staging = tmp_path / "must-not-exist"
    with pytest.raises(preparer.RunPreparationError, match=match):
        preparer.prepare_decisive_run(
            arm="FULL",
            stage=stage,
            seed=seed,
            physical_gpu=gpu,
            output_dir=tmp_path / "output",
            staging_dir=staging,
        )
    assert not staging.exists()


@pytest.mark.parametrize("attempt", [0, -1, True, 1.5])
def test_nonpositive_or_noninteger_attempt_is_rejected_before_writing(
    tmp_path: Path, attempt: object
) -> None:
    staging = tmp_path / f"invalid-attempt-{attempt!s}"
    with pytest.raises(preparer.RunPreparationError, match="positive integer"):
        preparer.prepare_decisive_run(
            arm="FULL",
            stage="S1",
            seed=314159,
            attempt=attempt,  # type: ignore[arg-type]
            physical_gpu=0,
            output_dir=tmp_path / "output",
            staging_dir=staging,
        )
    assert not staging.exists()


def test_s2_rejects_second_attempt_without_exact_retry_reason(
    tmp_path: Path,
) -> None:
    staging = tmp_path / "s2-attempt-2"
    with pytest.raises(preparer.RunPreparationError, match="requires retry_reason"):
        preparer.prepare_decisive_run(
            arm="FULL",
            stage="S2",
            seed=42,
            attempt=2,
            physical_gpu=0,
            output_dir=tmp_path / "output",
            staging_dir=staging,
        )
    assert not staging.exists()


def test_s2_rejects_wrong_retry_reason_and_attempt_above_two(
    tmp_path: Path,
) -> None:
    for attempt, reason, match in [
        (2, "some_other_reason", "requires retry_reason"),
        (3, preparer.S2_RETRY_REASON, "exactly 1 or 2"),
    ]:
        staging = tmp_path / f"s2-attempt-{attempt}-{reason}"
        with pytest.raises(preparer.RunPreparationError, match=match):
            preparer.prepare_decisive_run(
                arm="FULL",
                stage="S2",
                seed=42,
                attempt=attempt,
                retry_reason=reason,
                physical_gpu=0,
                output_dir=tmp_path / f"output-{attempt}",
                staging_dir=staging,
            )
        assert not staging.exists()


def test_s2_a2_exact_numeric_verifier_retry_is_claim_eligible_and_finalizable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, output, staging = _prepare(
        tmp_path,
        arm="FULL",
        stage="S2",
        seed=42,
        attempt=2,
        retry_reason=preparer.S2_RETRY_REASON,
        suffix="-governed-retry",
    )
    assert plan["run_id"] == "P04-G050-S2-FULL-S42-A2"
    assert plan["claim_eligible"] is True
    assert plan["retry_reason"] == preparer.S2_RETRY_REASON
    assert plan["supersedes_attempt"] == 1

    checkpoint = output / "run" / "model-epoch=01-val_loss=0.2000.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"governed retry fixture checkpoint")
    monkeypatch.setattr(preparer, "_query_gpu_model", lambda gpu: "fixture RTX 4090")
    meta = preparer.finalize_run_meta(
        launch_plan=staging / "launch_plan.json",
        resolved_config=staging / "resolved_config.yaml",
        checkpoint=checkpoint,
        started_at="2026-08-01T13:00:00+08:00",
        ended_at="2026-08-01T13:01:00+08:00",
        runtime_seconds=60.0,
        exit_code=0,
    )
    assert meta["run_id"] == plan["run_id"]
    assert meta["claim_eligible"] is True
    assert meta["retry_reason"] == preparer.S2_RETRY_REASON
    assert meta["supersedes_attempt"] == 1


def test_s2_refuses_existing_output_and_all_stages_refuse_existing_staging(
    tmp_path: Path,
) -> None:
    output = tmp_path / "existing-output"
    output.mkdir()
    staging = tmp_path / "new-staging"
    with pytest.raises(FileExistsError, match="existing evidence output"):
        preparer.prepare_decisive_run(
            arm="FULL",
            stage="S2",
            seed=42,
            physical_gpu=0,
            output_dir=output,
            staging_dir=staging,
        )
    assert not staging.exists()

    existing_staging = tmp_path / "existing-staging"
    existing_staging.mkdir()
    marker = existing_staging / "marker"
    marker.write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError, match="existing staging path"):
        preparer.prepare_decisive_run(
            arm="FULL",
            stage="S1",
            seed=314159,
            physical_gpu=0,
            output_dir=output,
            staging_dir=existing_staging,
        )
    assert marker.read_text(encoding="utf-8") == "preserve"


def test_path_overlap_is_rejected_without_creating_either_path(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    staging = output / "preparation"
    with pytest.raises(preparer.RunPreparationError, match="must not overlap"):
        preparer.prepare_decisive_run(
            arm="FULL",
            stage="S2",
            seed=42,
            physical_gpu=0,
            output_dir=output,
            staging_dir=staging,
        )
    assert not output.exists()
    assert not staging.exists()


def test_frozen_config_hash_mismatch_fails_closed_without_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    relative = "configs/experiments/p04/decisive_full.yaml"
    monkeypatch.setitem(preparer.FROZEN_CONFIG_SHA256, relative, "0" * 64)
    staging = tmp_path / "staging"

    with pytest.raises(preparer.RunPreparationError, match="frozen config SHA-256 mismatch"):
        preparer.prepare_decisive_run(
            arm="FULL",
            stage="S2",
            seed=42,
            physical_gpu=0,
            output_dir=tmp_path / "output",
            staging_dir=staging,
        )
    assert not staging.exists()


def test_finalize_meta_writes_complete_packager_contract_exclusively(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, output, staging = _prepare(tmp_path, arm="FULL", seed=42)
    checkpoint = output / "nested" / "model-epoch=07-val_loss=0.1000.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"deterministic fixture checkpoint")
    monkeypatch.setattr(preparer, "_query_gpu_model", lambda gpu: "fixture RTX 4090")

    meta = preparer.finalize_run_meta(
        launch_plan=staging / "launch_plan.json",
        resolved_config=staging / "resolved_config.yaml",
        checkpoint=checkpoint,
        started_at="2026-08-01T12:00:00+08:00",
        ended_at="2026-08-01T12:01:00+08:00",
        runtime_seconds=60.0,
        exit_code=0,
    )
    run_meta_path = staging / "run_meta.yaml"
    persisted = yaml.safe_load(run_meta_path.read_text(encoding="utf-8"))

    assert persisted == {key: value for key, value in meta.items() if key != "run_meta"}
    assert persisted["run_id"] == plan["run_id"]
    assert set(REQUIRED_RUN_META_FIELDS).issubset(persisted)
    assert persisted["status"] == "completed"
    assert persisted["dataset"] == "P04_SYNTHETIC"
    assert persisted["arm"] == "FULL"
    assert persisted["gpu_model"] == "fixture RTX 4090"
    assert persisted["fallback_used"] is False
    assert persisted["checkpoint_sha256"] == _sha256(checkpoint)
    assert persisted["resolved_config_sha256"] == _sha256(
        staging / "resolved_config.yaml"
    )
    assert persisted["source_metadata_sha256"] == _sha256(
        preparer.GENERATOR_MANIFEST
    )
    assert persisted["derived_metadata_sha256"] == _sha256(preparer.METADATA_FILE)

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        preparer.finalize_run_meta(
            launch_plan=staging / "launch_plan.json",
            resolved_config=staging / "resolved_config.yaml",
            checkpoint=checkpoint,
            started_at="2026-08-01T12:00:00+08:00",
            ended_at="2026-08-01T12:01:00+08:00",
            runtime_seconds=60.0,
            exit_code=0,
        )


def test_finalize_meta_rejects_nonzero_exit_and_multiple_checkpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, output, staging = _prepare(tmp_path, arm="HOMO", seed=42)
    first = output / "first.ckpt"
    output.mkdir()
    first.write_bytes(b"first")
    monkeypatch.setattr(preparer, "_query_gpu_model", lambda gpu: "fixture GPU")

    with pytest.raises(preparer.RunPreparationError, match="exit_code=0"):
        preparer.finalize_run_meta(
            launch_plan=staging / "launch_plan.json",
            resolved_config=staging / "resolved_config.yaml",
            checkpoint=first,
            started_at="2026-08-01T12:00:00+08:00",
            ended_at="2026-08-01T12:01:00+08:00",
            runtime_seconds=60.0,
            exit_code=1,
        )
    assert not (staging / "run_meta.yaml").exists()

    (output / "second.ckpt").write_bytes(b"second")
    with pytest.raises(preparer.RunPreparationError, match="exactly the supplied checkpoint"):
        preparer.finalize_run_meta(
            launch_plan=staging / "launch_plan.json",
            resolved_config=staging / "resolved_config.yaml",
            checkpoint=first,
            started_at="2026-08-01T12:00:00+08:00",
            ended_at="2026-08-01T12:01:00+08:00",
            runtime_seconds=60.0,
            exit_code=0,
        )
    assert not (staging / "run_meta.yaml").exists()


def test_cli_parser_defaults_to_strict_s2() -> None:
    args = preparer.build_parser().parse_args(
        [
            "--arm",
            "FULL",
            "--seed",
            "42",
            "--physical-gpu",
            "0",
            "--output-dir",
            "/tmp/output",
            "--staging-dir",
            "/tmp/staging",
        ]
    )
    assert args.stage == "S2"
    assert args.attempt == 1
