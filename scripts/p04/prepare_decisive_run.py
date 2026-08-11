"""Prepare, validate, and publish one P04 decisive-run launch contract.

This command is deliberately non-executing.  It resolves the maintained arm
configuration through PHMFactory's public merge/override path, validates the
frozen protocol and local governed artifacts, and atomically publishes exactly
two preparation artifacts: ``resolved_config.yaml`` and ``launch_plan.json``.

S2 is evidence-intent and refuses both an existing training output path and an
existing staging path.  S1 is an engineering-only compatibility mode for the
already registered seed 314159: it may refer to an existing training output
directory read-only, but still refuses an existing staging path.
"""

from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import errno
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from phmfactory.config import resolve_config


SCHEMA_ID = "p04.decisive-run-preparation.v1"
SCHEMA_VERSION = "1.0.0"
PAPER_ID = "P04"
GOAL_ID = "P04-G050"
EXPERIMENT_ID = "E-MINDEC"
DATASET = "P04_SYNTHETIC"
CONDA_ENVIRONMENT = "LQ_signal"
ALLOWED_ARMS = ("FULL", "HOMO", "RAND")
ALLOWED_PHYSICAL_GPUS = (0, 1)
FROZEN_S2_SEEDS = (42, 123, 456, 789, 1024)
S1_ENGINEERING_SEED = 314159
S2_RETRY_REASON = "pre_evidence_numeric_verifier_defect"
EXPECTED_TRAINABLE_PARAMETERS = 62_069

FROZEN_S2_RANDOM_ROLE_PERMUTATIONS: dict[int, list[int]] = {
    42: [1, 2, 3, 0],
    123: [2, 3, 0, 1],
    456: [3, 0, 1, 2],
    789: [1, 3, 0, 2],
    1024: [2, 0, 3, 1],
}
S1_ENGINEERING_RANDOM_ROLE_PERMUTATION = [1, 2, 3, 0]

VIBENCH_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATHS: dict[str, Path] = {
    "FULL": VIBENCH_ROOT / "configs/experiments/p04/decisive_full.yaml",
    "HOMO": VIBENCH_ROOT / "configs/experiments/p04/decisive_homogeneous.yaml",
    "RAND": VIBENCH_ROOT / "configs/experiments/p04/decisive_random_role.yaml",
}
DATA_ROOT = VIBENCH_ROOT / "data/derived/p04/synthetic_v1"
DATA_CACHE_ROOT = VIBENCH_ROOT / "data/cache/p04/synthetic_v1"
PARTITION_MANIFEST = DATA_ROOT / "partition_manifest.json"
GENERATOR_MANIFEST = DATA_ROOT / "generator_manifest.json"
METADATA_FILE = DATA_ROOT / "metadata.csv"
LOCAL_OVERRIDE = VIBENCH_ROOT / "configs/local/local.yaml"

# These byte hashes cover P04-owned protocol inputs. Shared base configs are
# deliberately excluded: they are allowed to evolve on ``dev`` and are checked
# through the resolved semantic contract in ``_validate_resolved_config``.
# All paths are relative to VIBENCH_ROOT.
FROZEN_CONFIG_SHA256: dict[str, str] = {
    "configs/experiments/p04/decisive_full.yaml": (
        "e60b0f00be3d60d989253eb23d7f16491cb3f821af5b8d286699ed1a252f032a"
    ),
    "configs/experiments/p04/decisive_homogeneous.yaml": (
        "2f5e6caf024920acf95786a01cd5a4ab8d876cfffba8384e42c72bea13eb8610"
    ),
    "configs/experiments/p04/decisive_random_role.yaml": (
        "8fde7467db5f7af90e17b70a76cc263fdcafac99977b6fd1834e6f0b541d4968"
    ),
    "configs/base/environment/base.yaml": (
        "c10982a67e87c1293d1a44ba7fc3fc10202fee33e9fd1f03aa05b1d762afa514"
    ),
    "configs/base/model/role_constrained_moe.yaml": (
        "61e9ec95717f2141f7d2c1abc15b145560728f3eabf03a680ae53a18fce49fd3"
    ),
    "configs/base/task/classification.yaml": (
        "fa7e9f27c38804a6aec22fa2483b6fab5f1937a3f57072c2a29ea64e8d296fa1"
    ),
    "configs/base/trainer/default_single_gpu.yaml": (
        "c9ca475d53b0131576602e21b44d0959059ac2ac248c60e51ef573a88cf66c6a"
    ),
}

EXPECTED_RUNTIME_BINDINGS: dict[str, str] = {
    "generator_source_sha256": (
        "747420e82c6ece64731481458a4d261dfb2863cbef6a2acea694bbbe42288f1a"
    ),
    "generator_manifest_sha256": (
        "39f37e326a11e7d0b59c91458c4f15eebf2d5f436ab6904b1a922ec2dc36c68e"
    ),
    "partition_manifest_sha256": (
        "5788451a4666762fac53a96008e982e4d202f46e0b834e4cee7c81b9a32effd9"
    ),
}

# This inventory supplements ``git diff`` because the G050 implementation may
# legitimately still be untracked.  Its aggregate is deterministic and records
# the exact launch-critical implementation/config byte state.
CODE_ARTIFACT_PATHS = (
    "main.py",
    "phmfactory/config.py",
    "src/Pipeline_01_Fault_Diagnosis.py",
    "src/configs/config_utils.py",
    "src/utils/config_utils.py",
    "src/data_factory/data_factory.py",
    "src/data_factory/splitting.py",
    "src/data_factory/reader/P04_Synthetic.py",
    "src/model_factory/MoE/M_04_RoleConstrainedMoE.py",
    "src/trainer_factory/Default_trainer.py",
    "scripts/p04/generate_synthetic.py",
    "scripts/p04/collect_checkpoint_outputs.py",
    "scripts/p04/evaluate_role_identification.py",
    "scripts/p04/evaluate_predictions.py",
    "scripts/p04/package_decisive_run.py",
    "scripts/p04/aggregate_decisive.py",
    "scripts/p04/prepare_decisive_run.py",
    *tuple(FROZEN_CONFIG_SHA256),
)

SHA256_RE = re.compile(r"[0-9a-f]{64}")
GIT_COMMIT_RE = re.compile(r"[0-9a-f]{40}")


class RunPreparationError(ValueError):
    """Raised when a requested run is not safe to prepare."""


def _lexists(path: Path) -> bool:
    return os.path.lexists(os.fspath(path))


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_regular_file(path: Path, description: str) -> Path:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise FileNotFoundError(
            f"{description} must be a regular non-symlink file: {expanded}"
        )
    resolved = expanded.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(
            f"{description} must be a regular non-symlink file: {resolved}"
        )
    return resolved


def _require_directory(path: Path, description: str) -> Path:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise FileNotFoundError(
            f"{description} must be a non-symlink directory: {expanded}"
        )
    resolved = expanded.resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(
            f"{description} must be a non-symlink directory: {resolved}"
        )
    return resolved


def _relative(path: Path) -> str:
    return path.resolve().relative_to(VIBENCH_ROOT).as_posix()


def _inventory_hash(hashes: Mapping[str, str]) -> str:
    material = "".join(
        f"{hashes[path]}  {path}\n" for path in sorted(hashes)
    ).encode("utf-8")
    return _sha256_bytes(material)


def _validate_frozen_config_files() -> dict[str, str]:
    observed: dict[str, str] = {}
    for relative, expected in FROZEN_CONFIG_SHA256.items():
        path = _require_regular_file(VIBENCH_ROOT / relative, "frozen config input")
        digest = _sha256_file(path)
        if digest != expected:
            raise RunPreparationError(
                f"frozen config SHA-256 mismatch for {relative}: "
                f"expected {expected}, got {digest}"
            )
        observed[relative] = digest
    return observed


def _load_generator_manifest() -> Mapping[str, Any]:
    manifest_path = _require_regular_file(GENERATOR_MANIFEST, "generator manifest")
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RunPreparationError(f"cannot parse generator manifest: {exc}") from exc
    if not isinstance(value, Mapping):
        raise RunPreparationError("generator manifest must be a JSON mapping")
    return value


def _validate_runtime_artifacts() -> dict[str, str]:
    _require_directory(DATA_ROOT, "immutable synthetic data root")
    _require_directory(DATA_CACHE_ROOT, "isolated synthetic cache root")
    manifest_path = _require_regular_file(PARTITION_MANIFEST, "partition manifest")
    generator_source = _require_regular_file(
        VIBENCH_ROOT / "scripts/p04/generate_synthetic.py", "generator source"
    )
    evaluator_source = _require_regular_file(
        VIBENCH_ROOT / "scripts/p04/evaluate_role_identification.py",
        "held-out evaluator source",
    )
    actual = {
        "generator_source_sha256": _sha256_file(generator_source),
        "generator_manifest_sha256": _sha256_file(GENERATOR_MANIFEST),
        "partition_manifest_sha256": _sha256_file(manifest_path),
        "held_out_evaluator_source_sha256": _sha256_file(evaluator_source),
    }
    # Generator/data bytes are protocol-frozen.  The evaluator is being sealed
    # as code, so its current byte hash is computed here and must match every
    # arm config rather than being duplicated as another mutable constant.
    for key, expected in EXPECTED_RUNTIME_BINDINGS.items():
        if actual[key] != expected:
            raise RunPreparationError(
                f"runtime binding {key} mismatch: expected {expected}, got {actual[key]}"
            )

    generator_manifest = _load_generator_manifest()
    source = generator_manifest.get("source")
    content_hashes = generator_manifest.get("content_hashes")
    if not isinstance(source, Mapping) or not isinstance(content_hashes, Mapping):
        raise RunPreparationError(
            "generator manifest must contain source and content_hashes mappings"
        )
    if source.get("sha256") != actual["generator_source_sha256"]:
        raise RunPreparationError("generator manifest source SHA-256 is not current")
    if (
        content_hashes.get("partition_manifest_sha256")
        != actual["partition_manifest_sha256"]
    ):
        raise RunPreparationError(
            "generator manifest partition-manifest SHA-256 is not current"
        )
    return actual


def _code_artifact_inventory() -> tuple[dict[str, str], str]:
    hashes: dict[str, str] = {}
    for relative in CODE_ARTIFACT_PATHS:
        if relative in hashes:
            continue
        path = _require_regular_file(
            VIBENCH_ROOT / relative, f"launch-critical artifact {relative}"
        )
        hashes[relative] = _sha256_file(path)
    return hashes, _inventory_hash(hashes)


def _git_state() -> tuple[str, str]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=VIBENCH_ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ).stdout.strip()
        diff = subprocess.run(
            ["git", "diff", "--binary", "HEAD"],
            cwd=VIBENCH_ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RunPreparationError(f"cannot capture nested-repository Git state: {exc}") from exc
    if GIT_COMMIT_RE.fullmatch(commit) is None:
        raise RunPreparationError(f"nested-repository HEAD is not a full Git SHA: {commit!r}")
    return commit, _sha256_bytes(diff)


def _get(config: Mapping[str, Any], dotted: str) -> Any:
    value: Any = config
    traversed: list[str] = []
    for part in dotted.split("."):
        traversed.append(part)
        if not isinstance(value, Mapping) or part not in value:
            raise RunPreparationError(
                f"resolved config is missing {'.'.join(traversed)}"
            )
        value = value[part]
    return value


def _require_config_value(
    config: Mapping[str, Any], dotted: str, expected: Any
) -> None:
    observed = _get(config, dotted)
    if type(observed) is not type(expected) or observed != expected:
        raise RunPreparationError(
            f"resolved config {dotted} must equal {expected!r}; got {observed!r}"
        )


def _validate_source_random_role_table(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    entries = _get(config, "protocol.random_role_prior_permutations")
    if not isinstance(entries, list):
        raise RunPreparationError("RAND seed/permutation table must be a list")
    normalized: list[dict[str, Any]] = []
    seen: set[int] = set()
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise RunPreparationError("RAND seed/permutation entry must be a mapping")
        seed = entry.get("seed")
        permutation = entry.get("permutation")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed in seen:
            raise RunPreparationError("RAND seed/permutation table has an invalid seed")
        if (
            not isinstance(permutation, list)
            or any(isinstance(value, bool) or not isinstance(value, int) for value in permutation)
            or sorted(permutation) != [0, 1, 2, 3]
            or any(index == value for index, value in enumerate(permutation))
        ):
            raise RunPreparationError(
                f"RAND permutation for seed {seed} must be fixed-point-free"
            )
        seen.add(seed)
        normalized.append({"seed": seed, "permutation": list(permutation)})
    for seed, expected in FROZEN_S2_RANDOM_ROLE_PERMUTATIONS.items():
        matches = [entry for entry in normalized if entry["seed"] == seed]
        if len(matches) != 1 or matches[0]["permutation"] != expected:
            raise RunPreparationError(
                f"RAND frozen permutation mismatch for S2 seed {seed}"
            )
    return normalized


def _json_override(key: str, value: Any) -> str:
    return f"{key}={json.dumps(value, ensure_ascii=False, separators=(',', ':'))}"


def _resolve_run_config(
    *, arm: str, stage: str, seed: int, output_dir: Path
) -> tuple[dict[str, Any], str, list[int] | None]:
    config_path = CONFIG_PATHS[arm]
    initial = resolve_config(config_path)
    if initial.path != config_path.resolve():
        raise RunPreparationError("public config resolver selected an unexpected source path")

    random_role_table: list[dict[str, Any]] | None = None
    permutation: list[int] | None = None
    if arm == "RAND":
        random_role_table = _validate_source_random_role_table(initial.data)
        permutation = (
            list(FROZEN_S2_RANDOM_ROLE_PERMUTATIONS[seed])
            if stage == "S2"
            else list(S1_ENGINEERING_RANDOM_ROLE_PERMUTATION)
        )

    evidence_intent = stage == "S2"
    overrides = [
        _json_override("environment.seed", seed),
        _json_override("environment.output_dir", str(output_dir)),
        _json_override("data.cache_dir", str(DATA_CACHE_ROOT.resolve())),
        _json_override("data.split.manifest_path", str(PARTITION_MANIFEST.resolve())),
        _json_override(
            "data.split.manifest_sha256",
            EXPECTED_RUNTIME_BINDINGS["partition_manifest_sha256"],
        ),
        _json_override("protocol.stage", stage),
        _json_override("protocol.arm", arm),
        _json_override("protocol.evidence_intent", evidence_intent),
        _json_override(
            "protocol.launch_status",
            "ready" if evidence_intent else "engineering_calibration",
        ),
        _json_override("trainer.device", "cuda"),
        _json_override("trainer.gpus", 1),
    ]
    if stage == "S1":
        overrides.extend(
            [
                _json_override("trainer.num_epochs", 1),
                _json_override("trainer.early_stopping", False),
            ]
        )
    if permutation is not None:
        overrides.extend(
            [
                _json_override("model.role_prior_permutation", permutation),
                _json_override(
                    "protocol.runtime_bindings.seed_specific_random_role_permutation",
                    permutation,
                ),
            ]
        )
        if stage == "S1":
            assert random_role_table is not None
            engineering_table = [
                entry for entry in random_role_table if entry["seed"] != seed
            ]
            engineering_table.append({"seed": seed, "permutation": permutation})
            overrides.append(
                _json_override(
                    "protocol.random_role_prior_permutations", engineering_table
                )
            )

    resolved = resolve_config(config_path, override_values=overrides)
    return resolved.data, _sha256_file(config_path), permutation


def _validate_capacity_matrix() -> tuple[int, str]:
    # Importing and instantiating on CPU verifies the implementation's actual
    # trainable parameter signature, while preserving the caller's torch RNG.
    import torch

    from src.configs.config_utils import load_config
    from src.model_factory.MoE.M_04_RoleConstrainedMoE import Model

    cpu_state = torch.random.get_rng_state()
    try:
        signatures: dict[str, tuple[tuple[str, tuple[int, ...]], ...]] = {}
        counts: dict[str, int] = {}
        for arm, config_path in CONFIG_PATHS.items():
            config = load_config(config_path)
            model = Model(config.model)
            signature = tuple(
                (name, tuple(parameter.shape))
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            )
            signatures[arm] = signature
            counts[arm] = sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            )
    finally:
        torch.random.set_rng_state(cpu_state)

    if not all(signatures[arm] == signatures["FULL"] for arm in ALLOWED_ARMS):
        raise RunPreparationError(
            "FULL/HOMO/RAND trainable parameter names or shapes are not identical"
        )
    if any(count != EXPECTED_TRAINABLE_PARAMETERS for count in counts.values()):
        raise RunPreparationError(
            f"three-arm trainable capacity must be {EXPECTED_TRAINABLE_PARAMETERS}; "
            f"got {counts}"
        )
    signature_bytes = json.dumps(
        signatures["FULL"], separators=(",", ":")
    ).encode("utf-8")
    return EXPECTED_TRAINABLE_PARAMETERS, _sha256_bytes(signature_bytes)


def _validate_resolved_config(
    config: Mapping[str, Any],
    *,
    arm: str,
    stage: str,
    seed: int,
    output_dir: Path,
    runtime_bindings: Mapping[str, str],
    permutation: list[int] | None,
) -> None:
    common_values = {
        "pipeline": "Pipeline_01_Fault_Diagnosis",
        "protocol.paper_id": PAPER_ID,
        "protocol.goal_id": GOAL_ID,
        "protocol.experiment_id": EXPERIMENT_ID,
        "protocol.stage": stage,
        "protocol.arm": arm,
        "protocol.evidence_intent": stage == "S2",
        "protocol.launch_status": (
            "ready" if stage == "S2" else "engineering_calibration"
        ),
        "protocol.current_runtime_blockers": [],
        "environment.seed": seed,
        "environment.iterations": 1,
        "environment.output_dir": str(output_dir),
        "data.data_dir": "data/derived/p04/synthetic_v1",
        "data.metadata_file": "metadata.csv",
        "data.cache_dir": str(DATA_CACHE_ROOT.resolve()),
        "data.batch_size": 64,
        "data.num_workers": 0,
        "data.normalization": "none",
        "data.window_size": 512,
        "data.stride": 512,
        "data.num_window": 1,
        "data.dtype": "float32",
        "data.split.strategy": "grouped_metadata",
        "data.split.group_key": "Split_group",
        "data.split.stratify_key": "Split_stratum",
        "data.split.seed": 240401,
        "data.split.test_policy": "partition",
        "data.split.manifest_path": str(PARTITION_MANIFEST.resolve()),
        "data.split.manifest_mode": "read_only",
        "data.split.manifest_sha256": runtime_bindings[
            "partition_manifest_sha256"
        ],
        "model.type": "MoE",
        "model.name": "M_04_RoleConstrainedMoE",
        "model.input_dim": 2,
        "model.num_classes": 4,
        "model.feature_dim": 64,
        "model.expert_hidden_channels": 32,
        "model.router_hidden_dim": 32,
        "model.dropout": 0.1,
        "model.routing_temperature": 1.0,
        "model.router_mode": "learned_prior",
        "model.low_cutoff": 0.12,
        "model.envelope_band": [0.2, 0.8],
        "model.filter_transition": 0.03,
        "model.role_prior_strength": 0.5,
        "model.role_prior_max": 1.0,
        "model.load_balance_weight": 0.01,
        "model.entropy_floor_weight": 0.01,
        "model.entropy_floor": 0.25,
        "task.optimizer": "adamw",
        "task.lr": 0.001,
        "task.weight_decay": 0.0001,
        "task.epochs": 50,
        "trainer.name": "Default_trainer",
        "trainer.monitor": "val_loss",
        "trainer.num_epochs": 50 if stage == "S2" else 1,
        "trainer.early_stopping": stage == "S2",
        "trainer.patience": 7,
        "trainer.min_delta": 0.0001,
        "trainer.deterministic": True,
        "trainer.device": "cuda",
        "trainer.gpus": 1,
        "trainer.test_after_fit": False,
    }
    for dotted, expected in common_values.items():
        _require_config_value(config, dotted, expected)

    expected_representation = "homogeneous_raw" if arm == "HOMO" else "role_constrained"
    _require_config_value(
        config, "model.expert_representation_mode", expected_representation
    )
    if arm == "RAND":
        assert permutation is not None
        _require_config_value(config, "model.role_prior_permutation", permutation)
        _require_config_value(
            config, "model.role_prior_assignment", "external_deranged"
        )
        _require_config_value(
            config,
            "protocol.runtime_bindings.seed_specific_random_role_permutation",
            permutation,
        )
        matches = [
            entry
            for entry in _get(config, "protocol.random_role_prior_permutations")
            if isinstance(entry, Mapping) and entry.get("seed") == seed
        ]
        if len(matches) != 1 or matches[0].get("permutation") != permutation:
            raise RunPreparationError(
                "resolved RAND config lacks exactly one seed-specific permutation binding"
            )
    else:
        _require_config_value(config, "model.role_prior_permutation", [0, 1, 2, 3])
        _require_config_value(config, "model.role_prior_assignment", "aligned")

    required_runtime_bindings = _get(config, "protocol.required_runtime_bindings")
    required = set(runtime_bindings)
    if arm == "RAND":
        required.add("seed_specific_random_role_permutation")
    if not isinstance(required_runtime_bindings, list) or set(required_runtime_bindings) != required:
        raise RunPreparationError(
            "resolved config required_runtime_bindings does not match the frozen arm contract"
        )
    for key, expected in runtime_bindings.items():
        _require_config_value(config, f"protocol.runtime_bindings.{key}", expected)


def _validate_request(
    arm: str,
    stage: str,
    seed: int,
    attempt: int,
    retry_reason: str | None,
    physical_gpu: int,
    output_dir: Path,
    staging_dir: Path,
) -> tuple[str, str]:
    normalized_arm = arm.upper()
    normalized_stage = stage.upper()
    if normalized_arm not in ALLOWED_ARMS:
        raise RunPreparationError(
            f"arm must be one of {', '.join(ALLOWED_ARMS)}; got {arm!r}"
        )
    if normalized_stage not in {"S1", "S2"}:
        raise RunPreparationError("stage must be S1 or S2")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise RunPreparationError("seed must be an integer")
    if normalized_stage == "S2" and seed not in FROZEN_S2_SEEDS:
        raise RunPreparationError(
            f"S2 seed must be one of {list(FROZEN_S2_SEEDS)}"
        )
    if normalized_stage == "S1" and seed != S1_ENGINEERING_SEED:
        raise RunPreparationError(
            f"S1 seed must equal engineering seed {S1_ENGINEERING_SEED}"
        )
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt <= 0:
        raise RunPreparationError("attempt must be a positive integer")
    if normalized_stage == "S2":
        if attempt == 1:
            if retry_reason is not None:
                raise RunPreparationError(
                    "S2 attempt=1 must not set retry_reason"
                )
        elif attempt == 2:
            if retry_reason != S2_RETRY_REASON:
                raise RunPreparationError(
                    "S2 attempt=2 requires retry_reason=" + S2_RETRY_REASON
                )
        else:
            raise RunPreparationError("S2 attempt must be exactly 1 or 2")
    elif retry_reason is not None:
        raise RunPreparationError("retry_reason is reserved for the governed S2 retry")
    if isinstance(physical_gpu, bool) or physical_gpu not in ALLOWED_PHYSICAL_GPUS:
        raise RunPreparationError(
            "physical_gpu must be exactly 0 or 1; physical GPU 2 is forbidden"
        )
    if output_dir == staging_dir or output_dir in staging_dir.parents or staging_dir in output_dir.parents:
        raise RunPreparationError("training output and staging paths must not overlap")
    return normalized_arm, normalized_stage


def _validate_destination_state(
    *, stage: str, output_dir: Path, staging_dir: Path
) -> bool:
    if _lexists(staging_dir):
        raise FileExistsError(f"refusing to overwrite existing staging path: {staging_dir}")
    output_exists = _lexists(output_dir)
    if stage == "S2" and output_exists:
        raise FileExistsError(
            f"refusing to prepare S2 with an existing evidence output path: {output_dir}"
        )
    if stage == "S1" and output_exists:
        if output_dir.is_symlink() or not output_dir.is_dir():
            raise FileExistsError(
                "existing S1 output reference must be a non-symlink directory: "
                f"{output_dir}"
            )
    return output_exists


def _publish_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is not None:
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        result = renameat2(
            -100,
            os.fsencode(source),
            -100,
            os.fsencode(destination),
            1,
        )
        if result == 0:
            return
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FileExistsError(
                error, os.strerror(error), os.fspath(destination)
            )
        if error not in {errno.ENOSYS, errno.EINVAL, errno.ENOTSUP}:
            raise OSError(error, os.strerror(error), os.fspath(destination))
    if _lexists(destination):
        raise FileExistsError(f"refusing to overwrite existing staging path: {destination}")
    os.rename(source, destination)


def _write_new_file(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _cleanup_temporary_directory(path: Path, parent: Path, prefix: str) -> None:
    if path.parent != parent or not path.name.startswith(prefix):
        raise RuntimeError("refusing to clean an unrecognized preparation directory")
    if path.exists():
        shutil.rmtree(path)


def prepare_decisive_run(
    *,
    arm: str,
    seed: int,
    attempt: int = 1,
    retry_reason: str | None = None,
    physical_gpu: int,
    output_dir: str | Path,
    staging_dir: str | Path,
    stage: str = "S2",
) -> dict[str, Any]:
    """Validate and atomically materialize one non-executing launch contract."""

    requested_output = Path(output_dir).expanduser()
    requested_staging = Path(staging_dir).expanduser()
    if requested_output.is_symlink() or requested_staging.is_symlink():
        raise RunPreparationError("training output and staging paths must not be symlinks")
    output = requested_output.resolve()
    staging = requested_staging.resolve()
    normalized_arm, normalized_stage = _validate_request(
        arm,
        stage,
        seed,
        attempt,
        retry_reason,
        physical_gpu,
        output,
        staging,
    )
    output_preexisted = _validate_destination_state(
        stage=normalized_stage, output_dir=output, staging_dir=staging
    )
    if _lexists(LOCAL_OVERRIDE):
        raise RunPreparationError(
            "configs/local/local.yaml exists and would silently alter the resolved launch"
        )

    config_hashes = _validate_frozen_config_files()
    runtime_bindings = _validate_runtime_artifacts()
    resolved, source_config_sha256, permutation = _resolve_run_config(
        arm=normalized_arm,
        stage=normalized_stage,
        seed=seed,
        output_dir=output,
    )
    _validate_resolved_config(
        resolved,
        arm=normalized_arm,
        stage=normalized_stage,
        seed=seed,
        output_dir=output,
        runtime_bindings=runtime_bindings,
        permutation=permutation,
    )
    parameter_count, parameter_signature_sha256 = _validate_capacity_matrix()
    code_artifacts, code_artifact_sha256 = _code_artifact_inventory()
    git_commit, git_diff_sha256 = _git_state()

    resolved_yaml = yaml.safe_dump(
        resolved,
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
    ).encode("utf-8")
    resolved_config_sha256 = _sha256_bytes(resolved_yaml)
    final_config = staging / "resolved_config.yaml"
    argv = [
        "conda",
        "run",
        "-n",
        CONDA_ENVIRONMENT,
        "env",
        f"CUDA_VISIBLE_DEVICES={physical_gpu}",
        "python",
        "main.py",
        "--config",
        str(final_config),
    ]
    run_id = (
        f"{GOAL_ID}-{normalized_stage}-{normalized_arm}-S{seed}-A{attempt}"
    )
    launch_plan: dict[str, Any] = {
        "schema_id": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "paper_id": PAPER_ID,
        "goal_id": GOAL_ID,
        "experiment_id": EXPERIMENT_ID,
        "stage": normalized_stage,
        "arm": normalized_arm,
        "training_seed": seed,
        "attempt": attempt,
        "retry_reason": retry_reason,
        "supersedes_attempt": (
            1 if normalized_stage == "S2" and attempt == 2 else None
        ),
        "claim_eligible": normalized_stage == "S2",
        "evidence_intent": normalized_stage == "S2",
        "launch_status": (
            "ready" if normalized_stage == "S2" else "engineering_calibration"
        ),
        "execute": False,
        "working_directory": str(VIBENCH_ROOT),
        "training_output_dir": str(output),
        "training_output_preexisted": output_preexisted,
        "staging_dir": str(staging),
        "resolved_config": str(final_config),
        "resolved_config_sha256": resolved_config_sha256,
        "source_config": _relative(CONFIG_PATHS[normalized_arm]),
        "source_config_sha256": source_config_sha256,
        "config_artifacts": config_hashes,
        "config_artifact_sha256": _inventory_hash(config_hashes),
        "runtime_bindings": runtime_bindings,
        "random_role_permutation": permutation,
        "conda_environment": CONDA_ENVIRONMENT,
        "physical_gpu_indices": [physical_gpu],
        "cuda_visible_devices": str(physical_gpu),
        "multi_gpu": False,
        "argv": argv,
        "command": shlex.join(argv),
        "git_commit": git_commit,
        "git_diff_sha256": git_diff_sha256,
        "git_diff_contract": (
            "SHA-256 of exact stdout bytes from `git diff --binary HEAD`; "
            "untracked files are excluded"
        ),
        "code_artifacts": code_artifacts,
        "code_artifact_sha256": code_artifact_sha256,
        "code_artifact_hash_contract": (
            "SHA-256 of sorted '<sha256>  <repo-relative-path>\\n' entries"
        ),
        "trainable_parameter_count": parameter_count,
        "trainable_parameter_signature_sha256": parameter_signature_sha256,
    }
    launch_json = (
        json.dumps(launch_plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")

    staging.parent.mkdir(parents=True, exist_ok=True)
    prefix = f".{staging.name}.preparing-"
    temporary = Path(tempfile.mkdtemp(prefix=prefix, dir=staging.parent)).resolve()
    published = False
    try:
        _write_new_file(temporary / "resolved_config.yaml", resolved_yaml)
        _write_new_file(temporary / "launch_plan.json", launch_json)
        # Recheck the evidence output immediately before publishing the plan.
        if normalized_stage == "S2" and _lexists(output):
            raise FileExistsError(
                f"S2 evidence output appeared during preparation: {output}"
            )
        _publish_noreplace(temporary, staging)
        published = True
    finally:
        if not published:
            _cleanup_temporary_directory(temporary, staging.parent, prefix)

    return launch_plan


def _load_json_mapping(path: Path, description: str) -> dict[str, Any]:
    source = _require_regular_file(path, description)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RunPreparationError(f"cannot parse {description}: {exc}") from exc
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise RunPreparationError(f"{description} must be a string-keyed mapping")
    return dict(value)


def _load_yaml_mapping(path: Path, description: str) -> dict[str, Any]:
    source = _require_regular_file(path, description)
    try:
        value = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise RunPreparationError(f"cannot parse {description}: {exc}") from exc
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise RunPreparationError(f"{description} must be a string-keyed mapping")
    return dict(value)


def _parse_iso8601(value: str, description: str) -> dt.datetime:
    if not isinstance(value, str) or not value:
        raise RunPreparationError(f"{description} must be a non-empty ISO-8601 string")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = dt.datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise RunPreparationError(f"{description} must be an ISO-8601 datetime") from exc
    if parsed.utcoffset() is None:
        raise RunPreparationError(f"{description} must include a UTC offset")
    return parsed


def _query_gpu_model(physical_gpu: int) -> str:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name",
                "--format=csv,noheader",
                f"--id={physical_gpu}",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RunPreparationError(
            f"cannot identify physical GPU {physical_gpu} with nvidia-smi: {exc}"
        ) from exc
    names = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(names) != 1:
        raise RunPreparationError(
            f"physical GPU {physical_gpu} must resolve to exactly one model name"
        )
    return names[0]


def _validate_plan_for_finalization(
    plan: Mapping[str, Any],
    *,
    plan_path: Path,
    config_path: Path,
) -> tuple[str, str, int, int, Path, list[int] | None]:
    if plan.get("schema_id") != SCHEMA_ID or plan.get("schema_version") != SCHEMA_VERSION:
        raise RunPreparationError("launch plan schema is not supported")
    if plan.get("execute") is not False:
        raise RunPreparationError("launch plan must remain non-executing")
    arm = plan.get("arm")
    stage = plan.get("stage")
    seed = plan.get("training_seed")
    attempt = plan.get("attempt")
    retry_reason = plan.get("retry_reason")
    devices = plan.get("physical_gpu_indices")
    if not isinstance(arm, str) or not isinstance(stage, str):
        raise RunPreparationError("launch plan arm/stage must be strings")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise RunPreparationError("launch plan training_seed must be an integer")
    if isinstance(attempt, bool) or not isinstance(attempt, int):
        raise RunPreparationError("launch plan attempt must be an integer")
    if retry_reason is not None and not isinstance(retry_reason, str):
        raise RunPreparationError("launch plan retry_reason must be null or a string")
    if (
        not isinstance(devices, list)
        or len(devices) != 1
        or isinstance(devices[0], bool)
        or not isinstance(devices[0], int)
    ):
        raise RunPreparationError("launch plan must contain one physical GPU")
    gpu = devices[0]

    staging = plan_path.parent.resolve()
    output_value = plan.get("training_output_dir")
    if not isinstance(output_value, str) or not output_value:
        raise RunPreparationError("launch plan training_output_dir is invalid")
    output = Path(output_value).expanduser().resolve()
    normalized_arm, normalized_stage = _validate_request(
        arm, stage, seed, attempt, retry_reason, gpu, output, staging
    )
    expected_run_id = (
        f"{GOAL_ID}-{normalized_stage}-{normalized_arm}-S{seed}-A{attempt}"
    )
    if plan.get("run_id") != expected_run_id:
        raise RunPreparationError("launch plan run_id disagrees with its attempt")
    expected_supersedes = (
        1 if normalized_stage == "S2" and attempt == 2 else None
    )
    if plan.get("supersedes_attempt") != expected_supersedes:
        raise RunPreparationError(
            "launch plan supersedes_attempt disagrees with its retry contract"
        )
    if plan.get("staging_dir") != str(staging):
        raise RunPreparationError("launch plan staging_dir does not match its location")
    if plan.get("resolved_config") != str(config_path):
        raise RunPreparationError("launch plan resolved_config path does not match input")
    if plan.get("claim_eligible") is not (normalized_stage == "S2"):
        raise RunPreparationError("launch plan claim_eligible disagrees with stage")
    if plan.get("evidence_intent") is not (normalized_stage == "S2"):
        raise RunPreparationError("launch plan evidence_intent disagrees with stage")
    if plan.get("conda_environment") != CONDA_ENVIRONMENT:
        raise RunPreparationError("launch plan conda environment is not LQ_signal")
    if plan.get("cuda_visible_devices") != str(gpu) or plan.get("multi_gpu") is not False:
        raise RunPreparationError("launch plan singleton GPU binding is invalid")
    expected_argv = [
        "conda",
        "run",
        "-n",
        CONDA_ENVIRONMENT,
        "env",
        f"CUDA_VISIBLE_DEVICES={gpu}",
        "python",
        "main.py",
        "--config",
        str(config_path),
    ]
    if plan.get("argv") != expected_argv or plan.get("command") != shlex.join(expected_argv):
        raise RunPreparationError("launch plan argv/command is not the exact frozen launch")
    if plan.get("working_directory") != str(VIBENCH_ROOT):
        raise RunPreparationError("launch plan working_directory is not nested Vibench")
    if plan.get("trainable_parameter_count") != EXPECTED_TRAINABLE_PARAMETERS:
        raise RunPreparationError("launch plan trainable parameter count is invalid")

    config_sha256 = _sha256_file(config_path)
    if plan.get("resolved_config_sha256") != config_sha256:
        raise RunPreparationError("resolved config SHA-256 disagrees with launch plan")
    config = _load_yaml_mapping(config_path, "resolved config")
    runtime_bindings = _validate_runtime_artifacts()
    permutation_value = plan.get("random_role_permutation")
    permutation: list[int] | None
    if normalized_arm == "RAND":
        if not isinstance(permutation_value, list):
            raise RunPreparationError("RAND launch plan is missing its permutation")
        permutation = list(permutation_value)
    else:
        if permutation_value is not None:
            raise RunPreparationError("non-RAND launch plan must not bind a permutation")
        permutation = None
    _validate_resolved_config(
        config,
        arm=normalized_arm,
        stage=normalized_stage,
        seed=seed,
        output_dir=output,
        runtime_bindings=runtime_bindings,
        permutation=permutation,
    )

    config_hashes = _validate_frozen_config_files()
    if plan.get("config_artifacts") != config_hashes:
        raise RunPreparationError("launch plan config artifact inventory is stale")
    if plan.get("config_artifact_sha256") != _inventory_hash(config_hashes):
        raise RunPreparationError("launch plan config aggregate SHA-256 is stale")
    code_artifacts, code_artifact_sha256 = _code_artifact_inventory()
    if plan.get("code_artifacts") != code_artifacts:
        raise RunPreparationError("launch plan code artifact inventory is stale")
    if plan.get("code_artifact_sha256") != code_artifact_sha256:
        raise RunPreparationError("launch plan code artifact aggregate SHA-256 is stale")
    git_commit, git_diff_sha256 = _git_state()
    if plan.get("git_commit") != git_commit or plan.get("git_diff_sha256") != git_diff_sha256:
        raise RunPreparationError("nested-repository Git state changed after preparation")
    return normalized_arm, normalized_stage, seed, gpu, output, permutation


def _unique_checkpoint(output: Path, supplied: Path) -> Path:
    output_root = _require_directory(output, "completed training output")
    checkpoint = _require_regular_file(supplied, "checkpoint")
    if checkpoint.stat().st_size == 0:
        raise RunPreparationError("checkpoint must not be empty")
    try:
        checkpoint.relative_to(output_root)
    except ValueError as exc:
        raise RunPreparationError(
            "checkpoint must be located beneath the planned training output"
        ) from exc

    candidates: list[Path] = []
    for candidate in output_root.rglob("*.ckpt"):
        if candidate.is_symlink() or not candidate.is_file():
            raise RunPreparationError(
                f"checkpoint inventory contains a non-regular entry: {candidate}"
            )
        candidates.append(candidate.resolve())
    if candidates != [checkpoint]:
        rendered = [str(path) for path in sorted(candidates)]
        raise RunPreparationError(
            f"training output must contain exactly the supplied checkpoint; got {rendered}"
        )
    return checkpoint


def _write_atomic_exclusive(path: Path, payload: bytes) -> None:
    if _lexists(path):
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.writing-", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    finally:
        if temporary.exists():
            temporary.unlink()


def finalize_run_meta(
    *,
    launch_plan: str | Path,
    resolved_config: str | Path,
    checkpoint: str | Path,
    started_at: str,
    ended_at: str,
    runtime_seconds: float,
    exit_code: int,
) -> dict[str, Any]:
    """Validate one completed launch and exclusively write packager run metadata."""

    if isinstance(exit_code, bool) or not isinstance(exit_code, int) or exit_code != 0:
        raise RunPreparationError("only a completed run with exit_code=0 can be finalized")
    if (
        isinstance(runtime_seconds, bool)
        or not isinstance(runtime_seconds, (int, float))
        or not math.isfinite(float(runtime_seconds))
        or float(runtime_seconds) < 0.0
    ):
        raise RunPreparationError("runtime_seconds must be finite and non-negative")
    started = _parse_iso8601(started_at, "started_at")
    ended = _parse_iso8601(ended_at, "ended_at")
    if ended < started:
        raise RunPreparationError("ended_at precedes started_at")

    plan_path = _require_regular_file(Path(launch_plan), "launch plan")
    config_path = _require_regular_file(Path(resolved_config), "resolved config")
    if plan_path.parent != config_path.parent:
        raise RunPreparationError("launch plan and resolved config must share one staging directory")
    plan = _load_json_mapping(plan_path, "launch plan")
    arm, stage, seed, gpu, output, _ = _validate_plan_for_finalization(
        plan, plan_path=plan_path, config_path=config_path
    )
    checkpoint_path = _unique_checkpoint(output, Path(checkpoint).expanduser().resolve())
    metadata_path = _require_regular_file(METADATA_FILE, "synthetic metadata")
    partition_path = _require_regular_file(PARTITION_MANIFEST, "partition manifest")
    generator_path = _require_regular_file(GENERATOR_MANIFEST, "generator manifest")
    gpu_model = _query_gpu_model(gpu)

    meta: dict[str, Any] = {
        "run_id": plan["run_id"],
        "experiment_id": EXPERIMENT_ID,
        "dataset": DATASET,
        "arm": arm,
        "stage": stage,
        "claim_eligible": stage == "S2",
        "retry_reason": plan["retry_reason"],
        "supersedes_attempt": plan["supersedes_attempt"],
        "status": "completed",
        "conda_environment": CONDA_ENVIRONMENT,
        "command": plan["command"],
        "working_directory": str(VIBENCH_ROOT),
        "physical_gpu_indices": [gpu],
        "cuda_visible_devices": str(gpu),
        "multi_gpu": False,
        "gpu_model": gpu_model,
        "gpu_count": 1,
        "precision": 32,
        "started_at": started_at,
        "ended_at": ended_at,
        "runtime_seconds": float(runtime_seconds),
        "exit_code": 0,
        "oom_or_failure_reason": None,
        "fallback_used": False,
        "git_commit": plan["git_commit"],
        "git_diff_sha256": plan["git_diff_sha256"],
        "resolved_config_sha256": plan["resolved_config_sha256"],
        "source_metadata_sha256": _sha256_file(generator_path),
        "derived_metadata_sha256": _sha256_file(metadata_path),
        "split_manifest_sha256": _sha256_file(partition_path),
        "code_artifact_sha256": plan["code_artifact_sha256"],
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "training_seed": seed,
        "split_seed": 240401,
    }
    if meta["source_metadata_sha256"] != plan["runtime_bindings"][
        "generator_manifest_sha256"
    ]:
        raise RunPreparationError("generator manifest changed after preparation")
    if meta["split_manifest_sha256"] != plan["runtime_bindings"][
        "partition_manifest_sha256"
    ]:
        raise RunPreparationError("partition manifest changed after preparation")

    destination = plan_path.parent / "run_meta.yaml"
    content = yaml.safe_dump(
        meta, allow_unicode=True, sort_keys=False, default_flow_style=False
    ).encode("utf-8")
    _write_atomic_exclusive(destination, content)
    return {**meta, "run_meta": str(destination)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare one strict P04 decisive run without launching training."
        )
    )
    parser.add_argument("--arm", required=True, choices=ALLOWED_ARMS)
    parser.add_argument("--stage", choices=("S1", "S2"), default="S2")
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--retry-reason", default=None)
    parser.add_argument(
        "--physical-gpu", required=True, type=int, choices=ALLOWED_PHYSICAL_GPUS
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--staging-dir", required=True, type=Path)
    return parser


def build_finalize_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prepare_decisive_run finalize-meta",
        description=(
            "Validate one completed prepared run and exclusively write run_meta.yaml."
        ),
    )
    parser.add_argument("--launch-plan", required=True, type=Path)
    parser.add_argument("--resolved-config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--started-at", required=True)
    parser.add_argument("--ended-at", required=True)
    parser.add_argument("--runtime-seconds", required=True, type=float)
    parser.add_argument("--exit-code", required=True, type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments[:1] == ["finalize-meta"]:
        args = build_finalize_parser().parse_args(arguments[1:])
        meta = finalize_run_meta(
            launch_plan=args.launch_plan,
            resolved_config=args.resolved_config,
            checkpoint=args.checkpoint,
            started_at=args.started_at,
            ended_at=args.ended_at,
            runtime_seconds=args.runtime_seconds,
            exit_code=args.exit_code,
        )
        print(
            json.dumps(
                {
                    "run_id": meta["run_id"],
                    "run_meta": meta["run_meta"],
                    "checkpoint_sha256": meta["checkpoint_sha256"],
                    "status": meta["status"],
                },
                sort_keys=True,
            )
        )
        return 0

    args = build_parser().parse_args(arguments)
    plan = prepare_decisive_run(
        arm=args.arm,
        stage=args.stage,
        seed=args.seed,
        attempt=args.attempt,
        retry_reason=args.retry_reason,
        physical_gpu=args.physical_gpu,
        output_dir=args.output_dir,
        staging_dir=args.staging_dir,
    )
    print(
        json.dumps(
            {
                "run_id": plan["run_id"],
                "staging_dir": plan["staging_dir"],
                "resolved_config_sha256": plan["resolved_config_sha256"],
                "claim_eligible": plan["claim_eligible"],
                "executed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
