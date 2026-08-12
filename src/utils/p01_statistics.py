"""Offline, fail-closed statistical scoring for the P01 frozen protocol.

This module consumes immutable ``predictions.npz`` artifacts and their JSON
sidecars.  It never trains a model and never rewrites prediction inputs.
"""

from __future__ import annotations

import hashlib
import csv
import json
import os
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np


SCORING_DERANGEMENT_SEED = 20260802
BOOTSTRAP_SEED = 20260801
SCORING_ALGORITHM = "p01_group_class_cyclic_derangement_v1"

ARM_REQUIRED_REPRESENTATIONS: dict[str, tuple[str, ...]] = {
    "FULL": (
        "shared_1d",
        "shared_2d",
        "private_1d",
        "private_2d",
        "reconstructed_1d",
        "reconstructed_2d",
    ),
    "TRAIN-MISPAIR": (
        "shared_1d",
        "shared_2d",
        "private_1d",
        "private_2d",
        "reconstructed_1d",
        "reconstructed_2d",
    ),
    "B4-GATTN": ("encoded_1d", "encoded_2d"),
}

ARM_ALIGNMENT_VIEWS: dict[str, tuple[str, str]] = {
    "FULL": ("shared_1d", "shared_2d"),
    "TRAIN-MISPAIR": ("shared_1d", "shared_2d"),
    "B4-GATTN": ("encoded_1d", "encoded_2d"),
}

DATASET_IDENTIFIERS: dict[str, tuple[str, int]] = {
    "CWRU": ("cwru", 1),
    "XJTU": ("xjtu", 2),
}
DATASET_OUTER_FOLDS: dict[str, int] = {"CWRU": 4, "XJTU": 5}
TRAINING_SEEDS = (42, 123, 456, 789, 1024)
FROZEN_SPLIT_MANIFEST_SHA256S: dict[str, tuple[str, ...]] = {
    "CWRU": (
        "ed14b16912d91fd7d92d81bfb6d4e0fcdabe9fdc9fe5c56d613dc9143f8cc202",
        "25a5a59d0839798d29ee589935e05391496c9c82428984d3d20c852ba2006da4",
        "3456eea39cd273026b0c6169645fab5806325a22ea32a09aed5ea57751766df5",
        "6399f21486854db9e90934c803ca500c44fd66c3cf69c84e029c548700eaeda3",
    ),
    "XJTU": (
        "1c376ac9831b3cfd836d8068788bda2c12d73a852cce6abbe7ee93884f9242c1",
        "b2e785dfcd3edd6163c9bf2475000d7c18bb59a66443595ab7f64a1cd1f8b0f1",
        "1652eb9f5f6cf8b2f7281b3ec7467cd669b3bf638342fd6c81b0ba6aec2d31cd",
        "107d194eba91c383cfd5abb42e2d889fce526f5c420a356eee6c06bfa12f4f43",
        "d349a44b7cbfbe0f108a91502fa4374b5f2ba469c25961a26553286f3f3e3d93",
    ),
}

PREDICTION_PROVENANCE_REQUIRED = (
    "protocol_id",
    "dataset_key",
    "dataset_slug",
    "dataset_id",
    "arm_id",
    "attempt_id",
    "outer_fold",
    "training_seed",
    "config_snapshot_path",
    "config_snapshot_sha256",
    "invocation_path",
    "invocation_sha256",
    "best_checkpoint_manifest_path",
    "best_checkpoint_manifest_sha256",
    "checkpoint_path",
    "checkpoint_sha256",
    "checkpoint_monitor",
    "checkpoint_mode",
    "checkpoint_score",
    "split_manifest_path",
    "split_manifest_payload_sha256",
    "code_state_identifier",
    "code_state_sha256",
    "data_snapshot_manifest_path",
    "data_snapshot_manifest_sha256",
    "data_payload_sha256",
    "trainer_metrics_manifest_path",
    "trainer_metrics_manifest_sha256",
    "trainer_metrics_path",
    "trainer_metrics_sha256",
)

ArtifactKey = Tuple[str, int, int]
MetricValues = Dict[ArtifactKey, np.ndarray]


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: Any, field: str) -> str:
    rendered = str(value)
    if len(rendered) != 64 or any(character not in "0123456789abcdef" for character in rendered):
        raise ValueError(f"{field} must be a lowercase SHA-256 hex digest")
    return rendered


def _verified_provenance_file(
    provenance: Mapping[str, Any], path_field: str, hash_field: str
) -> Path:
    raw_path = str(provenance.get(path_field, ""))
    if not raw_path:
        raise ValueError(f"Prediction provenance field {path_field} is empty")
    path = Path(raw_path)
    if not path.is_file():
        raise FileNotFoundError(f"Prediction provenance file is absent: {path}")
    expected = _require_sha256(provenance.get(hash_field), f"provenance.{hash_field}")
    if _sha256_file(path) != expected:
        raise ValueError(f"Prediction provenance file hash mismatch: {path}")
    return path


def _validate_complete_prediction_provenance(
    provenance: Mapping[str, Any], split_manifest_sha256: str
) -> None:
    missing = [field for field in PREDICTION_PROVENANCE_REQUIRED if field not in provenance]
    if missing:
        raise ValueError(
            "Prediction provenance lacks required fields: " + ", ".join(missing)
        )
    config_path = _verified_provenance_file(
        provenance, "config_snapshot_path", "config_snapshot_sha256"
    )
    invocation_path = _verified_provenance_file(
        provenance, "invocation_path", "invocation_sha256"
    )
    checkpoint_manifest_path = _verified_provenance_file(
        provenance,
        "best_checkpoint_manifest_path",
        "best_checkpoint_manifest_sha256",
    )
    checkpoint_path = _verified_provenance_file(
        provenance, "checkpoint_path", "checkpoint_sha256"
    )
    data_snapshot_path = _verified_provenance_file(
        provenance,
        "data_snapshot_manifest_path",
        "data_snapshot_manifest_sha256",
    )
    trainer_metrics_manifest_path = _verified_provenance_file(
        provenance,
        "trainer_metrics_manifest_path",
        "trainer_metrics_manifest_sha256",
    )
    trainer_metrics_path = _verified_provenance_file(
        provenance, "trainer_metrics_path", "trainer_metrics_sha256"
    )
    del config_path

    code_state_sha256 = _require_sha256(
        provenance.get("code_state_sha256"), "provenance.code_state_sha256"
    )
    data_payload_sha256 = _require_sha256(
        provenance.get("data_payload_sha256"), "provenance.data_payload_sha256"
    )
    if not str(provenance.get("code_state_identifier", "")):
        raise ValueError("Prediction provenance code_state_identifier is empty")
    if str(provenance.get("checkpoint_monitor")) != "val_loss":
        raise ValueError("Prediction checkpoint monitor must be val_loss")
    if str(provenance.get("checkpoint_mode")) != "min":
        raise ValueError("Prediction checkpoint mode must be min")
    try:
        checkpoint_score = float(provenance.get("checkpoint_score"))
    except (TypeError, ValueError) as exc:
        raise ValueError("Prediction checkpoint_score must be numeric") from exc
    if not np.isfinite(checkpoint_score):
        raise ValueError("Prediction checkpoint_score must be finite")

    split_path = Path(str(provenance.get("split_manifest_path", "")))
    if not split_path.is_file():
        raise FileNotFoundError(f"Prediction split manifest is absent: {split_path}")
    try:
        split_payload = json.loads(split_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Prediction split manifest is invalid JSON: {split_path}") from exc
    if not isinstance(split_payload, dict) or str(
        split_payload.get("manifest_payload_sha256", "")
    ) != split_manifest_sha256:
        raise ValueError("Prediction split manifest payload binding mismatch")
    split_without_hash = dict(split_payload)
    split_without_hash.pop("manifest_payload_sha256", None)
    if hashlib.sha256(_canonical_json_bytes(split_without_hash)).hexdigest() != split_manifest_sha256:
        raise ValueError("Prediction split manifest canonical payload hash mismatch")

    def _json_object(path: Path, label: str) -> dict[str, Any]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{label} is invalid JSON: {path}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{label} must be a JSON object")
        return value

    invocation = _json_object(invocation_path, "Invocation provenance")
    if str(invocation.get("code_state_sha256", "")) != code_state_sha256:
        raise ValueError("Invocation code-state binding mismatch")
    if str(invocation.get("config_snapshot_sha256", "")) != str(
        provenance.get("config_snapshot_sha256")
    ):
        raise ValueError("Invocation config-snapshot binding mismatch")
    try:
        invocation_seed = int(invocation.get("effective_seed"))
    except (TypeError, ValueError) as exc:
        raise ValueError("Invocation effective_seed is absent or invalid") from exc
    if invocation_seed != int(provenance["training_seed"]):
        raise ValueError("Invocation effective_seed binding mismatch")

    identity_fields = (
        "protocol_id",
        "dataset_key",
        "dataset_slug",
        "dataset_id",
        "arm_id",
        "attempt_id",
    )

    def _validate_paper_identity(container: Mapping[str, Any], label: str) -> None:
        paper = container.get("paper")
        if not isinstance(paper, dict):
            raise ValueError(f"{label} requires a paper identity object")
        for field in identity_fields:
            if str(paper.get(field, "")) != str(provenance.get(field, "")):
                raise ValueError(f"{label} paper identity mismatch for {field}")

    _validate_paper_identity(invocation, "Invocation")

    checkpoint_manifest = _json_object(
        checkpoint_manifest_path, "Best-checkpoint provenance"
    )
    checkpoint_bindings = {
        "path": str(checkpoint_path.resolve()),
        "sha256": str(provenance.get("checkpoint_sha256")),
        "monitor": str(provenance.get("checkpoint_monitor")),
        "mode": str(provenance.get("checkpoint_mode")),
    }
    for field, expected in checkpoint_bindings.items():
        observed = checkpoint_manifest.get(field)
        if field == "path":
            observed = str(Path(str(observed)).resolve())
        else:
            observed = str(observed)
        if observed != expected:
            raise ValueError(f"Best-checkpoint manifest binding mismatch for {field}")
    if float(checkpoint_manifest.get("score")) != checkpoint_score:
        raise ValueError("Best-checkpoint score binding mismatch")

    data_snapshot = _json_object(data_snapshot_path, "Data-snapshot provenance")
    _validate_paper_identity(data_snapshot, "Data snapshot")
    expected_snapshot_fields = {
        "data_payload_sha256": data_payload_sha256,
        "config_snapshot_sha256": str(provenance.get("config_snapshot_sha256")),
        "invocation_sha256": str(provenance.get("invocation_sha256")),
        "split_manifest_payload_sha256": split_manifest_sha256,
    }
    for field, expected in expected_snapshot_fields.items():
        if str(data_snapshot.get(field, "")) != expected:
            raise ValueError(f"Data-snapshot binding mismatch for {field}")
    cross_validation = split_payload.get("cross_validation")
    if not isinstance(cross_validation, dict) or int(
        cross_validation.get("outer_fold", -1)
    ) != int(provenance["outer_fold"]):
        raise ValueError("Prediction split manifest outer-fold binding mismatch")

    trainer_metrics_manifest = _json_object(
        trainer_metrics_manifest_path, "Trainer-metrics provenance"
    )
    if str(Path(str(trainer_metrics_manifest.get("metrics_path", ""))).resolve()) != str(
        trainer_metrics_path.resolve()
    ):
        raise ValueError("Trainer-metrics manifest path binding mismatch")
    if str(trainer_metrics_manifest.get("metrics_sha256", "")) != str(
        provenance.get("trainer_metrics_sha256")
    ):
        raise ValueError("Trainer-metrics manifest SHA binding mismatch")
    if (
        str(trainer_metrics_manifest.get("logger_type", "")) != "CSVLogger"
        or int(trainer_metrics_manifest.get("logger_version", -1)) != 0
    ):
        raise ValueError("Trainer-metrics manifest logger contract mismatch")
    with trainer_metrics_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not any(True for _ in reader):
            raise ValueError("Trainer metrics CSV must contain a header and data rows")


def _atomic_write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    temporary.write_text(rendered, encoding="utf-8")
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json_summary(path: str | Path, payload: Mapping[str, Any]) -> str:
    """Write a summary once and return its file SHA-256."""

    target = Path(path)
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite analysis summary: {target}")
    try:
        _atomic_write_new_json(target, payload)
    except FileExistsError as exc:
        raise FileExistsError(
            f"Concurrent analysis summary already exists: {target}"
        ) from exc
    return _sha256_file(target)


@dataclass(frozen=True)
class PredictionArtifact:
    path: Path
    manifest_path: Path
    artifact_sha256: str
    protocol_id: str
    dataset_key: str
    dataset_slug: str
    dataset_id: int
    arm_id: str
    attempt_id: int
    outer_fold: int
    training_seed: int
    split_manifest_sha256: str
    arrays: Mapping[str, np.ndarray]

    @property
    def key(self) -> ArtifactKey:
        return (self.arm_id, self.training_seed, self.outer_fold)

    @property
    def samples(self) -> int:
        return int(self.arrays["sample_key"].shape[0])


def load_prediction_artifact(path: str | Path) -> PredictionArtifact:
    """Load and internally validate one prediction artifact and its sidecar."""

    target = Path(path)
    manifest_path = target.with_suffix(".manifest.json")
    if not target.is_file():
        raise FileNotFoundError(f"Prediction artifact is absent: {target}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Prediction manifest is absent: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Prediction manifest is not valid JSON: {manifest_path}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("Prediction manifest must be a JSON object")
    if int(manifest.get("schema_version", -1)) != 1:
        raise ValueError("Unsupported prediction manifest schema_version")

    artifact_sha256 = _require_sha256(
        manifest.get("artifact_sha256"), "manifest.artifact_sha256"
    )
    if _sha256_file(target) != artifact_sha256:
        raise ValueError(f"Prediction artifact SHA-256 mismatch: {target}")
    if manifest.get("artifact") != target.name:
        raise ValueError("Prediction manifest artifact filename does not match its NPZ")

    try:
        with np.load(target, allow_pickle=False) as archive:
            arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    except (OSError, ValueError) as exc:
        raise ValueError(f"Prediction artifact is not a safe readable NPZ: {target}") from exc

    required = {
        "logits",
        "y_true",
        "y_pred",
        "file_id",
        "window_id",
        "group_id",
        "sample_key",
        "outer_fold",
        "training_seed",
    }
    missing = required - set(arrays)
    if missing:
        raise ValueError("Prediction artifact lacks arrays: " + ", ".join(sorted(missing)))
    manifest_arrays = manifest.get("arrays")
    if not isinstance(manifest_arrays, list) or set(map(str, manifest_arrays)) != set(arrays):
        raise ValueError("Prediction manifest array inventory does not match its NPZ")

    sample_key = np.asarray(arrays["sample_key"]).astype(str)
    if sample_key.ndim != 1 or sample_key.size == 0:
        raise ValueError("sample_key must be a non-empty rank-1 array")
    samples = int(sample_key.size)
    if int(manifest.get("samples", -1)) != samples:
        raise ValueError("Prediction manifest sample count does not match its NPZ")
    if len(set(sample_key.tolist())) != samples:
        raise ValueError("Prediction artifact contains duplicate sample keys")
    if any(not key for key in sample_key.tolist()):
        raise ValueError("Prediction artifact contains an empty sample key")

    for name, array in arrays.items():
        if array.ndim == 0 or int(array.shape[0]) != samples:
            raise ValueError(f"Array {name!r} is not sample-aligned")
        if array.dtype.kind in "fc" and not np.isfinite(array).all():
            raise ValueError(f"Array {name!r} contains non-finite values")

    logits = np.asarray(arrays["logits"])
    if logits.ndim != 2 or logits.shape[1] < 2:
        raise ValueError("logits must have shape [samples, classes>=2]")
    y_true = np.asarray(arrays["y_true"])
    y_pred = np.asarray(arrays["y_pred"])
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("y_true and y_pred must be rank-1 arrays")
    if not np.array_equal(y_true, y_true.astype(np.int64)):
        raise ValueError("y_true must contain integer labels")
    if not np.array_equal(y_pred, y_pred.astype(np.int64)):
        raise ValueError("y_pred must contain integer labels")
    if int(y_true.min()) < 0 or int(y_true.max()) >= int(logits.shape[1]):
        raise ValueError("y_true contains labels outside the logits class range")
    if not np.array_equal(logits.argmax(axis=1), y_pred.astype(np.int64)):
        raise ValueError("y_pred does not equal argmax(logits)")

    file_id = np.asarray(arrays["file_id"]).astype(str)
    group_id = np.asarray(arrays["group_id"]).astype(str)
    if any(not value for value in file_id.tolist()) or any(
        not value for value in group_id.tolist()
    ):
        raise ValueError("file_id and group_id must be non-empty")
    window_id = np.asarray(arrays["window_id"])
    if not np.array_equal(window_id, window_id.astype(np.int64)):
        raise ValueError("window_id must contain integers")
    expected_keys = np.asarray(
        [f"{file_value}:{int(window_value)}" for file_value, window_value in zip(file_id, window_id)],
        dtype=str,
    )
    if not np.array_equal(sample_key, expected_keys):
        raise ValueError("sample_key is not the canonical file_id:window_id key")

    outer_values = np.asarray(arrays["outer_fold"])
    seed_values = np.asarray(arrays["training_seed"])
    if not np.array_equal(outer_values, outer_values.astype(np.int64)):
        raise ValueError("outer_fold must contain integers")
    if not np.array_equal(seed_values, seed_values.astype(np.int64)):
        raise ValueError("training_seed must contain integers")
    if len(set(outer_values.astype(int).tolist())) != 1:
        raise ValueError("outer_fold must be constant inside one prediction artifact")
    if len(set(seed_values.astype(int).tolist())) != 1:
        raise ValueError("training_seed must be constant inside one prediction artifact")
    outer_fold = int(outer_values[0])
    training_seed = int(seed_values[0])
    if int(manifest.get("outer_fold", -1)) != outer_fold:
        raise ValueError("outer_fold differs between NPZ and manifest")
    if int(manifest.get("training_seed", -1)) != training_seed:
        raise ValueError("training_seed differs between NPZ and manifest")

    provenance = manifest.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("Prediction manifest provenance must be an object")
    protocol_id = str(provenance.get("protocol_id", ""))
    dataset_key = str(provenance.get("dataset_key", ""))
    dataset_slug = str(provenance.get("dataset_slug", ""))
    try:
        dataset_id = int(provenance.get("dataset_id"))
    except (TypeError, ValueError) as exc:
        raise ValueError("Prediction provenance dataset_id must be an integer") from exc
    arm_id = str(provenance.get("arm_id", ""))
    if not protocol_id or not dataset_key or not dataset_slug or not arm_id:
        raise ValueError(
            "Prediction provenance requires protocol_id, dataset_key, "
            "dataset_slug, dataset_id, and arm_id"
        )
    if dataset_key not in DATASET_IDENTIFIERS:
        raise ValueError(f"Unsupported P01 dataset_key: {dataset_key!r}")
    expected_slug, expected_id = DATASET_IDENTIFIERS[dataset_key]
    if dataset_slug != expected_slug or dataset_id != expected_id:
        raise ValueError(
            "Prediction provenance dataset_key/slug/id identifiers disagree"
        )
    raw_attempt_id = provenance.get("attempt_id")
    if (
        isinstance(raw_attempt_id, bool)
        or not isinstance(raw_attempt_id, int)
        or raw_attempt_id not in {0, 1}
    ):
        raise ValueError("Prediction provenance attempt_id must be integer 0 or 1")
    attempt_id = raw_attempt_id
    try:
        provenance_fold = int(provenance.get("outer_fold"))
        provenance_seed = int(provenance.get("training_seed"))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Prediction provenance requires integer outer_fold and training_seed"
        ) from exc
    if provenance_fold != outer_fold or provenance_seed != training_seed:
        raise ValueError("Prediction provenance fold/seed disagree with NPZ and manifest")
    split_manifest_sha256 = _require_sha256(
        provenance.get("split_manifest_payload_sha256"),
        "provenance.split_manifest_payload_sha256",
    )
    _validate_complete_prediction_provenance(provenance, split_manifest_sha256)

    frozen_arrays: dict[str, np.ndarray] = {}
    for name, array in arrays.items():
        array.setflags(write=False)
        frozen_arrays[name] = array
    return PredictionArtifact(
        path=target,
        manifest_path=manifest_path,
        artifact_sha256=artifact_sha256,
        protocol_id=protocol_id,
        dataset_key=dataset_key,
        dataset_slug=dataset_slug,
        dataset_id=dataset_id,
        arm_id=arm_id,
        attempt_id=attempt_id,
        outer_fold=outer_fold,
        training_seed=training_seed,
        split_manifest_sha256=split_manifest_sha256,
        arrays=frozen_arrays,
    )


@dataclass
class ArtifactGrid:
    protocol_id: str
    dataset_key: str
    dataset_slug: str
    dataset_id: int
    arms: tuple[str, ...]
    seeds: tuple[int, ...]
    folds: tuple[int, ...]
    artifacts: Mapping[ArtifactKey, PredictionArtifact]
    orders: Mapping[ArtifactKey, np.ndarray]
    reference: Mapping[int, Mapping[str, np.ndarray]]
    group_strata: Mapping[str, str]
    split_manifest_sha256s: Mapping[int, str]

    @property
    def classes(self) -> tuple[int, ...]:
        values: set[int] = set()
        for fold in self.folds:
            values.update(np.asarray(self.reference[fold]["y_true"]).astype(int).tolist())
        return tuple(sorted(values))

    def array(self, arm: str, seed: int, fold: int, name: str) -> np.ndarray:
        key = (arm, seed, fold)
        artifact = self.artifacts[key]
        if name not in artifact.arrays:
            raise ValueError(f"Artifact {key} lacks required array {name!r}")
        return np.asarray(artifact.arrays[name])[self.orders[key]]


def _sorted_rows(artifact: PredictionArtifact) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    keys = np.asarray(artifact.arrays["sample_key"]).astype(str)
    order = np.argsort(keys, kind="stable")
    names = ("sample_key", "file_id", "window_id", "group_id", "y_true")
    return order, {name: np.asarray(artifact.arrays[name])[order] for name in names}


def validate_artifact_grid(
    artifacts: Iterable[PredictionArtifact],
    *,
    protocol_id: str,
    dataset_key: str,
    dataset_slug: str,
    expected_arms: Sequence[str],
    expected_seeds: Sequence[int],
    expected_folds: Sequence[int],
    analysis_scope: str = "final_oof",
    group_strata_by_group: Mapping[str, str] | None = None,
    required_representations: Mapping[str, Sequence[str]] = ARM_REQUIRED_REPRESENTATIONS,
) -> ArtifactGrid:
    """Validate a complete crossed arm-by-seed-by-fold OOF artifact grid."""

    arms = tuple(str(value) for value in expected_arms)
    seeds = tuple(int(value) for value in expected_seeds)
    folds = tuple(int(value) for value in expected_folds)
    if not arms or not seeds or not folds:
        raise ValueError("Expected arms, seeds, and folds must all be non-empty")
    if len(set(arms)) != len(arms) or len(set(seeds)) != len(seeds) or len(set(folds)) != len(folds):
        raise ValueError("Expected arms, seeds, and folds must not contain duplicates")
    if dataset_key not in DATASET_IDENTIFIERS:
        raise ValueError(f"Unsupported P01 dataset_key: {dataset_key!r}")
    canonical_slug, canonical_id = DATASET_IDENTIFIERS[dataset_key]
    if dataset_slug != canonical_slug:
        raise ValueError("dataset_slug does not match dataset_key")
    if tuple(seeds) != TRAINING_SEEDS:
        raise ValueError(f"P01 training seeds must be exactly {TRAINING_SEEDS}")
    if analysis_scope == "final_oof":
        required_folds = tuple(range(DATASET_OUTER_FOLDS[dataset_key]))
        if folds != required_folds:
            raise ValueError(
                f"Final OOF analysis requires all folds {required_folds}, got {folds}"
            )
    elif analysis_scope == "g050_fold0":
        required_arms = {"FULL", "B4-GATTN", "TRAIN-MISPAIR"}
        if dataset_key != "CWRU" or folds != (0,) or set(arms) != required_arms:
            raise ValueError(
                "G050 requires CWRU fold 0 and exactly FULL, B4-GATTN, TRAIN-MISPAIR"
            )
    else:
        raise ValueError("analysis_scope must be 'final_oof' or 'g050_fold0'")

    indexed: dict[ArtifactKey, PredictionArtifact] = {}
    for artifact in artifacts:
        if artifact.protocol_id != protocol_id:
            raise ValueError(
                f"Protocol mismatch for {artifact.path}: {artifact.protocol_id!r}"
            )
        if (
            artifact.dataset_key != dataset_key
            or artifact.dataset_slug != dataset_slug
            or artifact.dataset_id != canonical_id
        ):
            raise ValueError(
                "Dataset key/slug/id mismatch for "
                f"{artifact.path}: {artifact.dataset_key}/"
                f"{artifact.dataset_slug}/{artifact.dataset_id}"
            )
        if artifact.key in indexed:
            raise ValueError(f"Duplicate artifact cell: {artifact.key}")
        indexed[artifact.key] = artifact

    expected_keys = {
        (arm, seed, fold) for arm in arms for seed in seeds for fold in folds
    }
    if set(indexed) != expected_keys:
        missing = sorted(expected_keys - set(indexed))
        unexpected = sorted(set(indexed) - expected_keys)
        raise ValueError(
            f"Incomplete OOF artifact grid; missing={missing}, unexpected={unexpected}"
        )

    for key, artifact in indexed.items():
        for name in required_representations.get(artifact.arm_id, ()):
            array_name = f"repr__{name}"
            if array_name not in artifact.arrays:
                raise ValueError(f"Artifact {key} lacks required representation {array_name}")

    reference: dict[int, dict[str, np.ndarray]] = {}
    orders: dict[ArtifactKey, np.ndarray] = {}
    split_hashes: dict[int, str] = {}
    reference_arm = arms[0]
    reference_seed = seeds[0]
    for fold in folds:
        key = (reference_arm, reference_seed, fold)
        order, rows = _sorted_rows(indexed[key])
        orders[key] = order
        reference[fold] = rows
        split_hashes[fold] = indexed[key].split_manifest_sha256
        for arm in arms:
            for seed in seeds:
                candidate_key = (arm, seed, fold)
                candidate_order, candidate_rows = _sorted_rows(indexed[candidate_key])
                orders[candidate_key] = candidate_order
                for name in rows:
                    reference_values = np.asarray(rows[name]).astype(str) if name in {"sample_key", "file_id", "group_id"} else np.asarray(rows[name])
                    candidate_values = np.asarray(candidate_rows[name]).astype(str) if name in {"sample_key", "file_id", "group_id"} else np.asarray(candidate_rows[name])
                    if not np.array_equal(reference_values, candidate_values):
                        raise ValueError(
                            f"Cross-arm/seed test rows differ at fold {fold} for {name}"
                        )
                if indexed[candidate_key].split_manifest_sha256 != split_hashes[fold]:
                    raise ValueError(
                        f"Split manifest hash differs across arms/seeds at fold {fold}"
                    )

    seen_keys: set[str] = set()
    seen_groups: set[str] = set()
    global_classes: set[int] = set()
    fold_classes: dict[int, set[int]] = {}
    for fold in folds:
        keys = set(np.asarray(reference[fold]["sample_key"]).astype(str).tolist())
        groups = set(np.asarray(reference[fold]["group_id"]).astype(str).tolist())
        if seen_keys & keys:
            raise ValueError("OOF sample keys occur in more than one outer fold")
        if seen_groups & groups:
            raise ValueError("OOF test groups occur in more than one outer fold")
        seen_keys.update(keys)
        seen_groups.update(groups)
        classes = set(np.asarray(reference[fold]["y_true"]).astype(int).tolist())
        fold_classes[fold] = classes
        global_classes.update(classes)
    if any(classes != global_classes for classes in fold_classes.values()):
        raise ValueError("Every outer fold must cover the complete observed class set")

    derived_strata: dict[str, str] = {}
    supplied = (
        {str(key): str(value) for key, value in group_strata_by_group.items()}
        if group_strata_by_group is not None
        else None
    )
    for fold in folds:
        groups = np.asarray(reference[fold]["group_id"]).astype(str)
        labels = np.asarray(reference[fold]["y_true"]).astype(int)
        if dataset_key == "CWRU":
            row_strata = labels.astype(str)
        elif supplied is not None:
            missing_groups = set(groups.tolist()) - set(supplied)
            if missing_groups:
                raise ValueError(
                    "External design-stratum mapping lacks groups: "
                    + ", ".join(sorted(missing_groups))
                )
            row_strata = np.asarray([supplied[group] for group in groups], dtype=str)
        else:
            key = (reference_arm, reference_seed, fold)
            if "design_stratum" not in indexed[key].arrays:
                raise ValueError(
                    "XJTU bootstrap requires explicit Domain_id design strata via "
                    "design_stratum arrays or group_strata_by_group"
                )
            row_strata = np.asarray(indexed[key].arrays["design_stratum"])[orders[key]].astype(str)
        if dataset_key == "XJTU":
            for arm in arms:
                for seed in seeds:
                    candidate_key = (arm, seed, fold)
                    candidate = indexed[candidate_key]
                    if "design_stratum" not in candidate.arrays:
                        if supplied is None:
                            raise ValueError(
                                f"Artifact {candidate_key} lacks required design_stratum"
                            )
                        continue
                    candidate_strata = np.asarray(
                        candidate.arrays["design_stratum"]
                    )[orders[candidate_key]].astype(str)
                    if not np.array_equal(candidate_strata, row_strata):
                        raise ValueError(
                            f"Design strata differ across XJTU arms/seeds at fold {fold}"
                        )
        for group, stratum in zip(groups, row_strata):
            previous = derived_strata.setdefault(group, stratum)
            if previous != stratum:
                raise ValueError(f"Group {group!r} belongs to multiple design strata")

    if supplied is not None and set(supplied) != seen_groups:
        raise ValueError("External design-stratum mapping must exactly cover OOF test groups")

    return ArtifactGrid(
        protocol_id=protocol_id,
        dataset_key=dataset_key,
        dataset_slug=dataset_slug,
        dataset_id=canonical_id,
        arms=arms,
        seeds=seeds,
        folds=folds,
        artifacts=indexed,
        orders=orders,
        reference=reference,
        group_strata=derived_strata,
        split_manifest_sha256s=split_hashes,
    )


def group_class_balanced_mean(
    values: np.ndarray,
    y_true: np.ndarray,
    group_id: np.ndarray,
    *,
    group_weights: Mapping[str, int | float] | None = None,
) -> float:
    """Mean group cell value within class, followed by a mean over classes."""

    value_array = np.asarray(values, dtype=float)
    labels = np.asarray(y_true).astype(int)
    groups = np.asarray(group_id).astype(str)
    if value_array.ndim != 1 or labels.ndim != 1 or groups.ndim != 1:
        raise ValueError("values, y_true, and group_id must be rank-1")
    if not (len(value_array) == len(labels) == len(groups)) or len(values) == 0:
        raise ValueError("values, y_true, and group_id must be non-empty and aligned")
    if not np.isfinite(value_array).all():
        raise ValueError("Metric values contain non-finite entries")

    class_means: list[float] = []
    for label in sorted(set(labels.tolist())):
        cell_values: list[float] = []
        cell_weights: list[float] = []
        for group in sorted(set(groups[labels == label].tolist())):
            weight = 1.0 if group_weights is None else float(group_weights.get(group, 0.0))
            if weight < 0:
                raise ValueError("Group bootstrap weights cannot be negative")
            if weight == 0:
                continue
            mask = (labels == label) & (groups == group)
            cell_values.append(float(value_array[mask].mean()))
            cell_weights.append(weight)
        if not cell_values:
            raise ValueError(f"No positively weighted groups remain for class {label}")
        class_means.append(float(np.average(cell_values, weights=cell_weights)))
    return float(np.mean(class_means))


def group_class_balanced_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group_id: np.ndarray,
) -> float:
    labels = np.asarray(y_true)
    predictions = np.asarray(y_pred)
    if labels.shape != predictions.shape:
        raise ValueError("y_true and y_pred must have identical shapes")
    return group_class_balanced_mean(
        (labels == predictions).astype(float), labels, group_id
    )


@dataclass(frozen=True)
class ScoringUniverse:
    path: Path
    file_sha256: str
    protocol_id: str
    dataset_key: str
    dataset_slug: str
    dataset_id: int
    folds: tuple[int, ...]
    split_manifests: tuple[Mapping[str, Any], ...]
    samples: tuple[Mapping[str, Any], ...]

    @property
    def split_manifest_sha256s(self) -> dict[int, str]:
        return {
            int(entry["outer_fold"]): str(entry["manifest_payload_sha256"])
            for entry in self.split_manifests
        }


def load_scoring_universe(
    path: str | Path,
    *,
    expected_split_sha256s: Sequence[str] | None = None,
) -> ScoringUniverse:
    """Load a complete, independent all-fold evaluation sample universe."""

    target = Path(path).resolve()
    if not target.is_file():
        raise FileNotFoundError(f"Scoring sample-universe file is absent: {target}")
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Scoring sample-universe file is invalid JSON: {target}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Scoring sample universe must be a JSON object")
    if int(payload.get("schema_version", -1)) != 1:
        raise ValueError("Unsupported scoring sample-universe schema_version")
    protocol_id = str(payload.get("protocol_id", ""))
    dataset_key = str(payload.get("dataset_key", ""))
    dataset_slug = str(payload.get("dataset_slug", ""))
    try:
        dataset_id = int(payload.get("dataset_id"))
    except (TypeError, ValueError) as exc:
        raise ValueError("Scoring sample-universe dataset_id must be an integer") from exc
    if not protocol_id or dataset_key not in DATASET_IDENTIFIERS:
        raise ValueError("Scoring sample universe has invalid protocol/dataset identity")
    expected_slug, expected_id = DATASET_IDENTIFIERS[dataset_key]
    if dataset_slug != expected_slug or dataset_id != expected_id:
        raise ValueError("Scoring sample-universe dataset key/slug/id disagree")

    raw_splits = payload.get("split_manifests")
    if not isinstance(raw_splits, list):
        raise ValueError("Scoring sample universe requires split_manifests")
    split_manifests: list[dict[str, Any]] = []
    split_test_ids: dict[int, set[str]] = {}
    split_test_groups: dict[int, set[str]] = {}
    for raw_entry in raw_splits:
        if not isinstance(raw_entry, dict):
            raise ValueError("Every split_manifests entry must be an object")
        try:
            fold = int(raw_entry["outer_fold"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Split-manifest entries require integer outer_fold") from exc
        manifest_path_value = str(raw_entry.get("path", ""))
        if not manifest_path_value:
            raise ValueError("Split-manifest entries require a non-empty path")
        manifest_hash = _require_sha256(
            raw_entry.get("manifest_payload_sha256"),
            "split_manifests.manifest_payload_sha256",
        )
        manifest_path = Path(manifest_path_value)
        if not manifest_path.is_absolute():
            manifest_path = manifest_path.resolve()
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Frozen split manifest bound by scoring universe is absent: {manifest_path}"
            )
        try:
            split_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Frozen split manifest is invalid JSON: {manifest_path}") from exc
        if not isinstance(split_payload, dict):
            raise ValueError("Frozen split manifest must be a JSON object")
        internal_hash = _require_sha256(
            split_payload.get("manifest_payload_sha256"),
            "frozen_split.manifest_payload_sha256",
        )
        unhashed_payload = dict(split_payload)
        del unhashed_payload["manifest_payload_sha256"]
        recomputed_hash = hashlib.sha256(
            _canonical_json_bytes(unhashed_payload)
        ).hexdigest()
        if internal_hash != recomputed_hash or internal_hash != manifest_hash:
            raise ValueError(
                f"Frozen split manifest payload hash mismatch: {manifest_path}"
            )
        cross_validation = split_payload.get("cross_validation")
        if not isinstance(cross_validation, dict) or int(
            cross_validation.get("outer_fold", -1)
        ) != fold:
            raise ValueError("Frozen split manifest outer_fold does not match universe")
        split_ids = split_payload.get("split_ids")
        split_groups = split_payload.get("split_groups")
        if not isinstance(split_ids, dict) or not isinstance(split_ids.get("test"), list):
            raise ValueError("Frozen split manifest requires split_ids.test")
        if not isinstance(split_groups, dict) or not isinstance(split_groups.get("test"), list):
            raise ValueError("Frozen split manifest requires split_groups.test")
        split_test_ids[fold] = {str(value) for value in split_ids["test"]}
        split_test_groups[fold] = {str(value) for value in split_groups["test"]}
        split_manifests.append(
            {
                "outer_fold": fold,
                "path": str(manifest_path),
                "manifest_payload_sha256": manifest_hash,
                "file_sha256": _sha256_file(manifest_path),
            }
        )
    split_manifests.sort(key=lambda entry: int(entry["outer_fold"]))
    observed_folds = tuple(int(entry["outer_fold"]) for entry in split_manifests)
    expected_folds = tuple(range(DATASET_OUTER_FOLDS[dataset_key]))
    if observed_folds != expected_folds:
        raise ValueError(
            "Scoring sample universe must bind every outer fold exactly once; "
            f"expected={expected_folds}, observed={observed_folds}"
        )
    if len({entry["path"] for entry in split_manifests}) != len(split_manifests):
        raise ValueError("Scoring sample-universe split paths must be unique")
    approved_hashes = tuple(
        FROZEN_SPLIT_MANIFEST_SHA256S[dataset_key]
        if expected_split_sha256s is None
        else map(str, expected_split_sha256s)
    )
    observed_hashes = tuple(
        str(entry["manifest_payload_sha256"]) for entry in split_manifests
    )
    if observed_hashes != approved_hashes:
        raise ValueError(
            "Scoring sample-universe split hashes differ from the frozen protocol"
        )

    raw_samples = payload.get("samples")
    if not isinstance(raw_samples, list) or not raw_samples:
        raise ValueError("Scoring sample universe requires non-empty samples")
    samples: list[dict[str, Any]] = []
    for raw_row in raw_samples:
        if not isinstance(raw_row, dict):
            raise ValueError("Every scoring sample row must be an object")
        sample_key = str(raw_row.get("sample_key", ""))
        group_id = str(raw_row.get("group_id", ""))
        try:
            y_true = int(raw_row["y_true"])
            outer_fold = int(raw_row["outer_fold"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "Scoring sample rows require integer y_true and outer_fold"
            ) from exc
        if not sample_key or not group_id or outer_fold not in expected_folds:
            raise ValueError("Scoring sample row has invalid key, group, or fold")
        samples.append(
            {
                "sample_key": sample_key,
                "group_id": group_id,
                "y_true": y_true,
                "outer_fold": outer_fold,
            }
        )
    samples.sort(key=lambda row: str(row["sample_key"]))
    keys = [str(row["sample_key"]) for row in samples]
    if len(set(keys)) != len(keys):
        raise ValueError("Scoring sample universe contains duplicate sample keys")
    row_folds = {int(row["outer_fold"]) for row in samples}
    if row_folds != set(expected_folds):
        raise ValueError("Scoring sample universe has no samples for one or more outer folds")
    group_folds: dict[str, int] = {}
    global_classes = {int(row["y_true"]) for row in samples}
    for row in samples:
        group = str(row["group_id"])
        fold = int(row["outer_fold"])
        previous = group_folds.setdefault(group, fold)
        if previous != fold:
            raise ValueError("Scoring sample-universe groups cross outer folds")
    for fold in expected_folds:
        classes = {
            int(row["y_true"]) for row in samples if int(row["outer_fold"]) == fold
        }
        if classes != global_classes:
            raise ValueError("Every scoring-universe fold must cover all observed classes")
        fold_rows = [row for row in samples if int(row["outer_fold"]) == fold]
        observed_groups = {str(row["group_id"]) for row in fold_rows}
        observed_file_ids: set[str] = set()
        for row in fold_rows:
            sample_key = str(row["sample_key"])
            try:
                file_id, window_id = sample_key.rsplit(":", 1)
                int(window_id)
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    "Scoring sample keys must use canonical file_id:window_id form"
                ) from exc
            if not file_id:
                raise ValueError("Scoring sample key contains an empty file_id")
            observed_file_ids.add(file_id)
        if observed_groups != split_test_groups[fold]:
            raise ValueError(
                f"Scoring universe group coverage differs from split fold {fold}"
            )
        if observed_file_ids != split_test_ids[fold]:
            raise ValueError(
                f"Scoring universe file coverage differs from split fold {fold}"
            )

    return ScoringUniverse(
        path=target,
        file_sha256=_sha256_file(target),
        protocol_id=protocol_id,
        dataset_key=dataset_key,
        dataset_slug=dataset_slug,
        dataset_id=dataset_id,
        folds=expected_folds,
        split_manifests=tuple(split_manifests),
        samples=tuple(samples),
    )


@dataclass(frozen=True)
class ScoringDerangement:
    path: Path
    file_sha256: str
    sample_universe_file_sha256: str
    sample_universe_sha256: str
    mapping_sha256: str
    mapping: Mapping[str, str]
    universe: ScoringUniverse


def freeze_scoring_derangement(
    universe_source: ScoringUniverse,
    path: str | Path,
    *,
    seed: int = SCORING_DERANGEMENT_SEED,
) -> ScoringDerangement:
    """Create once, or exactly validate, the dataset-level scoring mapping."""

    if _sha256_file(universe_source.path) != universe_source.file_sha256:
        raise ValueError("Scoring sample-universe file drifted after validation")
    for split_manifest in universe_source.split_manifests:
        split_path = Path(str(split_manifest["path"]))
        if _sha256_file(split_path) != str(split_manifest["file_sha256"]):
            raise ValueError(f"Frozen split manifest drifted: {split_path}")

    universe = [dict(row) for row in universe_source.samples]
    strata: dict[tuple[str, int], list[str]] = defaultdict(list)
    for row in universe:
        strata[(str(row["group_id"]), int(row["y_true"]))].append(
            str(row["sample_key"])
        )
    sample_universe_sha256 = hashlib.sha256(_canonical_json_bytes(universe)).hexdigest()

    rng = np.random.Generator(np.random.PCG64(int(seed)))
    mapping: dict[str, str] = {}
    for stratum in sorted(strata):
        keys = sorted(strata[stratum])
        if len(keys) < 2:
            raise ValueError(
                f"Scoring derangement stratum {stratum!r} has fewer than two samples"
            )
        shuffled = np.asarray(keys, dtype=str)[rng.permutation(len(keys))].tolist()
        for index, source in enumerate(shuffled):
            mapping[source] = shuffled[(index + 1) % len(shuffled)]
    if set(mapping) != {row["sample_key"] for row in universe}:
        raise AssertionError("Scoring derangement does not cover the OOF universe")
    if any(source == partner for source, partner in mapping.items()):
        raise AssertionError("Scoring derangement contains a self pair")
    if len(set(mapping.values())) != len(mapping):
        raise AssertionError("Scoring derangement is not bijective")

    pairs = [
        {"source": source, "partner": mapping[source]} for source in sorted(mapping)
    ]
    mapping_sha256 = hashlib.sha256(_canonical_json_bytes(pairs)).hexdigest()
    payload = {
        "schema_version": 1,
        "protocol_id": universe_source.protocol_id,
        "dataset_key": universe_source.dataset_key,
        "dataset_slug": universe_source.dataset_slug,
        "dataset_id": universe_source.dataset_id,
        "algorithm": SCORING_ALGORITHM,
        "rng": "numpy.PCG64",
        "seed": int(seed),
        "sample_universe_source": str(universe_source.path),
        "sample_universe_file_sha256": universe_source.file_sha256,
        "ordered_split_manifests": [dict(entry) for entry in universe_source.split_manifests],
        "sample_universe_sha256": sample_universe_sha256,
        "mapping_sha256": mapping_sha256,
        "sample_count": len(mapping),
        "self_pairs": 0,
        "partner_bijection": True,
        "mapping": pairs,
    }
    target = Path(path)
    expected_rendered = (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )
    if target.exists():
        existing_text = target.read_text(encoding="utf-8")
        try:
            existing = json.loads(existing_text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Frozen scoring manifest is invalid JSON: {target}") from exc
        if existing != payload or existing_text != expected_rendered:
            raise ValueError(
                f"Frozen scoring manifest does not match the validated input universe: {target}"
            )
    else:
        try:
            _atomic_write_new_json(target, payload)
        except FileExistsError:
            existing_text = target.read_text(encoding="utf-8")
            existing = json.loads(existing_text)
            if existing != payload or existing_text != expected_rendered:
                raise ValueError("Concurrent scoring manifest has a different payload")
    return ScoringDerangement(
        path=target,
        file_sha256=_sha256_file(target),
        sample_universe_file_sha256=universe_source.file_sha256,
        sample_universe_sha256=sample_universe_sha256,
        mapping_sha256=mapping_sha256,
        mapping=mapping,
        universe=universe_source,
    )


def validate_grid_against_scoring_universe(
    grid: ArtifactGrid, derangement: ScoringDerangement
) -> None:
    """Bind a full-OOF or G050 artifact subset to the frozen full universe."""

    universe = derangement.universe
    if (
        grid.protocol_id != universe.protocol_id
        or grid.dataset_key != universe.dataset_key
        or grid.dataset_slug != universe.dataset_slug
        or grid.dataset_id != universe.dataset_id
    ):
        raise ValueError("Artifact grid identity differs from scoring sample universe")
    split_hashes = universe.split_manifest_sha256s
    for fold in grid.folds:
        if grid.split_manifest_sha256s[fold] != split_hashes[fold]:
            raise ValueError(f"Artifact split hash differs from scoring universe at fold {fold}")
        expected = sorted(
            (
                str(row["sample_key"]),
                str(row["group_id"]),
                int(row["y_true"]),
            )
            for row in universe.samples
            if int(row["outer_fold"]) == fold
        )
        rows = grid.reference[fold]
        observed = sorted(
            zip(
                np.asarray(rows["sample_key"]).astype(str).tolist(),
                np.asarray(rows["group_id"]).astype(str).tolist(),
                np.asarray(rows["y_true"]).astype(int).tolist(),
            )
        )
        if observed != expected:
            raise ValueError(
                f"Artifact test rows do not exactly match scoring universe fold {fold}"
            )


def accuracy_metric_values(grid: ArtifactGrid) -> MetricValues:
    values: MetricValues = {}
    for arm in grid.arms:
        for seed in grid.seeds:
            for fold in grid.folds:
                labels = grid.array(arm, seed, fold, "y_true").astype(int)
                predictions = grid.array(arm, seed, fold, "y_pred").astype(int)
                values[(arm, seed, fold)] = (labels == predictions).astype(float)
    return values


def _row_cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_matrix = np.asarray(left, dtype=float).reshape(len(left), -1)
    right_matrix = np.asarray(right, dtype=float).reshape(len(right), -1)
    if left_matrix.shape != right_matrix.shape:
        raise ValueError("Alignment representations must have identical flattened shapes")
    left_norm = np.linalg.norm(left_matrix, axis=1)
    right_norm = np.linalg.norm(right_matrix, axis=1)
    if np.any(left_norm == 0) or np.any(right_norm == 0):
        raise ValueError("Alignment cosine is undefined for zero-norm representations")
    values = np.sum(left_matrix * right_matrix, axis=1) / (left_norm * right_norm)
    if not np.isfinite(values).all():
        raise ValueError("Alignment cosine contains non-finite values")
    return values


def paired_alignment_margin(
    representation_1d: np.ndarray,
    representation_2d: np.ndarray,
    sample_keys: Sequence[str],
    mapping: Mapping[str, str],
) -> np.ndarray:
    """Return row-level paired cosine minus frozen partner cosine."""

    keys = [str(key) for key in sample_keys]
    if len(keys) != len(set(keys)):
        raise ValueError("Alignment sample keys must be unique")
    if set(keys) != set(mapping):
        raise ValueError("Alignment mapping must exactly cover the supplied sample keys")
    index = {key: position for position, key in enumerate(keys)}
    try:
        partner_indices = np.asarray([index[mapping[key]] for key in keys], dtype=int)
    except KeyError as exc:
        raise ValueError("Alignment partner is absent from the supplied sample rows") from exc
    paired = _row_cosine(representation_1d, representation_2d)
    reference = _row_cosine(representation_1d, np.asarray(representation_2d)[partner_indices])
    return paired - reference


def alignment_metric_values(
    grid: ArtifactGrid,
    derangement: ScoringDerangement,
    *,
    arms: Sequence[str] | None = None,
) -> MetricValues:
    validate_grid_against_scoring_universe(grid, derangement)
    selected_arms = tuple(grid.arms if arms is None else map(str, arms))
    values: MetricValues = {}
    for arm in selected_arms:
        if arm not in ARM_ALIGNMENT_VIEWS:
            raise ValueError(f"No preregistered alignment representation pair for arm {arm!r}")
        left_name, right_name = ARM_ALIGNMENT_VIEWS[arm]
        for seed in grid.seeds:
            for fold in grid.folds:
                keys = grid.array(arm, seed, fold, "sample_key").astype(str).tolist()
                local_mapping = {key: derangement.mapping[key] for key in keys}
                if any(partner not in local_mapping for partner in local_mapping.values()):
                    raise ValueError("Scoring partner crosses an outer-fold artifact boundary")
                values[(arm, seed, fold)] = paired_alignment_margin(
                    grid.array(arm, seed, fold, f"repr__{left_name}"),
                    grid.array(arm, seed, fold, f"repr__{right_name}"),
                    keys,
                    local_mapping,
                )
    return values


def _cell_means(
    grid: ArtifactGrid,
    metric_values: MetricValues,
    arm: str,
    seed: int,
) -> dict[tuple[int, int, str], float]:
    cells: dict[tuple[int, int, str], float] = {}
    for fold in grid.folds:
        key = (arm, seed, fold)
        if key not in metric_values:
            raise ValueError(f"Metric values lack artifact cell {key}")
        values = np.asarray(metric_values[key], dtype=float)
        labels = np.asarray(grid.reference[fold]["y_true"]).astype(int)
        groups = np.asarray(grid.reference[fold]["group_id"]).astype(str)
        if values.ndim != 1 or len(values) != len(labels) or not np.isfinite(values).all():
            raise ValueError(f"Metric values for {key} are invalid or misaligned")
        for label in sorted(set(labels.tolist())):
            for group in sorted(set(groups[labels == label].tolist())):
                mask = (labels == label) & (groups == group)
                cells[(int(label), fold, group)] = float(values[mask].mean())
    return cells


def _weighted_cell_estimate(
    cells: Mapping[tuple[int, int, str], float],
    classes: Sequence[int],
    group_weights: Mapping[tuple[int, str], int],
) -> float:
    class_means: list[float] = []
    for label in classes:
        cell_values: list[float] = []
        cell_weights: list[int] = []
        for (cell_label, fold, group), value in cells.items():
            if cell_label != label:
                continue
            weight = int(group_weights.get((fold, group), 0))
            if weight > 0:
                cell_values.append(float(value))
                cell_weights.append(weight)
        if not cell_values:
            raise ValueError(f"Bootstrap draw omitted all groups for class {label}")
        class_means.append(float(np.average(cell_values, weights=cell_weights)))
    return float(np.mean(class_means))


def seed_metric_estimates(
    grid: ArtifactGrid,
    metric_values: MetricValues,
    arm: str,
) -> dict[int, float]:
    estimates: dict[int, float] = {}
    unit_weights = {
        (fold, group): 1
        for fold in grid.folds
        for group in set(np.asarray(grid.reference[fold]["group_id"]).astype(str).tolist())
    }
    for seed in grid.seeds:
        estimates[seed] = _weighted_cell_estimate(
            _cell_means(grid, metric_values, arm, seed),
            grid.classes,
            unit_weights,
        )
    return estimates


@dataclass(frozen=True)
class BootstrapMetricResult:
    point_estimate: float
    bootstrap_mean: float
    confidence_level: float
    interval_lower: float
    interval_upper: float
    interval_lower_mcse: float
    interval_upper_mcse: float


@dataclass(frozen=True)
class PairedBootstrapResult:
    replicates: int
    seed: int
    sampled_index_sha256: str
    metrics: Mapping[str, BootstrapMetricResult]
    replicate_effects: Mapping[str, np.ndarray]

    def summary(self) -> dict[str, Any]:
        return {
            "replicates": self.replicates,
            "seed": self.seed,
            "sampled_index_sha256": self.sampled_index_sha256,
            "metrics": {
                name: {
                    "point_estimate": result.point_estimate,
                    "bootstrap_mean": result.bootstrap_mean,
                    "confidence_level": result.confidence_level,
                    "interval_lower": result.interval_lower,
                    "interval_upper": result.interval_upper,
                    "interval_lower_mcse": result.interval_lower_mcse,
                    "interval_upper_mcse": result.interval_upper_mcse,
                    "endpoint_mcse_method": (
                        "empirical_quantile_local_spacing_bahadur_v1"
                    ),
                }
                for name, result in self.metrics.items()
            },
        }


def paired_hierarchical_bootstrap(
    grid: ArtifactGrid,
    metric_families: Mapping[str, MetricValues],
    arm_a: str,
    arm_b: str | None,
    *,
    replicates: int = 10000,
    seed: int = BOOTSTRAP_SEED,
    confidence_level: float = 0.95,
) -> PairedBootstrapResult:
    """Crossed bootstrap for a paired contrast or one absolute arm metric."""

    if arm_a not in grid.arms:
        raise ValueError("Bootstrap arm_a is absent from the validated grid")
    if arm_b is not None and (arm_a == arm_b or arm_b not in grid.arms):
        raise ValueError("Bootstrap contrast requires two distinct validated arms")
    if replicates <= 0:
        raise ValueError("Bootstrap replicates must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie strictly between zero and one")
    if not metric_families:
        raise ValueError("At least one metric family is required")

    cells: dict[str, dict[tuple[str, int], dict[tuple[int, int, str], float]]] = {}
    points: dict[str, float] = {}
    for metric_name, values in metric_families.items():
        metric_cells: dict[tuple[str, int], dict[tuple[int, int, str], float]] = {}
        selected_arms = (arm_a,) if arm_b is None else (arm_a, arm_b)
        for arm in selected_arms:
            for training_seed in grid.seeds:
                metric_cells[(arm, training_seed)] = _cell_means(
                    grid, values, arm, training_seed
                )
        cells[metric_name] = metric_cells
        estimate_a = seed_metric_estimates(grid, values, arm_a)
        if arm_b is None:
            points[metric_name] = float(np.mean(list(estimate_a.values())))
        else:
            estimate_b = seed_metric_estimates(grid, values, arm_b)
            points[metric_name] = float(
                np.mean(
                    [estimate_a[value] - estimate_b[value] for value in grid.seeds]
                )
            )

    groups_by_fold_stratum: dict[tuple[int, str], tuple[str, ...]] = {}
    for fold in grid.folds:
        groups = set(np.asarray(grid.reference[fold]["group_id"]).astype(str).tolist())
        by_stratum: dict[str, list[str]] = defaultdict(list)
        for group in groups:
            by_stratum[grid.group_strata[group]].append(group)
        for stratum, stratum_groups in by_stratum.items():
            groups_by_fold_stratum[(fold, stratum)] = tuple(sorted(stratum_groups))

    rng = np.random.Generator(np.random.PCG64(int(seed)))
    effects = {
        name: np.empty(replicates, dtype=float) for name in metric_families
    }
    sampled_index_digest = hashlib.sha256()
    for replicate in range(replicates):
        seed_draw = np.asarray(grid.seeds)[rng.integers(len(grid.seeds), size=len(grid.seeds))]
        if len(grid.folds) == 1:
            # G050 has no fold-resampling level; avoid even a degenerate RNG draw.
            fold_draw = np.asarray(grid.folds, dtype=int)
        else:
            fold_draw = np.asarray(grid.folds)[
                rng.integers(len(grid.folds), size=len(grid.folds))
            ]
        seed_weights = Counter(int(value) for value in seed_draw.tolist())
        group_weights: Counter[tuple[int, str]] = Counter()
        group_draw_record: list[dict[str, Any]] = []
        for fold_instance, fold in enumerate(fold_draw.astype(int).tolist()):
            for (candidate_fold, stratum), groups in sorted(groups_by_fold_stratum.items()):
                if candidate_fold != fold:
                    continue
                drawn = np.asarray(groups)[rng.integers(len(groups), size=len(groups))].tolist()
                for group in drawn:
                    group_weights[(fold, str(group))] += 1
                group_draw_record.append(
                    {
                        "fold_instance": fold_instance,
                        "fold": fold,
                        "stratum": stratum,
                        "groups": [str(group) for group in drawn],
                    }
                )
        sampled_index_digest.update(
            _canonical_json_bytes(
                {
                    "replicate": replicate,
                    "seeds": seed_draw.astype(int).tolist(),
                    "folds": fold_draw.astype(int).tolist(),
                    "group_draws": group_draw_record,
                }
            )
        )

        for metric_name, metric_cells in cells.items():
            seed_effects: list[float] = []
            seed_multiplicities: list[int] = []
            for training_seed, multiplicity in sorted(seed_weights.items()):
                value_a = _weighted_cell_estimate(
                    metric_cells[(arm_a, training_seed)], grid.classes, group_weights
                )
                if arm_b is None:
                    seed_effects.append(value_a)
                else:
                    value_b = _weighted_cell_estimate(
                        metric_cells[(arm_b, training_seed)],
                        grid.classes,
                        group_weights,
                    )
                    seed_effects.append(value_a - value_b)
                seed_multiplicities.append(multiplicity)
            effects[metric_name][replicate] = float(
                np.average(seed_effects, weights=seed_multiplicities)
            )

    alpha = 1.0 - confidence_level
    summaries: dict[str, BootstrapMetricResult] = {}
    for name, replicate_values in effects.items():
        lower_probability = alpha / 2.0
        upper_probability = 1.0 - alpha / 2.0
        lower, upper = np.quantile(
            replicate_values, [lower_probability, upper_probability]
        )
        lower_mcse = _quantile_endpoint_mcse(
            replicate_values, lower_probability
        )
        upper_mcse = _quantile_endpoint_mcse(
            replicate_values, upper_probability
        )
        replicate_values.setflags(write=False)
        summaries[name] = BootstrapMetricResult(
            point_estimate=points[name],
            bootstrap_mean=float(replicate_values.mean()),
            confidence_level=float(confidence_level),
            interval_lower=float(lower),
            interval_upper=float(upper),
            interval_lower_mcse=lower_mcse,
            interval_upper_mcse=upper_mcse,
        )
    return PairedBootstrapResult(
        replicates=int(replicates),
        seed=int(seed),
        sampled_index_sha256=sampled_index_digest.hexdigest(),
        metrics=summaries,
        replicate_effects=effects,
    )


def _quantile_endpoint_mcse(values: np.ndarray, probability: float) -> float:
    """Estimate percentile-endpoint MCSE using local empirical quantile spacing."""

    samples = np.asarray(values, dtype=float)
    if samples.ndim != 1 or samples.size < 2:
        raise ValueError("Endpoint MCSE requires at least two bootstrap replicates")
    probability_se = float(
        np.sqrt(probability * (1.0 - probability) / samples.size)
    )
    half_width = max(probability_se, 1.0 / samples.size)
    lower_probability = max(0.0, probability - half_width)
    upper_probability = min(1.0, probability + half_width)
    if upper_probability <= lower_probability:
        return 0.0
    lower_value, upper_value = np.quantile(
        samples, [lower_probability, upper_probability]
    )
    inverse_density = float(upper_value - lower_value) / (
        upper_probability - lower_probability
    )
    return max(0.0, probability_se * inverse_density)


def single_arm_hierarchical_bootstrap(
    grid: ArtifactGrid,
    metric_families: Mapping[str, MetricValues],
    arm: str,
    *,
    replicates: int = 10000,
    seed: int = BOOTSTRAP_SEED,
    confidence_level: float = 0.975,
) -> PairedBootstrapResult:
    """Bootstrap an absolute arm metric with the same crossed sampling contract."""

    return paired_hierarchical_bootstrap(
        grid,
        metric_families,
        arm,
        None,
        replicates=replicates,
        seed=seed,
        confidence_level=confidence_level,
    )


def collapse_diagnostic(
    grid: ArtifactGrid,
    *,
    arm: str = "FULL",
    views: Sequence[str] = ("shared_1d", "shared_2d"),
    threshold: float = 0.01,
) -> dict[str, Any]:
    """Compute checkpoint-local shared spread without cross-model concatenation."""

    if arm not in grid.arms:
        raise ValueError(f"Collapse arm {arm!r} is absent from the artifact grid")
    if threshold <= 0:
        raise ValueError("Collapse threshold must be positive")
    summaries: dict[str, Any] = {}
    passes = True
    for view in views:
        checkpoint_scores: list[dict[str, Any]] = []
        for training_seed in grid.seeds:
            for fold in grid.folds:
                representation = grid.array(
                    arm, training_seed, fold, f"repr__{view}"
                ).astype(float).reshape(
                    len(grid.reference[fold]["y_true"]), -1
                )
                labels = np.asarray(grid.reference[fold]["y_true"]).astype(int)
                groups = np.asarray(grid.reference[fold]["group_id"]).astype(str)
                standard_deviations: list[np.ndarray] = []
                for label in sorted(set(labels.tolist())):
                    for group in sorted(set(groups[labels == label].tolist())):
                        mask = (labels == label) & (groups == group)
                        if int(mask.sum()) < 2:
                            raise ValueError(
                                "Collapse diagnostic requires at least two windows in every "
                                f"group-class cell; seed={training_seed}, fold={fold}, "
                                f"group={group}, class={label}"
                            )
                        standard_deviations.append(
                            np.std(representation[mask], axis=0, ddof=1)
                        )
                flattened = np.concatenate(standard_deviations)
                if not np.isfinite(flattened).all():
                    raise ValueError("Collapse diagnostic produced non-finite sample SD")
                checkpoint_scores.append(
                    {
                        "training_seed": training_seed,
                        "outer_fold": fold,
                        "score": float(np.median(flattened)),
                    }
                )
        dataset_score = float(
            np.median([entry["score"] for entry in checkpoint_scores])
        )
        collapsed = dataset_score < threshold
        passes = passes and not collapsed
        summaries[view] = {
            "checkpoint_scores": checkpoint_scores,
            "dataset_median": dataset_score,
            "collapsed": collapsed,
        }
    return {
        "arm": arm,
        "threshold": float(threshold),
        "views": summaries,
        "passes_no_collapse": bool(passes),
    }


def exact_two_sided_sign_flip(seed_effects: Sequence[float]) -> float:
    """Enumerate the exact two-sided sign-flip sensitivity p-value."""

    values = np.asarray(seed_effects, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("seed_effects must be a finite non-empty vector")
    observed = abs(float(values.mean()))
    exceedances = 0
    for assignment in range(1 << len(values)):
        signs = np.asarray(
            [1.0 if assignment & (1 << index) else -1.0 for index in range(len(values))]
        )
        if abs(float(np.mean(signs * values))) >= observed - 1e-15:
            exceedances += 1
    return float(exceedances / (1 << len(values)))
