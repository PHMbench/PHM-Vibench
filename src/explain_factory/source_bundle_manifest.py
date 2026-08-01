"""Canonical, fail-closed source-bundle manifests for P02.

The ``p02.source-bundle-manifest.v1`` object binds one frozen sibling-paper
model/checkpoint stratum to immutable software, data-contract, adapter,
artifact, preprocessing, target, and score identities.  It is deliberately a
pure manifest layer: locators are recorded but no source artifact is opened and
the module is not connected to the maintained runtime.

All declared collections are normalized before hashing.  ``manifest_sha256``
is the SHA-256 of compact canonical JSON for every other manifest field and is
therefore excluded from its own digest.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any


SOURCE_BUNDLE_MANIFEST_SCHEMA = "p02.source-bundle-manifest.v1"
TARGET_ID = "original-predicted-class-v1"
TARGET_SEMANTICS = "class fixed on the unperturbed model input"
SCORE_ID = "softmax-target-probability-v1"
SCORE_SEMANTICS = "softmax probability of the fixed target class"

_HASH_FIELD = "manifest_sha256"
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")

_BASE_KEYS = frozenset(
    {
        "schema",
        "source",
        "provenance",
        "references",
        "adapter",
        "methods",
        "artifacts",
        "preprocessing",
        "target",
        "score",
    }
)
_DERIVED_KEYS = frozenset({"derived"})
_CONTENT_KEYS = _BASE_KEYS | _DERIVED_KEYS

_SOURCE_KEYS = frozenset(
    {
        "paper_id",
        "run_id",
        "model_id",
        "model_architecture_id",
        "model_architecture_sha256",
        "checkpoint_id",
        "checkpoint_sha256",
        "model_seed",
    }
)
_PROVENANCE_KEYS = frozenset({"code", "config", "environment"})
_HASH_IDENTITY_KEYS = frozenset({"id", "sha256"})
_REFERENCE_KEYS = frozenset(
    {
        "dataset_id",
        "dataset_release_id",
        "dataset_release_sha256",
        "task_transform_id",
        "task_transform_sha256",
        "split_manifest_sha256",
        "sample_cohort_manifest_sha256",
    }
)
_ADAPTER_KEYS = frozenset(
    {
        "adapter_id",
        "adapter_version",
        "adapter_sha256",
        "input_kind",
        "output_kind",
        "capabilities",
        "custom_source_fork",
        "source_specific_metric_branch",
    }
)
_METHOD_KEYS = frozenset(
    {
        "method_id",
        "method_role",
        "method_version",
        "implementation_sha256",
        "capabilities",
        "output_artifact_ids",
    }
)
_ARTIFACT_KEYS = frozenset(
    {"artifact_id", "role", "locator", "sha256", "media_type"}
)
_PREPROCESSING_KEYS = frozenset(
    {"preprocessing_id", "learned_parameters", "fit_split", "train_fit_proof"}
)
_TRAIN_FIT_PROOF_KEYS = frozenset(
    {"artifact_id", "artifact_sha256", "split_manifest_sha256"}
)
_TARGET_KEYS = frozenset(
    {
        "target_id",
        "semantics",
        "sample_ids_sha256",
        "values_artifact_id",
        "values_artifact_sha256",
    }
)
_SCORE_KEYS = frozenset(
    {
        "score_id",
        "semantics",
        "sample_ids_sha256",
        "values_artifact_id",
        "values_artifact_sha256",
    }
)
_DERIVED_VALUE_KEYS = frozenset(
    {
        "artifact_list_sha256",
        "method_list_sha256",
        "learned_parameters_sha256",
        "preprocessing_identity_sha256",
        "target_policy_sha256",
        "target_identity_sha256",
        "score_policy_sha256",
        "score_identity_sha256",
        "shared_protocol_identity_sha256",
        "source_block_identity_sha256",
    }
)


class SourceBundleManifestError(ValueError):
    """Raised when a source bundle is incomplete, unsafe, or tampered."""


def _require_mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SourceBundleManifestError(f"{location} must be a JSON object")
    if not all(isinstance(key, str) for key in value):
        raise SourceBundleManifestError(f"{location} keys must be strings")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], expected: frozenset[str], location: str
) -> None:
    actual = frozenset(value)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing:
        raise SourceBundleManifestError(
            f"{location} is missing required keys: {', '.join(missing)}"
        )
    if unknown:
        raise SourceBundleManifestError(
            f"{location} has unknown keys: {', '.join(unknown)}"
        )


def _require_nonempty_string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value:
        raise SourceBundleManifestError(f"{location} must be a non-empty string")
    if value != value.strip():
        raise SourceBundleManifestError(
            f"{location} must not have leading or trailing whitespace"
        )
    return value


def _require_sha256(value: Any, location: str) -> str:
    digest = _require_nonempty_string(value, location)
    if _SHA256_RE.fullmatch(digest) is None:
        raise SourceBundleManifestError(
            f"{location} must be a lowercase 64-character SHA-256 digest"
        )
    return digest


def _require_array(value: Any, location: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(
        value, (str, bytes, bytearray)
    ):
        raise SourceBundleManifestError(f"{location} must be a JSON array")
    return value


def _require_string_array(value: Any, location: str) -> list[str]:
    values = [
        _require_nonempty_string(item, f"{location}[{index}]")
        for index, item in enumerate(_require_array(value, location))
    ]
    if not values:
        raise SourceBundleManifestError(f"{location} must not be empty")
    if len(values) != len(set(values)):
        raise SourceBundleManifestError(f"{location} must not contain duplicates")
    return sorted(values)


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SourceBundleManifestError(
            "manifest is not canonical JSON data"
        ) from exc


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _detached_json(value: Any, location: str) -> Any:
    try:
        encoded = _canonical_json_bytes(value)
        return json.loads(encoded.decode("utf-8"))
    except SourceBundleManifestError as exc:
        raise SourceBundleManifestError(
            f"{location} must contain only canonical JSON data"
        ) from exc


def _normalize_hash_identity(value: Any, location: str) -> dict[str, str]:
    identity = _require_mapping(value, location)
    _require_exact_keys(identity, _HASH_IDENTITY_KEYS, location)
    return {
        "id": _require_nonempty_string(identity["id"], f"{location}.id"),
        "sha256": _require_sha256(identity["sha256"], f"{location}.sha256"),
    }


def _normalize_source(value: Any) -> dict[str, Any]:
    source = _require_mapping(value, "manifest.source")
    _require_exact_keys(source, _SOURCE_KEYS, "manifest.source")
    seed = source["model_seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise SourceBundleManifestError(
            "manifest.source.model_seed must be a non-negative integer"
        )
    return {
        "paper_id": _require_nonempty_string(
            source["paper_id"], "manifest.source.paper_id"
        ),
        "run_id": _require_nonempty_string(source["run_id"], "manifest.source.run_id"),
        "model_id": _require_nonempty_string(
            source["model_id"], "manifest.source.model_id"
        ),
        "model_architecture_id": _require_nonempty_string(
            source["model_architecture_id"],
            "manifest.source.model_architecture_id",
        ),
        "model_architecture_sha256": _require_sha256(
            source["model_architecture_sha256"],
            "manifest.source.model_architecture_sha256",
        ),
        "checkpoint_id": _require_nonempty_string(
            source["checkpoint_id"], "manifest.source.checkpoint_id"
        ),
        "checkpoint_sha256": _require_sha256(
            source["checkpoint_sha256"], "manifest.source.checkpoint_sha256"
        ),
        "model_seed": seed,
    }


def _normalize_provenance(value: Any) -> dict[str, Any]:
    provenance = _require_mapping(value, "manifest.provenance")
    _require_exact_keys(provenance, _PROVENANCE_KEYS, "manifest.provenance")
    return {
        name: _normalize_hash_identity(
            provenance[name], f"manifest.provenance.{name}"
        )
        for name in ("code", "config", "environment")
    }


def _normalize_references(value: Any) -> dict[str, str]:
    references = _require_mapping(value, "manifest.references")
    _require_exact_keys(references, _REFERENCE_KEYS, "manifest.references")
    normalized = {
        "dataset_id": _require_nonempty_string(
            references["dataset_id"], "manifest.references.dataset_id"
        ),
        "dataset_release_id": _require_nonempty_string(
            references["dataset_release_id"],
            "manifest.references.dataset_release_id",
        ),
        "task_transform_id": _require_nonempty_string(
            references["task_transform_id"],
            "manifest.references.task_transform_id",
        ),
    }
    for key in (
        "dataset_release_sha256",
        "task_transform_sha256",
        "split_manifest_sha256",
        "sample_cohort_manifest_sha256",
    ):
        normalized[key] = _require_sha256(
            references[key], f"manifest.references.{key}"
        )
    return normalized


def _normalize_adapter(value: Any) -> dict[str, Any]:
    adapter = _require_mapping(value, "manifest.adapter")
    _require_exact_keys(adapter, _ADAPTER_KEYS, "manifest.adapter")
    custom_fork = adapter["custom_source_fork"]
    if not isinstance(custom_fork, bool):
        raise SourceBundleManifestError(
            "manifest.adapter.custom_source_fork must be a boolean"
        )
    if custom_fork:
        raise SourceBundleManifestError("custom source fork is forbidden")
    metric_branch = adapter["source_specific_metric_branch"]
    if not isinstance(metric_branch, bool):
        raise SourceBundleManifestError(
            "manifest.adapter.source_specific_metric_branch must be a boolean"
        )
    if metric_branch:
        raise SourceBundleManifestError("source-specific metric branch is forbidden")
    return {
        "adapter_id": _require_nonempty_string(
            adapter["adapter_id"], "manifest.adapter.adapter_id"
        ),
        "adapter_version": _require_nonempty_string(
            adapter["adapter_version"], "manifest.adapter.adapter_version"
        ),
        "adapter_sha256": _require_sha256(
            adapter["adapter_sha256"], "manifest.adapter.adapter_sha256"
        ),
        "input_kind": _require_nonempty_string(
            adapter["input_kind"], "manifest.adapter.input_kind"
        ),
        "output_kind": _require_nonempty_string(
            adapter["output_kind"], "manifest.adapter.output_kind"
        ),
        "capabilities": _require_string_array(
            adapter["capabilities"], "manifest.adapter.capabilities"
        ),
        "custom_source_fork": False,
        "source_specific_metric_branch": False,
    }


def _normalize_artifacts(value: Any) -> list[dict[str, str]]:
    artifacts = _require_array(value, "manifest.artifacts")
    if not artifacts:
        raise SourceBundleManifestError("manifest.artifacts must not be empty")
    normalized: list[dict[str, str]] = []
    artifact_ids: set[str] = set()
    locators: set[str] = set()
    for index, raw_artifact in enumerate(artifacts):
        location = f"manifest.artifacts[{index}]"
        artifact = _require_mapping(raw_artifact, location)
        _require_exact_keys(artifact, _ARTIFACT_KEYS, location)
        normalized_artifact = {
            "artifact_id": _require_nonempty_string(
                artifact["artifact_id"], f"{location}.artifact_id"
            ),
            "role": _require_nonempty_string(artifact["role"], f"{location}.role"),
            "locator": _require_nonempty_string(
                artifact["locator"], f"{location}.locator"
            ),
            "sha256": _require_sha256(artifact["sha256"], f"{location}.sha256"),
            "media_type": _require_nonempty_string(
                artifact["media_type"], f"{location}.media_type"
            ),
        }
        artifact_id = normalized_artifact["artifact_id"]
        locator = normalized_artifact["locator"]
        if artifact_id in artifact_ids:
            raise SourceBundleManifestError(f"duplicate artifact_id {artifact_id!r}")
        if locator in locators:
            raise SourceBundleManifestError(f"duplicate artifact locator {locator!r}")
        artifact_ids.add(artifact_id)
        locators.add(locator)
        normalized.append(normalized_artifact)
    return sorted(normalized, key=lambda item: item["artifact_id"])


def _normalize_methods(value: Any) -> list[dict[str, Any]]:
    methods = _require_array(value, "manifest.methods")
    if not methods:
        raise SourceBundleManifestError("manifest.methods must not be empty")
    normalized: list[dict[str, Any]] = []
    method_ids: set[str] = set()
    for index, raw_method in enumerate(methods):
        location = f"manifest.methods[{index}]"
        method = _require_mapping(raw_method, location)
        _require_exact_keys(method, _METHOD_KEYS, location)
        normalized_method = {
            "method_id": _require_nonempty_string(
                method["method_id"], f"{location}.method_id"
            ),
            "method_role": _require_nonempty_string(
                method["method_role"], f"{location}.method_role"
            ),
            "method_version": _require_nonempty_string(
                method["method_version"], f"{location}.method_version"
            ),
            "implementation_sha256": _require_sha256(
                method["implementation_sha256"],
                f"{location}.implementation_sha256",
            ),
            "capabilities": _require_string_array(
                method["capabilities"], f"{location}.capabilities"
            ),
            "output_artifact_ids": _require_string_array(
                method["output_artifact_ids"], f"{location}.output_artifact_ids"
            ),
        }
        if normalized_method["method_role"] != "real":
            raise SourceBundleManifestError(
                f"{location}.method_role must equal 'real'; controls are not "
                "source methods"
            )
        method_id = normalized_method["method_id"]
        if method_id in method_ids:
            raise SourceBundleManifestError(f"duplicate method_id {method_id!r}")
        method_ids.add(method_id)
        normalized.append(normalized_method)
    return sorted(normalized, key=lambda item: item["method_id"])


def _normalize_preprocessing(value: Any) -> dict[str, Any]:
    preprocessing = _require_mapping(value, "manifest.preprocessing")
    _require_exact_keys(
        preprocessing, _PREPROCESSING_KEYS, "manifest.preprocessing"
    )
    fit_split = _require_nonempty_string(
        preprocessing["fit_split"], "manifest.preprocessing.fit_split"
    )
    if fit_split != "train":
        raise SourceBundleManifestError(
            "manifest.preprocessing.fit_split must equal 'train'"
        )
    learned_parameters = _require_mapping(
        preprocessing["learned_parameters"],
        "manifest.preprocessing.learned_parameters",
    )
    proof = _require_mapping(
        preprocessing["train_fit_proof"],
        "manifest.preprocessing.train_fit_proof",
    )
    _require_exact_keys(
        proof,
        _TRAIN_FIT_PROOF_KEYS,
        "manifest.preprocessing.train_fit_proof",
    )
    return {
        "preprocessing_id": _require_nonempty_string(
            preprocessing["preprocessing_id"],
            "manifest.preprocessing.preprocessing_id",
        ),
        "learned_parameters": _detached_json(
            learned_parameters, "manifest.preprocessing.learned_parameters"
        ),
        "fit_split": "train",
        "train_fit_proof": {
            "artifact_id": _require_nonempty_string(
                proof["artifact_id"],
                "manifest.preprocessing.train_fit_proof.artifact_id",
            ),
            "artifact_sha256": _require_sha256(
                proof["artifact_sha256"],
                "manifest.preprocessing.train_fit_proof.artifact_sha256",
            ),
            "split_manifest_sha256": _require_sha256(
                proof["split_manifest_sha256"],
                "manifest.preprocessing.train_fit_proof.split_manifest_sha256",
            ),
        },
    }


def _normalize_target(value: Any) -> dict[str, str]:
    target = _require_mapping(value, "manifest.target")
    _require_exact_keys(target, _TARGET_KEYS, "manifest.target")
    target_id = _require_nonempty_string(
        target["target_id"], "manifest.target.target_id"
    )
    semantics = _require_nonempty_string(
        target["semantics"], "manifest.target.semantics"
    )
    if target_id != TARGET_ID:
        raise SourceBundleManifestError(
            f"manifest.target.target_id must equal {TARGET_ID!r}"
        )
    if semantics != TARGET_SEMANTICS:
        raise SourceBundleManifestError(
            f"manifest.target.semantics must equal {TARGET_SEMANTICS!r}"
        )
    return {
        "target_id": TARGET_ID,
        "semantics": TARGET_SEMANTICS,
        "sample_ids_sha256": _require_sha256(
            target["sample_ids_sha256"], "manifest.target.sample_ids_sha256"
        ),
        "values_artifact_id": _require_nonempty_string(
            target["values_artifact_id"],
            "manifest.target.values_artifact_id",
        ),
        "values_artifact_sha256": _require_sha256(
            target["values_artifact_sha256"],
            "manifest.target.values_artifact_sha256",
        ),
    }


def _normalize_score(value: Any) -> dict[str, str]:
    score = _require_mapping(value, "manifest.score")
    _require_exact_keys(score, _SCORE_KEYS, "manifest.score")
    score_id = _require_nonempty_string(score["score_id"], "manifest.score.score_id")
    semantics = _require_nonempty_string(
        score["semantics"], "manifest.score.semantics"
    )
    if score_id != SCORE_ID:
        raise SourceBundleManifestError(
            f"manifest.score.score_id must equal {SCORE_ID!r}"
        )
    if semantics != SCORE_SEMANTICS:
        raise SourceBundleManifestError(
            f"manifest.score.semantics must equal {SCORE_SEMANTICS!r}"
        )
    return {
        "score_id": SCORE_ID,
        "semantics": SCORE_SEMANTICS,
        "sample_ids_sha256": _require_sha256(
            score["sample_ids_sha256"], "manifest.score.sample_ids_sha256"
        ),
        "values_artifact_id": _require_nonempty_string(
            score["values_artifact_id"],
            "manifest.score.values_artifact_id",
        ),
        "values_artifact_sha256": _require_sha256(
            score["values_artifact_sha256"],
            "manifest.score.values_artifact_sha256",
        ),
    }


def _audit_cross_references(base: Mapping[str, Any]) -> None:
    artifacts = {item["artifact_id"]: item for item in base["artifacts"]}
    source = base["source"]
    checkpoint = artifacts.get(source["checkpoint_id"])
    if checkpoint is None:
        raise SourceBundleManifestError(
            "source checkpoint_id must reference an artifact"
        )
    if checkpoint["role"] != "checkpoint":
        raise SourceBundleManifestError(
            "source checkpoint artifact must have role='checkpoint'"
        )
    if checkpoint["sha256"] != source["checkpoint_sha256"]:
        raise SourceBundleManifestError(
            "source checkpoint_sha256 does not match its artifact"
        )

    proof = base["preprocessing"]["train_fit_proof"]
    proof_artifact = artifacts.get(proof["artifact_id"])
    if proof_artifact is None:
        raise SourceBundleManifestError(
            "train-fit proof artifact_id must reference an artifact"
        )
    if proof_artifact["role"] != "preprocessing_fit_proof":
        raise SourceBundleManifestError(
            "train-fit proof artifact must have role='preprocessing_fit_proof'"
        )
    if proof_artifact["sha256"] != proof["artifact_sha256"]:
        raise SourceBundleManifestError(
            "train-fit proof hash does not match its artifact"
        )
    if proof["split_manifest_sha256"] != base["references"]["split_manifest_sha256"]:
        raise SourceBundleManifestError(
            "train-fit proof must reference the bundle split_manifest_sha256"
        )

    for section, required_role in (
        ("target", "target_vector"),
        ("score", "original_score_vector"),
    ):
        binding = base[section]
        artifact = artifacts.get(binding["values_artifact_id"])
        if artifact is None:
            raise SourceBundleManifestError(
                f"{section} values_artifact_id must reference an artifact"
            )
        if artifact["role"] != required_role:
            raise SourceBundleManifestError(
                f"{section} values artifact must have role={required_role!r}"
            )
        if artifact["sha256"] != binding["values_artifact_sha256"]:
            raise SourceBundleManifestError(
                f"{section} values artifact hash does not match its artifact"
            )
    if base["target"]["sample_ids_sha256"] != base["score"]["sample_ids_sha256"]:
        raise SourceBundleManifestError(
            "target and score values must use the identical ordered sample IDs"
        )

    claimed_outputs: dict[str, str] = {}
    adapter_capabilities = set(base["adapter"]["capabilities"])
    for method in base["methods"]:
        unsupported_capabilities = sorted(
            set(method["capabilities"]) - adapter_capabilities
        )
        if unsupported_capabilities:
            raise SourceBundleManifestError(
                f"method {method['method_id']!r} declares capabilities absent "
                f"from its adapter: {unsupported_capabilities}"
            )
        for artifact_id in method["output_artifact_ids"]:
            if artifact_id not in artifacts:
                raise SourceBundleManifestError(
                    f"method {method['method_id']!r} references unknown artifact "
                    f"{artifact_id!r}"
                )
            if artifacts[artifact_id]["role"] != "explanation":
                raise SourceBundleManifestError(
                    f"method output artifact {artifact_id!r} must have "
                    "role='explanation'"
                )
            previous = claimed_outputs.get(artifact_id)
            if previous is not None:
                raise SourceBundleManifestError(
                    f"artifact {artifact_id!r} is claimed by duplicate methods "
                    f"{previous!r} and {method['method_id']!r}"
                )
            claimed_outputs[artifact_id] = method["method_id"]
    for artifact_id, artifact in artifacts.items():
        if artifact["role"] == "explanation" and artifact_id not in claimed_outputs:
            raise SourceBundleManifestError(
                f"explanation artifact {artifact_id!r} must be claimed by exactly "
                "one method"
            )


def _normalize_base(root: Mapping[str, Any]) -> dict[str, Any]:
    _require_exact_keys(root, _BASE_KEYS, "manifest")
    if root["schema"] != SOURCE_BUNDLE_MANIFEST_SCHEMA:
        raise SourceBundleManifestError(
            f"manifest.schema must equal {SOURCE_BUNDLE_MANIFEST_SCHEMA!r}"
        )
    base = {
        "schema": SOURCE_BUNDLE_MANIFEST_SCHEMA,
        "source": _normalize_source(root["source"]),
        "provenance": _normalize_provenance(root["provenance"]),
        "references": _normalize_references(root["references"]),
        "adapter": _normalize_adapter(root["adapter"]),
        "methods": _normalize_methods(root["methods"]),
        "artifacts": _normalize_artifacts(root["artifacts"]),
        "preprocessing": _normalize_preprocessing(root["preprocessing"]),
        "target": _normalize_target(root["target"]),
        "score": _normalize_score(root["score"]),
    }
    _audit_cross_references(base)
    return base


def _derive_contract(base: Mapping[str, Any]) -> dict[str, str]:
    learned_parameters_sha256 = _canonical_sha256(
        base["preprocessing"]["learned_parameters"]
    )
    preprocessing_identity_sha256 = _canonical_sha256(base["preprocessing"])
    target_policy_sha256 = _canonical_sha256(
        {
            "target_id": base["target"]["target_id"],
            "semantics": base["target"]["semantics"],
        }
    )
    target_identity_sha256 = _canonical_sha256(base["target"])
    score_policy_sha256 = _canonical_sha256(
        {
            "score_id": base["score"]["score_id"],
            "semantics": base["score"]["semantics"],
        }
    )
    score_identity_sha256 = _canonical_sha256(base["score"])
    shared_protocol_identity = {
        "dataset_id": base["references"]["dataset_id"],
        "dataset_release_id": base["references"]["dataset_release_id"],
        "dataset_release_sha256": base["references"]["dataset_release_sha256"],
        "task_transform_id": base["references"]["task_transform_id"],
        "task_transform_sha256": base["references"]["task_transform_sha256"],
        "split_manifest_sha256": base["references"]["split_manifest_sha256"],
        "sample_cohort_manifest_sha256": base["references"][
            "sample_cohort_manifest_sha256"
        ],
        "target_policy_sha256": target_policy_sha256,
        "score_policy_sha256": score_policy_sha256,
    }
    shared_protocol_identity_sha256 = _canonical_sha256(shared_protocol_identity)
    source_block_identity = {
        "shared_protocol_identity_sha256": shared_protocol_identity_sha256,
        "model_architecture_id": base["source"]["model_architecture_id"],
        "model_architecture_sha256": base["source"]["model_architecture_sha256"],
        "model_id": base["source"]["model_id"],
        "checkpoint_sha256": base["source"]["checkpoint_sha256"],
        "model_seed": base["source"]["model_seed"],
        "preprocessing_identity_sha256": preprocessing_identity_sha256,
        "target_identity_sha256": target_identity_sha256,
        "score_identity_sha256": score_identity_sha256,
    }
    return {
        "artifact_list_sha256": _canonical_sha256(base["artifacts"]),
        "method_list_sha256": _canonical_sha256(base["methods"]),
        "learned_parameters_sha256": learned_parameters_sha256,
        "preprocessing_identity_sha256": preprocessing_identity_sha256,
        "target_policy_sha256": target_policy_sha256,
        "target_identity_sha256": target_identity_sha256,
        "score_policy_sha256": score_policy_sha256,
        "score_identity_sha256": score_identity_sha256,
        "shared_protocol_identity_sha256": shared_protocol_identity_sha256,
        "source_block_identity_sha256": _canonical_sha256(source_block_identity),
    }


def _assemble_content(base: Mapping[str, Any]) -> dict[str, Any]:
    return {**base, "derived": _derive_contract(base)}


def _normalize_content(manifest: Mapping[str, Any]) -> dict[str, Any]:
    root = _require_mapping(manifest, "manifest")
    content = {key: value for key, value in root.items() if key != _HASH_FIELD}
    _require_exact_keys(content, _CONTENT_KEYS, "manifest")
    base = _normalize_base({key: content[key] for key in _BASE_KEYS})
    canonical = _assemble_content(base)
    declared_derived = _require_mapping(content["derived"], "manifest.derived")
    _require_exact_keys(declared_derived, _DERIVED_VALUE_KEYS, "manifest.derived")
    for key in _DERIVED_VALUE_KEYS:
        _require_sha256(declared_derived[key], f"manifest.derived.{key}")
    if _canonical_json_bytes(declared_derived) != _canonical_json_bytes(
        canonical["derived"]
    ):
        raise SourceBundleManifestError(
            "manifest.derived does not match values derived from bundle bindings"
        )
    return canonical


def compute_source_bundle_manifest_sha256(manifest: Mapping[str, Any]) -> str:
    """Hash canonical content while excluding ``manifest_sha256`` itself."""

    content = _normalize_content(manifest)
    return hashlib.sha256(_canonical_json_bytes(content)).hexdigest()


def build_source_bundle_manifest(
    *,
    source: Mapping[str, Any],
    provenance: Mapping[str, Any],
    references: Mapping[str, Any],
    adapter: Mapping[str, Any],
    methods: Sequence[Mapping[str, Any]],
    artifacts: Sequence[Mapping[str, Any]],
    preprocessing: Mapping[str, Any],
    target: Mapping[str, Any],
    score: Mapping[str, Any],
) -> dict[str, Any]:
    """Build and self-hash one complete source-bundle manifest."""

    base = _normalize_base(
        {
            "schema": SOURCE_BUNDLE_MANIFEST_SCHEMA,
            "source": source,
            "provenance": provenance,
            "references": references,
            "adapter": adapter,
            "methods": methods,
            "artifacts": artifacts,
            "preprocessing": preprocessing,
            "target": target,
            "score": score,
        }
    )
    content = _assemble_content(base)
    content[_HASH_FIELD] = hashlib.sha256(_canonical_json_bytes(content)).hexdigest()
    return content


def validate_source_bundle_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Return detached canonical content after all hash and policy audits."""

    root = _require_mapping(manifest, "manifest")
    _require_exact_keys(root, _CONTENT_KEYS | {_HASH_FIELD}, "manifest")
    declared_hash = _require_sha256(root[_HASH_FIELD], f"manifest.{_HASH_FIELD}")
    content = _normalize_content(root)
    computed_hash = hashlib.sha256(_canonical_json_bytes(content)).hexdigest()
    if declared_hash != computed_hash:
        raise SourceBundleManifestError(
            "manifest.manifest_sha256 does not match canonical bundle content"
        )
    content[_HASH_FIELD] = declared_hash
    return content


def dumps_source_bundle_manifest(manifest: Mapping[str, Any]) -> str:
    """Serialize a validated manifest as compact canonical UTF-8 JSON."""

    canonical = validate_source_bundle_manifest(manifest)
    return _canonical_json_bytes(canonical).decode("utf-8")


def _reject_duplicate_object_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SourceBundleManifestError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> None:
    raise SourceBundleManifestError(
        f"non-finite JSON number {value!r} is not allowed"
    )


def loads_source_bundle_manifest(payload: str | bytes | bytearray) -> dict[str, Any]:
    """Parse and validate JSON while rejecting duplicate object keys."""

    if isinstance(payload, (bytes, bytearray)):
        try:
            payload = bytes(payload).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise SourceBundleManifestError(
                "manifest bytes must be valid UTF-8"
            ) from exc
    if not isinstance(payload, str):
        raise SourceBundleManifestError(
            "manifest payload must be text or UTF-8 bytes"
        )
    try:
        decoded = json.loads(
            payload,
            object_pairs_hook=_reject_duplicate_object_keys,
            parse_constant=_reject_nonfinite_constant,
        )
    except SourceBundleManifestError:
        raise
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise SourceBundleManifestError("manifest payload is not valid JSON") from exc
    return validate_source_bundle_manifest(decoded)
