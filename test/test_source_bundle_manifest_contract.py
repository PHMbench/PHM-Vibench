"""Contract tests for canonical P02 source-bundle manifests."""

from __future__ import annotations

import copy

import pytest

from src.explain_factory.source_bundle_manifest import (
    SOURCE_BUNDLE_MANIFEST_SCHEMA,
    SourceBundleManifestError,
    build_source_bundle_manifest,
    compute_source_bundle_manifest_sha256,
    dumps_source_bundle_manifest,
    loads_source_bundle_manifest,
    validate_source_bundle_manifest,
)


def _digest(character: str) -> str:
    return character * 64


def _inputs() -> dict[str, object]:
    return {
        "source": {
            "paper_id": "P07",
            "run_id": "P07-E7-seed-0",
            "model_id": "XOANOperatorPath",
            "model_architecture_id": "xoan-operator-path-v1",
            "model_architecture_sha256": _digest("a"),
            "checkpoint_id": "checkpoint",
            "checkpoint_sha256": _digest("1"),
            "model_seed": 0,
        },
        "provenance": {
            "code": {"id": "p07-code", "sha256": _digest("2")},
            "config": {"id": "p07-config", "sha256": _digest("3")},
            "environment": {"id": "LQ_signal-lock", "sha256": _digest("4")},
        },
        "references": {
            "dataset_id": "CWRU",
            "dataset_release_id": "cwru-release-v1",
            "dataset_release_sha256": _digest("5"),
            "task_transform_id": "cwru-four-class-v1",
            "task_transform_sha256": _digest("6"),
            "split_manifest_sha256": _digest("7"),
            "sample_cohort_manifest_sha256": _digest("8"),
        },
        "adapter": {
            "adapter_id": "p07-fixed-output-v1",
            "adapter_version": "1.0.0",
            "adapter_sha256": _digest("9"),
            "input_kind": "source_dense_attribution",
            "output_kind": "temporal_attribution",
            "capabilities": [
                "topk_support",
                "deletion",
                "dense_attribution",
                "paired_stability",
            ],
            "custom_source_fork": False,
            "source_specific_metric_branch": False,
        },
        "methods": [
            {
                "method_id": "method-b",
                "method_role": "real",
                "method_version": "1.0.0",
                "implementation_sha256": _digest("b"),
                "capabilities": [
                    "deletion",
                    "dense_attribution",
                    "paired_stability",
                    "topk_support",
                ],
                "output_artifact_ids": ["attribution-b"],
            },
            {
                "method_id": "method-a",
                "method_role": "real",
                "method_version": "1.0.0",
                "implementation_sha256": _digest("a"),
                "capabilities": [
                    "deletion",
                    "dense_attribution",
                    "paired_stability",
                    "topk_support",
                ],
                "output_artifact_ids": ["attribution-a"],
            },
        ],
        "artifacts": [
            {
                "artifact_id": "train-fit-proof",
                "role": "preprocessing_fit_proof",
                "locator": "artifacts/train_fit_proof.json",
                "sha256": _digest("d"),
                "media_type": "application/json",
            },
            {
                "artifact_id": "attribution-b",
                "role": "explanation",
                "locator": "artifacts/attribution_b.npy",
                "sha256": _digest("e"),
                "media_type": "application/x-npy",
            },
            {
                "artifact_id": "checkpoint",
                "role": "checkpoint",
                "locator": "artifacts/model.ckpt",
                "sha256": _digest("1"),
                "media_type": "application/octet-stream",
            },
            {
                "artifact_id": "attribution-a",
                "role": "explanation",
                "locator": "artifacts/attribution_a.npy",
                "sha256": _digest("f"),
                "media_type": "application/x-npy",
            },
            {
                "artifact_id": "target-vector",
                "role": "target_vector",
                "locator": "artifacts/target_vector.npy",
                "sha256": _digest("0"),
                "media_type": "application/x-npy",
            },
            {
                "artifact_id": "original-score-vector",
                "role": "original_score_vector",
                "locator": "artifacts/original_score_vector.npy",
                "sha256": _digest("c"),
                "media_type": "application/x-npy",
            },
        ],
        "preprocessing": {
            "preprocessing_id": "train-standardization-v1",
            "learned_parameters": {
                "mean": [0.1],
                "scale": [1.5],
                "dtype": "float64",
            },
            "fit_split": "train",
            "train_fit_proof": {
                "artifact_id": "train-fit-proof",
                "artifact_sha256": _digest("d"),
                "split_manifest_sha256": _digest("7"),
            },
        },
        "target": {
            "target_id": "original-predicted-class-v1",
            "semantics": "class fixed on the unperturbed model input",
            "sample_ids_sha256": _digest("8"),
            "values_artifact_id": "target-vector",
            "values_artifact_sha256": _digest("0"),
        },
        "score": {
            "score_id": "softmax-target-probability-v1",
            "semantics": "softmax probability of the fixed target class",
            "sample_ids_sha256": _digest("8"),
            "values_artifact_id": "original-score-vector",
            "values_artifact_sha256": _digest("c"),
        },
    }


def _build(inputs: dict[str, object] | None = None) -> dict[str, object]:
    values = inputs if inputs is not None else _inputs()
    return build_source_bundle_manifest(**values)  # type: ignore[arg-type]


def test_round_trip_is_canonical_compact_and_order_independent() -> None:
    inputs = _inputs()
    manifest = _build(inputs)
    payload = dumps_source_bundle_manifest(manifest)
    declared_hash = manifest["manifest_sha256"]

    reordered_inputs = _inputs()
    reordered_inputs["methods"] = list(reversed(reordered_inputs["methods"]))  # type: ignore[arg-type]
    reordered_inputs["artifacts"] = list(reversed(reordered_inputs["artifacts"]))  # type: ignore[arg-type]
    reordered = _build(reordered_inputs)

    assert manifest == reordered
    assert manifest["schema"] == SOURCE_BUNDLE_MANIFEST_SCHEMA
    assert loads_source_bundle_manifest(payload) == manifest
    assert loads_source_bundle_manifest(payload.encode("utf-8")) == manifest
    assert compute_source_bundle_manifest_sha256(manifest) == declared_hash
    manifest["manifest_sha256"] = _digest("0")
    assert compute_source_bundle_manifest_sha256(manifest) == declared_hash
    assert '": ' not in payload
    assert ", " not in payload
    assert "\n" not in payload


def test_derived_hashes_and_cross_references_are_bound() -> None:
    manifest = _build()

    assert [item["artifact_id"] for item in manifest["artifacts"]] == [  # type: ignore[index]
        "attribution-a",
        "attribution-b",
        "checkpoint",
        "original-score-vector",
        "target-vector",
        "train-fit-proof",
    ]
    assert [item["method_id"] for item in manifest["methods"]] == [  # type: ignore[index]
        "method-a",
        "method-b",
    ]
    assert set(manifest["derived"]) == {  # type: ignore[arg-type]
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
    assert all(len(value) == 64 for value in manifest["derived"].values())  # type: ignore[union-attr]


def test_shared_and_source_block_hashes_have_exact_frozen_boundaries() -> None:
    baseline = _build()
    baseline_shared = baseline["derived"]["shared_protocol_identity_sha256"]  # type: ignore[index]
    baseline_source_block = baseline["derived"]["source_block_identity_sha256"]  # type: ignore[index]

    invariant_mutations = {
        "source paper": lambda value: value["source"].__setitem__("paper_id", "P08"),
        "source run": lambda value: value["source"].__setitem__("run_id", "other-run"),
        "provenance": lambda value: value["provenance"]["code"].__setitem__(
            "sha256", _digest("0")
        ),
        "adapter": lambda value: value["adapter"].__setitem__(
            "adapter_sha256", _digest("0")
        ),
        "method": lambda value: value["methods"][0].__setitem__(
            "method_id", "renamed-method"
        ),
        "explanation artifact": lambda value: value["artifacts"][1].__setitem__(
            "sha256", _digest("0")
        ),
    }
    for label, mutate in invariant_mutations.items():
        inputs = _inputs()
        mutate(inputs)  # type: ignore[arg-type]
        variant = _build(inputs)
        assert variant["manifest_sha256"] != baseline["manifest_sha256"], label
        assert (
            variant["derived"]["shared_protocol_identity_sha256"] == baseline_shared  # type: ignore[index]
        ), label
        assert (
            variant["derived"]["source_block_identity_sha256"] == baseline_source_block  # type: ignore[index]
        ), label

    def change_split(value: dict[str, object]) -> None:
        value["references"]["split_manifest_sha256"] = _digest("0")  # type: ignore[index]
        value["preprocessing"]["train_fit_proof"]["split_manifest_sha256"] = _digest("0")  # type: ignore[index]

    def change_checkpoint(value: dict[str, object]) -> None:
        value["source"]["checkpoint_sha256"] = _digest("0")  # type: ignore[index]
        value["artifacts"][2]["sha256"] = _digest("0")  # type: ignore[index]

    def change_target_values(value: dict[str, object]) -> None:
        value["target"]["values_artifact_sha256"] = _digest("6")  # type: ignore[index]
        value["artifacts"][4]["sha256"] = _digest("6")  # type: ignore[index]

    def change_score_values(value: dict[str, object]) -> None:
        value["score"]["values_artifact_sha256"] = _digest("6")  # type: ignore[index]
        value["artifacts"][5]["sha256"] = _digest("6")  # type: ignore[index]

    shared_key_mutations = {
        "dataset": lambda value: value["references"].__setitem__(
            "dataset_id", "XJTU"
        ),
        "task": lambda value: value["references"].__setitem__(
            "task_transform_id", "xjtu-four-class-v1"
        ),
        "split": change_split,
        "cohort": lambda value: value["references"].__setitem__(
            "sample_cohort_manifest_sha256", _digest("0")
        ),
    }
    for label, mutate in shared_key_mutations.items():
        inputs = _inputs()
        mutate(inputs)  # type: ignore[arg-type]
        variant = _build(inputs)
        assert (
            variant["derived"]["shared_protocol_identity_sha256"] != baseline_shared  # type: ignore[index]
        ), label
        assert (
            variant["derived"]["source_block_identity_sha256"] != baseline_source_block  # type: ignore[index]
        ), label

    source_block_key_mutations = {
        "architecture": lambda value: value["source"].__setitem__(
            "model_architecture_sha256", _digest("0")
        ),
        "model": lambda value: value["source"].__setitem__(
            "model_id", "different-model"
        ),
        "checkpoint": change_checkpoint,
        "seed": lambda value: value["source"].__setitem__("model_seed", 1),
        "preprocessing": lambda value: value["preprocessing"][
            "learned_parameters"
        ].__setitem__("mean", [0.2]),
        "target values": change_target_values,
        "score values": change_score_values,
    }
    for label, mutate in source_block_key_mutations.items():
        inputs = _inputs()
        mutate(inputs)  # type: ignore[arg-type]
        variant = _build(inputs)
        assert (
            variant["derived"]["shared_protocol_identity_sha256"] == baseline_shared  # type: ignore[index]
        ), label
        assert (
            variant["derived"]["source_block_identity_sha256"] != baseline_source_block  # type: ignore[index]
        ), label


@pytest.mark.parametrize(
    ("container", "missing_key"),
    [
        ("root", "score"),
        ("source", "checkpoint_sha256"),
        ("provenance", "environment"),
        ("references", "sample_cohort_manifest_sha256"),
        ("adapter", "custom_source_fork"),
        ("method", "implementation_sha256"),
        ("artifact", "sha256"),
        ("preprocessing", "fit_split"),
        ("proof", "split_manifest_sha256"),
        ("target", "semantics"),
    ],
)
def test_missing_required_keys_fail_closed(container: str, missing_key: str) -> None:
    inputs = _inputs()
    if container == "root":
        manifest = _build(inputs)
        del manifest[missing_key]
        with pytest.raises(SourceBundleManifestError, match="missing required keys"):
            validate_source_bundle_manifest(manifest)
        return
    elif container in {"source", "provenance", "references", "adapter", "preprocessing", "target"}:
        del inputs[container][missing_key]  # type: ignore[index]
    elif container == "method":
        del inputs["methods"][0][missing_key]  # type: ignore[index]
    elif container == "artifact":
        del inputs["artifacts"][0][missing_key]  # type: ignore[index]
    else:
        del inputs["preprocessing"]["train_fit_proof"][missing_key]  # type: ignore[index]

    with pytest.raises(SourceBundleManifestError, match="missing required keys"):
        _build(inputs)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["source"].__setitem__("checkpoint_sha256", "not-a-sha"),
        lambda value: value["provenance"]["code"].__setitem__("sha256", "A" * 64),
        lambda value: value["references"].__setitem__("split_manifest_sha256", "7" * 63),
        lambda value: value["adapter"].__setitem__("adapter_sha256", "sha256:" + "9" * 64),
        lambda value: value["methods"][0].__setitem__("implementation_sha256", "x" * 64),
        lambda value: value["artifacts"][0].__setitem__("sha256", "d" * 65),
    ],
)
def test_non_sha256_bindings_are_rejected(mutate: object) -> None:
    inputs = _inputs()
    mutate(inputs)  # type: ignore[operator]
    with pytest.raises(SourceBundleManifestError, match="SHA-256"):
        _build(inputs)


@pytest.mark.parametrize("field", ["artifact_id", "locator"])
def test_duplicate_artifacts_are_rejected(field: str) -> None:
    inputs = _inputs()
    inputs["artifacts"][1][field] = inputs["artifacts"][0][field]  # type: ignore[index]

    with pytest.raises(SourceBundleManifestError, match="duplicate artifact"):
        _build(inputs)


def test_duplicate_methods_are_rejected() -> None:
    inputs = _inputs()
    inputs["methods"][1]["method_id"] = inputs["methods"][0]["method_id"]  # type: ignore[index]

    with pytest.raises(SourceBundleManifestError, match="duplicate method_id"):
        _build(inputs)


@pytest.mark.parametrize("role", ["random-permutation-control", "label-leaking-control"])
def test_control_roles_cannot_enter_the_real_source_method_set(role: str) -> None:
    inputs = _inputs()
    inputs["methods"][0]["method_role"] = role  # type: ignore[index]

    with pytest.raises(SourceBundleManifestError, match="must equal 'real'"):
        _build(inputs)


def test_one_artifact_cannot_be_claimed_by_two_methods() -> None:
    inputs = _inputs()
    inputs["methods"][1]["output_artifact_ids"] = ["attribution-b"]  # type: ignore[index]

    with pytest.raises(SourceBundleManifestError, match="claimed by duplicate methods"):
        _build(inputs)


@pytest.mark.parametrize(
    ("flag", "message"),
    [
        ("custom_source_fork", "custom source fork is forbidden"),
        ("source_specific_metric_branch", "source-specific metric branch is forbidden"),
    ],
)
def test_adapter_escape_hatches_are_rejected(flag: str, message: str) -> None:
    inputs = _inputs()
    inputs["adapter"][flag] = True  # type: ignore[index]

    with pytest.raises(SourceBundleManifestError, match=message):
        _build(inputs)


def test_method_capabilities_must_be_supported_by_the_adapter() -> None:
    inputs = _inputs()
    inputs["methods"][0]["capabilities"].append("unsupported-capability")  # type: ignore[index]

    with pytest.raises(SourceBundleManifestError, match="absent from its adapter"):
        _build(inputs)


@pytest.mark.parametrize("fit_split", ["validation", "test", "TRAIN"])
def test_preprocessing_must_be_fit_on_train(fit_split: str) -> None:
    inputs = _inputs()
    inputs["preprocessing"]["fit_split"] = fit_split  # type: ignore[index]

    with pytest.raises(SourceBundleManifestError, match="must equal 'train'"):
        _build(inputs)


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("target", "target_id", "ground-truth-class-v1"),
        ("target", "semantics", "ground-truth class from evaluation labels"),
        ("score", "score_id", "target-logit-v1"),
        ("score", "semantics", "raw logit of the fixed target class"),
    ],
)
def test_target_and_score_must_match_the_exact_approved_policy(
    section: str, field: str, value: str
) -> None:
    inputs = _inputs()
    inputs[section][field] = value  # type: ignore[index]

    with pytest.raises(SourceBundleManifestError, match="must equal"):
        _build(inputs)


def test_checkpoint_and_train_fit_proof_must_match_artifacts_and_split() -> None:
    inputs = _inputs()
    inputs["artifacts"][2]["sha256"] = _digest("c")  # type: ignore[index]
    with pytest.raises(SourceBundleManifestError, match="checkpoint_sha256"):
        _build(inputs)

    inputs = _inputs()
    inputs["preprocessing"]["train_fit_proof"]["artifact_sha256"] = _digest("c")  # type: ignore[index]
    with pytest.raises(SourceBundleManifestError, match="train-fit proof hash"):
        _build(inputs)

    inputs = _inputs()
    inputs["preprocessing"]["train_fit_proof"]["split_manifest_sha256"] = _digest("c")  # type: ignore[index]
    with pytest.raises(SourceBundleManifestError, match="bundle split_manifest_sha256"):
        _build(inputs)


@pytest.mark.parametrize(
    ("section", "artifact_index", "field", "value", "message"),
    [
        ("target", 4, "role", "labels", "role='target_vector'"),
        ("score", 5, "role", "logits", "role='original_score_vector'"),
        ("target", 4, "sha256", _digest("6"), "hash does not match"),
        ("score", 5, "sha256", _digest("6"), "hash does not match"),
    ],
)
def test_target_and_score_value_artifacts_are_cross_checked(
    section: str, artifact_index: int, field: str, value: str, message: str
) -> None:
    inputs = _inputs()
    inputs["artifacts"][artifact_index][field] = value  # type: ignore[index]

    with pytest.raises(SourceBundleManifestError, match=message):
        _build(inputs)


def test_target_and_score_value_vectors_share_ordered_sample_ids() -> None:
    inputs = _inputs()
    inputs["score"]["sample_ids_sha256"] = _digest("6")  # type: ignore[index]

    with pytest.raises(SourceBundleManifestError, match="identical ordered sample IDs"):
        _build(inputs)


def test_unknown_method_artifact_is_rejected() -> None:
    inputs = _inputs()
    inputs["methods"][0]["output_artifact_ids"] = ["missing"]  # type: ignore[index]

    with pytest.raises(SourceBundleManifestError, match="unknown artifact"):
        _build(inputs)


def test_method_output_must_be_an_explanation_artifact() -> None:
    inputs = _inputs()
    inputs["methods"][0]["output_artifact_ids"] = ["train-fit-proof"]  # type: ignore[index]

    with pytest.raises(SourceBundleManifestError, match="role='explanation'"):
        _build(inputs)


def test_every_explanation_artifact_must_be_claimed_by_one_method() -> None:
    inputs = _inputs()
    inputs["artifacts"].append(  # type: ignore[union-attr]
        {
            "artifact_id": "unclaimed-attribution",
            "role": "explanation",
            "locator": "artifacts/unclaimed.npy",
            "sha256": _digest("c"),
            "media_type": "application/x-npy",
        }
    )

    with pytest.raises(SourceBundleManifestError, match="claimed by exactly one method"):
        _build(inputs)


def test_manifest_and_derived_tampering_are_rejected() -> None:
    manifest = _build()
    tampered_derived = copy.deepcopy(manifest)
    tampered_derived["derived"]["artifact_list_sha256"] = _digest("0")
    with pytest.raises(SourceBundleManifestError, match="derived does not match"):
        validate_source_bundle_manifest(tampered_derived)

    tampered_content = copy.deepcopy(manifest)
    tampered_content["source"]["run_id"] = "changed-run"
    with pytest.raises(SourceBundleManifestError, match="manifest_sha256"):
        validate_source_bundle_manifest(tampered_content)


def test_duplicate_json_keys_are_rejected_before_validation() -> None:
    payload = (
        '{"schema":"p02.source-bundle-manifest.v1",'
        '"schema":"p02.source-bundle-manifest.v1"}'
    )

    with pytest.raises(SourceBundleManifestError, match="duplicate JSON object key"):
        loads_source_bundle_manifest(payload)
