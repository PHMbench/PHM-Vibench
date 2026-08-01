"""Contract tests for the outcome-free P02 E1 binding manifest."""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Mapping, Sequence

import pytest

from src.explain_factory.e1_binding_manifest import (
    APPROVED_DEFINITION_SHA256,
    E1_BINDING_MANIFEST_SCHEMA,
    EXPECTED_DATASETS,
    EXPECTED_SEEDS,
    EXPECTED_SOURCES,
    METRIC_REGISTRY_SHA256,
    REQUIRED_CAPABILITIES,
    E1BindingManifestError,
    build_e1_binding_manifest,
    compute_e1_binding_manifest_sha256,
    dumps_e1_binding_manifest,
    loads_e1_binding_manifest,
    validate_e1_binding_manifest,
)
from src.explain_factory.source_bundle_manifest import (
    SourceBundleManifestError,
    build_source_bundle_manifest,
)


PRIMARY_METHODS = ("method-a", "method-b")
THREE_METHODS = (*PRIMARY_METHODS, "method-c")


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _source_bundle(
    source: str,
    dataset: str,
    seed: int,
    *,
    method_ids: Sequence[str] = THREE_METHODS,
    capabilities: Sequence[str] = REQUIRED_CAPABILITIES,
    method_capability_overrides: Mapping[str, Sequence[str]] | None = None,
    method_identity_overrides: Mapping[str, Mapping[str, str]] | None = None,
    method_role_overrides: Mapping[str, str] | None = None,
    source_family_overrides: Mapping[str, str] | None = None,
    reference_overrides: Mapping[str, str] | None = None,
    sample_ids_sha256_override: str | None = None,
    cell_variant: str = "",
    target_semantics: str = "class fixed on the unperturbed model input",
    score_semantics: str = "softmax probability of the fixed target class",
) -> dict[str, object]:
    dataset_slug = dataset.lower()
    references = {
        "dataset_id": dataset,
        "dataset_release_id": f"{dataset_slug}-release-v1",
        "dataset_release_sha256": _sha(f"{dataset}:release"),
        "task_transform_id": f"{dataset_slug}-closed-set-task-v1",
        "task_transform_sha256": _sha(f"{dataset}:task"),
        "split_manifest_sha256": _sha(f"{dataset}:split"),
        "sample_cohort_manifest_sha256": _sha(f"{dataset}:cohort"),
    }
    if reference_overrides:
        references.update(reference_overrides)

    cell_identity = f"{source}:{dataset}:{seed}:{cell_variant}"
    checkpoint_sha256 = _sha(f"{cell_identity}:checkpoint")
    proof_sha256 = _sha(f"{cell_identity}:train-fit-proof")
    target_values_sha256 = _sha(f"{cell_identity}:target-values")
    score_values_sha256 = _sha(f"{cell_identity}:score-values")
    sample_ids_sha256 = sample_ids_sha256_override or _sha(
        f"{dataset}:ordered-sample-ids"
    )
    methods = []
    explanation_artifacts = []
    for method_id in method_ids:
        identity_override = (method_identity_overrides or {}).get(method_id, {})
        artifact_id = f"explanation-{method_id}"
        methods.append(
            {
                "method_id": method_id,
                "method_role": (method_role_overrides or {}).get(
                    method_id, "real"
                ),
                "method_version": identity_override.get(
                    "method_version", "1.0.0"
                ),
                "implementation_sha256": identity_override.get(
                    "implementation_sha256", _sha(f"implementation:{method_id}")
                ),
                "capabilities": list(
                    (method_capability_overrides or {}).get(
                        method_id, REQUIRED_CAPABILITIES
                    )
                ),
                "output_artifact_ids": [artifact_id],
            }
        )
        explanation_artifacts.append(
            {
                "artifact_id": artifact_id,
                "role": "explanation",
                "locator": f"artifacts/{method_id}.npy",
                "sha256": _sha(
                    f"{cell_identity}:explanation:{method_id}"
                ),
                "media_type": "application/x-npy",
            }
        )

    source_identity = {
            "paper_id": source,
            "run_id": f"{source}-E1-{dataset}-seed-{seed}{cell_variant}",
            "model_architecture_id": f"{source}-architecture-v1",
            "model_architecture_sha256": _sha(f"{source}:architecture"),
            "model_id": f"{source}-model-family",
            "checkpoint_id": "checkpoint",
            "checkpoint_sha256": checkpoint_sha256,
            "model_seed": seed,
    }
    if source_family_overrides:
        source_identity.update(source_family_overrides)

    return build_source_bundle_manifest(
        source=source_identity,
        provenance={
            "code": {
                "id": f"{source}-code",
                "sha256": _sha(f"{source}:code"),
            },
            "config": {
                "id": f"{source}-{dataset}-seed-{seed}-config",
                "sha256": _sha(f"{cell_identity}:config"),
            },
            "environment": {
                "id": "LQ_signal-lock",
                "sha256": _sha("LQ_signal:environment"),
            },
        },
        references=references,
        adapter={
            "adapter_id": f"{source.lower()}-adapter-v1",
            "adapter_version": "1.0.0",
            "adapter_sha256": _sha(f"{source}:adapter"),
            "input_kind": "source_dense_attribution",
            "output_kind": "temporal_attribution",
            "capabilities": list(capabilities),
            "custom_source_fork": False,
            "source_specific_metric_branch": False,
        },
        methods=methods,
        artifacts=[
            {
                "artifact_id": "checkpoint",
                "role": "checkpoint",
                "locator": "artifacts/model.ckpt",
                "sha256": checkpoint_sha256,
                "media_type": "application/octet-stream",
            },
            {
                "artifact_id": "train-fit-proof",
                "role": "preprocessing_fit_proof",
                "locator": "artifacts/train_fit_proof.json",
                "sha256": proof_sha256,
                "media_type": "application/json",
            },
            {
                "artifact_id": "target-vector",
                "role": "target_vector",
                "locator": "artifacts/target_vector.npy",
                "sha256": target_values_sha256,
                "media_type": "application/x-npy",
            },
            {
                "artifact_id": "original-score-vector",
                "role": "original_score_vector",
                "locator": "artifacts/original_score_vector.npy",
                "sha256": score_values_sha256,
                "media_type": "application/x-npy",
            },
            *explanation_artifacts,
        ],
        preprocessing={
            "preprocessing_id": f"{source.lower()}-train-standardization-v1",
            "learned_parameters": {
                "mean": [seed + (0.1 if source == "P07" else 0.2)],
                "scale": [1.0 + seed / 100.0],
                "dtype": "float64",
            },
            "fit_split": "train",
            "train_fit_proof": {
                "artifact_id": "train-fit-proof",
                "artifact_sha256": proof_sha256,
                "split_manifest_sha256": references["split_manifest_sha256"],
            },
        },
        target={
            "target_id": "original-predicted-class-v1",
            "semantics": target_semantics,
            "sample_ids_sha256": sample_ids_sha256,
            "values_artifact_id": "target-vector",
            "values_artifact_sha256": target_values_sha256,
        },
        score={
            "score_id": "softmax-target-probability-v1",
            "semantics": score_semantics,
            "sample_ids_sha256": sample_ids_sha256,
            "values_artifact_id": "original-score-vector",
            "values_artifact_sha256": score_values_sha256,
        },
    )


def _bundles(
    *, method_ids: Sequence[str] = THREE_METHODS
) -> list[dict[str, object]]:
    return [
        _source_bundle(source, dataset, seed, method_ids=method_ids)
        for source in EXPECTED_SOURCES
        for dataset in EXPECTED_DATASETS
        for seed in EXPECTED_SEEDS
    ]


def _replace_cell(
    bundles: list[dict[str, object]],
    replacement: dict[str, object],
    *,
    source: str = "P07",
    dataset: str = "CWRU",
    seed: int = 0,
) -> None:
    for index, bundle in enumerate(bundles):
        if (
            bundle["source"]["paper_id"],  # type: ignore[index]
            bundle["references"]["dataset_id"],  # type: ignore[index]
            bundle["source"]["model_seed"],  # type: ignore[index]
        ) == (source, dataset, seed):
            bundles[index] = replacement
            return
    raise AssertionError("test helper could not find the requested cell")


def _build(
    bundles: Sequence[Mapping[str, object]] | None = None,
    *,
    expected_sources: Sequence[str] = EXPECTED_SOURCES,
    expected_datasets: Sequence[str] = EXPECTED_DATASETS,
    expected_seeds: Sequence[int] = EXPECTED_SEEDS,
    primary_method_ids: Sequence[str] = PRIMARY_METHODS,
    expected_common_real_method_ids: Sequence[str] = THREE_METHODS,
    approved_definition_sha256: str | None = None,
    metric_registry_sha256: str | None = None,
) -> dict[str, object]:
    return build_e1_binding_manifest(
        approved_definition_sha256=(
            approved_definition_sha256 or APPROVED_DEFINITION_SHA256
        ),
        metric_registry_sha256=(metric_registry_sha256 or METRIC_REGISTRY_SHA256),
        expected_sources=expected_sources,
        expected_datasets=expected_datasets,
        expected_seeds=expected_seeds,
        primary_method_ids=primary_method_ids,
        expected_common_real_method_ids=expected_common_real_method_ids,
        source_bundles=bundles if bundles is not None else _bundles(),
    )


def test_round_trip_is_canonical_compact_and_input_order_independent() -> None:
    baseline = _build()
    reordered = _build(
        list(reversed(_bundles())),
        expected_sources=list(reversed(EXPECTED_SOURCES)),
        expected_datasets=list(reversed(EXPECTED_DATASETS)),
        expected_seeds=list(reversed(EXPECTED_SEEDS)),
        primary_method_ids=list(reversed(PRIMARY_METHODS)),
        expected_common_real_method_ids=list(reversed(THREE_METHODS)),
    )
    payload = dumps_e1_binding_manifest(reordered)

    assert reordered == baseline
    assert baseline["schema"] == E1_BINDING_MANIFEST_SCHEMA
    assert baseline["approved_definition_sha256"] == APPROVED_DEFINITION_SHA256
    assert baseline["metric_registry_sha256"] == METRIC_REGISTRY_SHA256
    assert baseline["primary_method_ids"] == list(PRIMARY_METHODS)
    assert baseline["expected_common_real_method_ids"] == list(THREE_METHODS)
    assert loads_e1_binding_manifest(payload) == baseline
    assert loads_e1_binding_manifest(payload.encode("utf-8")) == baseline
    assert compute_e1_binding_manifest_sha256(baseline) == baseline["manifest_sha256"]
    assert '": ' not in payload
    assert ", " not in payload
    assert "\n" not in payload


def test_complete_crossing_and_derived_preflight_are_explicit() -> None:
    manifest = _build()
    derived = manifest["derived"]
    crossing = derived["crossing_audit"]  # type: ignore[index]
    observed = crossing["observed_cells"]

    assert crossing["axis_sizes"] == {"sources": 2, "datasets": 2, "seeds": 10}
    assert crossing["expected_cell_count"] == 40
    assert crossing["observed_cell_count"] == 40
    assert len(crossing["expected_cells"]) == 40
    assert len(observed) == 40
    assert crossing["missing_cells"] == []
    assert crossing["extra_cells"] == []
    assert crossing["duplicate_cells"] == []
    assert crossing["complete"] is True
    assert observed[0]["source_paper_id"] == "P07"
    assert observed[0]["dataset_id"] == "CWRU"
    assert observed[0]["model_seed"] == 0
    assert observed[-1]["source_paper_id"] == "P08"
    assert observed[-1]["dataset_id"] == "XJTU"
    assert observed[-1]["model_seed"] == 9
    assert derived["ordered_bundle_manifest_sha256s"] == [
        cell["source_bundle_manifest_sha256"] for cell in observed
    ]
    assert len(set(derived["ordered_bundle_manifest_sha256s"])) == 40
    assert derived["common_method_ids"] == list(THREE_METHODS)
    assert set(derived["common_method_identity_map"]) == set(THREE_METHODS)
    assert all(
        len(identity["implementation_sha256"]) == 64
        for identity in derived["common_method_identity_map"].values()
    )
    assert all(
        identity["method_role"] == "real"
        for identity in derived["common_method_identity_map"].values()
    )
    assert set(derived["source_family_identity_map"]) == set(EXPECTED_SOURCES)
    assert derived["d_at_least_three_methods_estimable"] is True
    assert len(derived["comparison_protocol_sha256"]) == 64


def test_d_preflight_is_false_when_only_two_methods_are_common() -> None:
    manifest = _build(
        _bundles(method_ids=PRIMARY_METHODS),
        expected_common_real_method_ids=PRIMARY_METHODS,
    )

    assert manifest["derived"]["common_method_ids"] == list(PRIMARY_METHODS)  # type: ignore[index]
    assert (
        manifest["derived"]["d_at_least_three_methods_estimable"] is False  # type: ignore[index]
    )


def test_source_model_checkpoint_and_preprocessing_may_vary_by_stratum() -> None:
    manifest = _build()
    bundles = manifest["source_bundles"]
    p07 = bundles[0]
    p08 = bundles[20]

    assert p07["source"]["model_id"] != p08["source"]["model_id"]
    assert (
        p07["source"]["model_architecture_sha256"]
        != p08["source"]["model_architecture_sha256"]
    )
    assert p07["source"]["checkpoint_sha256"] != p08["source"]["checkpoint_sha256"]
    assert (
        p07["derived"]["preprocessing_identity_sha256"]
        != p08["derived"]["preprocessing_identity_sha256"]
    )
    assert len(manifest["derived"]["dataset_shared_identities"]) == 2  # type: ignore[index]
    families = manifest["derived"]["source_family_identity_map"]  # type: ignore[index]
    assert families["P07"]["model_id"] == p07["source"]["model_id"]
    assert families["P08"]["model_id"] == p08["source"]["model_id"]


@pytest.mark.parametrize(
    ("source_family_overrides", "field"),
    [
        ({"model_id": "drifted-model-family"}, "model_id"),
        (
            {"model_architecture_id": "drifted-architecture-v2"},
            "model_architecture_id",
        ),
        (
            {"model_architecture_sha256": _sha("drifted-architecture")},
            "model_architecture_sha256",
        ),
    ],
)
def test_source_family_identity_must_be_constant_across_twenty_cells(
    source_family_overrides: Mapping[str, str], field: str
) -> None:
    bundles = _bundles()
    _replace_cell(
        bundles,
        _source_bundle(
            "P08",
            "XJTU",
            9,
            source_family_overrides=source_family_overrides,
        ),
        source="P08",
        dataset="XJTU",
        seed=9,
    )

    with pytest.raises(E1BindingManifestError, match=field):
        _build(bundles)


@pytest.mark.parametrize("missing_capability", REQUIRED_CAPABILITIES)
def test_every_cell_requires_all_metric_capabilities(missing_capability: str) -> None:
    bundles = _bundles()
    capabilities = [
        item for item in REQUIRED_CAPABILITIES if item != missing_capability
    ]
    method_overrides = {method_id: capabilities for method_id in THREE_METHODS}
    _replace_cell(
        bundles,
        _source_bundle(
            "P07",
            "CWRU",
            0,
            capabilities=capabilities,
            method_capability_overrides=method_overrides,
        ),
    )

    with pytest.raises(E1BindingManifestError, match="lacks required capabilities"):
        _build(bundles)


@pytest.mark.parametrize("missing_capability", REQUIRED_CAPABILITIES)
def test_every_primary_method_requires_all_metric_capabilities(
    missing_capability: str,
) -> None:
    bundles = _bundles()
    incomplete = [
        item for item in REQUIRED_CAPABILITIES if item != missing_capability
    ]
    _replace_cell(
        bundles,
        _source_bundle(
            "P07",
            "CWRU",
            0,
            method_capability_overrides={"method-b": incomplete},
        ),
    )

    with pytest.raises(
        E1BindingManifestError, match="expected common real method.*lacks"
    ):
        _build(bundles)


def test_expected_common_method_cannot_silently_lose_metric_eligibility() -> None:
    bundles = _bundles()
    _replace_cell(
        bundles,
        _source_bundle(
            "P07",
            "CWRU",
            0,
            method_capability_overrides={
                "method-c": tuple(REQUIRED_CAPABILITIES[:-1])
            },
        ),
    )
    with pytest.raises(
        E1BindingManifestError, match="expected common real method.*lacks"
    ):
        _build(bundles)


@pytest.mark.parametrize(
    ("method_id", "identity_override"),
    [
        ("method-b", {"method_version": "2.0.0"}),
        (
            "method-b",
            {"implementation_sha256": _sha("different-method-b-implementation")},
        ),
        ("method-c", {"method_version": "2.0.0"}),
        (
            "method-c",
            {"implementation_sha256": _sha("different-method-c-implementation")},
        ),
    ],
)
def test_common_method_identity_must_match_in_all_forty_cells(
    method_id: str,
    identity_override: Mapping[str, str],
) -> None:
    bundles = _bundles()
    _replace_cell(
        bundles,
        _source_bundle(
            "P08",
            "XJTU",
            9,
            method_identity_overrides={method_id: identity_override},
        ),
        source="P08",
        dataset="XJTU",
        seed=9,
    )

    with pytest.raises(E1BindingManifestError, match="method identity mismatch"):
        _build(bundles)


def test_every_cell_requires_every_primary_method() -> None:
    bundles = _bundles()
    _replace_cell(
        bundles,
        _source_bundle("P07", "CWRU", 0, method_ids=("method-a", "method-c")),
    )

    with pytest.raises(E1BindingManifestError, match="missing expected common"):
        _build(bundles)


def test_undeclared_extra_real_method_is_rejected() -> None:
    bundles = _bundles()
    _replace_cell(
        bundles,
        _source_bundle(
            "P07", "CWRU", 0, method_ids=(*THREE_METHODS, "method-d")
        ),
    )

    with pytest.raises(E1BindingManifestError, match="extra undeclared real methods"):
        _build(bundles)


@pytest.mark.parametrize(
    "method_role", ["random_permutation_control", "label_leaking_control"]
)
def test_non_real_method_roles_are_rejected_by_source_schema(
    method_role: str,
) -> None:
    with pytest.raises(SourceBundleManifestError, match="method_role must equal 'real'"):
        _source_bundle(
            "P07",
            "CWRU",
            0,
            method_role_overrides={"method-c": method_role},
        )


@pytest.mark.parametrize(
    ("reference_overrides", "field"),
    [
        ({"dataset_release_id": "other-release"}, "dataset_release_id"),
        (
            {"dataset_release_sha256": _sha("other-release")},
            "dataset_release_sha256",
        ),
        ({"task_transform_id": "other-task"}, "task_transform_id"),
        (
            {"task_transform_sha256": _sha("other-task")},
            "task_transform_sha256",
        ),
        (
            {"split_manifest_sha256": _sha("other-split")},
            "split_manifest_sha256",
        ),
        (
            {"sample_cohort_manifest_sha256": _sha("other-cohort")},
            "sample_cohort_manifest_sha256",
        ),
    ],
)
def test_dataset_identity_must_be_shared_across_sources_and_seeds(
    reference_overrides: Mapping[str, str],
    field: str,
) -> None:
    bundles = _bundles()
    replacement = _source_bundle(
        "P08",
        "CWRU",
        0,
        reference_overrides=reference_overrides,
    )
    _replace_cell(bundles, replacement, source="P08")

    with pytest.raises(E1BindingManifestError, match=field):
        _build(bundles)


def test_ordered_sample_ids_must_be_shared_across_sources_and_seeds() -> None:
    bundles = _bundles()
    replacement = _source_bundle(
        "P08",
        "CWRU",
        0,
        sample_ids_sha256_override=_sha("different-ordered-sample-ids"),
    )
    _replace_cell(bundles, replacement, source="P08")

    with pytest.raises(E1BindingManifestError, match="sample_ids_sha256"):
        _build(bundles)


def test_ep_v1_policies_are_shared_while_bound_value_vectors_may_vary() -> None:
    manifest = _build()
    shared = {
        identity["dataset_id"]: identity
        for identity in manifest["derived"]["dataset_shared_identities"]  # type: ignore[index]
    }

    for bundle in manifest["source_bundles"]:  # type: ignore[union-attr]
        dataset = bundle["references"]["dataset_id"]
        assert (
            bundle["target"]["sample_ids_sha256"]
            == shared[dataset]["sample_ids_sha256"]
        )
        assert (
            bundle["score"]["sample_ids_sha256"]
            == shared[dataset]["sample_ids_sha256"]
        )
        assert (
            bundle["derived"]["target_policy_sha256"]
            == shared[dataset]["target_policy_sha256"]
        )
        assert (
            bundle["derived"]["score_policy_sha256"]
            == shared[dataset]["score_policy_sha256"]
        )
        assert (
            bundle["derived"]["shared_protocol_identity_sha256"]
            == shared[dataset]["shared_protocol_identity_sha256"]
        )

    cwru_bundles = [
        bundle
        for bundle in manifest["source_bundles"]  # type: ignore[union-attr]
        if bundle["references"]["dataset_id"] == "CWRU"
    ]
    assert len(
        {bundle["derived"]["target_identity_sha256"] for bundle in cwru_bundles}
    ) == 20
    assert len(
        {bundle["derived"]["score_identity_sha256"] for bundle in cwru_bundles}
    ) == 20


def test_missing_duplicate_and_unexpected_crossing_cells_fail_closed() -> None:
    bundles = _bundles()
    with pytest.raises(E1BindingManifestError, match="missing 1 cells"):
        _build(bundles[:-1])

    duplicate = _bundles()
    duplicate.append(copy.deepcopy(duplicate[0]))
    with pytest.raises(E1BindingManifestError, match="duplicate source-bundle cell"):
        _build(duplicate)

    unexpected = _bundles()
    _replace_cell(unexpected, _source_bundle("P09", "CWRU", 0))
    with pytest.raises(E1BindingManifestError, match="not an expected source"):
        _build(unexpected)


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    [
        ("expected_sources", ("P07",), "must contain exactly"),
        ("expected_datasets", ("CWRU", "CWRU"), "must not contain duplicates"),
        ("expected_seeds", tuple(range(9)), "integers 0 through 9"),
        (
            "expected_seeds",
            (*EXPECTED_SEEDS, 9),
            "must not contain duplicates",
        ),
        ("primary_method_ids", ("method-a",), "at least 2"),
        (
            "primary_method_ids",
            THREE_METHODS,
            "exactly 2 unique method IDs",
        ),
        (
            "primary_method_ids",
            ("method-a", "method-a"),
            "must not contain duplicates",
        ),
        (
            "expected_common_real_method_ids",
            ("method-a",),
            "at least 2",
        ),
        (
            "expected_common_real_method_ids",
            ("method-a", "method-a"),
            "must not contain duplicates",
        ),
        (
            "expected_common_real_method_ids",
            ("method-a", "method-c"),
            "primary_method_ids must be a subset",
        ),
    ],
)
def test_declared_axes_and_primary_pair_are_exact(
    keyword: str, value: Sequence[object], message: str
) -> None:
    arguments: dict[str, object] = {keyword: value}
    with pytest.raises(E1BindingManifestError, match=message):
        _build(**arguments)  # type: ignore[arg-type]


def test_nested_source_bundle_must_already_be_valid_and_untampered() -> None:
    bundles = _bundles()
    bundles[0]["source"]["run_id"] = "tampered-run"  # type: ignore[index]

    with pytest.raises(E1BindingManifestError, match="not a valid.*manifest_sha256"):
        _build(bundles)


def test_protocol_profile_hashes_are_fixed_and_bundle_hashes_are_bound() -> None:
    baseline = _build()
    with pytest.raises(E1BindingManifestError, match="approved EP-V1 digest"):
        _build(approved_definition_sha256=_sha("different-definition"))
    with pytest.raises(E1BindingManifestError, match="approved EP-V1 digest"):
        _build(metric_registry_sha256=_sha("different-registry"))

    bundles = _bundles()
    replacement = _source_bundle("P07", "CWRU", 0, cell_variant="-rerun")
    _replace_cell(bundles, replacement)
    changed_bundle = _build(bundles)

    assert (
        baseline["derived"]["comparison_protocol_sha256"]  # type: ignore[index]
        != changed_bundle["derived"]["comparison_protocol_sha256"]  # type: ignore[index]
    )


def test_manifest_and_derived_tampering_are_rejected() -> None:
    manifest = _build()
    tampered_derived = copy.deepcopy(manifest)
    tampered_derived["derived"]["common_method_ids"] = ["method-a", "method-b"]  # type: ignore[index]
    with pytest.raises(E1BindingManifestError, match="derived does not match"):
        validate_e1_binding_manifest(tampered_derived)

    tampered_content = copy.deepcopy(manifest)
    tampered_content["approved_definition_sha256"] = _sha("tampered-definition")
    with pytest.raises(E1BindingManifestError, match="approved EP-V1 digest"):
        validate_e1_binding_manifest(tampered_content)

    wrong_hash = copy.deepcopy(manifest)
    wrong_hash["manifest_sha256"] = "0" * 64
    with pytest.raises(E1BindingManifestError, match="manifest_sha256"):
        validate_e1_binding_manifest(wrong_hash)


def test_missing_unknown_and_non_sha_bindings_fail_closed() -> None:
    manifest = _build()
    del manifest["metric_registry_sha256"]
    with pytest.raises(E1BindingManifestError, match="missing required keys"):
        validate_e1_binding_manifest(manifest)

    manifest = _build()
    manifest["unexpected"] = "value"
    with pytest.raises(E1BindingManifestError, match="unknown keys"):
        validate_e1_binding_manifest(manifest)

    with pytest.raises(E1BindingManifestError, match="SHA-256"):
        _build(approved_definition_sha256="not-a-sha")


def test_duplicate_json_keys_are_rejected_before_validation() -> None:
    payload = (
        '{"schema":"p02.e1-binding-manifest.v1",'
        '"schema":"p02.e1-binding-manifest.v1"}'
    )

    with pytest.raises(E1BindingManifestError, match="duplicate JSON object key"):
        loads_e1_binding_manifest(payload)
