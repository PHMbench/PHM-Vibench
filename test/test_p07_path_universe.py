from __future__ import annotations

import ast
import copy
import hashlib
import inspect
from collections import Counter

import pytest
import torch

import src.utils.p07_protocol.path_universe as path_universe_module
from src.utils.p07_protocol.path_universe import (
    COMPOSITION_SPLIT_COUNTS,
    CORRUPTION_SEED_DOMAIN,
    CORRUPTION_SEED_BY_OPTIMIZATION_SEED,
    EQUIVALENCE_REWRITE_WHITELIST,
    GENERATOR_SEED_NAMESPACES,
    OPERATORS,
    OPTIMIZATION_SEEDS,
    PathRecord,
    SAMPLES_PER_GENERATOR_SEED,
    SINGLETON_POSITION_MINIMUMS,
    build_composition_split_manifest,
    build_path_universe_manifest,
    build_seed_namespace_manifest,
    canonical_json_bytes,
    canonical_json_sha256,
    canonicalize_path,
    derive_sample_seed,
    enumerate_equivalence_classes,
    enumerate_path_records,
    load_composition_split_manifest,
    load_path_universe_manifest,
    make_sample_id,
    oracle_apply_operator,
    oracle_execute_path,
    path_universe_manifest_json,
    primary_classes,
    registered_sample_ids,
    strict_canonical_json_loads,
    validate_composition_split_manifest,
    validate_path_universe_manifest,
    validate_raw_path,
    validate_sample_id,
    validate_seed_namespace_manifest,
)


EXPECTED_UNIVERSE_SHA256 = "60b4907005403eaad082b50c169170f5433eb3bc9ec33fea999d6450af9ad338"
EXPECTED_SEED_SHA256 = "0ecdb2747616732e8246298b51684b219dd277c9d15f517168c750e55ec765d1"
EXPECTED_SPLIT_SHA256 = "ebe91496c8a50d39f9ae072337dec2dec1ae435b328315b8bd08980ed0f569ce"


def test_operator_and_equivalence_whitelists_are_exact() -> None:
    assert OPERATORS == ("I", "D1", "ABS", "SQUARE", "MA3", "HT")
    assert [item["rewrite_id"] for item in EQUIVALENCE_REWRITE_WHITELIST] == [
        "identity_elision",
        "abs_idempotence",
        "abs_of_square",
        "square_of_abs",
        "abs_of_hilbert_envelope",
        "abs_after_ma3_abs",
        "abs_after_ma3_square",
        "abs_after_ma3_hilbert_envelope",
    ]
    assert [item["expression"] for item in EQUIVALENCE_REWRITE_WHITELIST] == [
        "I(a)->a",
        "ABS(ABS(a))->ABS(a)",
        "ABS(SQUARE(a))->SQUARE(a)",
        "SQUARE(ABS(a))->SQUARE(a)",
        "ABS(HT(a))->HT(a)",
        "ABS(MA3(ABS(a)))->MA3(ABS(a))",
        "ABS(MA3(SQUARE(a)))->MA3(SQUARE(a))",
        "ABS(MA3(HT(a)))->MA3(HT(a))",
    ]


@pytest.mark.parametrize(
    ("raw_path", "expected"),
    [
        (("I", "I", "I"), ()),
        (("I", "D1", "I"), ("D1",)),
        (("ABS", "ABS", "I"), ("ABS",)),
        (("SQUARE", "ABS", "I"), ("SQUARE",)),
        (("ABS", "SQUARE", "I"), ("SQUARE",)),
        (("HT", "ABS", "I"), ("HT",)),
        (("ABS", "SQUARE", "ABS"), ("SQUARE",)),
        (("ABS", "MA3", "ABS"), ("ABS", "MA3")),
        (("SQUARE", "MA3", "ABS"), ("SQUARE", "MA3")),
        (("HT", "MA3", "ABS"), ("HT", "MA3")),
        (("D1", "MA3", "ABS"), ("D1", "MA3", "ABS")),
    ],
)
def test_canonicalization_applies_only_frozen_rewrites(
    raw_path: tuple[str, str, str], expected: tuple[str, ...]
) -> None:
    assert canonicalize_path(raw_path) == expected


def test_universe_is_exactly_216_paths_and_116_classes() -> None:
    records = enumerate_path_records()
    classes = enumerate_equivalence_classes()

    assert len(records) == 6**3 == 216
    assert len({record.raw_path for record in records}) == 216
    assert len({record.raw_path_id for record in records}) == 216
    assert len(classes) == 116
    assert sum(item.multiplicity for item in classes) == 216
    assert Counter(item.multiplicity for item in classes) == {
        1: 90,
        3: 6,
        4: 6,
        5: 8,
        6: 3,
        7: 2,
        12: 1,
    }
    assert sum(item.ambiguous for item in classes) == 26
    assert sum(not item.ambiguous for item in classes) == 90
    assert [item.canonical_expression for item in classes].count("x") == 1


def test_path_and_class_ids_are_full_sha256_and_stable() -> None:
    by_path = {record.raw_path: record for record in enumerate_path_records()}
    identity = by_path[("I", "I", "I")]
    d1 = by_path[("D1", "I", "I")]

    assert identity.raw_path_id == (
        "P07-RAW-ad216af41bc3d4e5064e54fbdac70339ceedd7f6083951bef23bae733a81a366"
    )
    assert identity.class_id == (
        "P07-SEM-a09837670ac99299b785f758c37c166ad6d817e5f34b9191b7ea9a5c25bf9380"
    )
    assert d1.class_id == (
        "P07-SEM-de0b625097cc2fb7938335e267c7ad642478ff1fe8bb3fab75e633ff57e6123d"
    )
    assert identity.raw_path_id.removeprefix("P07-RAW-") == identity.raw_path_sha256
    assert identity.class_id.removeprefix("P07-SEM-") == identity.class_sha256


def test_path_universe_manifest_is_canonical_hash_bound_and_frozen() -> None:
    manifest = build_path_universe_manifest()

    assert manifest["manifest_sha256"] == EXPECTED_UNIVERSE_SHA256
    assert canonical_json_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    ) == EXPECTED_UNIVERSE_SHA256
    assert validate_path_universe_manifest(manifest) == manifest
    serialized = path_universe_manifest_json()
    assert serialized.encode("utf-8") == canonical_json_bytes(manifest)
    assert load_path_universe_manifest(serialized) == manifest


def test_path_universe_manifest_validation_fails_closed_on_tampering() -> None:
    manifest = build_path_universe_manifest()
    stale_hash = copy.deepcopy(manifest)
    stale_hash["class_count"] = 120
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_path_universe_manifest(stale_hash)

    rehashed = copy.deepcopy(stale_hash)
    rehashed["manifest_sha256"] = canonical_json_sha256(
        {key: value for key, value in rehashed.items() if key != "manifest_sha256"}
    )
    with pytest.raises(ValueError, match="frozen universe"):
        validate_path_universe_manifest(rehashed)

    extra_key = copy.deepcopy(manifest)
    extra_key["unregistered"] = True
    with pytest.raises(ValueError, match="invalid key set"):
        validate_path_universe_manifest(extra_key)


def test_strict_json_rejects_noncanonical_duplicate_and_nonfinite_values() -> None:
    canonical = canonical_json_bytes({"a": 1, "b": [True, None]})
    assert strict_canonical_json_loads(canonical) == {"a": 1, "b": [True, None]}
    with pytest.raises(ValueError, match="not in canonical"):
        strict_canonical_json_loads('{"b":[], "a":1}')
    with pytest.raises(ValueError, match="Duplicate"):
        strict_canonical_json_loads('{"a":1,"a":2}')
    with pytest.raises(ValueError, match="Non-finite"):
        strict_canonical_json_loads('{"a":NaN}')
    with pytest.raises(ValueError, match="Non-finite"):
        strict_canonical_json_loads('{"a":1e999}')
    with pytest.raises(TypeError, match="Unsupported canonical JSON"):
        canonical_json_bytes(("tuple",))  # type: ignore[arg-type]


def test_primary_selection_and_composition_split_are_exact_and_disjoint() -> None:
    primary = primary_classes()
    manifest = build_composition_split_manifest()
    splits = manifest["composition_splits"]

    assert len(primary) == 72
    assert sum(item.ambiguous for item in primary) == 26
    assert sum(not item.ambiguous for item in primary) == 46
    assert all(item.canonical_expression != "x" for item in primary)
    assert manifest["manifest_sha256"] == EXPECTED_SPLIT_SHA256

    expected = {
        "train": (36, 13, 23),
        "validation": (18, 7, 11),
        "test": (18, 6, 12),
    }
    class_sets: dict[str, set[str]] = {}
    assert isinstance(splits, dict)
    for split, (count, ambiguous, singleton) in expected.items():
        item = splits[split]
        assert isinstance(item, dict)
        assert (item["count"], item["ambiguous_count"], item["singleton_count"]) == (
            count,
            ambiguous,
            singleton,
        )
        class_sets[split] = set(item["class_ids"])
    assert class_sets["train"].isdisjoint(class_sets["validation"])
    assert class_sets["train"].isdisjoint(class_sets["test"])
    assert class_sets["validation"].isdisjoint(class_sets["test"])
    assert set.union(*class_sets.values()) == {item.class_id for item in primary}
    by_id = {item.class_id: item for item in primary}
    for split, class_ids in class_sets.items():
        singleton_paths = [
            by_id[class_id].members[0].raw_path
            for class_id in class_ids
            if not by_id[class_id].ambiguous
        ]
        for stage in range(3):
            counts = Counter(path[stage] for path in singleton_paths)
            assert all(
                counts[operator] >= SINGLETON_POSITION_MINIMUMS[split]
                for operator in OPERATORS[1:]
            )


def test_composition_split_manifest_is_deterministic_and_fails_closed() -> None:
    first = build_composition_split_manifest()
    second = build_composition_split_manifest()
    assert first == second
    assert validate_composition_split_manifest(first) == first
    serialized = canonical_json_bytes(first)
    assert load_composition_split_manifest(serialized) == first

    tampered = copy.deepcopy(first)
    composition_splits = tampered["composition_splits"]
    assert isinstance(composition_splits, dict)
    train = composition_splits["train"]
    assert isinstance(train, dict)
    class_ids = train["class_ids"]
    assert isinstance(class_ids, list)
    class_ids[0] = "P07-SEM-" + "0" * 64
    tampered["manifest_sha256"] = canonical_json_sha256(
        {key: value for key, value in tampered.items() if key != "manifest_sha256"}
    )
    with pytest.raises(ValueError, match="frozen split"):
        validate_composition_split_manifest(tampered)


def test_seed_namespaces_and_all_sample_ids_are_disjoint_and_stable() -> None:
    assert GENERATOR_SEED_NAMESPACES == {
        "fit": (1103, 1109),
        "validation": (2203, 2207),
        "test": (3301, 3307),
    }
    assert SAMPLES_PER_GENERATOR_SEED == {"fit": 256, "validation": 128, "test": 256}
    assert OPTIMIZATION_SEEDS == (
        7,
        20,
        31,
        42,
        100,
        113,
        127,
        139,
        151,
        163,
        179,
        193,
        211,
        227,
        241,
        257,
        271,
        283,
        307,
        331,
        347,
        367,
        389,
        409,
        449,
    )
    expected_corruption = {
        seed: int(
            hashlib.sha256(f"{CORRUPTION_SEED_DOMAIN}|{seed}".encode("ascii")).hexdigest()[
                :16
            ],
            16,
        )
        % (2**63)
        for seed in OPTIMIZATION_SEEDS
    }
    assert CORRUPTION_SEED_DOMAIN == "P07-E8-corruption-v2"
    assert CORRUPTION_SEED_BY_OPTIMIZATION_SEED == expected_corruption
    assert len(set(expected_corruption.values())) == len(OPTIMIZATION_SEEDS)
    generator_seeds = [seed for values in GENERATOR_SEED_NAMESPACES.values() for seed in values]
    assert len(generator_seeds) == len(set(generator_seeds))
    assert set(generator_seeds).isdisjoint(OPTIMIZATION_SEEDS)
    assert set(generator_seeds).isdisjoint(CORRUPTION_SEED_BY_OPTIMIZATION_SEED.values())
    assert set(OPTIMIZATION_SEEDS).isdisjoint(
        CORRUPTION_SEED_BY_OPTIMIZATION_SEED.values()
    )

    sample_ids = set(registered_sample_ids())
    assert len(sample_ids) == 1280
    expected = (
        "P07-SAMPLE-46294594199ec1bf2eef8781ce3c9b9d86cd270410c4f2dcd5f667dc8d09ccff"
    )
    assert make_sample_id("fit", 1103, 0) == expected
    assert derive_sample_seed(expected, "noise", "snr=20", 32) == 907859620655585262
    assert derive_sample_seed(expected, "root") != derive_sample_seed(expected, "noise")
    assert validate_sample_id(expected) == expected


def test_seed_manifest_is_hash_bound_and_invalid_sample_requests_fail_closed() -> None:
    manifest = build_seed_namespace_manifest()
    assert manifest["manifest_sha256"] == EXPECTED_SEED_SHA256
    assert manifest["corruption_seed_derivation"] == {
        "algorithm": "sha256_first_16_hex_mod_2^63",
        "domain": "P07-E8-corruption-v2",
    }
    assert manifest["optimization_seed_policy"] == {
        "approximate_paired_mde_dz": 0.69,
        "maximum_directional_family_items": 6,
        "paired_seed_count": 25,
        "power_not_claimed_at_dz": 0.5,
        "practical_target_dz": 0.8,
        "status": "practical_floor_not_formal_power_guarantee",
    }
    assert validate_seed_namespace_manifest(manifest) == manifest
    with pytest.raises(ValueError, match="Unknown generator split"):
        make_sample_id("dev", 1103, 0)
    with pytest.raises(ValueError, match="not registered"):
        make_sample_id("fit", 2203, 0)
    with pytest.raises(ValueError, match="sample_index"):
        make_sample_id("validation", 2203, 128)
    with pytest.raises(TypeError, match="sample_index"):
        make_sample_id("fit", 1103, True)
    with pytest.raises(ValueError, match="sample_id"):
        derive_sample_seed("unbound", "noise")
    with pytest.raises(ValueError, match="absent from the registered"):
        derive_sample_seed("P07-SAMPLE-" + "0" * 64, "noise")
    with pytest.raises(TypeError, match="components"):
        derive_sample_seed(make_sample_id("fit", 1103, 0), "noise", 1.5)  # type: ignore[arg-type]


def test_independent_oracle_matches_elementary_operator_definitions() -> None:
    x = torch.tensor([[[1.0], [2.0], [4.0], [8.0]]], dtype=torch.float64)

    assert oracle_apply_operator("I", x) is x
    assert torch.equal(
        oracle_apply_operator("D1", x),
        torch.tensor([[[0.0], [1.0], [2.0], [4.0]]], dtype=torch.float64),
    )
    assert torch.equal(oracle_apply_operator("ABS", -x), x)
    assert torch.equal(oracle_apply_operator("SQUARE", x), x.square())
    expected_ma3 = torch.tensor(
        [[[4.0 / 3.0], [7.0 / 3.0], [14.0 / 3.0], [20.0 / 3.0]]],
        dtype=torch.float64,
    )
    assert torch.allclose(oracle_apply_operator("MA3", x), expected_ma3, atol=1e-12, rtol=0)
    envelope = oracle_apply_operator("HT", x)
    assert envelope.shape == x.shape
    assert envelope.dtype == x.dtype
    assert torch.isfinite(envelope).all()
    assert torch.all(envelope >= 0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_all_equivalence_class_members_match_under_independent_oracle(
    dtype: torch.dtype,
) -> None:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260801)
    x = torch.randn(2, 33, 2, generator=generator, dtype=dtype)
    atol = 1e-5 if dtype == torch.float32 else 1e-11
    rtol = 1e-5 if dtype == torch.float32 else 1e-11

    for item in enumerate_equivalence_classes():
        reference = oracle_execute_path(x, item.members[0].raw_path)
        for member in item.members[1:]:
            observed = oracle_execute_path(x, member.raw_path)
            assert torch.allclose(observed, reference, atol=atol, rtol=rtol), (
                item.canonical_expression,
                item.members[0].raw_path,
                member.raw_path,
            )


def test_all_216_paths_form_the_complete_independent_oracle_closure() -> None:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(81723)
    x = torch.randn(7, 97, 2, generator=generator, dtype=torch.float64)
    groups: dict[str, list[PathRecord]] = {}

    for record in enumerate_path_records():
        output = oracle_execute_path(x, record.raw_path).contiguous()
        digest = hashlib.sha256(output.numpy().tobytes()).hexdigest()
        groups.setdefault(digest, []).append(record)

    assert len(groups) == 116
    assert sum(len(members) for members in groups.values()) == 216
    assert all(
        len({member.class_id for member in members}) == 1
        for members in groups.values()
    )
    assert Counter(len(members) for members in groups.values()) == {
        1: 90,
        3: 6,
        4: 6,
        5: 8,
        6: 3,
        7: 2,
        12: 1,
    }


def test_oracle_and_path_validation_fail_closed() -> None:
    x = torch.ones(1, 8, 1)
    with pytest.raises(ValueError, match="exactly 3"):
        validate_raw_path(("I", "D1"))
    with pytest.raises(ValueError, match="Operator must be one of"):
        validate_raw_path(("I", "D1", "MA5"))
    with pytest.raises(TypeError, match="not text"):
        validate_raw_path("I,D1,ABS")
    with pytest.raises(ValueError, match="batch,length,channels"):
        oracle_execute_path(torch.ones(2, 8), ("I", "I", "I"))
    with pytest.raises(TypeError, match="float32 or torch.float64"):
        oracle_execute_path(torch.ones(1, 8, 1, dtype=torch.int64), ("I", "I", "I"))
    nonfinite = x.clone()
    nonfinite[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        oracle_execute_path(nonfinite, ("I", "I", "I"))
    huge = torch.full((1, 8, 1), 1e30)
    with pytest.raises(ValueError, match="produced non-finite"):
        oracle_execute_path(huge, ("SQUARE", "SQUARE", "SQUARE"))


def test_oracle_source_has_no_dependency_on_model_core_or_executor() -> None:
    source = inspect.getsource(path_universe_module)
    tree = ast.parse(source)
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    imported_modules.update(
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )
    assert not any("model_factory" in module for module in imported_modules)

    forbidden_calls = {"_apply_operator", "execute_paths", "forward_evidence"}
    observed_calls: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            observed_calls.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            observed_calls.add(node.func.attr)
    assert observed_calls.isdisjoint(forbidden_calls)


def test_registered_split_count_table_matches_manifest() -> None:
    assert COMPOSITION_SPLIT_COUNTS == {
        "train": {"ambiguous": 13, "singleton": 23},
        "validation": {"ambiguous": 7, "singleton": 11},
        "test": {"ambiguous": 6, "singleton": 12},
    }
    assert SINGLETON_POSITION_MINIMUMS == {"train": 2, "validation": 1, "test": 1}
