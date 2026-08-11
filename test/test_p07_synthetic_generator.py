from __future__ import annotations

import ast
import copy
import hashlib
import inspect
from dataclasses import replace

import pytest
import torch

import src.utils.p07_protocol.synthetic_generator as generator_module
from src.utils.p07_protocol.path_universe import (
    GENERATOR_SEED_NAMESPACES,
    canonical_json_bytes,
    canonical_json_sha256,
    make_sample_id,
    oracle_execute_path,
)
from src.utils.p07_protocol.synthetic_generator import (
    CHANNEL_COUNT,
    CIRCULAR_SHIFTS,
    NUISANCE_CELLS,
    NUISANCE_ORDER,
    SCALE_LEVELS,
    SEQUENCE_LENGTH,
    SNR_LEVELS_DB,
    NuisanceCell,
    SyntheticEpisode,
    apply_normalization,
    apply_nuisance,
    build_episode_manifest,
    build_nuisance_manifest,
    build_synthetic_generator_manifest,
    estimate_normalization_artifact,
    generate_root_batch,
    generate_root_signal,
    generate_synthetic_episode,
    load_normalization_artifact,
    load_nuisance_manifest,
    load_synthetic_generator_manifest,
    nuisance_manifest_json,
    synthetic_generator_manifest_json,
    validate_episode_manifest,
    validate_nuisance_cell,
    validate_nuisance_manifest,
    validate_synthetic_generator_manifest,
)


EXPECTED_GENERATOR_MANIFEST_SHA256 = (
    "26884d3cdff9437ff804988eb2212695736ded5eb80d178a90a1eabe49551b82"
)
EXPECTED_NUISANCE_MANIFEST_SHA256 = (
    "9b2aa503f168594c5ba9588c510b1cd9d8c11dc7f094adbd05b9e14ddfb9044b"
)
EXPECTED_FIRST_ROOT_SHA256 = (
    "161aad15d83cc82566cd7a83e7830c5331ce113d8514653d1b6dfce0a32ad125"
)
EXPECTED_EIGHT_FIT_NORMALIZATION_SHA256 = (
    "c89ed80fd82e550b2be5e3b82eee2f76842dcb3f174c02d0b7063366c2989ca4"
)


def _ids(split: str, count: int, *, seed_offset: int = 0) -> tuple[str, ...]:
    seed = GENERATOR_SEED_NAMESPACES[split][seed_offset]
    return tuple(make_sample_id(split, seed, index) for index in range(count))


def _cell(snr_db: int | None, scale: float, shift: int) -> NuisanceCell:
    return next(
        item
        for item in NUISANCE_CELLS
        if (item.snr_db, item.scale, item.circular_shift) == (snr_db, scale, shift)
    )


def _fit_artifact(count: int = 8):
    return estimate_normalization_artifact(_ids("fit", count))


def test_manifests_are_canonical_self_hashed_and_frozen() -> None:
    generator_manifest = build_synthetic_generator_manifest()
    nuisance_manifest = build_nuisance_manifest()

    assert generator_manifest["manifest_sha256"] == EXPECTED_GENERATOR_MANIFEST_SHA256
    assert nuisance_manifest["manifest_sha256"] == EXPECTED_NUISANCE_MANIFEST_SHA256
    assert synthetic_generator_manifest_json() == canonical_json_bytes(
        generator_manifest
    ).decode("utf-8")
    assert nuisance_manifest_json() == canonical_json_bytes(nuisance_manifest).decode(
        "utf-8"
    )
    assert load_synthetic_generator_manifest(synthetic_generator_manifest_json()) == (
        generator_manifest
    )
    assert load_nuisance_manifest(nuisance_manifest_json()) == nuisance_manifest
    assert validate_synthetic_generator_manifest(generator_manifest) == generator_manifest
    assert validate_nuisance_manifest(nuisance_manifest) == nuisance_manifest


def test_static_manifests_fail_closed_on_tamper_or_noncanonical_encoding() -> None:
    tampered_generator = copy.deepcopy(build_synthetic_generator_manifest())
    tampered_generator["tensor_contract"]["sequence_length"] = 255
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_synthetic_generator_manifest(tampered_generator)

    tampered_nuisance = copy.deepcopy(build_nuisance_manifest())
    tampered_nuisance["cells"][0]["scale"] = 9.0
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_nuisance_manifest(tampered_nuisance)

    with pytest.raises(ValueError, match="canonical"):
        load_synthetic_generator_manifest(" " + synthetic_generator_manifest_json())
    with pytest.raises(ValueError, match="canonical"):
        load_nuisance_manifest(nuisance_manifest_json() + "\n")


def test_nuisance_grid_is_exact_ordered_cartesian_product() -> None:
    expected = [
        (snr_db, scale, shift)
        for snr_db in SNR_LEVELS_DB
        for scale in SCALE_LEVELS
        for shift in CIRCULAR_SHIFTS
    ]
    observed = [
        (item.snr_db, item.scale, item.circular_shift) for item in NUISANCE_CELLS
    ]

    assert NUISANCE_ORDER == (
        "normalize",
        "scale",
        "circular_shift",
        "additive_noise",
    )
    assert observed == expected
    assert len(NUISANCE_CELLS) == 27
    assert len({item.cell_id for item in NUISANCE_CELLS}) == 27
    assert len({item.cell_sha256 for item in NUISANCE_CELLS}) == 27
    assert all(validate_nuisance_cell(item) is item for item in NUISANCE_CELLS)

    original = NUISANCE_CELLS[0]
    forged = replace(original, scale=9.0)
    with pytest.raises(ValueError, match="frozen 27-cell grid"):
        validate_nuisance_cell(forged)


def test_root_signal_is_deterministic_finite_and_dtype_stable() -> None:
    sample_id = _ids("fit", 1)[0]
    first64 = generate_root_signal(sample_id, dtype=torch.float64)
    second64 = generate_root_signal(sample_id, dtype=torch.float64)
    first32 = generate_root_signal(sample_id, dtype=torch.float32)

    assert first64.shape == (1, SEQUENCE_LENGTH, CHANNEL_COUNT)
    assert first64.dtype == torch.float64
    assert first32.dtype == torch.float32
    assert torch.isfinite(first64).all()
    assert torch.equal(first64, second64)
    assert torch.equal(first32, first64.float())
    assert hashlib.sha256(first64.numpy().tobytes()).hexdigest() == (
        EXPECTED_FIRST_ROOT_SHA256
    )


def test_registered_sample_namespaces_and_batch_order_are_disjoint() -> None:
    sample_ids = (
        _ids("fit", 1)[0],
        _ids("validation", 1)[0],
        _ids("test", 1)[0],
    )
    batch = generate_root_batch(sample_ids)
    reverse = generate_root_batch(tuple(reversed(sample_ids)))

    assert len({item.numpy().tobytes() for item in batch}) == 3
    assert torch.equal(batch, reverse.flip(0))
    assert not torch.equal(batch[0], batch[1])
    assert not torch.equal(batch[1], batch[2])


def test_root_generation_rejects_invalid_ids_duplicates_and_dtype() -> None:
    sample_id = _ids("fit", 1)[0]
    with pytest.raises(ValueError, match="unique"):
        generate_root_batch((sample_id, sample_id))
    with pytest.raises(ValueError, match="registered"):
        generate_root_signal("P07-SAMPLE-" + "0" * 64)
    with pytest.raises(TypeError, match="float32 or torch.float64"):
        generate_root_signal(sample_id, dtype=torch.float16)
    with pytest.raises(ValueError, match="nonempty"):
        generate_root_batch(())


def test_normalization_fit_is_deterministic_canonical_and_effective() -> None:
    fit_ids = _ids("fit", 8)
    artifact = estimate_normalization_artifact(fit_ids)
    reverse_artifact = estimate_normalization_artifact(tuple(reversed(fit_ids)))
    roots = generate_root_batch(artifact.fit_sample_ids)
    normalized = apply_normalization(
        roots,
        artifact,
        expected_artifact_sha256=artifact.artifact_sha256,
    )

    assert artifact == reverse_artifact
    assert artifact.artifact_sha256 == EXPECTED_EIGHT_FIT_NORMALIZATION_SHA256
    assert artifact.to_json() == canonical_json_bytes(artifact.manifest()).decode("utf-8")
    assert load_normalization_artifact(artifact.to_json()) == artifact
    assert torch.allclose(
        normalized.mean(dim=(0, 1)), torch.zeros(CHANNEL_COUNT, dtype=torch.float64), atol=1e-15
    )
    assert torch.allclose(
        normalized.square().mean(dim=(0, 1)).sqrt(),
        torch.ones(CHANNEL_COUNT, dtype=torch.float64),
        atol=1e-15,
    )


def test_normalization_rejects_nonfit_samples_and_artifact_tamper() -> None:
    with pytest.raises(ValueError, match="fit samples"):
        estimate_normalization_artifact(_ids("validation", 2))

    artifact = _fit_artifact(4)
    parsed = copy.deepcopy(artifact.manifest())
    parsed["artifact"]["mean"][0] += 1.0
    with pytest.raises(ValueError, match="hash mismatch"):
        load_normalization_artifact(canonical_json_bytes(parsed))

    rehashed_tamper = replace(
        artifact,
        mean=(artifact.mean[0] + 1.0, artifact.mean[1]),
    )
    roots = generate_root_batch(artifact.fit_sample_ids)
    with pytest.raises(ValueError, match="Pinned normalization"):
        apply_normalization(
            roots,
            rehashed_tamper,
            expected_artifact_sha256=artifact.artifact_sha256,
        )
    with pytest.raises(ValueError, match="canonical"):
        load_normalization_artifact(" " + artifact.to_json())


@pytest.mark.parametrize(
    ("mutation", "exception", "match"),
    [
        (lambda x: x[:, :-1], ValueError, "BLC shape"),
        (lambda x: x.to(torch.float16), TypeError, "float32 or torch.float64"),
        (
            lambda x: x.clone().index_fill_(1, torch.tensor([0]), float("nan")),
            ValueError,
            "non-finite",
        ),
    ],
)
def test_normalization_apply_rejects_invalid_tensor_contract(
    mutation, exception: type[Exception], match: str
) -> None:
    artifact = _fit_artifact(4)
    roots = generate_root_batch(artifact.fit_sample_ids)
    with pytest.raises(exception, match=match):
        apply_normalization(
            mutation(roots),
            artifact,
            expected_artifact_sha256=artifact.artifact_sha256,
        )
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        apply_normalization(
            roots,
            artifact,
            expected_artifact_sha256=artifact.artifact_sha256.upper(),
        )


def test_infinite_snr_nuisance_is_exact_scale_then_circular_shift() -> None:
    artifact = _fit_artifact(4)
    test_ids = _ids("test", 2)
    roots = generate_root_batch(test_ids)
    normalized = apply_normalization(
        roots, artifact, expected_artifact_sha256=artifact.artifact_sha256
    )
    cell = _cell(None, 2.0, 32)

    observed = apply_nuisance(normalized, test_ids, cell)
    expected = torch.roll(normalized * 2.0, shifts=32, dims=1)
    assert torch.equal(observed, expected)


@pytest.mark.parametrize("snr_db", [20, 10])
def test_finite_noise_has_exact_requested_per_sample_channel_snr(snr_db: int) -> None:
    artifact = _fit_artifact(4)
    sample_ids = _ids("test", 3)
    normalized = apply_normalization(
        generate_root_batch(sample_ids),
        artifact,
        expected_artifact_sha256=artifact.artifact_sha256,
    )
    cell = _cell(snr_db, 1.0, 0)
    observed = apply_nuisance(normalized, sample_ids, cell)
    repeated = apply_nuisance(normalized, sample_ids, cell)
    noise = observed - normalized
    measured = 20.0 * torch.log10(
        normalized.square().mean(dim=1).sqrt() / noise.square().mean(dim=1).sqrt()
    )

    assert torch.equal(observed, repeated)
    assert torch.allclose(measured, torch.full_like(measured, float(snr_db)), atol=1e-11)


def test_noise_seed_is_bound_to_sample_and_full_nuisance_cell() -> None:
    artifact = _fit_artifact(4)
    sample_ids = _ids("test", 3)
    normalized = apply_normalization(
        generate_root_batch(sample_ids),
        artifact,
        expected_artifact_sha256=artifact.artifact_sha256,
    )
    cell = _cell(20, 1.0, 0)
    output = apply_nuisance(normalized, sample_ids, cell)
    reversed_output = apply_nuisance(
        normalized.flip(0), tuple(reversed(sample_ids)), cell
    )
    other_cell = _cell(20, 1.0, 32)
    shifted_output = apply_nuisance(normalized, sample_ids, other_cell)

    assert torch.equal(output, reversed_output.flip(0))
    assert not torch.equal(output - normalized, shifted_output - torch.roll(normalized, 32, 1))


def test_nuisance_rejects_shape_nonfinite_zero_rms_and_identity_mismatch() -> None:
    sample_ids = _ids("test", 2)
    valid = generate_root_batch(sample_ids)
    cell = _cell(20, 1.0, 0)
    with pytest.raises(ValueError, match="count"):
        apply_nuisance(valid, sample_ids[:1], cell)
    with pytest.raises(ValueError, match="unique"):
        apply_nuisance(valid, (sample_ids[0], sample_ids[0]), cell)
    with pytest.raises(ValueError, match="BLC shape"):
        apply_nuisance(valid[:, :-1], sample_ids, cell)
    bad = valid.clone()
    bad[0, 0, 0] = float("inf")
    with pytest.raises(ValueError, match="non-finite"):
        apply_nuisance(bad, sample_ids, cell)
    with pytest.raises(ValueError, match="positive signal RMS"):
        apply_nuisance(torch.zeros_like(valid), sample_ids, cell)
    with pytest.raises(ValueError, match="frozen 27-cell grid"):
        apply_nuisance(valid, sample_ids, replace(cell, cell_sha256="0" * 64))


def test_episode_input_order_and_target_are_oracle_bound(monkeypatch) -> None:
    artifact = _fit_artifact(4)
    sample_ids = _ids("test", 2)
    path = ("D1", "MA3", "ABS")
    cell = _cell(10, 0.5, -32)
    original_oracle = generator_module._path_protocol.oracle_execute_path
    calls: list[tuple[tuple[int, ...], tuple[str, ...]]] = []

    def recording_oracle(x: torch.Tensor, raw_path):
        calls.append((tuple(x.shape), tuple(raw_path)))
        return original_oracle(x, raw_path)

    monkeypatch.setattr(
        generator_module._path_protocol, "oracle_execute_path", recording_oracle
    )
    episode = generate_synthetic_episode(
        sample_ids,
        path,
        artifact,
        cell,
        expected_normalization_sha256=artifact.artifact_sha256,
    )

    roots = generate_root_batch(sample_ids)
    normalized = apply_normalization(
        roots, artifact, expected_artifact_sha256=artifact.artifact_sha256
    )
    expected_input = apply_nuisance(normalized, sample_ids, cell)
    assert calls
    assert calls[0] == ((2, SEQUENCE_LENGTH, CHANNEL_COUNT), path)
    assert torch.equal(episode.input, expected_input)
    assert torch.equal(episode.target, original_oracle(episode.input, path))


def test_episode_manifest_and_oracle_binding_fail_closed() -> None:
    artifact = _fit_artifact(4)
    episode = generate_synthetic_episode(
        _ids("test", 2),
        ("HT", "ABS", "I"),
        artifact,
        _cell(None, 1.0, 0),
        expected_normalization_sha256=artifact.artifact_sha256,
        dtype=torch.float32,
    )
    manifest = build_episode_manifest(episode)
    assert validate_episode_manifest(manifest, episode) == manifest
    payload = dict(manifest)
    digest = payload.pop("manifest_sha256")
    assert canonical_json_sha256(payload) == digest

    tampered = copy.deepcopy(manifest)
    tampered["raw_path"][0] = "I"
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_episode_manifest(tampered, episode)

    wrong_target = replace(episode, target=episode.target + 1.0)
    with pytest.raises(ValueError, match="independent oracle"):
        build_episode_manifest(wrong_target)

    forged_hash_episode = SyntheticEpisode(
        sample_ids=episode.sample_ids,
        raw_path=episode.raw_path,
        nuisance_cell_id=episode.nuisance_cell_id,
        normalization_artifact_sha256="NOT-A-HASH",
        input=episode.input,
        target=episode.target,
    )
    with pytest.raises(ValueError, match="normalization hash"):
        build_episode_manifest(forged_hash_episode)


def test_generator_source_has_no_private_executor_or_model_dependency() -> None:
    source = inspect.getsource(generator_module)
    tree = ast.parse(source)
    imported_modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported_modules.append(node.module or "")

    assert not any("core" in name or "model" in name for name in imported_modules)
    assert "G030" not in source
    assert "_path_protocol.oracle_execute_path(nuisance_input, path)" in source
    assert oracle_execute_path is not None
