from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest
import torch

import src.explain_factory.p05_trace_export as trace_export
from src.explain_factory.p05_trace_export import P05TraceBatch, export_p05_trace_package
from src.model_factory.X_model.UXFD.fuzzy.fuzzy_reasoner import FuzzyTrace


CONFIG_HASH = "a" * 64
CHECKPOINT_HASH = "b" * 64
MODEL_HASH = "c" * 64


def _make_batch(prefix: str, offset: int = 0) -> P05TraceBatch:
    batch_size = 2
    num_features = 2
    num_memberships = 2
    num_rules = 2
    num_classes = 3
    normalized_firing = torch.tensor([[0.75, 0.25], [0.20, 0.80]])
    consequents = torch.tensor(
        [[0.20, -0.10, 0.30], [-0.40, 0.50, 0.10]],
        dtype=torch.float32,
    )
    contributions = normalized_firing.unsqueeze(-1) * consequents.unsqueeze(0)
    fuzzy_logits = contributions.sum(dim=1)
    non_fuzzy_logits = torch.tensor(
        [[1.0 + offset, 0.2, -0.3], [0.1, 1.2 + offset, 0.4]],
        dtype=torch.float32,
    )
    fuzzy_scale = 0.5
    logits = non_fuzzy_logits + fuzzy_scale * fuzzy_logits
    trace = FuzzyTrace(
        reduced_features=torch.tensor([[0.1, 0.2], [0.3, 0.4]]) + offset,
        membership_values=torch.tensor(
            [
                [[0.8, 0.2], [0.6, 0.4]],
                [[0.3, 0.7], [0.9, 0.1]],
            ]
        ),
        centers=torch.tensor([[-1.0, 1.0], [-0.5, 0.5]]),
        widths=torch.tensor([[0.7, 0.8], [0.9, 1.0]]),
        antecedent_probabilities=torch.full(
            (num_rules, num_features, num_memberships),
            0.5,
        ),
        antecedent_memberships=torch.full(
            (batch_size, num_rules, num_features),
            0.6,
        ),
        log_rule_firing=torch.log(torch.tensor([[0.6, 0.4], [0.3, 0.7]])),
        rule_firing=torch.tensor([[0.6, 0.4], [0.3, 0.7]]),
        normalized_rule_firing=normalized_firing,
        rule_consequents=consequents,
        rule_contributions=contributions,
        fuzzy_logits=fuzzy_logits,
        rule_mask=torch.ones((batch_size, num_rules), dtype=torch.bool),
        consequent_permutation=torch.arange(num_rules),
    )
    return P05TraceBatch(
        sample_id=[f"{prefix}-sample-{offset}", f"{prefix}-sample-{offset + 1}"],
        record_id=[f"{prefix}-record", f"{prefix}-record"],
        group_id=[f"{prefix}-group", f"{prefix}-group"],
        window_start=torch.tensor([offset * 10, offset * 10 + 5]),
        window_end=torch.tensor([offset * 10 + 5, offset * 10 + 10]),
        y=torch.tensor([0, 1]),
        logits=logits,
        non_fuzzy_logits=non_fuzzy_logits,
        fuzzy_scale=fuzzy_scale,
        fuzzy_trace=trace,
    )


def _export(package, batches):
    return export_p05_trace_package(
        package,
        batches,
        config_sha256=CONFIG_HASH,
        checkpoint_sha256=CHECKPOINT_HASH,
        model_sha256=MODEL_HASH,
    )


def test_trace_export_success_and_semantic_idempotent_reuse(tmp_path) -> None:
    package = tmp_path / "trace-package"
    batches = [_make_batch("a", 0), _make_batch("b", 10)]

    created = _export(package, batches)
    npz_before = created.npz_path.read_bytes()
    manifest_before = created.manifest_path.read_bytes()
    reused = _export(package, batches)

    assert created.status == "created"
    assert reused.status == "reused"
    assert reused.npz_path.read_bytes() == npz_before
    assert reused.manifest_path.read_bytes() == manifest_before
    manifest = json.loads(manifest_before)
    assert manifest["schema_name"] == "p05.complete_fuzzy_trace"
    assert manifest["sample_count"] == 4
    assert manifest["provenance"] == {
        "checkpoint_sha256": CHECKPOINT_HASH,
        "config_sha256": CONFIG_HASH,
        "model_sha256": MODEL_HASH,
    }
    assert manifest["content"]["npz_sha256"] == created.npz_sha256
    assert manifest["content"]["semantic_sha256"] == created.semantic_sha256
    with np.load(created.npz_path, allow_pickle=False) as arrays:
        assert set(arrays.files) == set(manifest["arrays"])
        for name in arrays.files:
            if name in {"sample_id", "record_id", "group_id"}:
                assert arrays[name].dtype.kind == "U"
            else:
                assert arrays[name].dtype == np.dtype("<f8")
                assert np.isfinite(arrays[name]).all()
    assert not list(tmp_path.glob(".trace-package.*.tmp"))


def test_trace_export_rejects_per_sample_reconstruction_drift(tmp_path) -> None:
    batch = _make_batch("drift")
    bad_logits = batch.logits.clone()
    bad_logits[1, 2] += 1.0e-3
    bad = replace(batch, logits=bad_logits)
    package = tmp_path / "drift-package"

    with pytest.raises(ValueError, match="reconstruction.*sample_id='drift-sample-1'"):
        _export(package, [bad])

    assert not package.exists()


def test_trace_export_rejects_duplicate_sample_ids_across_batches(tmp_path) -> None:
    first = _make_batch("duplicate", 0)
    second = _make_batch("other", 0)
    second = replace(
        second,
        sample_id=[first.sample_id[1], "otherwise-unique"],
    )

    with pytest.raises(ValueError, match="duplicate sample_id"):
        _export(tmp_path / "duplicate-package", [first, second])


def test_trace_export_rejects_nan_in_complete_trace(tmp_path) -> None:
    batch = _make_batch("nan")
    reduced = batch.fuzzy_trace.reduced_features.clone()
    reduced[0, 0] = float("nan")
    bad_trace = replace(batch.fuzzy_trace, reduced_features=reduced)

    with pytest.raises(FloatingPointError, match="reduced_features.*non-finite"):
        _export(
            tmp_path / "nan-package",
            [replace(batch, fuzzy_trace=bad_trace)],
        )


def test_trace_export_refuses_conflicting_existing_content(tmp_path) -> None:
    package = tmp_path / "conflict-package"
    original = _make_batch("conflict")
    _export(package, [original])
    npz_before = (package / "trace_arrays.npz").read_bytes()
    manifest_before = (package / "manifest.json").read_bytes()
    changed = replace(original, record_id=["changed-record", "changed-record"])

    with pytest.raises(FileExistsError, match="conflicts"):
        _export(package, [changed])

    assert (package / "trace_arrays.npz").read_bytes() == npz_before
    assert (package / "manifest.json").read_bytes() == manifest_before


def test_trace_export_refuses_symlink_target(tmp_path) -> None:
    real_directory = tmp_path / "real"
    real_directory.mkdir()
    target = tmp_path / "linked-package"
    target.symlink_to(real_directory, target_is_directory=True)

    with pytest.raises(FileExistsError, match="symlink"):
        _export(target, [_make_batch("symlink")])

    assert not list(real_directory.iterdir())


def test_trace_export_write_failure_leaves_no_half_package(tmp_path, monkeypatch) -> None:
    package = tmp_path / "failed-package"

    def _fail_save(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("synthetic write failure")

    monkeypatch.setattr(trace_export.np, "savez", _fail_save)
    with pytest.raises(RuntimeError, match="synthetic write failure"):
        _export(package, [_make_batch("failure")])

    assert not package.exists()
    assert not list(tmp_path.glob(".failed-package.*.tmp"))
