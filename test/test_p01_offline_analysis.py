"""Synthetic estimator checks, not evidence of real D1 performance."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.p01 import analyze_d1 as analysis
from experiments.p01.fusion_data import summarize_rows


def predictions() -> dict[str, np.ndarray]:
    # g1 has two acquisitions with unequal window counts, g2 only one.
    raw = np.asarray([[.9, .1], [.1, .9], [.1, .9], [.8, .2], [.2, .8]])
    candidate = np.asarray([[.7, .3], [.6, .4], [.6, .4], [.3, .7], [.4, .6]])
    return dict(raw_probs=raw, candidate_probs=candidate, deployed_probs=candidate,
                raw_log_probs=np.log(raw), candidate_log_probs=np.log(candidate),
                deployed_log_probs=np.log(candidate), labels=np.asarray([0, 0, 0, 1, 1]),
                group_ids=np.asarray(["g1", "g1", "g1", "g2", "g2"]),
                acquisition_ids=np.asarray(["a1", "a2", "a2", "a3", "a4"]),
                window_ids=np.asarray(["0", "0", "1", "0", "0"]),
                domains=np.asarray(["d0", "d0", "d0", "d0", "d1"]),
                class_names=np.asarray(["healthy", "fault"]))


def save(tmp_path: Path, p=None, *, arm="O", seed=42, role="direct", alpha=1.0):
    name = f"{arm}_{seed}_{role}"
    path = tmp_path / f"{name}.npz"
    if p is None:
        p = predictions()
    np.savez(path, **p)
    return dict(name=name, path=str(path), arm=arm, seed=seed, split="test", alpha=alpha, role=role)


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def test_hierarchy_and_confusion_match_existing_evaluator(tmp_path):
    artifact = analysis.load_artifact(save(tmp_path))
    group = next(row for row in artifact.groups if row["domain"] == "d0" and row["unit_id"] == "g1" and row["predictor"] == "raw")
    # Equal acquisitions (not three equally weighted windows).
    assert group["brier"] == pytest.approx((.02 + 1.62) / 2)
    rows = [row for row in artifact.groups if row["domain"] == "d0" and row["predictor"] == "raw"]
    got = analysis.condition_metrics(rows, ["g1", "g2"], np.ones((1, 2), int))
    canonical = next(row for row in summarize_rows(artifact.acquisitions, 2)
                     if row["domain"] == "d0" and row["predictor"] == "raw")
    for metric in analysis.METRICS:
        assert got[metric][0] == pytest.approx(canonical[metric])
    cm = group["confusion_matrix"] + rows[1]["confusion_matrix"]
    expected_f1 = np.mean(2 * cm.diagonal() / (cm.sum(0) + cm.sum(1)))
    assert got["macro_f1"][0] == pytest.approx(expected_f1)
    # Averaging group-specific F1 is a different, disallowed estimator.
    f1s = []
    for row in rows:
        matrix = row["confusion_matrix"]
        denominator = matrix.sum(0) + matrix.sum(1)
        f1s.append(np.divide(2 * matrix.diagonal(), denominator, out=np.zeros(2), where=denominator != 0).mean())
    assert not np.isclose(got["macro_f1"][0], np.mean(f1s))


def test_bootstrap_incidence_and_weighted_evaluator_parity(tmp_path):
    incidence = {"g1": {"d0", "d1"}, "g2": {"d0", "d1"}, "g3": {"d0"}, "g4": {"d0"}}
    groups, counts = analysis.bootstrap_counts(incidence, repeats=17)
    assert groups == ["g1", "g2", "g3", "g4"]
    np.testing.assert_array_equal(counts[:, :2].sum(1), 2)
    np.testing.assert_array_equal(counts[:, 2:].sum(1), 2)
    np.testing.assert_array_equal(counts, analysis.bootstrap_counts(incidence, repeats=17)[1])
    artifact = analysis.load_artifact(save(tmp_path))
    part = [row for row in artifact.groups if row["domain"] == "d0" and row["predictor"] == "raw"]
    weights = np.asarray([[2, 1], [0, 3], [3, 0]])
    result = analysis.condition_metrics(part, ["g1", "g2"], weights)
    for index, multiplicities in enumerate(weights):
        repeated = []
        for group, count in zip(["g1", "g2"], multiplicities):
            for repeat in range(count):
                repeated.extend(dict(row, unit_id=f"{group}_copy{repeat}") for row in artifact.acquisitions
                                if row["domain"] == "d0" and row["predictor"] == "raw" and row["unit_id"] == group)
        expected = summarize_rows(repeated, 2)[0]
        for metric in analysis.METRICS:
            assert result[metric][index] == pytest.approx(expected[metric])


def test_ce_uses_stable_log_probabilities_without_clipping(tmp_path):
    p = predictions()
    for predictor in analysis.PREDICTORS:
        p[predictor + "_probs"][0] = [0., 1.]
        p[predictor + "_log_probs"][0] = [-1000., 0.]
    artifact = analysis.load_artifact(save(tmp_path, p))
    row = next(row for row in artifact.acquisitions if row["acquisition_id"] == "a1" and row["predictor"] == "raw")
    assert row["ce"] == 1000.


@pytest.mark.parametrize("problem", ["duplicate", "log_prob", "class_order", "mixture"])
def test_invalid_prediction_artifacts_fail(tmp_path, problem):
    p = predictions()
    if problem == "duplicate":
        p["window_ids"][2] = "0"
    elif problem == "log_prob":
        p["candidate_log_probs"][0, 0] = -np.inf
    elif problem == "class_order":
        p["raw_class_names"] = p["class_names"][::-1]
    else:
        p["deployed_probs"] = p["raw_probs"].copy()
        p["deployed_log_probs"] = p["raw_log_probs"].copy()
    with pytest.raises(ValueError):
        analysis.load_artifact(save(tmp_path, p))


def test_alignment_uses_full_window_identity_not_row_order(tmp_path):
    left = analysis.load_artifact(save(tmp_path))
    p = predictions()
    for key in p:
        if key != "class_names":
            p[key] = p[key][::-1]
    right = analysis.load_artifact(save(tmp_path, p, arm="RC"))
    analysis._aligned(left, right)
    right.arrays["labels"][0] = 1
    with pytest.raises(ValueError, match="labels"):
        analysis._aligned(left, right)


def test_complete_contrasts_and_zero_adoption(tmp_path):
    specs = [save(tmp_path, arm=arm, seed=seed) for arm in analysis.CORE for seed in analysis.SEEDS]
    p = predictions()
    p["deployed_probs"] = p["raw_probs"].copy()
    p["deployed_log_probs"] = p["raw_log_probs"].copy()
    specs.append(save(tmp_path, p, arm="RC", role="adopted", alpha=0.))
    report = analysis.run(specs, tmp_path / "analysis", condition_sets={"test": {"source": ["d0"], "unseen": ["d1"]}},
                          adoption_mode="independent")
    assert report["status"] == "complete"
    assert report["bootstrap_repeats"] == 2000
    contrasts = read_csv(tmp_path / "analysis/paired_contrasts.csv")
    assert {row["contrast"] for row in contrasts} == {*analysis.CONTRASTS, "Delta_adoption", "Delta_adoption_vs_raw"}
    assert all(float(row["value"]) == 0 for row in contrasts if row["contrast"] in analysis.CONTRASTS)
    assert all(float(row["value"]) == 0 for row in contrasts if row["contrast"] == "Delta_adoption_vs_raw")
    adoption = next(row for row in contrasts if row["contrast"] == "Delta_adoption" and row["scope"] == "condition:d0" and row["metric"] == "brier")
    metrics = read_csv(tmp_path / "analysis/metrics.csv")
    values = {row["predictor"]: float(row["value"]) for row in metrics
              if row["role"] == "adopted" and row["scope"] == "condition:d0" and row["metric"] == "brier"}
    assert float(adoption["value"]) == pytest.approx(values["raw"] - values["candidate"])
    mechanism = read_csv(tmp_path / "analysis/mechanism.csv")
    adopted = [row for row in mechanism if row["role"] == "adopted" and row["predictor"] == "deployed"]
    assert all(float(row["alpha_sqrt_A"]) == 0 and float(row["prediction_change_rate"]) == 0 for row in adopted)


def test_missing_seed_does_not_create_complete_mean(tmp_path):
    specs = [save(tmp_path, arm="O")]
    missing = dict(specs[0], name="O_123_direct", seed=123, path=str(tmp_path / "missing.npz"))
    report = analysis.run([*specs, missing], tmp_path / "partial",
                          condition_sets={"test": {"source": ["d0"], "unseen": ["d1"]}})
    assert report["status"] == "partial"
    assert report["missing_artifacts"] == [missing]
    rows = read_csv(tmp_path / "partial/seed_summary.csv")
    assert all(row["finite_seed_mean"] == "" and row["seed_sd"] == "" for row in rows)
    assert json.loads((tmp_path / "partial/analysis.json").read_text())["coverage"][0]["complete_core"] is False


def test_zero_direction_has_no_fabricated_quality(tmp_path):
    p = predictions()
    p["candidate_probs"] = p["raw_probs"].copy()
    p["candidate_log_probs"] = p["raw_log_probs"].copy()
    p["deployed_probs"] = p["raw_probs"].copy()
    p["deployed_log_probs"] = p["raw_log_probs"].copy()
    analysis.run([save(tmp_path, p)], tmp_path / "zero",
                 condition_sets={"test": {"source": ["d0"], "unseen": ["d1"]}})
    rows = read_csv(tmp_path / "zero/mechanism.csv")
    assert all(float(row["A"]) == 0 and row["b_over_sqrt_A"] == "" for row in rows)


def test_same_readout_stays_seed42_source_validation(tmp_path):
    specs = [dict(save(tmp_path, arm=arm), split="validation") for arm in ("MLP16", "MLP16_all")]
    analysis.run(specs, tmp_path / "source", condition_sets={"validation": {"source": ["d0", "d1"]}})
    rows = read_csv(tmp_path / "source/paired_contrasts.csv")
    diagnostic = [row for row in rows if row["contrast"] == "Delta_same_readout"]
    assert diagnostic and all(row["seed"] == "42" and row["split"] == "validation" for row in diagnostic)
    with pytest.raises(ValueError, match="source-validation"):
        analysis.load_artifact(dict(specs[1], split="test"))
    with pytest.raises(ValueError, match="source-validation"):
        analysis.load_artifact(dict(specs[1], seed=123))


def test_physical_groups_cannot_cross_partitions(tmp_path):
    spec = save(tmp_path)
    with pytest.raises(ValueError, match="crosses analyzed partitions"):
        analysis.run([spec, dict(spec, split="validation")], tmp_path / "leaked",
                     condition_sets={"test": {"all": ["d0", "d1"]}, "validation": {"all": ["d0", "d1"]}})


def source_predictions():
    p = predictions()
    p.pop("deployed_probs")
    p.pop("deployed_log_probs")
    classes = p.pop("class_names")
    p["raw_class_names"] = classes
    p["candidate_class_names"] = classes.copy()
    # Training-mixture arrays must never be mistaken for the direct candidate.
    p["training_tau_mixture_probs"] = p["raw_probs"].copy()
    p["training_tau_mixture_log_probs"] = p["raw_log_probs"].copy()
    return p


def test_source_trainer_artifacts_are_analyzed_without_rewriting_or_inference(tmp_path):
    source_runs = []
    unchanged = {}
    diagnostics = ["linear", "MLP8", "MLP32", "MLP16_all"] + [f"only_view_{i}" for i in range(7)]
    slots = [(arm, seed) for arm in analysis.CORE for seed in analysis.SEEDS] + [(arm, 42) for arm in diagnostics]
    for arm, seed in slots:
        root = tmp_path / arm / f"seed_{seed}"
        root.mkdir(parents=True)
        path = root / "selected_source_validation_windows.npz"
        p = source_predictions()
        p.update(arm=np.asarray(arm), seed=np.asarray(seed), checkpoint=np.asarray(str(root / "selected_candidate.pt")),
                 benchmark_commit=np.asarray("original-training-commit"), model_config=np.asarray(str(root / "model_config.yaml")))
        np.savez(path, **p)
        (root / "model_config.yaml").write_text("model: synthetic-test-fixture\n")
        unchanged[path] = path.read_bytes()
        source_runs.append(f"{arm}:{seed}:{root}")
    descriptors = analysis.source_run_descriptors(source_runs)
    assert len(descriptors) == 26
    artifact = analysis.load_artifact(descriptors[0])
    np.testing.assert_array_equal(artifact.arrays["deployed_probs"], artifact.arrays["candidate_probs"])
    assert not np.array_equal(artifact.arrays["deployed_probs"], artifact.arrays["training_tau_mixture_probs"])
    output = tmp_path / "analysis"
    report = analysis.run(descriptors, output, condition_sets={"validation": {"source": ["d0", "d1"]}})
    assert report["status"] == "complete"
    recorded = json.loads((output / "inputs.json").read_text())["predictions"]
    assert len(recorded) == 26
    assert all(row["benchmark_commit"] == "original-training-commit" for row in recorded)
    assert all(row["checkpoint"].endswith("selected_candidate.pt") for row in recorded)
    assert all(row["model_config"] == row["source_snapshots"]["model_config.yaml"] for row in recorded)
    assert all("not independent adoption" in row["deployed_alias"] for row in recorded)
    assert all(path.read_bytes() == content for path, content in unchanged.items())
    contrasts = read_csv(output / "paired_contrasts.csv")
    assert {row["contrast"] for row in contrasts} == {*analysis.CONTRASTS, "Delta_same_readout"}


@pytest.mark.parametrize("override", [{"split": "test"}, {"alpha": .5}, {"role": "adopted"}])
def test_source_alias_cannot_fill_missing_deployment_for_other_roles(tmp_path, override):
    spec = dict(save(tmp_path, source_predictions()), split="validation")
    with pytest.raises(ValueError, match="validation/direct/alpha=1"):
        analysis.load_artifact(dict(spec, **override))


def test_partial_deployment_is_not_silently_repaired(tmp_path):
    p = source_predictions()
    p["deployed_probs"] = p["candidate_probs"]
    with pytest.raises(ValueError, match="Missing deployed"):
        analysis.load_artifact(dict(save(tmp_path, p), split="validation"))


def test_source_descriptor_keeps_explicit_missing_slots(tmp_path):
    run_directory = tmp_path / "not_finished"
    spec = analysis.source_run_descriptors([f"RO:456:{run_directory}"])[0]
    assert spec["arm"] == "RO" and spec["seed"] == 456
    assert spec["path"] == str(run_directory / "selected_source_validation_windows.npz")
    assert not run_directory.exists()
