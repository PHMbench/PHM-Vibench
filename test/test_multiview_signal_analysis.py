"""Synthetic saved predictions test estimators, not PHM performance claims."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from src.task_factory.task.classification.multiview_signal.analysis import analyze, plot


SEEDS = [11, 23, 37]
CLASSES = ["c0", "c1", "c2"]
TSPN_METHODS = ["TSPN"] + [
    f"{readout}/{variant}"
    for readout in ("intercept", "native_matched", "native_full", "signal_program",
                    "signal_transient", "signal_band", "signal_modulation", "signal_recurrence")
    for variant in ("raw", "projected", "guarded")
]
METHODS = TSPN_METHODS + ["ResNet1D", "TCN"]


def _saved_predictions(root: Path, protocol: str = "op1", *, permute: bool = True):
    """Three complete fitted seeds over unequal window counts per entity."""
    root.mkdir(parents=True)
    sizes = 1 + np.arange(18) % 4
    unit_index = np.repeat(np.arange(18), sizes)
    unit = np.array([f"entity-{index:02d}" for index in unit_index])
    labels = unit_index % 3
    items = np.array([f"window-{index:03d}" for index in range(len(labels))])
    prepared = {"study": {"protocol": protocol, "training_seeds": SEEDS,
                           "class_names": CLASSES,
                           "baseline_models": {"TSPN": {}, "ResNet1D": {}, "TCN": {}}}}
    banks, runs = [], []
    for seed_index, seed in enumerate(SEEDS):
        rng = np.random.default_rng(seed)
        shared_noise = rng.normal(size=(len(labels), 3))
        probabilities = []
        for method_index in range(len(METHODS)):
            logits = shared_noise.copy()
            margin = .55 + .1 * (method_index % 5) + .15 * seed_index
            logits[np.arange(len(labels)), labels] += margin
            logits[:, method_index % 3] += .08 * np.sin(unit_index + method_index)
            values = np.exp(logits - logits.max(axis=1, keepdims=True))
            probabilities.append(values / values.sum(axis=1, keepdims=True))
        bank = np.stack(probabilities, axis=1)
        banks.append(bank)
        for model_index, model in enumerate(prepared["study"]["baseline_models"]):
            if model == "TSPN":
                columns, names = list(range(25)), TSPN_METHODS
            else:
                columns, names = [METHODS.index(model)], [model]
            order = (np.random.default_rng(seed + 100 * model_index).permutation(len(labels))
                     if permute else np.arange(len(labels)))
            path = root / f"{model}-{seed}.npz"
            np.savez_compressed(path, predictions=bank[order][:, columns], methods=np.array(names),
                                y=labels[order], unit=unit[order], item=items[order],
                                class_names=np.array(CLASSES), seed=np.array(seed))
            runs.append({"model": model, "seed": seed, "predictions": str(path)})
    return prepared, {"runs": list(reversed(runs))}, np.stack(banks), labels, unit


def _macro_f1(confusion: np.ndarray) -> np.ndarray:
    true_positive = np.diagonal(confusion, axis1=-2, axis2=-1)
    denominator = confusion.sum(axis=-1) + confusion.sum(axis=-2)
    return np.divide(2 * true_positive, denominator,
                     out=np.zeros_like(true_positive), where=denominator > 0).mean(axis=-1)


def _reference(bank: np.ndarray, labels: np.ndarray, unit: np.ndarray, bootstrap_seed: int):
    """Direct entity means and confusion totals, independent of production metrics."""
    units = np.unique(unit)
    seeds, _, methods, classes = bank.shape
    brier = np.zeros((seeds, len(units), methods))
    accuracy = np.zeros_like(brier)
    confusion = np.zeros((seeds, len(units), methods, classes, classes))
    unit_labels = []
    for index, entity in enumerate(units):
        rows = np.flatnonzero(unit == entity)
        unit_labels.append(int(labels[rows[0]]))
        for seed in range(seeds):
            probability = bank[seed, rows]
            residual = probability - np.eye(classes)[labels[rows], None]
            brier[seed, index] = np.square(residual).sum(axis=2).mean(axis=0)
            decision = probability.argmax(axis=2)
            accuracy[seed, index] = (decision == labels[rows, None]).mean(axis=0)
            for method in range(methods):
                for item, label in enumerate(labels[rows]):
                    confusion[seed, index, method, label, decision[item, method]] += 1 / len(rows)
    per_seed = {
        "brier": brier.mean(axis=1),
        "accuracy": accuracy.mean(axis=1),
        "macro_f1": _macro_f1(confusion.mean(axis=1)),
    }
    rng = np.random.default_rng(bootstrap_seed)
    stratum = np.array([CLASSES[label] for label in unit_labels])
    draws = np.concatenate([
        rng.choice(indexes, size=(2000, len(indexes)), replace=True)
        for label in np.unique(stratum)
        for indexes in [np.flatnonzero(stratum == label)]
    ], axis=1)
    samples = {
        "brier": brier.mean(axis=0)[draws].mean(axis=1),
        "accuracy": accuracy.mean(axis=0)[draws].mean(axis=1),
        # F1 is nonlinear: recompute each fitted seed's F1 before averaging seeds.
        "macro_f1": np.mean([_macro_f1(confusion[seed][draws].mean(axis=1))
                              for seed in range(seeds)], axis=0),
    }
    return per_seed, samples, confusion


def _read_summary(output: Path):
    return json.loads((output / "analysis.json").read_text())


def _rewrite(path: Path, update):
    with np.load(path, allow_pickle=False) as archive:
        values = {key: archive[key].copy() for key in archive.files}
    update(values)
    np.savez_compressed(path, **values)


def test_complete_multiseed_external_baselines_allow_row_permutation_and_equal_entity_metrics(tmp_path):
    prepared, evaluation, bank, labels, unit = _saved_predictions(tmp_path / "predictions")
    output = tmp_path / "analysis"
    output.mkdir()
    paths = analyze(prepared, evaluation, output)
    assert isinstance(paths, dict)
    for filename in ("analysis.json", "summary_metrics.csv", "paired_effects.csv",
                     "seed_metrics.csv", "contribution_results.csv"):
        assert (output / filename).is_file()
        assert (output / filename).stat().st_size > 0
    summary = _read_summary(output)
    assert summary["methods"] == METHODS
    assert summary["training_seeds"] == SEEDS
    assert summary["independent_units"] == 18
    assert summary["bootstrap"] == 2000
    assert summary["bootstrap_seed"] == 20260922
    assert summary["singleton_strata"] == []
    per_seed, _, _ = _reference(bank, labels, unit, 20260922)
    for metric, expected in per_seed.items():
        result = summary["metrics"][metric]
        np.testing.assert_allclose(result["per_seed"], expected, atol=1e-14)
        np.testing.assert_allclose(result["mean"], expected.mean(axis=0), atol=1e-14)
        np.testing.assert_allclose(result["seed_sd"], expected.std(axis=0, ddof=1), atol=1e-14)
    # Unequal window counts must affect neither the independent sample size nor unit mass.
    naive_accuracy = (bank.argmax(axis=3) == labels[None, :, None]).mean(axis=1)
    assert np.max(abs(naive_accuracy - per_seed["accuracy"])) > .01
    with (output / "paired_effects.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    primary = [row for row in rows if row["role"] == "primary"]
    assert [(row["metric"], row["method"], row["comparator"]) for row in primary] == [
        ("brier", "signal_program/guarded", "TSPN")]
    assert {row["comparator"] for row in rows} >= {"ResNet1D", "TCN"}
    figures = tmp_path / "figures"
    figures.mkdir()
    plotted = plot(output / "summary_metrics.csv", figures)
    assert Path(plotted["figure"]).is_file() and Path(plotted["pdf"]).is_file()


def test_incomplete_or_semantically_invalid_saved_predictions_fail_without_aggregation(tmp_path):
    cases = ("missing_baseline_seed", "coordinate_mismatch", "invalid_probability",
             "missing_tspn_identity", "duplicate_item", "class_order", "seed_identity",
             "absent_class", "mixed_entity_labels")
    for case in cases:
        prepared, evaluation, _, _, _ = _saved_predictions(tmp_path / case, permute=False)
        run = next(row for row in evaluation["runs"] if row["model"] == "TSPN" and row["seed"] == 23)
        path = Path(run["predictions"])
        if case == "missing_baseline_seed":
            evaluation["runs"] = [row for row in evaluation["runs"]
                                   if not (row["model"] == "TCN" and row["seed"] == 23)]
        elif case == "coordinate_mismatch":
            _rewrite(path, lambda value: value["item"].__setitem__(0, "different-window"))
        elif case == "invalid_probability":
            _rewrite(path, lambda value: value["predictions"].__setitem__((0, 0, 0), np.nan))
        elif case == "missing_tspn_identity":
            def missing_identity(value):
                value["methods"] = value["methods"][:-1]
                value["predictions"] = value["predictions"][:, :-1]
            _rewrite(path, missing_identity)
        elif case == "duplicate_item":
            _rewrite(path, lambda value: value["item"].__setitem__(0, value["item"][1]))
        elif case == "class_order":
            _rewrite(path, lambda value: value.__setitem__("class_names", value["class_names"][::-1]))
        elif case == "seed_identity":
            _rewrite(path, lambda value: value.__setitem__("seed", np.array(99)))
        else:
            for current in evaluation["runs"]:
                def invalid_labels(value):
                    if case == "absent_class":
                        value["y"][value["y"] == 2] = 1
                    else:
                        # Entity 1 has two windows: modify only one across every saved run.
                        value["y"][1] = 0
                _rewrite(Path(current["predictions"]), invalid_labels)
        output = tmp_path / f"rejected-{case}"
        output.mkdir()
        with pytest.raises(ValueError):
            analyze(prepared, evaluation, output)
        assert not (output / "analysis.json").exists(), case


@pytest.mark.parametrize("protocol,bootstrap_seed", [("op1", 20260922), ("condition_dg", 20261013)])
def test_paired_class_stratified_bootstrap_recomputes_f1_for_fixed_trained_seeds(tmp_path, protocol, bootstrap_seed):
    prepared, evaluation, bank, labels, unit = _saved_predictions(tmp_path / "predictions", protocol)
    output = tmp_path / "analysis"
    output.mkdir()
    analyze(prepared, evaluation, output)
    summary = _read_summary(output)
    per_seed, samples, confusion = _reference(bank, labels, unit, bootstrap_seed)
    assert summary["bootstrap_seed"] == bootstrap_seed
    for metric, values in samples.items():
        low, high = np.quantile(values, [.025, .975], axis=0)
        np.testing.assert_allclose(summary["metrics"][metric]["ci_low"], low, atol=1e-14)
        np.testing.assert_allclose(summary["metrics"][metric]["ci_high"], high, atol=1e-14)
    # Pooling confusion over seeds before F1 changes the estimator.
    pooled_f1 = _macro_f1(confusion.mean(axis=(0, 1)))
    assert np.max(abs(pooled_f1 - per_seed["macro_f1"].mean(axis=0))) > 1e-3
    for metric in ("brier", "accuracy", "macro_f1"):
        rows = [row for row in summary["comparisons"] if row["metric"] == metric
                and row["method"] == "signal_program/guarded" and row["comparator"] == "TSPN"]
        assert len(rows) == 1
        row = rows[0]
        method, comparator = METHODS.index(row["method"]), METHODS.index(row["comparator"])
        delta = samples[metric][:, method] - samples[metric][:, comparator]
        low, high = np.quantile(delta, [.025, .975])
        seed_delta = per_seed[metric][:, method] - per_seed[metric][:, comparator]
        assert row["mean_difference"] == pytest.approx(seed_delta.mean(), abs=1e-14)
        assert row["seed_sd"] == pytest.approx(seed_delta.std(ddof=1), abs=1e-14)
        np.testing.assert_allclose([row["ci_low"], row["ci_high"]], [low, high], atol=1e-14)
        assert row["higher_is_better"] is (metric != "brier")
        assert row["role"] == ("primary" if metric == "brier" else "secondary")
