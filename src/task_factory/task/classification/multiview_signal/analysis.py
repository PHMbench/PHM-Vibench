"""Paired physical-entity inference from already frozen P19 predictions.

This stage loads no raw signals, fits no model and chooses no hyperparameters.
Training seeds describe repeated fits; whole physical entities are the sampling
units. Intervals condition on the fixed trained models and class composition.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .metrics import (aggregate_metrics, group_means, paired_brier_summary,
                      validate_labels, validate_probabilities)
from .readouts import _ORIGINAL_READOUTS, _SINGLE_READOUTS

_METRICS = ("brier", "accuracy", "macro_f1")
_SIGNAL_METHODS = ["TSPN"] + [
    f"{name}/{stage}" for name in _ORIGINAL_READOUTS + _SINGLE_READOUTS
    for stage in ("raw", "projected", "guarded")
]


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("An empty planned result table cannot be reported")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _archive(path: str | Path, model: str, seed: int,
             classes: list[str]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        required = {"predictions", "methods", "y", "unit", "item", "class_names", "seed"}
        if not required <= set(data.files):
            raise ValueError(f"Prediction archive lacks fields: {sorted(required - set(data.files))}")
        values = {name: data[name].copy() for name in required}
    saved_seed = values["seed"]
    if saved_seed.shape != () or saved_seed.dtype.kind not in "iu" or int(saved_seed) != seed:
        raise ValueError("Prediction archive and registered training seed differ")
    if values["class_names"].ndim != 1 or values["class_names"].tolist() != classes:
        raise ValueError("Prediction class coordinates differ from the frozen vocabulary")
    methods = values["methods"]
    expected = _SIGNAL_METHODS if model == "TSPN" else [model]
    if methods.ndim != 1 or methods.tolist() != expected:
        raise ValueError(f"Prediction method coordinates differ from the complete {model} identities")
    prediction = values["predictions"]
    if prediction.ndim != 3 or prediction.shape[1:] != (len(expected), len(classes)):
        raise ValueError("Predictions must have aligned [items,methods,fixed classes] axes")
    n = len(prediction)
    values["y"] = validate_labels(values["y"], n, len(classes))
    for name in ("item", "unit"):
        identifiers = values[name]
        if (identifiers.shape != (n,) or identifiers.dtype.kind not in "iuUS"
                or (name == "item" and identifiers.dtype.kind not in "US")
                or (identifiers.dtype.kind in "US" and
                    np.any(np.char.str_len(np.char.strip(identifiers)) == 0))):
            raise ValueError(f"Explicit aligned nonempty {name} coordinates are required")
    if len(np.unique(values["item"])) != n:
        raise ValueError("Prediction items must be unique; duplicate windows cannot be counted twice")
    for method in range(len(expected)):
        validate_probabilities(prediction[:, method])
    order = np.argsort(values["item"])
    for name in ("predictions", "y", "unit", "item"):
        values[name] = values[name][order]
    return values


def _merge(prepared: Mapping[str, Any], evaluation: Mapping[str, Any]
           ) -> tuple[np.ndarray, dict[str, np.ndarray], list[str], list[int]]:
    study = prepared["study"]
    seeds = list(study["training_seeds"])
    if seeds != [11, 23, 37]:
        raise ValueError("Analysis requires registered training seeds 11/23/37, not successful subsets")
    models = list(study["baseline_models"])
    if "TSPN" not in models or len(set(models)) != len(models):
        raise ValueError("The registered model comparison must include frozen TSPN")
    classes = list(study["class_names"])
    if (len(classes) < 2 or len(set(classes)) != len(classes)
            or any(not isinstance(name, str) or not name.strip() for name in classes)):
        raise ValueError("A distinct fixed class vocabulary is required")
    rows = evaluation["runs"]
    index = {}
    for row in rows:
        key = (row["model"], row["seed"])
        if key in index:
            raise ValueError("Duplicate model/training-seed evaluation run")
        index[key] = row["predictions"]
    expected = {(model, seed) for model in models for seed in seeds}
    if set(index) != expected:
        raise ValueError("Evaluation must contain every registered model and training seed exactly once")
    # TSPN's original identity order stays first even if YAML orders models differently.
    models = ["TSPN"] + [model for model in models if model != "TSPN"]
    reference = None
    method_names: list[str] = []
    seed_predictions = []
    for seed in seeds:
        blocks, current_methods = [], []
        for model in models:
            values = _archive(index[(model, seed)], model, seed, classes)
            if reference is None:
                reference = values
            else:
                for coordinate in ("item", "y", "unit", "class_names"):
                    if not np.array_equal(reference[coordinate], values[coordinate]):
                        raise ValueError(f"Prediction coordinates differ across models/seeds: {coordinate}")
            blocks.append(values["predictions"])
            current_methods.extend(values["methods"].tolist())
        if method_names and current_methods != method_names:
            raise ValueError("Prediction method coordinates differ across training seeds")
        method_names = current_methods
        if len(set(method_names)) != len(method_names):
            raise ValueError("Registered models produce duplicate method identities")
        seed_predictions.append(np.concatenate(blocks, axis=1))
    if reference is None:
        raise ValueError("No complete registered predictions")
    if set(reference["y"]) != set(range(len(classes))):
        raise ValueError("Frozen target population lacks declared class support")
    if "roles" in prepared:
        declared = set(prepared["roles"]["test"]["unit_by_record"].values())
        if set(map(str, reference["unit"])) != declared:
            raise ValueError("Predictions omit or add physical entities from the frozen target population")
    return np.stack(seed_predictions), reference, method_names, seeds


def _bootstrap(groups: np.ndarray, strata: Mapping[str, str], seed: int
               ) -> tuple[np.ndarray, list[str]]:
    labels = np.asarray([strata[str(group)] for group in groups])
    rng = np.random.default_rng(seed)
    blocks, singleton = [], []
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        blocks.append(rng.choice(indices, size=(2000, len(indices)), replace=True))
        if len(indices) == 1:
            singleton.append(str(label))
    return np.concatenate(blocks, axis=1), singleton


def _f1(confusion: np.ndarray) -> np.ndarray:
    denominator = confusion.sum(axis=-2) + confusion.sum(axis=-1)
    numerator = 2. * np.diagonal(confusion, axis1=-2, axis2=-1)
    scores = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
    return scores.mean(axis=-1)


def _contrasts(methods: list[str]) -> list[tuple[str, str, str]]:
    comparisons = [
        ("signal_program/guarded", "TSPN", "primary"),
        ("signal_program/projected", "TSPN", "projected_diagnostic"),
        ("signal_program/projected", "intercept/projected", "intercept_control"),
        ("signal_program/projected", "native_matched/projected", "matched_native_control"),
        ("signal_program/projected", "native_full/projected", "full_native_control"),
    ]
    comparisons.extend(("signal_program/projected", f"{view}/projected", "single_view_control")
                       for view in _SINGLE_READOUTS)
    comparisons.extend(("signal_program/guarded", method, "external_baseline")
                       for method in methods if method != "TSPN" and "/" not in method)
    if any(method not in methods or comparator not in methods for method, comparator, _ in comparisons):
        raise ValueError("A predeclared controlled comparison is missing")
    return comparisons


def analyze(prepared: Mapping[str, Any], evaluation: Mapping[str, Any],
            output: Path) -> dict[str, Any]:
    """Report every registered model/seed using the same paired entity draws."""
    output = Path(output)
    if not output.is_dir():
        raise ValueError("Analysis output directory must already exist")
    protocol = prepared["study"]["protocol"]
    if protocol not in {"op1", "condition_dg"}:
        raise ValueError("Analysis requires a frozen OP1 or condition DG protocol")
    prediction, coordinates, methods, seeds = _merge(prepared, evaluation)
    units, inverse, counts = np.unique(coordinates["unit"], return_inverse=True, return_counts=True)
    if len(units) < 2:
        raise ValueError("At least two independent physical entities are required")
    classes = coordinates["class_names"].tolist()
    strata = {}
    for unit in units:
        labels = np.unique(coordinates["y"][coordinates["unit"] == unit])
        if len(labels) != 1:
            raise ValueError("Bearing-stratified analysis requires exactly one class per physical entity")
        strata[str(unit)] = classes[int(labels[0])]
    bootstrap_seed = 20260922 if protocol == "op1" else 20261013
    draws, singleton = _bootstrap(units, strata, bootstrap_seed)
    n_seeds, _, n_methods, n_classes = prediction.shape
    unit_brier, unit_accuracy, confusions, seed_rows = [], [], [], []
    for seed_index, seed in enumerate(seeds):
        probabilities = prediction[seed_index]
        residual = probabilities - np.eye(n_classes)[coordinates["y"]][:, None, :]
        brier = group_means(np.square(residual).sum(axis=-1), coordinates["unit"])
        predicted = probabilities.argmax(axis=-1)
        accuracy = group_means((predicted == coordinates["y"][:, None]).astype(float), coordinates["unit"])
        confusion = np.zeros((len(units), n_methods, n_classes, n_classes), dtype=np.float64)
        for method in range(n_methods):
            np.add.at(confusion[:, method], (inverse, coordinates["y"], predicted[:, method]), 1. / counts[inverse])
            measured = aggregate_metrics(probabilities[:, method], coordinates["y"], coordinates["unit"])
            seed_rows.append({"method": methods[method], "seed": seed,
                              **{metric: measured[metric] for metric in _METRICS},
                              "independent_units": len(units)})
        unit_brier.append(brier)
        unit_accuracy.append(accuracy)
        confusions.append(confusion)
    unit_brier, unit_accuracy, confusions = np.stack(unit_brier), np.stack(unit_accuracy), np.stack(confusions)
    brier_summary = paired_brier_summary(unit_brier, methods, units, bootstrap=2000,
                                         seed=bootstrap_seed, unit_strata=strata)
    # Only the predeclared contrasts below are scientific comparisons.
    brier_summary.pop("comparisons")
    # Each bootstrap replicate is an equal-entity weighted confusion matrix;
    # average nonlinear F1 only after computing each trained model's F1.
    multiplicity = np.zeros((len(draws), len(units)), dtype=np.float64)
    np.add.at(multiplicity, (np.arange(len(draws))[:, None], draws), 1. / len(units))
    sampled_confusions = np.einsum("bg,sgmcd->sbmcd", multiplicity, confusions, optimize=True)
    per_seed = {"brier": unit_brier.mean(axis=1), "accuracy": unit_accuracy.mean(axis=1),
                "macro_f1": _f1(confusions.mean(axis=1))}
    bootstrap_values = {"brier": unit_brier.mean(axis=0)[draws].mean(axis=1),
                        "accuracy": unit_accuracy.mean(axis=0)[draws].mean(axis=1),
                        "macro_f1": _f1(sampled_confusions).mean(axis=0)}
    metric_summary = {}
    summary_rows = []
    for metric in _METRICS:
        point = per_seed[metric].mean(axis=0)
        dispersion = np.std(per_seed[metric], axis=0, ddof=1)
        low, high = np.quantile(bootstrap_values[metric], [.025, .975], axis=0)
        metric_summary[metric] = {"mean": point.tolist(), "seed_sd": dispersion.tolist(),
                                  "ci_low": low.tolist(), "ci_high": high.tolist(),
                                  "per_seed": per_seed[metric].tolist()}
        summary_rows.extend({"method": method, "metric": metric, "mean": float(point[index]),
                             "seed_sd": float(dispersion[index]), "ci_low": float(low[index]),
                             "ci_high": float(high[index]), "independent_units": len(units),
                             "training_seeds": len(seeds)}
                            for index, method in enumerate(methods))
    comparisons = []
    for method, comparator, role in _contrasts(methods):
        candidate, baseline = methods.index(method), methods.index(comparator)
        for metric in _METRICS:
            seed_effect = per_seed[metric][:, candidate] - per_seed[metric][:, baseline]
            if metric in {"brier", "accuracy"}:
                values = unit_brier if metric == "brier" else unit_accuracy
                effect = values.mean(axis=0)[:, candidate] - values.mean(axis=0)[:, baseline]
                samples = effect[draws].mean(axis=1)
                point = float(effect.mean())
            else:
                samples = bootstrap_values[metric][:, candidate] - bootstrap_values[metric][:, baseline]
                point = float(seed_effect.mean())
            low, high = np.quantile(samples, [.025, .975])
            comparisons.append({"metric": metric, "method": method, "comparator": comparator,
                                "mean_difference": point, "ci_low": float(low), "ci_high": float(high),
                                "seed_sd": float(np.std(seed_effect, ddof=1)),
                                "higher_is_better": metric != "brier",
                                "role": "secondary" if role == "primary" and metric != "brier" else role})
    boundary = ("Conditional on the fixed trained models and registered physical-entity class composition; "
                "training seeds are not independent entities; singleton-stratum variance is unestimated.")
    summary = {"protocol": protocol, "methods": methods, "training_seeds": seeds,
               "independent_units": len(units), "units": units.tolist(), "bootstrap": 2000,
               "bootstrap_seed": bootstrap_seed, "singleton_strata": singleton,
               "interval_scope": boundary, "metrics": metric_summary, "comparisons": comparisons,
               "primary_comparison": {"method": "signal_program/guarded", "comparator": "TSPN", "metric": "brier"},
               "brier_paired_summary": brier_summary,
               "interpretation": "Measured effects do not by themselves establish mechanism or general DG superiority."}
    contribution_rows = []
    for comparison in comparisons:
        if comparison["metric"] != "brier":
            continue
        role = comparison["role"]
        contribution = ("Frozen-baseline risk comparison" if role == "primary" else
                        "External baseline context" if role == "external_baseline" else
                        "Signal information beyond the frozen backbone")
        contribution_rows.append({"contribution": contribution,
            "controlled_contrast": comparison["method"] + " minus " + comparison["comparator"],
            "metric": "entity_equal_full_brier", "mean_difference": comparison["mean_difference"],
            "ci_low": comparison["ci_low"], "ci_high": comparison["ci_high"],
            "observation": "Measured candidate-minus-comparator risk; a negative value means lower Brier.",
            "claim_boundary": boundary})
    paths = {"summary": output / "analysis.json", "summary_metrics": output / "summary_metrics.csv",
             "paired_effects": output / "paired_effects.csv", "seed_metrics": output / "seed_metrics.csv",
             "contribution_results": output / "contribution_results.csv"}
    _json(paths["summary"], summary)
    _csv(paths["summary_metrics"], summary_rows)
    _csv(paths["paired_effects"], comparisons)
    _csv(paths["seed_metrics"], seed_rows)
    _csv(paths["contribution_results"], contribution_rows)
    return {"status": "completed", **{name: str(path) for name, path in paths.items()}}


def plot(summary_csv: Path, output: Path) -> dict[str, Any]:
    """Plot saved summary estimates and intervals without fitting or resampling."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    output = Path(output)
    if not output.is_dir():
        raise ValueError("Plot output directory must already exist")
    with Path(summary_csv).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    values = {}
    methods = []
    for row in rows:
        method, metric = row["method"], row["metric"]
        key = (method, metric)
        if key in values:
            raise ValueError("Duplicate method/metric summary row")
        estimate = tuple(float(row[name]) for name in ("mean", "ci_low", "ci_high"))
        if metric not in _METRICS or not np.isfinite(estimate).all() or estimate[1] > estimate[2]:
            raise ValueError("Invalid saved summary estimates or confidence interval")
        values[key] = estimate
        if method not in methods:
            methods.append(method)
    selected = [method for method in methods
                if method in {"TSPN", "signal_program/guarded"} or method.endswith("/projected")
                or "/" not in method]
    if not selected or any((method, metric) not in values for method in selected for metric in _METRICS):
        raise ValueError("Plot requires the complete saved metric table")
    figure = Figure(figsize=(14, max(5., .35 * len(selected) + 1.5)))
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 3)
    labels = {"TSPN": "TSPN", "signal_program/guarded": "Signal views, guarded",
              "signal_program/projected": "Signal views, projected",
              "intercept/projected": "Intercept, projected",
              "native_matched/projected": "Native features, matched",
              "native_full/projected": "Native features, full",
              "signal_transient/projected": "Transient view",
              "signal_band/projected": "Band-power view",
              "signal_modulation/projected": "Modulation view",
              "signal_recurrence/projected": "Recurrence view"}
    for axis, metric in zip(axes, _METRICS):
        for index, method in enumerate(selected):
            mean, low, high = values[(method, metric)]
            color = "#1766A5" if method == "signal_program/guarded" else "#444444"
            axis.hlines(index, low, high, color=color, linewidth=1.5)
            axis.plot(mean, index, "o", color=color, markersize=4)
        axis.set_yticks(np.arange(len(selected)))
        axis.set_yticklabels([labels.get(method, method) for method in selected] if axis is axes[0] else [])
        axis.set_ylim(len(selected) - .5, -.5)
        axis.set_xlim(0, 2 if metric == "brier" else 1)
        axis.set_xlabel({"brier": "Full Brier (lower is better)", "accuracy": "Accuracy",
                         "macro_f1": "Macro-F1"}[metric])
        axis.grid(axis="x", alpha=.2)
        axis.spines[["top", "right"]].set_visible(False)
    figure.text(.03, .025, "Points: mean over registered training seeds. Lines: 95% paired whole-entity, class-stratified bootstrap intervals.\n"
                "Intervals condition on fixed trained models; singleton-stratum variance is unestimated.", fontsize=9)
    figure.subplots_adjust(left=.23, right=.98, bottom=.15, top=.97, wspace=.15)
    png, pdf = output / "comparison_metrics.png", output / "comparison_metrics.pdf"
    figure.savefig(png, dpi=200, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    return {"status": "completed", "figure": str(png), "pdf": str(pdf)}
