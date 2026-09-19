"""Render D1 diagnostic previews from completed, saved analysis tables only.

This module deliberately does not import an evaluator, model, checkpoint loader,
or data reader. It neither estimates metrics nor resamples observations.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


SEEDS = (42, 123, 456)
ARMS = ("MLP16", "O", "UO", "RO", "RC")
CONTRASTS = {
    "Delta_pipe": "Pipeline: O − MLP16",
    "Delta_loss": "Loss: RC − O",
    "Delta_prior": "Prior: RO − UO",
    "Delta_response": "Response: RC − RO",
}
METRICS = {"brier": "Brier", "ce": "Cross-entropy"}
STYLE = {
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "savefig.dpi": 300,
}
SCOPE_STYLE = {"source": ("#0072B2", "o"), "unseen": ("#D55E00", "s")}
Row = dict[str, str]


def _table(root: Path, name: str) -> list[Row]:
    with (root / name).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Required saved table is empty: {name}")
    return rows


def _one(rows: list[Row], **keys: str) -> Row:
    selected = [row for row in rows if all(row.get(key) == value for key, value in keys.items())]
    if len(selected) != 1:
        raise ValueError(f"Expected one saved row for {keys}; found {len(selected)}.")
    return selected[0]


def _number(row: Row, column: str) -> float:
    value = float(row[column])
    if not math.isfinite(value):
        raise ValueError(f"Non-finite saved {column}: {row}")
    return value


def _summary(row: Row) -> None:
    if row.get("status") != "complete" or json.loads(row["seeds"]) != list(SEEDS):
        raise ValueError("Plots require the complete fixed three-seed summary; no partial mean is substituted.")
    for key in ("finite_seed_mean", "seed_sd", "lower", "upper"):
        _number(row, key)
    if _number(row, "seed_sd") < 0 or _number(row, "lower") > _number(row, "upper"):
        raise ValueError("Invalid saved dispersion or interval.")


def _prepare(root: Path, split: str) -> dict[str, Any]:
    if split not in {"validation", "test"}:
        raise ValueError("Choose validation or test explicitly; assessment is not a plotting population.")
    scopes = ("source",) if split == "validation" else ("source", "unseen")
    report = json.loads((root / "analysis.json").read_text(encoding="utf-8"))
    if report["bootstrap_repeats"] != 2000 or report["analysis_seed"] != 20260919:
        raise ValueError("Saved analysis does not use the frozen D1 bootstrap policy.")
    coverage = [row for row in report["coverage"] if row["split"] == split]
    if len(coverage) != 1 or not coverage[0]["complete_core"]:
        raise ValueError(f"Saved analysis does not contain complete core coverage for {split}.")
    inputs = json.loads((root / "inputs.json").read_text(encoding="utf-8"))
    conditions = inputs["condition_sets"][split]
    if any(not conditions.get(scope) for scope in scopes):
        raise ValueError(f"Missing explicit condition population for {scopes}.")
    contrasts = _table(root, "contrast_seed_summary.csv")
    paired = _table(root, "paired_contrasts.csv")
    seeds = _table(root, "seed_summary.csv")
    mechanism = _table(root, "mechanism.csv")
    for scope in scopes:
        for metric in METRICS:
            for contrast in CONTRASTS:
                _summary(_one(contrasts, split=split, scope=scope, metric=metric, contrast=contrast))
                for seed in SEEDS:
                    _number(_one(paired, split=split, scope=scope, metric=metric,
                                 contrast=contrast, seed=str(seed)), "value")
            for arm in ARMS:
                _summary(_one(seeds, split=split, scope=scope, metric=metric, arm=arm))
    adopted = [row for row in mechanism if row["split"] == split and row["role"] == "adopted"
               and row["predictor"] == "deployed" and row["scope"] in scopes]
    if split == "test" and not adopted:
        raise ValueError("Test adoption preview requires actual adopted mechanism rows; zero is not a substitute.")
    if adopted:
        if len({(row["name"], row["arm"], row["seed"], row["alpha"]) for row in adopted}) != 1:
            raise ValueError("Adoption preview requires one frozen selected predictor and coefficient.")
        for scope in scopes:
            row = _one(adopted, scope=scope)
            for key in ("A", "sqrt_A", "alpha", "alpha_sqrt_A"):
                if _number(row, key) < 0:
                    raise ValueError(f"Negative saved {key}.")
            if _number(row, "alpha") > 1:
                raise ValueError("Saved adoption coefficient is outside [0, 1].")
            if _number(row, "A") > 0:
                _number(row, "b_over_sqrt_A")
            elif row["b_over_sqrt_A"] != "":
                raise ValueError("A = 0 must retain an undefined direction-quality value, not zero.")
    return dict(split=split, scopes=scopes, report=report, conditions=conditions,
                contrasts=contrasts, paired=paired, seeds=seeds, adopted=adopted)


def _save(fig: Any, root: Path, stem: str) -> None:
    try:
        for extension in ("svg", "pdf", "png"):
            fig.savefig(root / f"{stem}.{extension}", dpi=300, bbox_inches="tight")
    finally:
        plt.close(fig)


def _contrast_figure(data: dict[str, Any], root: Path) -> None:
    scopes, split = data["scopes"], data["split"]
    fig, axes = plt.subplots(2, len(scopes), figsize=(5.3 * len(scopes), 6.6), squeeze=False)
    fig.subplots_adjust(left=.23 if len(scopes) == 1 else .16, right=.98, top=.90, bottom=.13,
                        hspace=.65, wspace=.65)
    for i, (metric, label) in enumerate(METRICS.items()):
        for j, scope in enumerate(scopes):
            ax = axes[i, j]
            color, marker = SCOPE_STYLE[scope]
            ax.axvline(0, color=".45", linewidth=.8, linestyle="--")
            for y, contrast in enumerate(CONTRASTS):
                row = _one(data["contrasts"], split=split, scope=scope, metric=metric, contrast=contrast)
                # Draw interval endpoints directly: a percentile interval need not contain its point estimate.
                ax.hlines(y, _number(row, "lower"), _number(row, "upper"), color=color, linewidth=1.5)
                ax.plot(_number(row, "finite_seed_mean"), y, marker=marker, color=color, markersize=5)
                for seed, offset in zip(SEEDS, (-.13, 0., .13)):
                    point = _one(data["paired"], split=split, scope=scope, metric=metric,
                                 contrast=contrast, seed=str(seed))
                    ax.plot(_number(point, "value"), y + offset, ".", color=".5", markersize=3, zorder=1)
            ax.set(yticks=range(len(CONTRASTS)), yticklabels=list(CONTRASTS.values()),
                   ylim=(3.5, -.5), xlabel=f"Δ {label} (treatment − control)",
                   title=f"{chr(65 + i * len(scopes) + j)}  {scope.capitalize()} · {label}")
            ax.grid(axis="x", alpha=.15)
    handles = [Line2D([], [], color="#0072B2", marker="o", label="Fixed three-seed mean; descriptive 95% group interval"),
               Line2D([], [], color=".5", marker=".", linestyle="none", label="Individual training-seed point")]
    fig.legend(handles=handles, loc="lower center", fontsize=8, frameon=False)
    fig.suptitle(f"D1 {split}: frozen main contrasts — diagnostic preview", fontsize=11)
    _save(fig, root, "main_contrasts")


def _seed_figure(data: dict[str, Any], root: Path) -> None:
    scopes, split = data["scopes"], data["split"]
    fig, axes = plt.subplots(2, len(scopes), figsize=(4.1 * len(scopes), 6.5), squeeze=False)
    fig.subplots_adjust(left=.17 if len(scopes) == 1 else .10, right=.97, top=.90, bottom=.11,
                        hspace=.60, wspace=.35)
    for i, (metric, label) in enumerate(METRICS.items()):
        for j, scope in enumerate(scopes):
            ax = axes[i, j]
            color, marker = SCOPE_STYLE[scope]
            for y, arm in enumerate(ARMS):
                row = _one(data["seeds"], split=split, scope=scope, metric=metric, arm=arm)
                ax.errorbar(_number(row, "finite_seed_mean"), y, xerr=_number(row, "seed_sd"),
                            fmt=marker, color=color, markersize=5, capsize=3, linewidth=1.2)
            ax.set(yticks=range(len(ARMS)), yticklabels=ARMS, ylim=(4.5, -.5), xlabel=label,
                   title=f"{chr(65 + i * len(scopes) + j)}  {scope.capitalize()} · {label}")
            ax.grid(axis="x", alpha=.15)
    fig.text(.5, .025, "Fixed training seeds 42, 123, 456: mean ± sample SD (not a confidence interval)",
             ha="center", fontsize=8)
    fig.suptitle(f"D1 {split}: direct-candidate seed dispersion — diagnostic preview", fontsize=11)
    _save(fig, root, "seed_dispersion")


def _adoption_figure(data: dict[str, Any], root: Path) -> None:
    columns = {"A": r"$A$", "sqrt_A": r"$\sqrt{A}$", "b_over_sqrt_A": r"$b/\sqrt{A}$",
               "alpha": r"$\alpha$", "alpha_sqrt_A": r"$\alpha\sqrt{A}$"}
    fig, axes = plt.subplots(1, len(columns), figsize=(11, 2.9))
    fig.subplots_adjust(left=.08, right=.99, top=.72, bottom=.23, wspace=.60)
    for ax, (column, label) in zip(axes, columns.items()):
        has_value = any(row[column] != "" for row in data["adopted"])
        if has_value:
            ax.axvline(0, color=".45", linewidth=.8, linestyle="--")
        for y, scope in enumerate(data["scopes"]):
            row = _one(data["adopted"], scope=scope)
            if column == "b_over_sqrt_A" and row[column] == "":
                ax.text(.05, y, "undefined\n(A = 0)", transform=ax.get_yaxis_transform(), fontsize=8, va="center")
                continue
            color, marker = SCOPE_STYLE[scope]
            ax.plot(_number(row, column), y, marker=marker, color=color, markersize=6)
        ax.set(yticks=range(len(data["scopes"])), yticklabels=data["scopes"] if ax is axes[0] else [],
               ylim=(len(data["scopes"]) - .5, -.5), xlabel=label)
        ax.tick_params(axis="x", labelsize=8)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(-2, 3), useOffset=False)
        if not has_value:
            ax.set_xticks([])
            ax.spines["bottom"].set_visible(False)
        if column == "alpha":
            ax.set_xlim(-.05, 1.05)
    selected = data["adopted"][0]
    fig.suptitle(f"D1 {data['split']}: adoption mechanism — diagnostic preview\n"
                 f"Selected {selected['arm']}, seed {selected['seed']}, α = {selected['alpha']}", fontsize=11)
    fig.text(.5, .035, "Saved point estimates; undefined direction is omitted. No fitted curve or uncertainty is inferred.",
             ha="center", fontsize=8)
    _save(fig, root, "adoption_mechanism")


def run(analysis: str | Path, output: str | Path, *, split: str) -> list[str]:
    """Plot complete saved tables into a new directory without changing them."""
    source, destination = Path(analysis).resolve(), Path(output)
    data = _prepare(source, split)
    destination.mkdir(parents=True, exist_ok=False)
    with plt.rc_context(STYLE):
        _contrast_figure(data, destination)
        _seed_figure(data, destination)
        if data["adopted"]:
            _adoption_figure(data, destination)
    population = "; ".join(f"{scope}: conditions {', '.join(map(str, data['conditions'][scope]))}"
                           for scope in data["scopes"])
    captions = ["# D1 diagnostic previews", "",
                "These are exploratory visual summaries of the frozen analysis, not publication-validated figures. "
                "No target-journal specification has been applied. No significance stars or ranking selection is used.", "",
                f"Source analysis directory: `{source}`. Partition: `{split}`. Population: {population}.", "",
                "Losses average windows within acquisitions, acquisitions equally within physical groups, "
                "and physical groups equally within conditions; named populations average their conditions equally. "
                "Plotting reads saved tables only and does not recompute metrics or bootstrap intervals.", "",
                "## main_contrasts", "",
                "Sources: `contrast_seed_summary.csv` and `paired_contrasts.csv`. Each marker is the saved finite "
                "three-seed mean of the labeled treatment-minus-control contrast; negative Brier or CE favors "
                "the treatment. Thin horizontal intervals are descriptive 95% paired global-physical-group "
                "bootstrap intervals (2,000 draws, analysis seed 20260919, stratified by condition-incidence "
                "mask, shared resampling across arms and training seeds). Small gray points are individual "
                "training-seed estimates, not independent physical specimens. The zero line denotes no contrast. "
                "Intervals are conditional on frozen predictors and can be unstable with few groups. "
                "Delta_pipe is a processing-pipeline contrast, not an isolated representation-causality claim.", "",
                "## seed_dispersion", "",
                "Source: `seed_summary.csv`, direct candidates only. Markers and bars are the saved mean ± sample "
                "SD across fixed training seeds 42, 123, and 456. SD describes training-seed dispersion; "
                "it is not a confidence interval and the seeds are not independent specimens.", ""]
    if data["adopted"]:
        captions.extend(["## adoption_mechanism", "",
                         "Source: `mechanism.csv`, actual adopted role and deployed predictor only. Panels show saved "
                         "A, sqrt(A), b/sqrt(A), alpha, and alpha sqrt(A) for the selected candidate relative to "
                         "the frozen reference. They are point estimates without inferred error bars. "
                         "When A = 0, b/sqrt(A) is undefined and omitted explicitly; alpha = 0 remains zero. "
                         "Selection and coefficient are already frozen; this figure selects neither.", ""])
    else:
        captions.extend(["No adoption figure: this source-validation analysis has no adopted predictor.", ""])
    (destination / "CAPTIONS.md").write_text("\n".join(captions), encoding="utf-8")
    return sorted(path.name for path in destination.iterdir())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", required=True, help="Completed analyze_d1 output directory")
    parser.add_argument("--output", required=True, help="New private diagnostic-preview directory")
    parser.add_argument("--split", required=True, choices=("validation", "test"))
    args = parser.parse_args()
    print(json.dumps(run(args.analysis, args.output, split=args.split), indent=2))


if __name__ == "__main__":
    main()
